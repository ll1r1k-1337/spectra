// @ts-check
import crypto from 'node:crypto';
import { SpectraLogger } from '../cli/logger.js';
import { cosineSimilarity } from '../ai/embeddings.js';

/**
 * @typedef {import('../../types/index.js').ISpectraRuntime} ISpectraRuntime
 * @typedef {import('../../types/index.js').IDirectedGraph} IDirectedGraph
 * @typedef {import('../../types/index.js').IToolRegistry} IToolRegistry
 * @typedef {import('../../types/index.js').IGraphDeterminismVerifier} IGraphDeterminismVerifier
 * @typedef {import('../../types/index.js').IStateOntologyManager} IStateOntologyManager
 * @typedef {import('../../types/index.js').ISpectraLogger} ISpectraLogger
 * @typedef {import('../../types/index.js').IPrecedentStore} IPrecedentStore
 * @typedef {import('../../types/index.js').ILocalEmbeddingEngine} ILocalEmbeddingEngine
 * @typedef {import('../../types/index.js').ILayaSystem1Arbiter} ILayaSystem1Arbiter
 * @typedef {import('../../types/index.js').ILangChainSystem2Arbiter} ILangChainSystem2Arbiter
 * @typedef {import('../../types/index.js').StateObject} StateObject
 * @typedef {import('../../types/index.js').DeadlockType} DeadlockType
 * @typedef {import('../../types/index.js').DeadlockIncident} DeadlockIncident
 * @typedef {import('../../types/index.js').MutationPlan} MutationPlan
 * @typedef {import('../../types/index.js').System1Verdict} System1Verdict
 * @typedef {import('../../types/index.js').SessionRunResult} SessionRunResult
 * @typedef {import('../../types/index.js').PrecedentRecord} PrecedentRecord
 */

/**
 * @implements {ISpectraRuntime}
 */
export class SpectraRuntime {
  /**
   * @param {{
   *   graph: IDirectedGraph;
   *   toolRegistry: IToolRegistry;
   *   verifier: IGraphDeterminismVerifier;
   *   ontology: IStateOntologyManager;
   *   logger?: ISpectraLogger;
   *   precedentStore?: IPrecedentStore | null;
   *   embedder?: ILocalEmbeddingEngine | null;
   *   system1Arbiter?: ILayaSystem1Arbiter | null;
   *   system2Arbiter?: ILangChainSystem2Arbiter | null;
   *   hostTools?: Map<string, (state: StateObject) => Promise<object>> | null;
   *   config?: Partial<import('../../types/index.js').SpectraConfig['runtime']>;
   * }} params
   */
  constructor({
    graph,
    toolRegistry,
    verifier,
    ontology,
    logger = new SpectraLogger(),
    precedentStore = null,
    embedder = null,
    system1Arbiter = null,
    system2Arbiter = null,
    hostTools = null,
    config = {}
  }) {
    this.graph = graph;
    this.toolRegistry = toolRegistry;
    this.verifier = verifier;
    this.ontology = ontology;
    this.logger = logger;
    this.precedentStore = precedentStore;
    this.embedder = embedder;
    this.system1Arbiter = system1Arbiter;
    this.system2Arbiter = system2Arbiter;
    this.hostTools = hostTools;
    this.maxSteps = config.hardStepLimit ?? 50;
    this.loopThreshold = config.stateHashLoopThreshold ?? 2;
    this.alphaPpr = config.pprTeleportAlpha ?? 0.15;
    /** @type {PrecedentRecord[]} */
    this.precedentMemory = [];
    this.llmCalls = 0;
  }

  /**
   * После ремонта CHECK_FAILED новый узел должен вернуть управление в узел-предшественник,
   * если тот (напр. persist_artifacts) читает ключи, которые производит новый узел, —
   * иначе запись на диск будет пропущена и проверка снова упадёт.
   * @param {string | null} predecessorNodeId
   * @param {string} failedNodeId
   * @param {string[]} newNodeWrites
   * @returns {string}
   */
  _resolveReverifyTarget(predecessorNodeId, failedNodeId, newNodeWrites) {
    const predecessor = predecessorNodeId ? this.graph.nodes.get(predecessorNodeId) : null;
    if (
      predecessor &&
      predecessor.type === 'tool' &&
      (predecessor.reads || []).some((k) => newNodeWrites.includes(k))
    ) {
      return predecessor.id;
    }
    return failedNodeId;
  }

  /**
   * Преемптивный guard: сработать, ПОКА новые результаты узла отсутствуют.
   * Строим только по ключам, которые новый узел пишет и которые читает check
   * (не инвертируем список ошибок вида syntaxErrors — «пустой список ошибок» = успех).
   * @param {string} checkNodeId
   * @param {string[]} newNodeWrites
   * @returns {string}
   */
  _buildPreemptiveGuard(checkNodeId, newNodeWrites) {
    const checkNode = this.graph.nodes.get(checkNodeId);
    const checkReads = checkNode?.reads && checkNode.reads.length > 0 ? checkNode.reads : ['artifacts'];
    let produceKeys = (newNodeWrites || []).filter((k) => checkReads.includes(k));
    if (produceKeys.length === 0) produceKeys = newNodeWrites && newNodeWrites.length > 0 ? newNodeWrites : checkReads;
    const readyExpr = produceKeys
      .map((k) => `(Array.isArray(state.${k}) ? state.${k}.length > 0 : Boolean(state.${k}))`)
      .join(' && ');
    return `!(${readyExpr})`;
  }

  /**
   * @param {string} nodeId
   * @param {StateObject} state
   */
  _hashNodeState(nodeId, state) {
    return crypto.createHash('sha256').update(JSON.stringify({ nodeId, state })).digest('hex');
  }

  /**
   * Определения 10 & 11: Гибридный поиск прецедента по косинусному сходству вектора инцидента v_inc
   * и реляционному заземлению Psi_adapt (индекс Жаккара по контрактам R, W) внутри каузального одеяла Маркова.
   * @param {DeadlockType} deadlockType
   * @param {string} centerNodeId
   * @param {number[]} queryVector
   * @param {PrecedentRecord[]} [candidatesPool]
   */
  findRelationallyGroundedPrecedent(deadlockType, centerNodeId, queryVector, candidatesPool) {
    const blanket = this.graph.getCausalMarkovBlanket(centerNodeId, this.alphaPpr);
    const pool = candidatesPool !== undefined ? candidatesPool : this.precedentMemory;
    let bestCandidate = null;
    let bestHybridScore = 0;

    for (const prec of pool) {
      if (prec.deadlockType !== deadlockType || prec.isAntipattern) continue;
      const vecSim =
        typeof prec.similarity === 'number'
          ? prec.similarity
          : prec.incidentVector && queryVector.length > 0
            ? cosineSimilarity(queryVector, prec.incidentVector)
            : 0;

      for (const candidateNodeId of blanket.keys()) {
        const node = this.graph.nodes.get(candidateNodeId);
        if (!node) continue;
        const rSet = new Set(node.reads.map((k) => this.ontology.findCanonical(k)));
        const wSet = new Set(node.writes.map((k) => this.ontology.findCanonical(k)));
        const anchorR = new Set((prec.anchorReads || []).map((k) => this.ontology.findCanonical(k)));
        const anchorW = new Set((prec.anchorWrites || []).map((k) => this.ontology.findCanonical(k)));

        const interR = [...rSet].filter((x) => anchorR.has(x)).length;
        const unionR = new Set([...rSet, ...anchorR]).size || 1;
        const interW = [...wSet].filter((x) => anchorW.has(x)).length;
        const unionW = new Set([...wSet, ...anchorW]).size || 1;

        const rScore = anchorR.size === 0 && rSet.size === 0 ? 1 : interR / unionR;
        const wScore = anchorW.size === 0 && wSet.size === 0 ? 1 : interW / unionW;
        const jaccard = 0.5 * rScore + 0.5 * wScore;
        const hybridScore = 0.5 * jaccard + 0.5 * Math.max(0, vecSim);

        if (jaccard >= 0.8 && vecSim >= 0.65 && hybridScore > bestHybridScore) {
          bestHybridScore = hybridScore;
          bestCandidate = {
            precedent: prec,
            targetNodeId: candidateNodeId,
            jaccard,
            vectorSimilarity: Number(vecSim.toFixed(4)),
            hybridScore: Number(hybridScore.toFixed(4))
          };
        }
      }
    }
    return bestCandidate;
  }

  /**
   * Выполняет откат состояния и курсора исполнения с вызовом Saga-компенсаторов в node:vm.
   * @param {Array<{ step: number; nodeId: string; stateBefore: StateObject; stateAfter: StateObject }>} trajectory
   * @param {StateObject} stateBeforeCurrentStep
   * @param {StateObject} currentState
   * @param {string} currentNodeId
   * @param {number} [stepsBack]
   * @param {Map<string, number>} [stateHashCounts]
   * @returns {{ restoredState: StateObject; restoredNodeId: string }}
   */
  rollbackTrajectoryAndCursor(
    trajectory,
    stateBeforeCurrentStep,
    currentState,
    currentNodeId,
    stepsBack = 1,
    stateHashCounts
  ) {
    this.logger.section('ROLLBACK', `Запуск Saga-отката траектории на ${stepsBack} шаг(ов) назад`);
    if (stateHashCounts) {
      stateHashCounts.delete(this._hashNodeState(currentNodeId, stateBeforeCurrentStep));
      stateHashCounts.delete(this._hashNodeState(currentNodeId, currentState));
    }

    let stateCursor = structuredClone(stateBeforeCurrentStep);
    let nodeCursor = currentNodeId;
    const minIdx = Math.max(0, trajectory.length - stepsBack);

    for (let i = trajectory.length - 1; i >= minIdx; i--) {
      const stepEntry = trajectory[i];
      const node = this.graph.nodes.get(stepEntry.nodeId);
      if (node?.hasBarrier) {
        this.logger.warn('ROLLBACK', `Остановка отката на узле "${node.id}": барьер необратимости Barrier(v)=1`);
        break;
      }
      if (node?.type === 'tool' && node.idempotency === 'non_idempotent') {
        const tool = this.toolRegistry.tools.get(node.config?.toolId || node.id);
        if (tool?.sagaCompensatorCode) {
          this.logger.info('ROLLBACK', `Вызов компенсирующего Saga-оператора Phi^-1 для узла "${node.id}"`);
          const compensatedDelta = this.toolRegistry.executeSandboxed(tool.sagaCompensatorCode, stepEntry.stateAfter);
          stateCursor = { ...stepEntry.stateAfter, ...compensatedDelta };
        } else {
          stateCursor = structuredClone(stepEntry.stateBefore);
        }
      } else {
        stateCursor = structuredClone(stepEntry.stateBefore);
      }
      nodeCursor = stepEntry.nodeId;
      if (stateHashCounts) {
        const prevHash = this._hashNodeState(nodeCursor, stateCursor);
        stateHashCounts.delete(prevHash);
      }
      trajectory.pop();
      this.logger.info('ROLLBACK', `Курсор возвращен на узел "${nodeCursor}" (шаг ${stepEntry.step})`);
    }
    return { restoredState: stateCursor, restoredNodeId: nodeCursor };
  }

  /**
   * @param {StateObject} [initialState]
   * @returns {Promise<SessionRunResult>}
   */
  async runSession(initialState = {}) {
    if (!this.embedder || !this.system1Arbiter || !this.system2Arbiter) {
      throw new Error(
        'SpectraRuntime требует инициализированные боевые экземпляры embedder (@huggingface/transformers), ' +
          'system1Arbiter (@receptron/laya) и system2Arbiter (@langchain/openai).'
      );
    }

    const startTimeMs = performance.now();
    this.llmCalls = 0;

    if (this.precedentStore) {
      if (this.precedentMemory.length === 0) {
        const loaded = await this.precedentStore.loadAllPrecedents();
        this.precedentMemory.push(...loaded);
        if (loaded.length > 0) {
          this.logger.info('MEMORY', `Загружено ${loaded.length} выученных прецедентов из PostgreSQL pgvector`);
        }
      }
      const savedOntology = await this.precedentStore.loadOntologyKeys();
      if (savedOntology.length > 0) {
        this.ontology.loadFromJSON(savedOntology);
      }
    }

    let state = structuredClone(initialState);
    let currentNodeId = this.graph.startNode || 'start';
    /** @type {string | null} */
    let lastValidNodeId = null;
    /** @type {Array<{ step: number; nodeId: string; stateBefore: StateObject; stateAfter: StateObject }>} */
    const trajectory = [];
    /** @type {Map<string, number>} */
    const stateHashCounts = new Map();
    /** @type {SessionRunResult['appliedMutations']} */
    const appliedMutations = [];
    let rollbacksCount = 0;
    /** @type {Array<{ record: PrecedentRecord, deadlock: DeadlockIncident }>} */
    const pendingPrecedents = [];
    /** @type {Set<string>} */
    const repairedCheckNodes = new Set();

    this.logger.banner(
      `ЗАПУСК СЕССИИ SPECTRA (${this.logger.sessionId})`,
      `Цель: ${String(initialState.goal || 'N/A')} | StructHash: ${this.graph.computeStructuralHash().slice(0, 16)}`
    );
    this.logger.logTopologyAscii(this.graph, currentNodeId);

    for (let step = 0; step < this.maxSteps; step++) {
      const stateBefore = structuredClone(state);
      const h = this._hashNodeState(currentNodeId, stateBefore);
      stateHashCounts.set(h, (stateHashCounts.get(h) || 0) + 1);

      /** @type {DeadlockIncident | null} */
      let deadlock = null;
      if ((stateHashCounts.get(h) || 0) >= this.loopThreshold) {
        deadlock = {
          type: 'STATE_HASH_LOOP',
          nodeId: currentNodeId,
          lastValidNodeId,
          step,
          stateSnapshot: structuredClone(stateBefore)
        };
      }

      const node = this.graph.nodes.get(currentNodeId);
      const stepStartMs = performance.now();

      if (!deadlock && node) {
        try {
          if (node.type === 'check') {
            const assertFn =
              typeof node.config?.assertFn === 'function'
                ? node.config.assertFn
                : this.graph.compileGuardExpression(node.config?.assertExpr || 'true');
            const passed = Boolean(assertFn(state));
            this.logger.logStepExecution({
              step,
              maxSteps: this.maxSteps,
              node,
              durationMs: performance.now() - stepStartMs,
              stateBefore,
              stateAfter: state
            });
            if (!passed) {
              deadlock = {
                type: 'CHECK_FAILED',
                nodeId: currentNodeId,
                lastValidNodeId,
                step,
                error: `Контракт проверки в узле "${currentNodeId}" вернул false`,
                stateSnapshot: structuredClone(state)
              };
            }
          } else if (node.type === 'tool') {
            const toolId = node.config?.toolId || node.id;
            const tool = this.toolRegistry.tools.get(toolId);
            let rawOut = state;
            const hostTool = this.hostTools ? this.hostTools.get(toolId) : undefined;
            if (hostTool) {
              rawOut = /** @type {StateObject} */ (await hostTool(state));
            } else if (tool?.sourceCode) {
              rawOut = this.toolRegistry.executeSandboxed(tool.sourceCode, state);
            } else if (typeof node.config?.sourceCode === 'string') {
              rawOut = this.toolRegistry.executeSandboxed(node.config.sourceCode, state);
            } else if (typeof node.config?.handler === 'function') {
              rawOut = await node.config.handler(state);
            }
            state =
              rawOut && typeof rawOut === 'object' ? { ...stateBefore, ...rawOut } : structuredClone(stateBefore);

            for (const wKey of node.writes) {
              this.ontology.registerStagingKey(wKey, 0.25);
            }

            this.logger.logStepExecution({
              step,
              maxSteps: this.maxSteps,
              node,
              durationMs: performance.now() - stepStartMs,
              stateBefore,
              stateAfter: state
            });
          } else if (node.type === 'llm') {
            this.llmCalls++;
            const llmDelta =
              typeof node.config?.llmHandler === 'function'
                ? await node.config.llmHandler(state, node.modelTier || 'frontier')
                : await this.system2Arbiter.executeLlmNode(node, state, node.modelTier || 'frontier');
            state = { ...stateBefore, ...(llmDelta || {}) };
            for (const wKey of node.writes) {
              this.ontology.registerStagingKey(wKey, 0.25);
            }
            this.logger.logStepExecution({
              step,
              maxSteps: this.maxSteps,
              node,
              durationMs: performance.now() - stepStartMs,
              stateBefore,
              stateAfter: state
            });
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          this.logger.error('FAST_LOOP', `Ошибка выполнения узла "${currentNodeId}": ${message}`);
          deadlock = {
            type: 'TOOL_EXECUTION_ERROR',
            nodeId: currentNodeId,
            lastValidNodeId,
            step,
            error: message,
            stateSnapshot: structuredClone(stateBefore)
          };
        }
      }

      if (!deadlock && this.graph.goalNodes.has(currentNodeId)) {
        const ver = this.verifier.verify(this.graph);
        if (this.precedentStore) {
          await this.precedentStore.saveOntologyKeys(this.ontology.toJSON());
        }
        if (this.precedentStore && pendingPrecedents.length > 0) {
          for (const p of pendingPrecedents) {
            await this.precedentStore.savePrecedent(p.record, p.deadlock);
          }
          await this.precedentStore.saveGraphVersion(
            this.graph.toJSON(),
            this.graph.computeStructuralHash(),
            ver.omegaPenalty
          );
        }
        const durationMs = Number((performance.now() - startTimeMs).toFixed(2));
        this.logger.banner(
          `🏁 ЦЕЛЬ ДОСТИГНУТА НА УЗЛЕ "${currentNodeId}" ЗА ${step + 1} ШАГ(ОВ)`,
          `Время: ${durationMs} мс | Мутаций: ${appliedMutations.length} | Штраф Ω(G): ${ver.omegaPenalty}`
        );
        return {
          sessionId: this.logger.sessionId,
          goalReached: true,
          steps: step + 1,
          rollbacksCount,
          finalState: state,
          omegaPenalty: ver.omegaPenalty,
          durationMs,
          appliedMutations,
          llmCalls: this.llmCalls
        };
      }

      if (!deadlock) {
        const outgoing = this.graph.edges.get(currentNodeId) || [];
        const evaluations = outgoing.map((e) => {
          let matched = false;
          try {
            matched = Boolean(e.guard(state));
          } catch {
            matched = false;
          }
          return {
            to: e.to,
            priority: e.priority,
            guardExpr: e.guardExpr,
            matched
          };
        });

        this.logger.logGuardEvaluation(currentNodeId, evaluations);
        const activeEdges = evaluations.filter((ev) => ev.matched);

        if (activeEdges.length === 0) {
          deadlock = {
            type: 'NO_VALID_TRANSITION',
            nodeId: currentNodeId,
            lastValidNodeId,
            step,
            error: `Ни одно из ${outgoing.length} исходящих ребер из "${currentNodeId}" не активно`,
            stateSnapshot: structuredClone(state)
          };
        } else {
          trajectory.push({
            step,
            nodeId: currentNodeId,
            stateBefore,
            stateAfter: structuredClone(state)
          });
          lastValidNodeId = currentNodeId;
          const nextNodeId = activeEdges[0].to;
          this.logger.info(
            'ROUTING',
            `Переход по ребру: "${currentNodeId}" ──► "${nextNodeId}" (priority=${activeEdges[0].priority})`
          );
          currentNodeId = nextNodeId;
          continue;
        }
      }

      // ======================================================================
      // КОНТУР 2: SLOW LOOP (Мета-уровень восстановления при тупике)
      // ======================================================================
      const blanket = this.graph.getCausalMarkovBlanket(currentNodeId, this.alphaPpr);
      this.logger.logDeadlock(deadlock, blanket, h);
      this.logger.section('SLOW_LOOP', 'Векторизация инцидента, поиск прецедентов в pgvector и синтез восстановления');

      const incidentVector = await this.embedder.computeIncidentVector({
        errorText: `${deadlock.type}: ${deadlock.error || ''}`,
        markovBlanketSignature: Array.from(blanket.keys()).join('->'),
        stateSchemaSignature: Object.keys(state).sort().join(',')
      });
      this.logger.debug('EMBEDDINGS', `Сформирован мультимодальный вектор тупика v_inc (dim=${incidentVector.length})`);

      const vectorCandidates = this.precedentStore
        ? await this.precedentStore.searchSimilarPrecedents(deadlock.type, incidentVector, 0.65)
        : undefined;

      const grounded = repairedCheckNodes.has(currentNodeId)
        ? null
        : this.findRelationallyGroundedPrecedent(
            deadlock.type,
            currentNodeId,
            incidentVector,
            vectorCandidates
          );
      /** @type {'MEMORY_GROUNDED' | 'SYSTEM2_SYNTHESIZED'} */
      let mutationSource = 'SYSTEM2_SYNTHESIZED';
      /** @type {MutationPlan} */
      let mutationPlan;

      if (grounded) {
        mutationSource = 'MEMORY_GROUNDED';
        this.logger.info(
          'MEMORY',
          `Найден прецедент в pgvector! Узел привязки Psi_adapt="${grounded.targetNodeId}" (CosSim=${grounded.vectorSimilarity}, Jaccard=${grounded.jaccard.toFixed(2)}, Hybrid=${grounded.hybridScore})`
        );
        mutationPlan = structuredClone(grounded.precedent.mutationPlan);
        if (deadlock.type === 'CHECK_FAILED') {
          mutationPlan.from = deadlock.lastValidNodeId || grounded.targetNodeId;
          mutationPlan.to = grounded.targetNodeId;
        } else {
          const goalNodeId = Array.from(this.graph.goalNodes)[0] || 'goal';
          mutationPlan.from = grounded.targetNodeId;
          if (mutationPlan.to === grounded.targetNodeId) {
            mutationPlan.to = goalNodeId;
          }
        }
        if (mutationPlan.node && this.graph.nodes.has(mutationPlan.node.id)) {
          mutationPlan.node.id = `${mutationPlan.node.id}_g${step}`;
        }
      } else {
        const goalNodeId = Array.from(this.graph.goalNodes)[0] || 'goal';
        this.logger.section(
          'SYSTEM_2',
          `Вызов LLM (${this.system2Arbiter.config.frontierModel}) для синтеза плана восстановления`
        );
        this.llmCalls++;
        mutationPlan = await this.system2Arbiter.synthesizeRecoveryMutation(
          deadlock,
          String(state.goal || ''),
          Array.from(this.graph.nodes.keys()),
          goalNodeId
        );
      }

      this.logger.info(
        'SLOW_LOOP',
        `Кандидат мутации [${mutationSource}]: operator=${mutationPlan.operator} (${mutationPlan.from} -> ${mutationPlan.to})`,
        {
          reasoning: mutationPlan.reasoning || 'N/A',
          newNodeId: mutationPlan.node?.id
        }
      );

      const s1Verdict = await this.system1Arbiter.evaluateMutation(
        deadlock,
        mutationPlan,
        String(state.goal || '')
      );
      this.logger.logSystem1Verdict(s1Verdict, {
        maxDrift: this.system1Arbiter.config.maxDrift,
        minCoherence: this.system1Arbiter.config.minCoherence,
        minApproveProb: this.system1Arbiter.config.minApproveProb
      });

      if (!s1Verdict.approved) {
        if (grounded) {
          grounded.precedent.isAntipattern = true;
        }
        this.logger.warn('SYSTEM_1', 'Мутация отклонена гейтом @receptron/laya! Выполнение отката...');
        const rb = this.rollbackTrajectoryAndCursor(
          trajectory,
          stateBefore,
          state,
          currentNodeId,
          1,
          stateHashCounts
        );
        state = rb.restoredState;
        currentNodeId = rb.restoredNodeId;
        lastValidNodeId = trajectory.length > 0 ? trajectory[trajectory.length - 1].nodeId : null;
        rollbacksCount++;
        continue;
      }

      if (deadlock.type === 'CHECK_FAILED' && mutationPlan.node && mutationPlan.node.type !== 'llm') {
        const checkReads = node?.reads && node.reads.length > 0 ? node.reads : ['artifacts'];
        const forged = checkReads.filter((k) => (mutationPlan.node?.writes || []).includes(k));
        if (forged.length > 0) {
          this.logger.warn('SYSTEM_1', `Мутация отклонена: узел типа "${mutationPlan.node.type}" подделывает ключ(и) проверки [${forged.join(', ')}] вместо реальной работы. Откат...`);
          const rb = this.rollbackTrajectoryAndCursor(
            trajectory,
            stateBefore,
            state,
            currentNodeId,
            1,
            stateHashCounts
          );
          state = rb.restoredState;
          currentNodeId = rb.restoredNodeId;
          lastValidNodeId = trajectory.length > 0 ? trajectory[trajectory.length - 1].nodeId : null;
          rollbacksCount++;
          continue;
        }
      }

      const candidateGraph = this.graph.clone();
      const compiledGuard = candidateGraph.compileGuardExpression(mutationPlan.guardExpr);
      const pendingToolCode = mutationPlan.toolCode || mutationPlan.node?.config?.sourceCode;

      if (mutationPlan.operator === 'addEdge') {
        candidateGraph.addEdge(
          mutationPlan.from,
          mutationPlan.to,
          mutationPlan.guardFn || compiledGuard,
          mutationPlan.priority ?? 0,
          { expr: mutationPlan.guardExpr || 'true', intervals: mutationPlan.intervals, preemptive: true }
        );
      } else if (
        (mutationPlan.operator === 'addNodeAndBridge' || mutationPlan.operator === 'registerTool') &&
        mutationPlan.node
      ) {
        if (pendingToolCode) {
          mutationPlan.node.config = { ...mutationPlan.node.config, sourceCode: pendingToolCode };
        }
        candidateGraph.addNode(mutationPlan.node);

        if (deadlock.type === 'CHECK_FAILED' && deadlock.lastValidNodeId) {
          const newNodeWrites = mutationPlan.node.writes || [];
          const preGuardExpr = this._buildPreemptiveGuard(deadlock.nodeId, newNodeWrites);
          const reverifyTargetId = this._resolveReverifyTarget(
            deadlock.lastValidNodeId,
            deadlock.nodeId,
            newNodeWrites
          );
          candidateGraph.addEdge(
            deadlock.lastValidNodeId,
            mutationPlan.node.id,
            candidateGraph.compileGuardExpression(preGuardExpr),
            0,
            { expr: preGuardExpr, preemptive: true }
          );
          candidateGraph.addEdge(mutationPlan.node.id, reverifyTargetId, () => true, 1, {
            expr:
              reverifyTargetId === deadlock.nodeId
                ? 'true /* reverify_check */'
                : `true /* re-run ${reverifyTargetId} */`
          });
        } else {
          const goalNodeId = Array.from(candidateGraph.goalNodes)[0] || 'goal';
          const safeTarget = mutationPlan.to === mutationPlan.from ? goalNodeId : mutationPlan.to;
          candidateGraph.addEdge(mutationPlan.from, mutationPlan.node.id, () => true, 0, {
            expr: 'true /* remediation_bridge */',
            preemptive: true
          });
          candidateGraph.addEdge(
            mutationPlan.node.id,
            safeTarget,
            mutationPlan.guardFn || compiledGuard,
            1,
            { expr: mutationPlan.guardExpr || 'true' }
          );
        }
      }

      this.logger.section('VERIFIER', 'Проверка мутированного графа G_cand статическим гейткипером');
      const checkResult = this.verifier.verify(candidateGraph);
      const structHash = candidateGraph.computeStructuralHash();
      this.logger.logVerifierReport(checkResult, structHash);

      if (checkResult.status === 'PASS') {
        if (mutationPlan.node && pendingToolCode) {
          this.toolRegistry.register({
            id: mutationPlan.node.id,
            reads: mutationPlan.node.reads,
            writes: mutationPlan.node.writes,
            sourceCode: pendingToolCode
          });
        }
        this.graph = candidateGraph;
        appliedMutations.push({
          deadlock,
          mutationPlan,
          omegaPenalty: checkResult.omegaPenalty,
          source: mutationSource
        });

        /** @type {PrecedentRecord} */
        const precedentRecord = {
          deadlockType: deadlock.type,
          incidentVector,
          anchorReads: node?.reads || [],
          anchorWrites: node?.writes || [],
          mutationPlan,
          isAntipattern: false
        };
        if (mutationSource === 'SYSTEM2_SYNTHESIZED') {
          this.precedentMemory.push(precedentRecord);
        }
        if (this.precedentStore && mutationSource === 'SYSTEM2_SYNTHESIZED') {
          pendingPrecedents.push({ record: precedentRecord, deadlock });
        }
        if (this.precedentStore) {
          await this.precedentStore.saveOntologyKeys(this.ontology.toJSON());
        }

        const prevCursor = currentNodeId;
        if (
          (mutationPlan.operator === 'addNodeAndBridge' || mutationPlan.operator === 'registerTool') &&
          mutationPlan.node
        ) {
          currentNodeId = mutationPlan.node.id;
        } else if (mutationPlan.operator === 'addEdge' && mutationPlan.from === currentNodeId) {
          currentNodeId = mutationPlan.to;
        }
        stateHashCounts.clear();
        if (deadlock.type === 'CHECK_FAILED') {
          repairedCheckNodes.add(deadlock.nodeId);
        }

        this.logger.info(
          'SLOW_LOOP',
          `Мутация принята! Курсор исполнения переведен: "${prevCursor}" ──► "${currentNodeId}"`
        );
        this.logger.logTopologyAscii(this.graph, currentNodeId);
      } else {
        if (grounded) {
          grounded.precedent.isAntipattern = true;
        }
        this.logger.error('VERIFIER', 'Верификатор заблокировал мутацию (HARD_REJECT). Выполнение отката...');
        const rb = this.rollbackTrajectoryAndCursor(
          trajectory,
          stateBefore,
          state,
          currentNodeId,
          1,
          stateHashCounts
        );
        state = rb.restoredState;
        currentNodeId = rb.restoredNodeId;
        lastValidNodeId = trajectory.length > 0 ? trajectory[trajectory.length - 1].nodeId : null;
        rollbacksCount++;
      }
    }

    const durationMs = Number((performance.now() - startTimeMs).toFixed(2));
    this.logger.error('FAST_LOOP', `Достигнут жесткий лимит шагов сессии T_max=${this.maxSteps}`);
    return {
      sessionId: this.logger.sessionId,
      goalReached: false,
      reason: 'HARD_LIMIT_EXCEEDED',
      steps: this.maxSteps,
      rollbacksCount,
      finalState: state,
      durationMs,
      appliedMutations,
      llmCalls: this.llmCalls
    };
  }

  /**
   * Контур 3 (Consolidation Loop / «Фаза сна»).
   */
  async runConsolidationSleep() {
    this.logger.section('SLEEP', 'Запуск Контура 3 (Consolidation Loop / Фаза сна)');
    if (this.precedentStore && this.ontology.keys.size === 0) {
      const savedOntology = await this.precedentStore.loadOntologyKeys();
      this.ontology.loadFromJSON(savedOntology);
    }

    const omegaBefore = this.verifier.verify(this.graph).omegaPenalty;
    const nodesBefore = this.graph.nodes.size;

    const { promoted: promotedKeys, evicted: evictedKeys } = this.ontology.runStagingGarbageCollection(0.1);
    this.graph.foldBisimilarBranches();

    const nodesAfter = this.graph.nodes.size;
    const foldedNodes = nodesBefore - nodesAfter;
    const verAfter = this.verifier.verify(this.graph);
    const omegaAfter = verAfter.omegaPenalty;

    if (this.precedentStore) {
      await this.precedentStore.saveGraphVersion(
        this.graph.toJSON(),
        this.graph.computeStructuralHash(),
        omegaAfter
      );
      await this.precedentStore.saveOntologyKeys(this.ontology.toJSON());
    }

    this.logger.info('SLEEP', 'Фаза консолидации завершена', {
      promotedCanonicalKeys: promotedKeys,
      evictedStagingKeys: evictedKeys,
      foldedBisimilarNodes: foldedNodes,
      omegaBefore,
      omegaAfter
    });
    return { promotedKeys, evictedKeys, foldedNodes, omegaBefore, omegaAfter };
  }
}
