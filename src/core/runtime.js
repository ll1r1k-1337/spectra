import crypto from 'node:crypto';
import { DirectedGraph } from './graph.js';
import { GraphDeterminismVerifier } from './verifier.js';
import { ToolRegistry } from './tools.js';
import { StateOntologyManager } from './ontology.js';
import { logger } from '../utils/logger.js';

export { DirectedGraph } from './graph.js';
export { GraphDeterminismVerifier } from './verifier.js';
export { ToolRegistry } from './tools.js';
export { StateOntologyManager } from './ontology.js';

/**
 * @typedef {import('#types').StateObject} StateObject
 * @typedef {import('#types').DeadlockIncident} DeadlockIncident
 * @typedef {import('#types').MutationPlan} MutationPlan
 * @typedef {import('#types').System1Verdict} System1Verdict
 * @typedef {import('#types').SessionRunResult} SessionRunResult
 * @typedef {import('#types').ModelTier} ModelTier
 * @typedef {import('#types').ISpectraRuntime} ISpectraRuntime
 */

/**
 * Section 5: Retroactive audit correction formula for offline policy evaluation.
 *
 * @param {number} oldDeltaU
 * @param {number} tdWeight
 * @param {number} rMax
 * @param {number} yExtActual
 * @param {number} qSemOld
 * @returns {number}
 */
export function applyRetroactiveAuditCorrection(oldDeltaU, tdWeight, rMax, yExtActual, qSemOld) {
  return oldDeltaU + tdWeight * rMax * (yExtActual - qSemOld);
}

/**
 * Section 5: Weighted Conformal Doubly Robust Offline Policy Evaluation (WC-LCB-DR-OPE).
 *
 * @param {Array<{ criticPrediction?: number, wasExecuted?: boolean, conformalQuantile?: number, sqDistToCalib?: number[], propensityScore?: number, retroCorrectedDeltaU?: number, actualDeltaU?: number }>} observations
 * @param {number[]} [calibrationResiduals=[]]
 * @param {number} [wMax=10.0]
 * @param {number} [xiOod=1.0]
 * @param {number} [alphaConf=0.1]
 * @param {number} [gammaRbf=2.0]
 * @returns {number}
 */
export function computeWeightedConformalDoublyRobustUtility(
  observations,
  calibrationResiduals = [],
  wMax = 10.0,
  xiOod = 1.0,
  alphaConf = 0.1,
  gammaRbf = 2.0
) {
  if (!observations.length) return 0.0;
  let sum = 0;

  for (const obs of observations) {
    const rHat = obs.criticPrediction ?? 0.0;
    const executed = obs.wasExecuted ? 1.0 : 0.0;

    let qConf = obs.conformalQuantile ?? 0.0;
    if (!obs.wasExecuted && calibrationResiduals.length > 0 && obs.sqDistToCalib) {
      const sqDist = obs.sqDistToCalib;
      const weighted = calibrationResiduals
        .map((r, idx) => ({
          r,
          w: Math.exp(-gammaRbf * (sqDist[idx] ?? 0.0))
        }))
        .sort((a, b) => a.r - b.r);
      const totalW = weighted.reduce((acc, item) => acc + item.w, 0) + 1e-9;
      let cumW = 0;
      for (const item of weighted) {
        cumW += item.w / totalW;
        if (cumW >= 1.0 - alphaConf) {
          qConf = item.r;
          break;
        }
      }
    }

    const oodPenalty = (1.0 - executed) * xiOod * qConf;
    const weight = Math.min(wMax, executed / Math.max(1e-4, obs.propensityScore ?? 1.0));
    const actualDeltaU = obs.retroCorrectedDeltaU ?? obs.actualDeltaU ?? 0.0;
    sum += rHat - oodPenalty + weight * (actualDeltaU - rHat);
  }
  return sum / observations.length;
}

/**
 * Section 5: Dual Lagrangian resource price updates under RPS and latency budget constraints.
 *
 * @param {Record<string, number>} prevDuals
 * @param {Record<string, number>} currentUsage
 * @param {Record<string, number>} targetBudgets
 * @param {number} [etaDual=0.05]
 * @returns {Record<string, number>}
 */
export function updateLagrangianResourcePrices(prevDuals, currentUsage, targetBudgets, etaDual = 0.05) {
  /** @type {Record<string, number>} */
  const nextDuals = {};
  for (const key of Object.keys(targetBudgets)) {
    const prev = prevDuals[key] ?? 0.0;
    const diff = (currentUsage[key] ?? 0.0) - targetBudgets[key];
    nextDuals[key] = Math.max(0.0, prev + etaDual * diff);
  }
  return nextDuals;
}

/**
 * Section 5: Model tier selection with token shadow price adaptation.
 *
 * @param {Record<string, { passRate?: number, lcbPassRate?: number }>} tierStats
 * @param {number} [epsilonTier=0.02]
 * @param {number} [tokenShadowPrice=1.0]
 * @returns {ModelTier}
 */
export function selectModelTierForNode(tierStats, epsilonTier = 0.02, tokenShadowPrice = 1.0) {
  const frontierPass = tierStats.frontier?.passRate ?? 0.98;
  const effectiveTolerance = Math.min(0.08, epsilonTier * Math.max(1.0, tokenShadowPrice));
  const threshold = frontierPass - effectiveTolerance;
  if ((tierStats.local_slm?.lcbPassRate ?? 0) >= threshold) return 'local_slm';
  if ((tierStats.light?.lcbPassRate ?? 0) >= threshold) return 'light';
  return 'frontier';
}

/**
 * Section 5: Calibrated CUSUM change detector with dynamic discount factor.
 *
 * @param {number} prevCusum
 * @param {boolean} isSuccess
 * @param {number} [p0=0.85]
 * @param {number} [nu=0.05]
 * @param {number} [threshold=3.5]
 * @param {number} [lambda=0.4]
 * @param {number} [criticEce=0.02]
 * @param {number} [epsilonCritic=0.08]
 * @returns {{ cusumStat: number, discountFactor: number, calibrationWeight: number }}
 */
export function updateCusumDiscountFactor(
  prevCusum,
  isSuccess,
  p0 = 0.85,
  nu = 0.05,
  threshold = 3.5,
  lambda = 0.4,
  criticEce = 0.02,
  epsilonCritic = 0.08
) {
  const wCal = Math.max(0.0, Math.min(1.0, 1.0 - criticEce / epsilonCritic));
  const residual = wCal * (p0 - (isSuccess ? 1.0 : 0.0));
  const nextCusum = Math.max(0, prevCusum + residual - nu);
  const discount = Math.exp(-lambda * Math.max(0, nextCusum - threshold));
  return { cusumStat: nextCusum, discountFactor: discount, calibrationWeight: wCal };
}

/**
 * Section 5: Paired Common Random Numbers (CRN) Simultaneous Perturbation Stochastic Approximation (SPSA).
 *
 * @param {Record<string, number>} currentPsi
 * @param {Record<string, number>} prevMomentum
 * @param {number[]} pairedOutcomesPlus
 * @param {number[]} pairedOutcomesMinus
 * @param {Record<string, number>} deltaVec
 * @param {number} [cStep=0.05]
 * @param {number} [etaMeta=0.05]
 * @param {number} [betaMom=0.9]
 * @param {Record<string, [number, number]>} [bounds={}]
 * @returns {{ nextPsi: Record<string, number>, nextMomentum: Record<string, number> }}
 */
export function computePairedCrnSpsaStep(
  currentPsi,
  prevMomentum,
  pairedOutcomesPlus,
  pairedOutcomesMinus,
  deltaVec,
  cStep = 0.05,
  etaMeta = 0.05,
  betaMom = 0.9,
  bounds = {}
) {
  const n = Math.max(1, pairedOutcomesPlus.length);
  let avgDiff = 0;
  for (let i = 0; i < n; i++) {
    avgDiff += (pairedOutcomesPlus[i] - pairedOutcomesMinus[i]) / (2 * cStep);
  }
  avgDiff /= n;

  /** @type {Record<string, number>} */
  const nextMomentum = {};
  /** @type {Record<string, number>} */
  const nextPsi = {};
  for (const key of Object.keys(currentPsi)) {
    const gradComp = avgDiff / (deltaVec[key] || 1);
    const m = betaMom * (prevMomentum[key] ?? 0) + (1 - betaMom) * gradComp;
    nextMomentum[key] = m;
    const raw = currentPsi[key] + etaMeta * m;
    const [low, high] = bounds[key] || [-Infinity, Infinity];
    nextPsi[key] = Math.min(high, Math.max(low, raw));
  }
  return { nextPsi, nextMomentum };
}

/**
 * @param {StateObject} before
 * @param {StateObject} after
 * @returns {Record<string, { before: any, after: any }>}
 */
function computeStateDiff(before, after) {
  /** @type {Record<string, { before: any, after: any }>} */
  const diff = {};
  const allKeys = new Set([...Object.keys(before || {}), ...Object.keys(after || {})]);
  for (const k of allKeys) {
    if (JSON.stringify(before?.[k]) !== JSON.stringify(after?.[k])) {
      diff[k] = { before: before?.[k], after: after?.[k] };
    }
  }
  return diff;
}

/**
 * SpectraRuntime orchestrates the Fast Loop execution, deadlock detection across all 5 classes,
 * Slow Loop recovery with relational grounding, System-1 gate, System-2 synthesis,
 * candidate verifier gating, and Saga rollback.
 *
 * @implements {ISpectraRuntime}
 */
export class SpectraRuntime {
  /**
   * @param {{
   *   graph: DirectedGraph,
   *   toolRegistry?: ToolRegistry,
   *   verifier?: GraphDeterminismVerifier,
   *   ontology?: StateOntologyManager,
   *   system1Evaluator?: ((deadlock: DeadlockIncident, mutationPlan: MutationPlan) => Promise<System1Verdict>) | null,
   *   system2Synthesizer?: ((deadlock: DeadlockIncident, state: StateObject, graph: DirectedGraph) => Promise<MutationPlan>) | null,
   *   config?: { hardStepLimit?: number, maxSteps?: number, stateHashLoopThreshold?: number, [key: string]: any }
   * }} params
   */
  constructor({
    graph,
    toolRegistry = new ToolRegistry(),
    verifier = new GraphDeterminismVerifier(),
    ontology = new StateOntologyManager(),
    system1Evaluator = null,
    system2Synthesizer = null,
    config = {}
  }) {
    /** @type {DirectedGraph} */
    this.graph = graph;
    /** @type {ToolRegistry} */
    this.toolRegistry = toolRegistry;
    /** @type {GraphDeterminismVerifier} */
    this.verifier = verifier;
    /** @type {StateOntologyManager} */
    this.ontology = ontology;
    /** @type {number} */
    this.maxSteps = config.hardStepLimit ?? config.maxSteps ?? 50;
    /** @type {number} */
    this.loopThreshold = config.stateHashLoopThreshold ?? 2;
    /** @type {Array<{ deadlockType: string, anchorReads: string[], anchorWrites: string[], mutationPlan: MutationPlan, isAntipattern?: boolean }>} */
    this.precedentMemory = [];

    /** @type {(deadlock: DeadlockIncident, mutationPlan: MutationPlan) => Promise<System1Verdict>} */
    this.system1Evaluator =
      system1Evaluator ||
      (async () => ({
        driftRisk: 0.12,
        coherence: 0.88,
        approveProb: 0.91,
        approved: true,
        rollbackAction: 'keep'
      }));

    /** @type {(deadlock: DeadlockIncident, state: StateObject, graph: DirectedGraph) => Promise<MutationPlan>} */
    this.system2Synthesizer =
      system2Synthesizer ||
      (async (deadlockCtx) => ({
        operator: 'addEdge',
        from: deadlockCtx.nodeId,
        to: Array.from(this.graph.goalNodes)[0] || 'goal',
        guardFn: (s) => Boolean(s?.goalCompleted),
        guardExpr: '(s) => Boolean(s.goalCompleted)',
        priority: 99
      }));
  }

  /**
   * Deterministic state hash for loop detection.
   *
   * @param {string} nodeId
   * @param {StateObject} state
   * @returns {string} SHA-256 hash hex
   */
  _hashNodeState(nodeId, state) {
    return crypto
      .createHash('sha256')
      .update(JSON.stringify({ nodeId, state }))
      .digest('hex');
  }

  /**
   * Definition 11: Relational grounding of precedent inside Causal Markov Blanket
   * using ontology canonical names and Jaccard read/write overlap.
   *
   * @param {string} deadlockType
   * @param {string} centerNodeId
   * @returns {{ precedent: any, targetNodeId: string, jaccard: number } | null}
   */
  findRelationallyGroundedPrecedent(deadlockType, centerNodeId) {
    const blanket = this.graph.getCausalMarkovBlanket(centerNodeId);
    logger.debug('SlowLoop', `Searching precedent memory for deadlock "${deadlockType}" around node "${centerNodeId}"`, {
      centerNodeId,
      precedentMemoryCount: this.precedentMemory.length,
      blanketNodes: Array.from(blanket.keys())
    });
    /** @type {{ precedent: any, targetNodeId: string, jaccard: number } | null} */
    let bestCandidate = null;
    let bestScore = 0;

    for (const prec of this.precedentMemory) {
      if (prec.deadlockType !== deadlockType || prec.isAntipattern) continue;
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

        const jaccard = 0.5 * (interR / unionR) + 0.5 * (interW / unionW);
        if (jaccard >= 0.8 && jaccard > bestScore) {
          bestScore = jaccard;
          bestCandidate = { precedent: prec, targetNodeId: candidateNodeId, jaccard };
        }
      }
    }
    if (bestCandidate) {
      logger.info('SlowLoop', `Grounded precedent found with Jaccard score ${bestScore.toFixed(3)} for node "${bestCandidate.targetNodeId}"`, {
        targetNodeId: bestCandidate.targetNodeId,
        score: bestScore,
        precedentPlan: bestCandidate.precedent.mutationPlan
      });
    } else {
      logger.debug('SlowLoop', `No precedent memory match found for deadlock "${deadlockType}" above threshold 0.8`);
    }
    return bestCandidate;
  }

  /**
   * Definition 2: Rollback execution trajectory invoking Saga compensators
   * backwards up to the nearest irreversible barrier.
   *
   * @param {Array<{ step: number, nodeId: string, stateBefore: StateObject, stateAfter: StateObject }>} trajectory
   * @param {StateObject} currentState
   * @param {number} [stepsBack=1]
   * @returns {StateObject}
   */
  rollbackTrajectory(trajectory, currentState, stepsBack = 1) {
    let stateCursor = structuredClone(currentState);
    const minIdx = Math.max(0, trajectory.length - stepsBack);
    logger.warn('SlowLoop', `Rolling back trajectory by ${stepsBack} step(s) (trajectory length: ${trajectory.length})`, {
      fromStep: trajectory[trajectory.length - 1]?.step,
      targetMinStep: trajectory[minIdx]?.step
    });

    for (let i = trajectory.length - 1; i >= minIdx; i--) {
      const stepEntry = trajectory[i];
      const node = this.graph.nodes.get(stepEntry.nodeId);
      if (node?.hasBarrier) {
        logger.warn('SlowLoop', `Irreversible barrier encountered at node "${node.id}"; terminating trajectory rollback`);
        break;
      }
      if (node?.type === 'tool' && node.idempotency === 'non_idempotent') {
        const tool = this.toolRegistry.tools.get(node.config?.toolId || node.id);
        if (tool?.sagaCompensatorCode) {
          logger.info('SlowLoop', `Executing Saga compensator for non-idempotent tool "${tool.id}" at step ${stepEntry.step}`);
          stateCursor = this.toolRegistry.executeSandboxed(tool.sagaCompensatorCode, stateCursor);
        }
      } else {
        logger.debug('SlowLoop', `Restoring pre-execution state for step ${stepEntry.step} (node "${stepEntry.nodeId}")`);
        stateCursor = structuredClone(stepEntry.stateBefore);
      }
    }
    return stateCursor;
  }

  /**
   * Primary Fast Loop execution with integrated Slow Loop recovery.
   *
   * @param {StateObject} [initialState={}]
   * @returns {Promise<SessionRunResult>}
   */
  async runSession(initialState = {}) {
    let state = structuredClone(initialState);
    let currentNodeId = this.graph.startNode;
    /** @type {Array<{ step: number, nodeId: string, stateBefore: StateObject, stateAfter: StateObject }>} */
    const trajectory = [];
    /** @type {Map<string, number>} */
    const stateHashCounts = new Map();
    /** @type {Array<{ deadlock: DeadlockIncident, mutationPlan: MutationPlan, omegaPenalty: number }>} */
    const appliedMutations = [];
    let rollbacksCount = 0;

    for (let step = 0; step < this.maxSteps; step++) {
      logger.debug('FastLoop', `Step ${step}: current node "${currentNodeId}"`, { step, nodeId: currentNodeId });

      if (!currentNodeId) {
        logger.warn('FastLoop', `Terminated at step ${step}: null current node`);
        return {
          goalReached: false,
          reason: 'NO_VALID_TRANSITION',
          steps: step,
          rollbacksCount,
          finalState: state,
          appliedMutations
        };
      }

      const h = this._hashNodeState(currentNodeId, state);

      /** @type {DeadlockIncident | null} */
      let deadlock = null;

      if (this.graph.goalNodes.has(currentNodeId)) {
        const goalNode = this.graph.nodes.get(currentNodeId);
        const goalAssertSatisfied = typeof goalNode?.config?.assertFn === 'function'
          ? Boolean(goalNode.config.assertFn(state))
          : true;
        const goalContractSatisfied = Boolean(state.goalCompleted) === true && goalAssertSatisfied;

        if (goalContractSatisfied) {
          logger.info('FastLoop', `Goal node "${currentNodeId}" reached at step ${step}`);
          const ver = this.verifier.verify(this.graph);
          return {
            goalReached: true,
            steps: step,
            rollbacksCount,
            finalState: state,
            omegaPenalty: ver.omegaPenalty,
            appliedMutations
          };
        }

        logger.warn('FastLoop', `Goal contract violation on node "${currentNodeId}": goalCompleted is false or goal assertion failed`, {
          nodeId: currentNodeId,
          state
        });
        deadlock = {
          type: 'INVALID_GOAL_STATE',
          nodeId: currentNodeId,
          step,
          error: 'Goal contract violation: goalCompleted is false or goal assertion failed',
          stateSnapshot: structuredClone(state)
        };
      }

      if (!deadlock) {
        stateHashCounts.set(h, (stateHashCounts.get(h) || 0) + 1);

        const currentHashCount = stateHashCounts.get(h) || 0;
        if (currentHashCount >= (this.loopThreshold || 2)) {
          logger.warn('FastLoop', `State hash collision detected for node "${currentNodeId}" (hit count: ${currentHashCount})`, {
            nodeId: currentNodeId,
            count: currentHashCount,
            stateHash: h
          });
          deadlock = {
            type: 'STATE_HASH_LOOP',
            nodeId: currentNodeId,
            step,
            stateSnapshot: structuredClone(state)
          };
        }
      }

      const node = this.graph.nodes.get(currentNodeId);
      const stateBefore = structuredClone(state);

      if (!deadlock && node) {
        try {
          if (node.type === 'check' && typeof node.config?.assertFn === 'function') {
            if (!node.config.assertFn(state)) {
              logger.warn('FastLoop', `Check assertion failed on node "${currentNodeId}"`, {
                nodeId: currentNodeId,
                assertExpr: node.config?.assertExpr,
                state
              });
              deadlock = {
                type: 'CHECK_FAILED',
                nodeId: currentNodeId,
                step,
                stateSnapshot: structuredClone(state)
              };
            }
          } else if (node.type === 'tool') {
            const toolId = node.config?.toolId || node.id;
            const tool = this.toolRegistry.tools.get(toolId);
            if (tool?.sourceCode) {
              state = this.toolRegistry.executeSandboxed(tool.sourceCode, state);
            } else if (node.config?.sourceCode) {
              state = this.toolRegistry.executeSandboxed(node.config.sourceCode, state);
            } else if (typeof node.config?.handler === 'function') {
              state = await node.config.handler(state);
            }
          } else if (node.type === 'llm' && typeof node.config?.llmHandler === 'function') {
            /** @type {import('../../types/domain.js').ModelTier} */
            const tier = node.modelTier || 'frontier';
            state = await node.config.llmHandler(state, tier);
          }
        } catch (err) {
          logger.error('FastLoop', `Tool execution exception in node "${currentNodeId}": ${err instanceof Error ? err.message : String(err)}`, {
            nodeId: currentNodeId,
            error: err
          });
          deadlock = {
            type: 'TOOL_EXECUTION_ERROR',
            nodeId: currentNodeId,
            error: err instanceof Error ? err.message : String(err),
            step,
            stateSnapshot: structuredClone(state)
          };
        }
      }

      const stateDiff = computeStateDiff(stateBefore, state);
      if (Object.keys(stateDiff).length > 0) {
        logger.debug('FastLoop', `State updated by node "${currentNodeId}"`, { diff: stateDiff });
      }

      trajectory.push({ step, nodeId: currentNodeId, stateBefore, stateAfter: structuredClone(state) });

      if (!deadlock) {
        const outgoing = this.graph.edges.get(currentNodeId) || [];
        const evaluatedEdges = [];
        const activeEdges = [];

        for (const e of outgoing) {
          let passed = false;
          let guardErr = null;
          try {
            passed = Boolean(e.guard(state));
          } catch (err) {
            guardErr = err instanceof Error ? err.message : String(err);
          }
          evaluatedEdges.push({
            to: e.to,
            priority: e.priority,
            guardExpr: e.guardExpr,
            passed,
            error: guardErr
          });
          if (passed) {
            activeEdges.push(e);
          }
        }

        logger.debug('FastLoop', `Candidate edges evaluated for node "${currentNodeId}"`, { candidateEdges: evaluatedEdges });

        if (activeEdges.length === 0) {
          logger.warn('FastLoop', `Edge starvation: no valid transitions from node "${currentNodeId}"`, {
            nodeId: currentNodeId,
            outgoingCount: outgoing.length
          });
          deadlock = {
            type: 'NO_VALID_TRANSITION',
            nodeId: currentNodeId,
            step,
            stateSnapshot: structuredClone(state)
          };
        } else {
          const chosenEdge = activeEdges[0];
          logger.info('FastLoop', `Chosen edge: "${currentNodeId}" -> "${chosenEdge.to}" (priority: ${chosenEdge.priority})`);
          currentNodeId = chosenEdge.to;
          continue;
        }
      }

      // Slow Loop Recovery
      logger.info('SlowLoop', `Initiating Slow Loop recovery for deadlock "${deadlock.type}" on node "${currentNodeId}"`, {
        step,
        nodeId: currentNodeId,
        deadlock
      });

      const blanket = this.graph.getCausalMarkovBlanket(currentNodeId);
      logger.info('SlowLoop', `Causal Markov blanket computed for node "${currentNodeId}"`, {
        centerNodeId: currentNodeId,
        blanket: Object.fromEntries(blanket)
      });

      const grounded = this.findRelationallyGroundedPrecedent(deadlock.type, currentNodeId);
      /** @type {MutationPlan} */
      let mutationPlan;
      if (grounded) {
        mutationPlan = { ...grounded.precedent.mutationPlan, from: grounded.targetNodeId };
        logger.info('SlowLoop', `Using grounded precedent mutation plan: ${mutationPlan.operator}`, { mutationPlan });
      } else {
        logger.info('SlowLoop', `Synthesizing recovery plan via System-2 for deadlock "${deadlock.type}"`);
        mutationPlan = await this.system2Synthesizer(deadlock, state, this.graph);
        logger.info('SlowLoop', `System-2 synthesis generated plan: ${mutationPlan.operator}`, { mutationPlan });
      }

      logger.info('SlowLoop', `Evaluating proposed mutation plan with System-1 gate`, { mutationPlan });
      const s1Verdict = await this.system1Evaluator(deadlock, mutationPlan);
      logger.info('SlowLoop', `System-1 gate evaluation: ${s1Verdict.approved ? 'APPROVED' : 'REJECTED'} (P(approve)=${s1Verdict.approveProb}, driftRisk=${s1Verdict.driftRisk}, coherence=${s1Verdict.coherence})`, { verdict: s1Verdict });

      if (s1Verdict.approveProb < 0.7 || s1Verdict.driftRisk > 0.35) {
        logger.warn('SlowLoop', `System-1 rejected mutation plan; initiating trajectory rollback`, { verdict: s1Verdict });
        state = this.rollbackTrajectory(trajectory, state, 1);
        rollbacksCount++;
        continue;
      }

      const candidateGraph = this.graph.clone();
      let nextTargetNode = mutationPlan.to;
      if (mutationPlan.operator === 'addEdge') {
        const isTargetGoal = candidateGraph.goalNodes.has(mutationPlan.to) || this.graph.goalNodes.has(mutationPlan.to);
        const targetNode = (isTargetGoal && candidateGraph.nodes.has('verify_goal_state') && mutationPlan.from !== 'verify_goal_state')
          ? 'verify_goal_state'
          : mutationPlan.to;
        nextTargetNode = targetNode;
        const isFinalGoal = candidateGraph.goalNodes.has(targetNode) || this.graph.goalNodes.has(targetNode);
        const guardFn = isFinalGoal
          ? /** @param {StateObject} s */ (s) => Boolean(s?.goalCompleted)
          : (mutationPlan.guardFn || (() => true));
        const guardExpr = isFinalGoal
          ? '(s) => Boolean(s.goalCompleted)'
          : (mutationPlan.guardExpr || '() => true');

        candidateGraph.addEdge(
          mutationPlan.from,
          targetNode,
          mutationPlan.priority ?? 50,
          guardFn,
          guardExpr,
          mutationPlan.intervals || {}
        );
      } else if (mutationPlan.operator === 'addNodeAndBridge' && mutationPlan.node) {
        candidateGraph.addNode(mutationPlan.node);
        const existingOut = candidateGraph.edges.get(mutationPlan.from) || [];
        const maxPrio = existingOut.reduce((max, e) => Math.max(max, e.priority), 0);
        candidateGraph.addEdge(mutationPlan.from, mutationPlan.node.id, maxPrio + 1, () => true);

        const isTargetGoal = candidateGraph.goalNodes.has(mutationPlan.to) || this.graph.goalNodes.has(mutationPlan.to);
        const targetNode = (isTargetGoal && candidateGraph.nodes.has('verify_goal_state') && mutationPlan.node.id !== 'verify_goal_state')
          ? 'verify_goal_state'
          : mutationPlan.to;
        nextTargetNode = targetNode;
        const isFinalGoal = candidateGraph.goalNodes.has(targetNode) || this.graph.goalNodes.has(targetNode);
        const bridgeGuard = isFinalGoal ? /** @param {StateObject} s */ (s) => Boolean(s?.goalCompleted) : () => true;
        const bridgeGuardExpr = isFinalGoal ? '(s) => Boolean(s.goalCompleted)' : '() => true';

        candidateGraph.addEdge(mutationPlan.node.id, targetNode, 1, bridgeGuard, bridgeGuardExpr);
      }

      // Verifier Gate: Determinism and transitive reachability
      const checkResult = this.verifier.verify(candidateGraph);
      logger.info('SlowLoop', `CEGIS candidate graph verification result: ${checkResult.status}`, {
        status: checkResult.status,
        omegaPenalty: checkResult.omegaPenalty,
        diagnostics: checkResult.diagnostics
      });

      const lyapunovCheck = this.verifier.verifyAmortizedLyapunovStability(
        this.graph.nodes.size,
        candidateGraph.nodes.size
      );
      logger.info('SlowLoop', `Lyapunov stability evaluation: mode=${lyapunovCheck.mode}, bank=${lyapunovCheck.bank.toFixed(2)}, deltaV=${lyapunovCheck.deltaV.toFixed(2)}`, {
        lyapunovCheck
      });

      if (checkResult.status === 'PASS' && lyapunovCheck.accepted) {
        this.graph = candidateGraph;
        stateHashCounts.delete(h);
        if (mutationPlan.operator === 'addNodeAndBridge' && mutationPlan.node) {
          currentNodeId = mutationPlan.node.id;
        } else if (mutationPlan.operator === 'addEdge') {
          currentNodeId = nextTargetNode;
        }
        appliedMutations.push({ deadlock, mutationPlan, omegaPenalty: checkResult.omegaPenalty });
        this.precedentMemory.push({
          deadlockType: deadlock.type,
          anchorReads: node?.reads || [],
          anchorWrites: node?.writes || [],
          mutationPlan,
          isAntipattern: false
        });
      } else {
        logger.warn('SlowLoop', `Candidate graph rejected by verifier (${checkResult.status}) or Lyapunov (${lyapunovCheck.mode}); rolling back`, {
          checkResult,
          lyapunovCheck
        });
        state = this.rollbackTrajectory(trajectory, state, 1);
        rollbacksCount++;
      }
    }

    return {
      goalReached: false,
      reason: 'HARD_LIMIT_EXCEEDED',
      steps: this.maxSteps,
      rollbacksCount,
      finalState: state,
      appliedMutations
    };
  }
}
