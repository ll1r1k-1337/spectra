// @ts-check
import fs from 'node:fs/promises';
import path from 'node:path';
import { runInteractiveWizard } from './wizard.js';
import { LocalEmbeddingEngine } from '../ai/embeddings.js';
import { LayaSystem1Arbiter } from '../ai/laya-system1.js';
import { LangChainSystem2Arbiter } from '../ai/langchain-system2.js';
import { PgVectorStore } from '../db/pg-store.js';
import {
  DirectedGraph,
  GraphDeterminismVerifier,
  ToolRegistry,
  StateOntologyManager,
  SpectraRuntime
} from '../core/runtime.js';
import { logger } from '../utils/logger.js';

/**
 * @typedef {import('../../types/index.js').SpectraConfig} SpectraConfig
 * @typedef {import('../../types/index.js').SessionRunResult} SessionRunResult
 */

/**
 * Loads existing spectra.config.json or triggers interactive wizard.
 * @param {string} [configPath]
 * @returns {Promise<SpectraConfig>}
 */
export async function loadOrInitConfig(configPath = 'spectra.config.json') {
  const cfgPath = path.resolve(process.cwd(), configPath);
  try {
    const raw = await fs.readFile(cfgPath, 'utf-8');
    return JSON.parse(raw);
  } catch {
    console.log('⚠️ spectra.config.json not found. Launching interactive setup...');
    return await runInteractiveWizard(cfgPath);
  }
}

/**
 * Builds the initial deterministic base graph G_0.
 * @returns {DirectedGraph}
 */
export function buildInitialGraph() {
  const graph = new DirectedGraph('session_graph');

  graph
    .addNode({
      id: 'start',
      type: 'tool',
      reads: ['goal'],
      writes: ['initialized'],
      config: {
        handler: async (s) => ({ ...s, initialized: true })
      }
    })
    .addNode({
      id: 'verify_goal_state',
      type: 'check',
      reads: ['goalCompleted'],
      writes: [],
      config: {
        assertFn: (s) => Boolean(s.goalCompleted)
      }
    })
    .addNode({
      id: 'goal',
      type: 'tool',
      reads: ['goalCompleted'],
      writes: [],
      config: {
        handler: async (s) => s
      }
    });

  graph.startNode = 'start';
  graph.goalNodes.add('goal');
  graph.addEdge('start', 'verify_goal_state', 1, () => true, '() => true');
  graph.addEdge(
    'verify_goal_state',
    'goal',
    1,
    (s) => Boolean(s.goalCompleted),
    '(s) => Boolean(s.goalCompleted)'
  );

  return graph;
}

/**
 * Orchestrates a complete goal execution session with System-1/2 arbitration and telemetry.
 * @param {string} goalText
 * @param {Partial<SpectraConfig> & { debug?: boolean, verbose?: boolean }} [customConfig]
 * @param {{ debug?: boolean, verbose?: boolean }} [options]
 * @returns {Promise<SessionRunResult>}
 */
export async function runSessionGoal(goalText, customConfig, options = {}) {
  const isDebug = Boolean(options.debug || options.verbose || customConfig?.debug || customConfig?.verbose);
  if (isDebug) {
    logger.setLevel('debug');
  } else if (customConfig?.logLevel) {
    logger.setLevel(customConfig.logLevel);
  }

  const config = customConfig ? /** @type {SpectraConfig} */ (customConfig) : await loadOrInitConfig();
  if (!isDebug && config.logLevel) {
    logger.setLevel(config.logLevel);
  }

  console.log(`\n============================================================`);
  console.log(`🚀 [SPECTRA RUN] Initiating Goal: "${goalText}"`);
  console.log(`============================================================\n`);

  // 1. Initialize local embeddings engine
  const embedder = new LocalEmbeddingEngine({
    modelName: config.embeddings?.modelName,
    cacheDir: config.embeddings?.cacheDir,
    weights: config.embeddings?.modalityWeights
  });
  await embedder.init();

  console.log(`🧠 [Embeddings] Vectorizing intent with ${embedder.modelName}...`);
  const intentVec = await embedder.embedText(goalText);
  console.log(`   ↳ Intent vector v_intent constructed (dim = ${intentVec.length})`);

  // 2. Initialize arbiters and persistent vector store
  const sys1 = new LayaSystem1Arbiter(config.system1);
  await sys1.init();

  const sys2 = new LangChainSystem2Arbiter(config.system2);

  const pgStore = new PgVectorStore(config.database?.connectionString);
  await pgStore.init();
  if (pgStore.isConnected) {
    console.log('🗄️  [Database] Connected to PostgreSQL + pgvector');
  } else {
    console.log('ℹ️  [Database] Running in disconnected/local memory mode');
  }

  // 3. Core graph runtime environment
  const tools = new ToolRegistry();
  const verifier = new GraphDeterminismVerifier();
  const ontology = new StateOntologyManager();
  const graph = buildInitialGraph();

  /** @type {Array<{ precedentId?: string, precedentData?: any }>} */
  const sessionPrecedents = [];

  const runtime = new SpectraRuntime({
    graph,
    toolRegistry: tools,
    verifier,
    ontology,
    config: config.runtime,
    system1Evaluator: async (deadlock, mutationPlan) => {
      const verdict = await sys1.evaluateMutation(deadlock, mutationPlan, goalText);
      console.log(
        `🛡️  [System-1 Laya Gate] P(approve)=${verdict.approveProb}, drift=${verdict.driftRisk}, coherence=${verdict.coherence}, verdict=${verdict.approved ? 'APPROVED' : 'REJECTED'}`
      );
      return verdict;
    },
    system2Synthesizer: async (deadlock, state, currentGraph) => {
      console.log(`\n⚡ [Deadlock Encountered: ${deadlock.type}] on node "${deadlock.nodeId}". Invoking System-2...`);
      const blanket = currentGraph.getCausalMarkovBlanket(deadlock.nodeId);
      const vInc = await embedder.computeIncidentVector({
        errorText: `${deadlock.type}: ${deadlock.error || ''}`,
        markovBlanketSignature: Array.from(blanket.keys()).join('->'),
        stateSchemaSignature: Object.keys(state).sort().join(',')
      });
      console.log(`   ↳ Multi-modal incident vector v_inc built (dim = ${vInc.length})`);

      // Query precedent memory if database connected
      let retrievedPlan = null;
      let retrievedPrecedentId = null;
      if (pgStore.isConnected) {
        const precedents = await pgStore.findSimilarPrecedents(vInc, deadlock.type, 3);
        if (precedents.length > 0 && precedents[0].distance < 0.2) {
          console.log(`   ↳ Reusing high-confidence precedent (${precedents[0].precedent_id}) dist=${precedents[0].distance.toFixed(3)}`);
          retrievedPlan = precedents[0].mutation_plan_json;
          retrievedPrecedentId = precedents[0].precedent_id;
        }
      }

      const plan =
        retrievedPlan ||
        (await sys2.synthesizeRecoveryMutation(
          { ...deadlock, step: 1, stateSnapshot: state },
          goalText,
          Array.from(currentGraph.nodes.keys()),
          'goal'
        ));

      if (plan.node?.config?.sourceCode) {
        tools.register({
          id: plan.node.id,
          reads: plan.node.reads,
          writes: plan.node.writes,
          sourceCode: plan.node.config.sourceCode
        });
      }

      if (retrievedPrecedentId) {
        sessionPrecedents.push({ precedentId: retrievedPrecedentId });
      } else {
        sessionPrecedents.push({
          precedentData: {
            deadlockType: deadlock.type,
            incidentVector: vInc,
            anchorReads: plan.node?.reads || [],
            anchorWrites: plan.node?.writes || [],
            mutationOperator: plan.operator,
            mutationPlan: plan
          }
        });
      }

      console.log(`🔧 [System-2 Plan] Operator: ${plan.operator} (${plan.from} -> ${plan.to})`);
      return plan;
    }
  });

  // 4. Execution of the Fast/Slow loop session
  const startTime = Date.now();
  const result = await runtime.runSession({ goal: goalText, goalCompleted: false });
  const duration = Date.now() - startTime;

  // Persist or update precedents based on final session outcome
  if (pgStore.isConnected && sessionPrecedents.length > 0) {
    for (const item of sessionPrecedents) {
      try {
        if (item.precedentId) {
          if (result.goalReached) {
            await pgStore.updatePrecedentOutcome(item.precedentId, true);
          } else {
            await pgStore.markPrecedentAntipattern(item.precedentId);
          }
        } else if (item.precedentData) {
          await pgStore.saveDeadlockPrecedent({
            ...item.precedentData,
            successCount: result.goalReached ? 1 : 0,
            failureCount: result.goalReached ? 0 : 1,
            isPermanentAntipattern: !result.goalReached
          });
        }
      } catch (err) {
        logger.warn('Runner', `Failed to update/save precedent outcome: ${err instanceof Error ? err.message : String(err)}`);
      }
    }
  }

  // 5. Output structured telemetry report
  console.log('\n============================================================');
  console.log(`🏁 [SPECTRA EXECUTION REPORT] Status: ${result.goalReached ? 'SUCCESS' : 'HALTED'}`);
  console.log('============================================================');
  console.log(`   • Goal Reached:        ${result.goalReached}`);
  console.log(`   • Steps Executed:      ${result.steps}`);
  console.log(`   • Slow Loop Mutations: ${result.appliedMutations.length}`);
  console.log(`   • Rollbacks Count:     ${result.rollbacksCount}`);
  console.log(`   • Omega Penalty:       ${result.omegaPenalty ?? 0}`);
  console.log(`   • Duration:            ${duration}ms`);
  console.log(`   • Final State:         ${JSON.stringify(result.finalState)}`);
  console.log('============================================================\n');

  await pgStore.close();
  return result;
}
