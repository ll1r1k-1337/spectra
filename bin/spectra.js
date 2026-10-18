#!/usr/bin/env node
// @ts-check
import fsSync from 'node:fs';
import { pathToFileURL } from 'node:url';
import { SpectraLogger } from '../src/cli/logger.js';
import { runInteractiveWizard } from '../src/cli/wizard.js';
import { parseCliArgs, resolveSpectraConfig, assertValidLlmCredentials } from '../src/cli/credentials.js';
import { LocalEmbeddingEngine } from '../src/ai/embeddings.js';
import { LayaSystem1Arbiter } from '../src/ai/laya-system1.js';
import { LangChainSystem2Arbiter } from '../src/ai/langchain-system2.js';
import { PgVectorPrecedentStore } from '../src/db/pg-store.js';
import { DirectedGraph } from '../src/core/graph.js';
import { GraphDeterminismVerifier } from '../src/core/verifier.js';
import { ToolRegistry } from '../src/core/tools.js';
import { StateOntologyManager } from '../src/core/ontology.js';
import { SpectraRuntime } from '../src/core/runtime.js';

/**
 * Создает базовый стартовый граф G_0.
 */
export function buildInitialGraph() {
  const graph = new DirectedGraph('session_graph');
  const goalPredicate =
    'Boolean(Array.isArray(state.artifacts) && state.artifacts.length > 0 && String((state.artifacts[0] || {}).content || "").trim().length >= 1)';
  graph
    .addNode({
      id: 'start',
      type: 'tool',
      reads: ['goal'],
      writes: ['initialized'],
      config: {
        sourceCode: '(s) => ({ ...s, initialized: true })'
      }
    })
    .addNode({
      id: 'verify_goal_state',
      type: 'check',
      reads: ['artifacts'],
      writes: [],
      config: {
        assertExpr: goalPredicate
      }
    })
    .addNode({
      id: 'goal',
      type: 'tool',
      reads: ['artifacts', 'summary'],
      writes: [],
      config: {
        sourceCode: '(s) => s'
      }
    });

  graph.startNode = 'start';
  graph.goalNodes.add('goal');
  graph.addEdge('start', 'verify_goal_state', graph.compileGuardExpression('Boolean(state.initialized)'), 1, {
    expr: 'Boolean(state.initialized)'
  });
  graph.addEdge('verify_goal_state', 'goal', graph.compileGuardExpression(goalPredicate), 1, {
    expr: goalPredicate
  });
  return graph;
}

/**
 * @param {string} goalText
 * @param {import('../types/index.js').CredentialsOverrides} overrides
 * @param {{ verbose: boolean; freshGraph: boolean }} flags
 */
async function handleRunCommand(goalText, overrides, flags) {
  const config = await resolveSpectraConfig(overrides);
  assertValidLlmCredentials(config.system2);

  const logger = new SpectraLogger({
    level: flags.verbose ? 'debug' : config.logging.level,
    traceStateDiffs: true,
    traceGuards: true,
    jsonlFilePath: config.logging.jsonlFilePath
  });

  logger.info('CLI', `Инициализация боевых подсистем Spectra для цели: "${goalText}"`, {
    llmBaseUrl: config.system2.baseUrl || 'https://api.openai.com/v1',
    frontierModel: config.system2.frontierModel
  });

  const store = new PgVectorPrecedentStore(config.database);
  await store.init();
  logger.info('DB', 'Успешно подключено к PostgreSQL 16 + pgvector');

  const embedder = await new LocalEmbeddingEngine({
    modelName: config.embeddings.modelName,
    cacheDir: config.embeddings.cacheDir,
    hfToken: config.embeddings.hfToken,
    weights: config.embeddings.modalityWeights
  }).init();

  const sys1 = await new LayaSystem1Arbiter(config.system1).init();
  const sys2 = new LangChainSystem2Arbiter(config.system2);

  try {
    const intentVec = await embedder.embedText(goalText);
    logger.info('EMBEDDINGS', `Векторизована цель v_intent (dim=${intentVec.length})`);

    const savedGraphJson = !flags.freshGraph ? await store.loadLatestGraph() : null;
    const graph = savedGraphJson ? DirectedGraph.fromJSON(savedGraphJson) : buildInitialGraph();
    if (savedGraphJson) {
      logger.info('MEMORY', `Загружена эволюционировавшая версия графа "${graph.id}" (узлов: ${graph.nodes.size})`);
    }

    const runtime = new SpectraRuntime({
      graph,
      toolRegistry: new ToolRegistry(),
      verifier: new GraphDeterminismVerifier(),
      ontology: new StateOntologyManager(),
      logger,
      precedentStore: store,
      embedder,
      system1Arbiter: sys1,
      system2Arbiter: sys2,
      config: config.runtime
    });

    const result = await runtime.runSession({ goal: goalText, artifacts: [] });
    logger.info('SUMMARY', 'Итоговый аудит сессии записан в JSONL', {
      sessionId: result.sessionId,
      goalReached: result.goalReached,
      steps: result.steps,
      mutationsCount: result.appliedMutations.length,
      mutationSources: result.appliedMutations.map((m) => m.source),
      rollbacksCount: result.rollbacksCount,
      omegaPenalty: result.omegaPenalty,
      logFile: logger.jsonlFilePath
    });
  } finally {
    await sys1.close();
    await store.close();
  }
}

/**
 * @param {import('../types/index.js').CredentialsOverrides} overrides
 */
async function handleVerifyCommand(overrides) {
  const config = await resolveSpectraConfig(overrides);
  const logger = new SpectraLogger({ level: 'info' });
  const store = new PgVectorPrecedentStore(config.database);
  await store.init();
  const savedGraphJson = await store.loadLatestGraph();
  const graph = savedGraphJson ? DirectedGraph.fromJSON(savedGraphJson) : buildInitialGraph();
  const verifier = new GraphDeterminismVerifier();

  logger.banner('СТАТИЧЕСКАЯ ВЕРИФИКАЦИЯ ГРАФА SPECTRA');
  logger.logTopologyAscii(graph);
  const report = verifier.verify(graph);
  logger.logVerifierReport(report, graph.computeStructuralHash());
  await store.close();
}

/**
 * @param {import('../types/index.js').CredentialsOverrides} overrides
 */
async function handleSleepCommand(overrides) {
  const config = await resolveSpectraConfig(overrides);
  const logger = new SpectraLogger({ level: 'info' });
  const store = new PgVectorPrecedentStore(config.database);
  await store.init();
  const savedGraphJson = await store.loadLatestGraph();
  const graph = savedGraphJson ? DirectedGraph.fromJSON(savedGraphJson) : buildInitialGraph();

  const runtime = new SpectraRuntime({
    graph,
    toolRegistry: new ToolRegistry(),
    verifier: new GraphDeterminismVerifier(),
    ontology: new StateOntologyManager(),
    logger,
    precedentStore: store
  });
  await runtime.runConsolidationSleep();
  await store.close();
}

async function main() {
  const { overrides, positional, flags } = parseCliArgs(process.argv.slice(2));
  const cmd = positional[0];
  const restArgs = positional.slice(1);

  if (cmd === 'init') {
    await runInteractiveWizard();
  } else if (cmd === 'run') {
    const goal = restArgs.join(' ').trim();
    if (!goal) {
      console.error(
        '❌ Укажите цель: spectra run "<goal>" [--api-key <key>] [--base-url <url>] [--model <name>] [--pg-url <db>] [--verbose] [--fresh-graph]'
      );
      process.exit(1);
    }
    await handleRunCommand(goal, overrides, flags);
  } else if (cmd === 'verify') {
    await handleVerifyCommand(overrides);
  } else if (cmd === 'sleep') {
    await handleSleepCommand(overrides);
  } else {
    console.log('Использование CLI Spectra:');
    console.log('  spectra init                                                                 — Интерактивная настройка конфигурации');
    console.log('  spectra run "<goal>" [--api-key <k>] [--base-url <u>] [--pg-url <db>] [-v]   — Запуск цикла с реальными LLM/Laya/PG');
    console.log('  spectra verify [--pg-url <db>]                                               — Статическая верификация актуального графа');
    console.log('  spectra sleep [--pg-url <db>]                                                — Запуск Контура 3 (Consolidation Loop)');
  }
}

const isExecutedDirectly = (() => {
  if (!process.argv[1]) return false;
  try {
    return import.meta.url === pathToFileURL(fsSync.realpathSync(process.argv[1])).href;
  } catch {
    return import.meta.url === pathToFileURL(process.argv[1]).href;
  }
})();

if (isExecutedDirectly) {
  main().catch((err) => {
    console.error('Критическая ошибка Spectra CLI:', err);
    process.exit(1);
  });
}
