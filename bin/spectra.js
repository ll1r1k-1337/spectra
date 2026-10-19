#!/usr/bin/env node
// @ts-check
import fsSync from 'node:fs';
import fs from 'node:fs/promises';
import path from 'node:path';
import vm from 'node:vm';
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
import { GOAL_PREDICATE_EXPR } from '../src/core/acceptance-gate.js';

const IGNORED_WORKSPACE_DIRS = new Set(['.git', 'node_modules', '.spectra', 'dist', 'build', 'coverage']);
const MAX_WORKSPACE_FILES = 300;
const MAX_WORKSPACE_FILE_CONTENT = 32 * 1024;
const MAX_WORKSPACE_TOTAL_CONTENT = 128 * 1024;
const MAX_WORKSPACE_READ_SIZE = 1024 * 1024;

/**
 * Host-инструмент: детерминированный обход workspace.
 * Возвращает только текстовые файлы; бинарники пропускаются, файлы > 1 MiB
 * попадают в список метаданных без содержимого.
 * @param {string} [baseDir]
 * @returns {Promise<{ workspaceFiles: import('../types/index.js').WorkspaceFile[] }>}
 */
export async function scanWorkspace(baseDir = process.cwd()) {
  const root = path.resolve(baseDir);
  /** @type {import('../types/index.js').WorkspaceFile[]} */
  const workspaceFiles = [];
  let totalContent = 0;

  /** @param {string} dir */
  async function walk(dir) {
    if (workspaceFiles.length >= MAX_WORKSPACE_FILES) return;
    /** @type {import('node:fs').Dirent[]} */
    let entries;
    try {
      entries = await fs.readdir(dir, { withFileTypes: true });
    } catch {
      return;
    }
    entries.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
    for (const entry of entries) {
      if (workspaceFiles.length >= MAX_WORKSPACE_FILES) break;
      const abs = path.join(dir, entry.name);
      if (entry.isDirectory()) {
        if (!IGNORED_WORKSPACE_DIRS.has(entry.name)) await walk(abs);
        continue;
      }
      if (!entry.isFile()) continue;
      let size = 0;
      try {
        const st = await fs.stat(abs);
        if (!st.isFile()) continue;
        size = st.size;
      } catch {
        continue;
      }
      const rel = path.relative(root, abs).split(path.sep).join('/');
      if (!rel || rel.startsWith('..') || path.isAbsolute(rel)) continue;
      /** @type {import('../types/index.js').WorkspaceFile} */
      const rec = { path: rel, size, ext: path.extname(entry.name).toLowerCase() };
      if (size > MAX_WORKSPACE_READ_SIZE) {
        workspaceFiles.push(rec);
        continue;
      }
      /** @type {Buffer} */
      let buf;
      try {
        buf = await fs.readFile(abs);
      } catch {
        workspaceFiles.push(rec);
        continue;
      }
      if (buf.subarray(0, Math.min(8192, buf.length)).includes(0)) continue;
      workspaceFiles.push(rec);
      const budget = MAX_WORKSPACE_TOTAL_CONTENT - totalContent;
      if (budget <= 0) {
        rec.contentTruncated = true;
        continue;
      }
      const take = Math.min(buf.length, MAX_WORKSPACE_FILE_CONTENT, budget);
      rec.content = buf.subarray(0, take).toString('utf8');
      if (take < buf.length) rec.contentTruncated = true;
      totalContent += take;
    }
  }

  await walk(root);
  return { workspaceFiles };
}

/**
 * Host-инструмент: материализует state.artifacts на диск и проверяет синтаксис .js.
 * Патчи { file, search, replace } применяются к артефактам или к файлам на диске.
 * Защита от path traversal: любой путь обязан остаться внутри baseDir.
 * @param {{ artifacts?: Array<{ name?: string, content?: string }>; patches?: Array<{ file?: string, search?: string, replace?: string }> }} state
 * @param {string} [baseDir]
 * @returns {Promise<{ persistedFiles: string[]; syntaxErrors: string[]; artifacts: Array<{ name: string, content: string }>, patches: Array<{ file: string, search: string, replace: string }> }>}
 */
export async function persistArtifactsToDisk(state, baseDir = process.cwd()) {
  const artifacts = Array.isArray(state.artifacts) ? state.artifacts : [];
  const patches = Array.isArray(state.patches) ? state.patches : [];
  const root = path.resolve(baseDir);
  /** @type {string[]} */ const persistedFiles = [];
  /** @type {string[]} */ const syntaxErrors = [];
  /** @type {Map<string, string>} */ const contents = new Map();

  for (const a of artifacts) {
    if (!a || typeof a.name !== 'string' || typeof a.content !== 'string') continue;
    contents.set(a.name.replace(/\\/g, '/').replace(/^\/+/, ''), a.content);
  }

  for (const p of patches) {
    if (!p || typeof p !== 'object') continue;
    const rel = typeof p.file === 'string' ? p.file.replace(/\\/g, '/').replace(/^\/+/, '') : '';
    const search = typeof p.search === 'string' ? p.search : '';
    const replace = typeof p.replace === 'string' ? p.replace : '';
    if (!rel) {
      syntaxErrors.push('patch: invalid or empty file path');
      continue;
    }
    const target = path.resolve(root, rel);
    if (target !== root && !target.startsWith(root + path.sep)) {
      throw new Error(`Security violation: patch path "${p.file}" escapes workspace directory`);
    }
    if (search === '') {
      syntaxErrors.push(`${rel}: patch search must not be empty`);
      continue;
    }
    let current = contents.get(rel);
    if (current === undefined) {
      try {
        current = await fs.readFile(target, 'utf8');
      } catch (e) {
        syntaxErrors.push(
          `${rel}: ${e instanceof Error && /** @type {NodeJS.ErrnoException} */ (e).code === 'ENOENT' ? 'patch target not found' : 'patch target unreadable'}`
        );
        continue;
      }
    }
    const idx = current.indexOf(search);
    if (idx < 0) {
      syntaxErrors.push(`${rel}: patch search not found`);
      continue;
    }
    contents.set(rel, current.slice(0, idx) + replace + current.slice(idx + search.length));
  }

  for (const [rel, content] of contents) {
    const target = path.resolve(root, rel);
    if (target !== root && !target.startsWith(root + path.sep)) {
      throw new Error(`Security violation: artifact path "${rel}" escapes workspace directory`);
    }
    await fs.mkdir(path.dirname(target), { recursive: true });
    await fs.writeFile(target, content, 'utf8');
    await fs.stat(target);
    persistedFiles.push(rel);
    if (rel.endsWith('.js')) {
      try {
        new vm.Script(content);
      } catch (e) {
        syntaxErrors.push(`${rel}: ${e instanceof Error ? e.message : String(e)}`);
      }
    }
  }

  return {
    persistedFiles,
    syntaxErrors,
    artifacts: Array.from(contents, ([name, content]) => ({ name, content })),
    // Патчи применены; очищаем, чтобы повторный прогон persist не накладывал их дважды.
    patches: []
  };
}

export const HOST_TOOLS = new Map(
  /** @type {Array<[string, (state: import('../types/index.js').StateObject) => Promise<object>]>} */ ([
    ['scan_workspace', () => scanWorkspace()],
    ['persist_artifacts_to_disk', (state) => persistArtifactsToDisk(state)]
  ])
);

/**
 * Вставляет сенсорную ноду scan_workspace в загруженный из памяти граф,
 * если она отсутствовала (графы, сохранённые до появления сенсора).
 * @param {DirectedGraph} graph
 */
export function ensureWorkspaceSensor(graph) {
  if (graph.nodes.has('scan_workspace') || !graph.nodes.has('acquire_artifacts') || !graph.startNode) return;
  const acquire = graph.nodes.get('acquire_artifacts');
  if (acquire) acquire.reads = Array.from(new Set([...acquire.reads, 'workspaceFiles']));

  const startEdges = graph.edges.get(graph.startNode) || [];
  const oldEdge = startEdges.find((e) => e.to === 'acquire_artifacts');
  graph.addNode({
    id: 'scan_workspace',
    type: 'tool',
    reads: [],
    writes: ['workspaceFiles'],
    config: { toolId: 'scan_workspace' }
  });
  graph.edges.set(
    graph.startNode,
    startEdges.filter((e) => e.to !== 'acquire_artifacts')
  );
  graph.addEdge(
    graph.startNode,
    'scan_workspace',
    oldEdge ? oldEdge.guard : () => true,
    oldEdge ? oldEdge.priority : 1,
    { expr: oldEdge ? oldEdge.guardExpr : 'true' }
  );
  graph.addEdge('scan_workspace', 'acquire_artifacts', () => true, 1, { expr: 'true' });
}

/**
 * Создает базовый стартовый граф G_0.
 */
export function buildInitialGraph() {
  const graph = new DirectedGraph('session_graph');
  const goalPredicate = GOAL_PREDICATE_EXPR;
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
      id: 'scan_workspace',
      type: 'tool',
      reads: [],
      writes: ['workspaceFiles'],
      config: {
        toolId: 'scan_workspace'
      }
    })
    .addNode({
      id: 'acquire_artifacts',
      type: 'llm',
      reads: ['goal', 'workspaceFiles'],
      writes: ['artifacts', 'summary'],
      config: {
        prompt:
      'You are the execution node. The session goal is given in Inputs (JSON) under "goal". ' +
      'Produce the REAL, COMPLETE, runnable deliverable that satisfies that goal as code/files ' +
      '(e.g. index.html, main.js, shaders.glsl where appropriate). ' +
      'Existing workspace files are provided in Inputs.workspaceFiles as { path, size, content }; content may be truncated. ' +
      'For edits prefer minimal "patches" ({ file, search, replace }) over rewriting whole files; never invent file contents. ' +
      'Return artifacts as an array of { name, content } entries for new/fully rewritten files, plus a short summary. ' +
      'Never ask questions and never return a placeholder, a "blocked" note, or a "no goal given" message — the goal IS provided.'
      }
    })
    .addNode({
      id: 'persist_artifacts',
      type: 'tool',
      reads: ['artifacts', 'patches', 'workspaceFiles'],
      writes: ['persistedFiles', 'syntaxErrors'],
      config: {
        toolId: 'persist_artifacts_to_disk'
      }
    })
    .addNode({
      id: 'verify_goal_state',
      type: 'check',
      reads: ['artifacts', 'persistedFiles', 'syntaxErrors'],
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
  graph.addEdge('start', 'scan_workspace', graph.compileGuardExpression('Boolean(state.initialized)'), 1, {
    expr: 'Boolean(state.initialized)'
  });
  graph.addEdge('scan_workspace', 'acquire_artifacts', graph.compileGuardExpression('true'), 1, {
    expr: 'true'
  });
  graph.addEdge('acquire_artifacts', 'persist_artifacts', graph.compileGuardExpression('true'), 1, {
    expr: 'true'
  });
  graph.addEdge('persist_artifacts', 'verify_goal_state', graph.compileGuardExpression('true'), 1, {
    expr: 'true'
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
    ensureWorkspaceSensor(graph);
    if (savedGraphJson) {
      logger.info('MEMORY', `Загружена эволюционировавшая версия графа "${graph.id}" (узлов: ${graph.nodes.size})`);
    }

    const toolRegistry = new ToolRegistry();
    for (const tool of (await store.loadRegisteredTools?.()) || []) {
      toolRegistry.register(tool);
    }

    const runtime = new SpectraRuntime({
      graph,
      toolRegistry,
      verifier: new GraphDeterminismVerifier(),
      ontology: new StateOntologyManager(),
      logger,
      precedentStore: store,
      embedder,
      system1Arbiter: sys1,
      system2Arbiter: sys2,
      hostTools: HOST_TOOLS,
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

/**
 * @param {import('../types/index.js').CredentialsOverrides} overrides
 */
async function handleStatusCommand(overrides) {
  const config = await resolveSpectraConfig(overrides);
  const logger = new SpectraLogger({ level: 'info' });
  const store = new PgVectorPrecedentStore(config.database);
  await store.init();
  try {
    const stats = (await store.getStats?.()) || {};
    logger.banner('СОСТОЯНИЕ ПАМЯТИ SPECTRA');
    console.log(JSON.stringify(stats, null, 2));
  } finally {
    await store.close();
  }
}

/**
 * @param {import('../types/index.js').CredentialsOverrides} overrides
 */
async function handleMemoryCleanCommand(overrides) {
  const config = await resolveSpectraConfig(overrides);
  const logger = new SpectraLogger({ level: 'info' });
  const store = new PgVectorPrecedentStore(config.database);
  await store.init();
  try {
    const pruned = (await store.pruneMemory?.()) || {};
    logger.banner('ОЧИСТКА ПАМЯТИ SPECTRA');
    console.log(JSON.stringify(pruned, null, 2));
  } finally {
    await store.close();
  }
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
  } else if (cmd === 'status') {
    await handleStatusCommand(overrides);
  } else if (cmd === 'memory-clean') {
    await handleMemoryCleanCommand(overrides);
  } else {
    console.log('Использование CLI Spectra:');
    console.log('  spectra init                                                                 — Интерактивная настройка конфигурации');
    console.log('  spectra run "<goal>" [--api-key <k>] [--base-url <u>] [--pg-url <db>] [-v]   — Запуск цикла с реальными LLM/Laya/PG');
    console.log('  spectra verify [--pg-url <db>]                                               — Статическая верификация актуального графа');
    console.log('  spectra sleep [--pg-url <db>]                                                — Запуск Контура 3 (Consolidation Loop)');
    console.log(' spectra status [--pg-url <db>] — Сводка по памяти (граф, прецеденты, инструменты, макросы)');
    console.log(' spectra memory-clean [--pg-url <db>] — Очистка антипаттернов и старых версий графа');
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
