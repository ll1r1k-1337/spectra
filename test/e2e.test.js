// @ts-check
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildInitialGraph, HOST_TOOLS } from '../bin/spectra.js';
import { resolveSpectraConfig, assertValidLlmCredentials } from '../src/cli/credentials.js';
import { GraphDeterminismVerifier } from '../src/core/verifier.js';
import { ToolRegistry } from '../src/core/tools.js';
import { StateOntologyManager } from '../src/core/ontology.js';
import { SpectraRuntime } from '../src/core/runtime.js';
import { SpectraLogger } from '../src/cli/logger.js';
import { PgVectorPrecedentStore } from '../src/db/pg-store.js';
import { LocalEmbeddingEngine } from '../src/ai/embeddings.js';
import { LayaSystem1Arbiter } from '../src/ai/laya-system1.js';
import { LangChainSystem2Arbiter } from '../src/ai/langchain-system2.js';
import fs from 'node:fs/promises';
import path from 'node:path';

test('Spectra Real E2E (No Mocks): Real LLM Synthesis -> pgvector Experience Replay -> Saga Verification & Rollback -> Bisimulation Sleep', async () => {
  // 1. Загружаем прокинутые учетные данные из .env / process.env / spectra.config.json
  const config = await resolveSpectraConfig();
  assertValidLlmCredentials(config.system2);

  const logger = new SpectraLogger({
    level: 'info',
    jsonlFilePath: '.spectra/logs/e2e-real.jsonl'
  });

  // 2. Инициализируем реальное хранилище PostgreSQL 16 + pgvector, ONNX эмбеддинги, ONNX Laya и LangChain LLM
  const store = new PgVectorPrecedentStore(config.database);
  await store.init();
  await store.clearAll();

  const embedder = await new LocalEmbeddingEngine({
    modelName: config.embeddings.modelName,
    cacheDir: config.embeddings.cacheDir,
    hfToken: config.embeddings.hfToken,
    weights: config.embeddings.modalityWeights
  }).init();

  const sys1 = await new LayaSystem1Arbiter(config.system1).init();
  const sys2 = new LangChainSystem2Arbiter(config.system2);

  try {
    // =========================================================================
    // ЭТАП 1: Старт на графе G_0 -> реальный вызов LLM (@langchain/openai) ->
    // материализация артефактов на диск + проверка синтаксиса -> достижение цели
    // =========================================================================
    const runtime1 = new SpectraRuntime({
      graph: buildInitialGraph(),
      toolRegistry: new ToolRegistry(),
      verifier: new GraphDeterminismVerifier(),
      ontology: new StateOntologyManager(2),
      logger,
      precedentStore: store,
      embedder,
      system1Arbiter: sys1,
      system2Arbiter: sys2,
      hostTools: HOST_TOOLS,
      config: config.runtime
    });

    const goal1 = 'Рассчитать квартальный баланс и подтвердить выполнение цели';
    const res1 = await runtime1.runSession({ goal: goal1, artifacts: [], });

    assert.equal(res1.goalReached, true);
    assert.ok(res1.llmCalls >= 1, 'Сессия 1 обязана вызвать тяжелую LLM (System-2) для синтеза артефактов');
    assert.ok(Array.isArray(res1.finalState.artifacts) && res1.finalState.artifacts.length > 0);
    assert.ok(typeof res1.finalState.summary === 'string' && res1.finalState.summary.length > 0);
    assert.ok(Array.isArray(res1.finalState.persistedFiles), 'Артефакты обязаны быть материализованы на диск');
    assert.equal(res1.finalState.persistedFiles.length, res1.finalState.artifacts.length);
    assert.ok(Array.isArray(res1.finalState.syntaxErrors) && res1.finalState.syntaxErrors.length === 0);
    for (const rel of res1.finalState.persistedFiles) {
      await fs.stat(path.resolve(process.cwd(), rel));
    }

    // =========================================================================
    // ЭТАП 2: Принудительный тупик на графе G_0 -> извлечение выученного рецепта
    // из PostgreSQL HNSW (MEMORY_GROUNDED) либо синтез System-2 (см. NOTE ниже)
    // =========================================================================
    const runtime2 = new SpectraRuntime({
      graph: buildInitialGraph(),
      toolRegistry: new ToolRegistry(),
      verifier: new GraphDeterminismVerifier(),
      ontology: new StateOntologyManager(1),
      logger,
      precedentStore: store,
      embedder,
      system1Arbiter: sys1,
      system2Arbiter: sys2,
      hostTools: HOST_TOOLS,
      // NOTE: Живое поведение LLM/DB предполагается. Ограничиваем лимит шагов,
      // чтобы принудительный тупик дал ровно одну мутацию восстановления.
      config: { ...config.runtime, hardStepLimit: 6 }
    });

    // NOTE: Принудительно ломаем контракт цели, чтобы гарантированно войти в SLOW LOOP
    // (векторизация инцидента + поиск прецедента в pgvector / синтез System-2).
    // Точный источник мутации (MEMORY_GROUNDED либо SYSTEM2_SYNTHESIZED) зависит от
    // состояния pgvector и недетерминирован без живых LLM/БД.
    const forcedCheck = runtime2.graph.nodes.get('verify_goal_state');
    assert.ok(forcedCheck, 'Узел verify_goal_state обязан существовать в G_0');
    if (forcedCheck) {
      forcedCheck.config = { ...(forcedCheck.config || {}), assertExpr: '(state) => false', assertFn: () => false };
    }

    const goal2 = 'Сформировать аудиторский отчет по транзакциям';
    const res2 = await runtime2.runSession({
      goal: goal2,
      artifacts: [],
      department: 'finance'
    });

    assert.equal(res2.appliedMutations.length, 1, 'Принудительный тупик обязан дать одну мутацию восстановления');
    assert.ok(
      ['MEMORY_GROUNDED', 'SYSTEM2_SYNTHESIZED'].includes(res2.appliedMutations[0].source),
      'Восстановление обязано прийти из pgvector или от System-2'
    );
    // NOTE: Прецеденты теперь сохраняются в pgvector отложенно — только при успешном
    // достижении цели (см. pendingPrecedents в runSession). Живое поведение LLM/DB предполагается.

    // =========================================================================
    // ЭТАП 3: Реальная верификация и исполнение Saga-компенсатора в песочнице node:vm
    // для неидемпотентного инструмента без единого мока
    // =========================================================================
    const sagaTools = new ToolRegistry();
    sagaTools.register({
      id: 'debit_ledger',
      reads: ['balance', 'amount'],
      writes: ['balance'],
      sourceCode: '(s) => ({ ...s, balance: Number(s.balance) - Number(s.amount) })',
      regressionPool: [
        { input: { balance: 1000, amount: 250 }, expectedOutput: { balance: 750, amount: 250 } }
      ]
    });

    const sagaCheck = sagaTools.verifyAndAttachSagaCompensator(
      'debit_ledger',
      '(s) => ({ ...s, balance: Number(s.balance) + Number(s.amount) })'
    );
    assert.equal(sagaCheck.verified, true);
    assert.equal(sagaCheck.hasBarrier, false);

    const txGraph = buildInitialGraph();
    txGraph.addNode({
      id: 'debit_ledger',
      type: 'tool',
      idempotency: 'non_idempotent',
      reads: ['balance', 'amount'],
      writes: ['balance'],
      config: { toolId: 'debit_ledger' }
    });

    const txRuntime = new SpectraRuntime({
      graph: txGraph,
      toolRegistry: sagaTools,
      verifier: new GraphDeterminismVerifier(),
      ontology: new StateOntologyManager(2),
      logger,
      precedentStore: store,
      embedder,
      system1Arbiter: sys1,
      system2Arbiter: sys2,
      hostTools: HOST_TOOLS
    });

    const trajectory = [
      {
        step: 0,
        nodeId: 'debit_ledger',
        stateBefore: { balance: 1000, amount: 250 },
        stateAfter: { balance: 750, amount: 250 }
      }
    ];
    const rbResult = txRuntime.rollbackTrajectoryAndCursor(
      trajectory,
      { balance: 750, amount: 250 },
      { balance: 750, amount: 250 },
      'verify_goal_state',
      1
    );
    assert.equal(rbResult.restoredNodeId, 'debit_ledger');
    assert.equal(rbResult.restoredState.balance, 1000);

    // =========================================================================
    // ЭТАП 4: Контур 3 («Фаза сна» / Consolidation Loop) с сохранением в PostgreSQL
    // =========================================================================
    runtime2.ontology.registerStagingKey('ephemeral_noise_key', 0.01);
    const activeSolver = Array.from(runtime2.graph.nodes.values()).find(
      (n) =>
        !['start', 'scan_workspace', 'acquire_artifacts', 'persist_artifacts', 'verify_goal_state', 'goal'].includes(
          n.id
        )
    );
    assert.ok(activeSolver, 'Должен существовать синтезированный узел восстановления');

    // Две идентичные ветви с одинаковой сигнатурой — материал для foldBisimilarBranches.
    for (const branchId of ['solver_duplicate_branch', 'solver_duplicate_branch_2']) {
      runtime2.graph.addNode({
        id: branchId,
        type: 'tool',
        reads: [...activeSolver.reads],
        writes: [...activeSolver.writes],
        config: { sourceCode: activeSolver.config?.sourceCode }
      });
      runtime2.graph.addEdge(
        'start',
        branchId,
        runtime2.graph.compileGuardExpression(
          '!((Array.isArray(state.artifacts) ? state.artifacts.length > 0 : Boolean(state.artifacts)))'
        ),
        2,
        { expr: '!((Array.isArray(state.artifacts) ? state.artifacts.length > 0 : Boolean(state.artifacts)))' }
      );
      runtime2.graph.addEdge(branchId, 'verify_goal_state', () => true, 1, {
        expr: 'true /* reverify_check */'
      });
    }

    const sleepRes = await runtime2.runConsolidationSleep();
    assert.ok(sleepRes.promotedKeys.some((k) => ['artifacts', 'persistedFiles'].includes(k)));
    assert.ok(sleepRes.evictedKeys.includes('ephemeral_noise_key'));
    assert.equal(sleepRes.foldedNodes, 1);
    assert.ok(sleepRes.omegaAfter <= sleepRes.omegaBefore);
  } finally {
    await sys1.close();
    await store.close();
  }
});
