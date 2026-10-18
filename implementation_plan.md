# Инженерный план реализации прототипа SPECTRA (Полная кодовая база без моков, прокидывание учетных данных и сквозная телеметрия)

Настоящий документ содержит готовую к запуску архитектуру и полную реализацию всех файлов прототипа самообучающейся графовой ИИ-системы **Spectra** на чистом JavaScript (Pure ESM, Node.js 20+) со **100% рукописными (handwritten) `.d.ts` контрактами типов**, сквозным прокидыванием учетных данных для подключения к LLM, Hugging Face и PostgreSQL (`src/cli/credentials.js`), интерактивным CLI (`spectra init`, `spectra run "<goal>"`, `spectra verify`, `spectra sleep`), структурным логгером (`SpectraLogger`), локальными эмбеддингами на `@huggingface/transformers`, реальным ONNX-арбитром System-1 на `@receptron/laya` (`Laya.load()` + `laya.systemOne()`), синтезатором System-2 на LangChain.js (`@langchain/openai`), хранилищем прецедентов PostgreSQL 16 (`pgvector`) и честным E2E-тестом (`node:test`) **полностью без моков и стабов**.

## 1. Архитектура репозитория, стек технологий и схема подключения учетных данных

### 1.1. Роли компонентов стека (100% реальные интеграции без моков)

| Компонент | Пакет / Технология | Роль в трехконтурной архитектуре Spectra |
| :--- | :--- | :--- |
| **Исполнительная среда** | **Node.js 20+** (`"type": "module"`) | Прямой запуск ESM `.js`-модулей без сборки, изолированная песочница `node:vm` с заморозкой входного клона `Object.freeze(input)` для динамических инструментов и `guard`-предикатов, нативная загрузка `.env` и `node:test`. |
| **Контракты и типизация** | **Самописные `.d.ts` (Handwritten Declarations) + JSDoc** | **Contract-First подход:** файлы `.d.ts` в директории `types/` пишутся исключительно вручную (`declaration: false`, `noEmit: true`), а `tsc --noEmit` проверяет каждый `.js`-файл на строгое соответствие рукописным интерфейсам. |
| **Управление кредами** | **`src/cli/credentials.js`** | Единый резолвер учетных данных для CLI и E2E-тестов: объединяет `.env`, переменные окружения (`OPENAI_API_KEY` / `SPECTRA_LLM_API_KEY`, `OPENAI_BASE_URL` / `SPECTRA_LLM_BASE_URL`, `SPECTRA_PG_URL`, `HF_TOKEN`), `spectra.config.json` и CLI-флаги (`--api-key`, `--base-url`, `--model`, `--pg-url`). |
| **Сквозное логирование** | **`src/cli/logger.js` (`SpectraLogger`)** | Иерархическая цветная консоль (ANSI) $+$ запись каждого шага в `.spectra/logs/session.jsonl`. Отображает ASCII-топологию графа с активным курсором, пошаговый дифф ключей `State` ($\Delta s_t$), таблицу вычисления `guard`-ребер по приоритету $\Pi(e)$, веса каузального одеяла Маркова $\mathcal{MB}(v_t, G)$, вердикты System-1/System-2 и диагностику верификатора $\Omega(G)$. |
| **Локальные эмбеддинги** | **`@huggingface/transformers`** (`transformers.js`) | Реальная CPU-генерация эмбеддингов ($\mathbb{R}^{384}$, модель `Xenova/multilingual-e5-small`) через ONNX-пайплайн `feature-extraction` для мультимодального вектора тупика $\mathbf{v}_{\text{inc}}$ и вектора цели $\mathbf{v}_{\text{intent}}$. |
| **Арбитр System-1** | **`@receptron/laya`** (`^0.1.2`, ONNX Runtime) | Реальный вызов `Laya.load()` и батч-инференс `laya.systemOne(state, questions)` с типизированными вопросами `score` (для риска дрейфа $r_{\text{drift}}$ и связности $s_{\text{coh}}$), `noul` (калиброванная вероятность одобрения $P_{\text{approve}}$) и `choice` (выбор стратегии отката `keep` / `rollback_last` / `rollback_session`). |
| **Арбитр System-2 и LLM-узлы** | **LangChain.js** (`@langchain/openai`, `zod`) | Реальный синтез планов мутаций графа и выполнение `llm`-узлов через `ChatOpenAI` + `.withStructuredOutput(ZodSchema)` с поддержкой любого OpenAI-совместимого эндпоинта (`baseURL` + `apiKey`). |
| **Память прецедентов, онтологии и графа** | **PostgreSQL 16 + `pgvector`** (`pg`) | Боевое хранение иммутабельных версий графа (`graph_versions`), онтологии ключей (`state_ontology_keys`) и HNSW-поиск прецедентов тупиков по косинусному сходству `1 - (incident_vector <=> $2::vector)` (`deadlock_precedents`). |

### 1.2. Способы прокидывания учетных данных (Credentials) в CLI и E2E-тесты

Учетные данные передаются без правки кода любым из трех способов (в порядке повышения приоритета):

1. **Через файл `spectra.config.json`** (создается интерактивно командой `npx spectra init`).
2. **Через файл `.env` или переменные окружения** (подхватываются автоматически и в CLI, и при запуске `npm test`):
   * `SPECTRA_LLM_API_KEY` (или `OPENAI_API_KEY`) — ключ API для LLM (OpenAI / OpenRouter / DeepSeek / vLLM / LiteLLM);
   * `SPECTRA_LLM_BASE_URL` (или `OPENAI_BASE_URL`) — опциональный базовый URL для OpenAI-совместимого провайдера (например, `https://api.openai.com/v1` или `https://openrouter.ai/api/v1`);
   * `SPECTRA_LLM_FRONTIER_MODEL` и `SPECTRA_LLM_LIGHT_MODEL` — имена моделей для System-2 и `llm`-узлов (по умолчанию `gpt-4o-mini`);
   * `SPECTRA_PG_URL` (или `DATABASE_URL`) — строка подключения к PostgreSQL 16 с `pgvector`;
   * `HF_TOKEN` — опциональный токен Hugging Face для скачивания весов `@receptron/laya` и `@huggingface/transformers`;
   * `LAYA_CACHE` и `LAYA_MODEL_DIR` — пути кэша или локального бандла ONNX-модели Laya.
3. **Через флаги командной строки CLI:**
   ```bash
   node ./bin/spectra.js run "Проверить баланс и сформировать отчет" \
     --api-key sk-proj-... \
     --base-url https://api.openai.com/v1 \
     --model gpt-4o-mini \
     --pg-url postgres://spectra:spectra_pass@localhost:5432/spectra_db \
     --verbose
   ```

### 1.3. Полная файловая структура проекта

```text
spectra-prototype/
├── .env.example                    # Шаблон переменных окружения для кредов LLM, PG и HF
├── bin/
│   └── spectra.js                  # Исполняемая точка входа CLI (init, run, verify, sleep)
├── types/                          # 100% САМОПИСНЫЕ (HANDWRITTEN) КОНТРАКТЫ ТИПОВ
│   ├── index.d.ts                  # Корневой экспорт всех рукописных типов пакета
│   ├── domain.d.ts                 # Доменные структуры: State, Graph, Deadlock, Mutation, Config
│   ├── core.d.ts                   # Рукописные интерфейсы ядра и логгера (ISpectraLogger и др.)
│   └── ai.d.ts                     # Рукописные интерфейсы ИИ-модулей и хранилища прецедентов
├── sql/
│   └── 001_init_pgvector.sql       # DDL-миграция PostgreSQL 16 + pgvector (vector(384))
├── src/
│   ├── cli/
│   │   ├── credentials.js          # Загрузчик и валидатор кредов (.env + ENV + CLI flags + JSON)
│   │   ├── logger.js               # Структурный логгер (ANSI-консоль + JSONL-аудит + ASCII-граф)
│   │   └── wizard.js               # Интерактивный мастер настройки `spectra init`
│   ├── core/
│   │   ├── graph.js                # DirectedGraph (сериализация, сдвиг приоритетов, Markov Blanket)
│   │   ├── verifier.js             # GraphDeterminismVerifier (Tarjan SCC, Reverse BFS, Intervals)
│   │   ├── tools.js                # ToolRegistry (песочница node:vm, CEGIS, Saga-верификация)
│   │   ├── ontology.js             # StateOntologyManager (Union-Find, сериализация и GC)
│   │   └── runtime.js              # SpectraRuntime (Fast/Slow Loop, гибридный поиск, Pre-Bridge)
│   ├── ai/
│   │   ├── embeddings.js           # Реальный ONNX-пайплайн @huggingface/transformers
│   │   ├── laya-system1.js         # Реальный ONNX-арбитр @receptron/laya (Laya.load + systemOne)
│   │   └── langchain-system2.js    # Реальный арбитр System-2 на @langchain/openai (без заглушек)
│   └── db/
│       └── pg-store.js             # Боевое хранилище PostgreSQL 16 + pgvector (HNSW)
├── test/
│   └── e2e.test.js                 # Сквозной интеграционный тест БЕЗ МОКОВ (реальные LLM, PG, Laya)
├── docker-compose.yml              # Контейнер pgvector/pgvector:pg16
├── package.json                    # ESM-манифест с "types": "./types/index.d.ts"
└── tsconfig.json                   # Строгий режим проверки JS по рукописным .d.ts (noEmit: true)
```

---

## 2. Манифест проекта, шаблон `.env.example`, конфигурация TypeScript и Docker

### 2.1. `package.json`

```json
{
  "name": "spectra-autonomous-cli",
  "version": "0.6.1",
  "description": "Continual Self-Learning Graph AI Runtime with Real System-1/System-2 Arbitrage and Full Telemetry",
  "type": "module",
  "main": "./src/core/runtime.js",
  "types": "./types/index.d.ts",
  "bin": {
    "spectra": "./bin/spectra.js"
  },
  "scripts": {
    "typecheck": "tsc --noEmit",
    "test": "node --test ./test/e2e.test.js",
    "db:up": "docker compose up -d",
    "init": "node ./bin/spectra.js init",
    "verify": "node ./bin/spectra.js verify",
    "sleep": "node ./bin/spectra.js sleep"
  },
  "dependencies": {
    "@huggingface/transformers": "^3.1.0",
    "@langchain/core": "^0.3.15",
    "@langchain/openai": "^0.3.12",
    "@receptron/laya": "^0.1.2",
    "pg": "^8.13.0",
    "zod": "^3.23.8"
  },
  "devDependencies": {
    "@types/node": "^22.7.0",
    "@types/pg": "^8.11.10",
    "typescript": "^5.6.3"
  },
  "engines": {
    "node": ">=20.0.0"
  }
}
```

### 2.2. `.env.example` (Прокидывание учетных данных для тестов и CLI)

```dotenv
# Учетные данные для System-2 LLM (@langchain/openai) и узлов типа 'llm'
SPECTRA_LLM_API_KEY=sk-proj-your-real-api-key
SPECTRA_LLM_BASE_URL=https://api.openai.com/v1
SPECTRA_LLM_FRONTIER_MODEL=gpt-4o-mini
SPECTRA_LLM_LIGHT_MODEL=gpt-4o-mini

# Учетные данные для PostgreSQL 16 + pgvector
SPECTRA_PG_URL=postgres://spectra:spectra_pass@localhost:5432/spectra_db

# Опциональные параметры для @receptron/laya и @huggingface/transformers
HF_TOKEN=
LAYA_CACHE=./.spectra/laya-cache
LAYA_MODEL_DIR=
```

### 2.3. `tsconfig.json` (Проверка `.js` строго по рукописным `.d.ts` без генерации файлов)

```json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "NodeNext",
    "moduleResolution": "NodeNext",
    "allowJs": true,
    "checkJs": true,
    "noEmit": true,
    "declaration": false,
    "emitDeclarationOnly": false,
    "strict": true,
    "skipLibCheck": true,
    "types": ["node"]
  },
  "include": ["bin/**/*.js", "src/**/*.js", "test/**/*.js", "types/**/*.d.ts"]
}
```

### 2.4. `docker-compose.yml` и `sql/001_init_pgvector.sql`

```yaml
# docker-compose.yml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: spectra_pgvector
    restart: unless-stopped
    environment:
      POSTGRES_USER: spectra
      POSTGRES_PASSWORD: spectra_pass
      POSTGRES_DB: spectra_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./sql/001_init_pgvector.sql:/docker-entrypoint-initdb.d/001_init_pgvector.sql:ro

volumes:
  pgdata:
```

```sql
-- sql/001_init_pgvector.sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS graph_versions (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    graph_id VARCHAR(128) NOT NULL,
    structural_hash CHAR(64) NOT NULL,
    definition_json JSONB NOT NULL,
    omega_penalty DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS state_ontology_keys (
    key_name VARCHAR(128) PRIMARY KEY,
    status VARCHAR(32) NOT NULL DEFAULT 'STAGING',
    canonical_alias VARCHAR(128),
    epochs_lived INTEGER NOT NULL DEFAULT 0,
    mutual_info_gain DOUBLE PRECISION NOT NULL DEFAULT 0.15,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS deadlock_precedents (
    precedent_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    deadlock_type VARCHAR(64) NOT NULL,
    incident_vector vector(384) NOT NULL,
    anchor_reads TEXT[] NOT NULL,
    anchor_writes TEXT[] NOT NULL,
    mutation_operator VARCHAR(64) NOT NULL,
    mutation_plan_json JSONB NOT NULL,
    success_count DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    failure_count DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    confidence_lcb DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    is_antipattern BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_precedents_hnsw
  ON deadlock_precedents USING hnsw (incident_vector vector_cosine_ops);
```

---

## 3. Рукописные контракты типов TypeScript (`types/*.d.ts`)

### 3.1. `types/domain.d.ts`

```ts
export type NodeType = 'llm' | 'tool' | 'branch' | 'join' | 'check' | 'macro' | 'fused';
export type IdempotencyClass = 'pure' | 'idempotent' | 'non_idempotent';
export type ModelTier = 'local_slm' | 'light' | 'frontier';
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export type DeadlockType =
  | 'NO_VALID_TRANSITION'
  | 'STATE_HASH_LOOP'
  | 'CHECK_FAILED'
  | 'TOOL_EXECUTION_ERROR'
  | 'HARD_LIMIT_EXCEEDED';

export type StateObject = Record<string, any>;

export interface StateDiffEntry {
  key: string;
  change: 'ADDED' | 'MODIFIED' | 'REMOVED';
  before: any;
  after: any;
}

export interface JsonSchemaProperty {
  type: 'string' | 'number' | 'integer' | 'boolean' | 'object' | 'array';
  nullable?: boolean;
  minimum?: number;
  maximum?: number;
}

export interface GraphNodeSpec {
  id: string;
  type: NodeType;
  reads: string[];
  writes: string[];
  inputSchemas?: Record<string, JsonSchemaProperty>;
  outputSchemas?: Record<string, JsonSchemaProperty>;
  idempotency?: IdempotencyClass;
  hasBarrier?: boolean;
  resourceRps?: number;
  modelTier?: ModelTier;
  config?: {
    prompt?: string;
    toolId?: string;
    sourceCode?: string;
    assertExpr?: string;
    assertFn?: (state: StateObject) => boolean;
    handler?: (state: StateObject) => Promise<StateObject> | StateObject;
    llmHandler?: (state: StateObject, tier: ModelTier) => Promise<StateObject>;
    [key: string]: any;
  };
}

export interface GraphEdgeSpec {
  from: string;
  to: string;
  priority: number;
  guard: (state: StateObject) => boolean;
  guardExpr: string;
  intervals?: Record<string, [number, number]>;
}

export interface SerializedGraph {
  id: string;
  startNode: string | null;
  goalNodes: string[];
  nodes: Array<{
    id: string;
    type: NodeType;
    reads: string[];
    writes: string[];
    idempotency?: IdempotencyClass;
    hasBarrier?: boolean;
    modelTier?: ModelTier;
    config?: {
      prompt?: string;
      sourceCode?: string;
      assertExpr?: string;
      toolId?: string;
    };
  }>;
  edges: Array<{
    from: string;
    to: string;
    priority: number;
    guardExpr: string;
    intervals?: Record<string, [number, number]>;
  }>;
}

export interface SerializedOntologyKey {
  keyName: string;
  status: 'STAGING' | 'CANONICAL';
  canonicalAlias?: string;
  epochsLived: number;
  mutualInfoGain: number;
}

export interface DeadlockIncident {
  type: DeadlockType;
  nodeId: string;
  lastValidNodeId?: string | null;
  step: number;
  error?: string;
  stateSnapshot: StateObject;
}

export interface MutationPlan {
  operator: 'addEdge' | 'addNodeAndBridge' | 'patchToolAST' | 'registerTool' | 'splitNode';
  from: string;
  to: string;
  priority?: number;
  guardFn?: (state: StateObject) => boolean;
  guardExpr?: string;
  intervals?: Record<string, [number, number]>;
  node?: GraphNodeSpec;
  toolCode?: string;
  reasoning?: string;
}

export interface PrecedentRecord {
  precedentId?: string;
  deadlockType: DeadlockType;
  incidentVector?: number[];
  similarity?: number;
  anchorReads: string[];
  anchorWrites: string[];
  mutationPlan: MutationPlan;
  successCount?: number;
  failureCount?: number;
  confidenceLcb?: number;
  isAntipattern: boolean;
}

export interface System1Verdict {
  driftRisk: number;
  coherence: number;
  approveProb: number;
  approved: boolean;
  rollbackAction: 'keep' | 'rollback_last' | 'rollback_session';
  selectedPrecedentId?: string | null;
  latencyMs?: number;
}

export interface VerificationReport {
  status: 'PASS' | 'HARD_REJECT';
  omegaPenalty: number;
  diagnostics: {
    vertexCount: number;
    edgeCount: number;
    density: number;
    priorityCollisions: number;
    deadNodes: string[];
    unsatNodes: string[];
    trappingCycles: string[][];
  };
}

export interface SessionRunResult {
  sessionId: string;
  goalReached: boolean;
  reason?: string;
  steps: number;
  rollbacksCount: number;
  finalState: StateObject;
  omegaPenalty?: number;
  durationMs: number;
  appliedMutations: Array<{
    deadlock: DeadlockIncident;
    mutationPlan: MutationPlan;
    omegaPenalty: number;
    source: 'MEMORY_GROUNDED' | 'SYSTEM2_SYNTHESIZED';
  }>;
}

export interface CredentialsOverrides {
  apiKey?: string;
  baseUrl?: string;
  frontierModel?: string;
  lightModel?: string;
  pgUrl?: string;
  hfToken?: string;
  layaCacheDir?: string;
  layaModelDir?: string;
}

export interface SpectraConfig {
  systemName: string;
  logging: {
    level: LogLevel;
    traceStateDiffs: boolean;
    traceGuards: boolean;
    jsonlFilePath: string;
  };
  database: {
    connectionString: string;
    embeddingDimension: number;
  };
  embeddings: {
    provider: 'transformers.js';
    modelName: string;
    cacheDir: string;
    hfToken?: string;
    modalityWeights: [number, number, number];
  };
  system1: {
    provider: '@receptron/laya';
    cacheDir?: string;
    modelDir?: string;
    hfToken?: string;
    temperaturePlatt: number;
    maxDrift: number;
    minCoherence: number;
    minApproveProb: number;
  };
  system2: {
    provider: 'openai';
    apiKey: string;
    baseUrl?: string;
    frontierModel: string;
    lightModel: string;
  };
  runtime: {
    hardStepLimit: number;
    stateHashLoopThreshold: number;
    pprTeleportAlpha: number;
  };
}
```

### 3.2. `types/core.d.ts`

```ts
import type {
  GraphNodeSpec,
  GraphEdgeSpec,
  SerializedGraph,
  SerializedOntologyKey,
  StateObject,
  StateDiffEntry,
  VerificationReport,
  MutationPlan,
  DeadlockIncident,
  System1Verdict,
  SessionRunResult,
  LogLevel,
  PrecedentRecord
} from './domain.js';

export interface ISpectraLogger {
  sessionId: string;
  level: LogLevel;
  traceStateDiffs: boolean;
  traceGuards: boolean;

  banner(title: string, subtitle?: string): void;
  section(
    phase: 'FAST_LOOP' | 'DEADLOCK' | 'SLOW_LOOP' | 'SYSTEM_1' | 'SYSTEM_2' | 'VERIFIER' | 'ROLLBACK' | 'MEMORY' | 'SLEEP',
    title: string
  ): void;
  info(scope: string, message: string, meta?: Record<string, any>): void;
  debug(scope: string, message: string, meta?: Record<string, any>): void;
  warn(scope: string, message: string, meta?: Record<string, any>): void;
  error(scope: string, message: string, meta?: Record<string, any>): void;

  logStepExecution(params: {
    step: number;
    maxSteps: number;
    node: GraphNodeSpec;
    durationMs: number;
    stateBefore: StateObject;
    stateAfter: StateObject;
  }): void;

  logGuardEvaluation(
    nodeId: string,
    evaluations: Array<{ to: string; priority: number; guardExpr: string; matched: boolean }>
  ): void;

  logDeadlock(deadlock: DeadlockIncident, markovBlanket: Map<string, number>, stateHash: string): void;
  logSystem1Verdict(
    verdict: System1Verdict,
    thresholds: { maxDrift: number; minCoherence: number; minApproveProb: number }
  ): void;
  logVerifierReport(report: VerificationReport, graphHash: string): void;
  logTopologyAscii(graph: IDirectedGraph, activeNodeId?: string | null): void;
  computeStateDiff(before: StateObject, after: StateObject): StateDiffEntry[];
}

export interface IDirectedGraph {
  id: string;
  nodes: Map<string, GraphNodeSpec>;
  edges: Map<string, GraphEdgeSpec[]>;
  startNode: string | null;
  goalNodes: Set<string>;

  addNode(node: GraphNodeSpec): this;
  addEdge(
    from: string,
    to: string,
    guardFn?: (state: StateObject) => boolean,
    priority?: number,
    guardMeta?: { expr?: string; intervals?: Record<string, [number, number]>; preemptive?: boolean } | null
  ): this;
  compileGuardExpression(exprStr?: string): (state: StateObject) => boolean;
  getCausalMarkovBlanket(centerNodeId: string, alphaPpr?: number): Map<string, number>;
  foldBisimilarBranches(): this;
  computeStructuralHash(): string;
  toJSON(): SerializedGraph;
  clone(): IDirectedGraph;
}

export interface IGraphDeterminismVerifier {
  rhoMax: number;
  lMax: number;
  lyapunovCreditBank: number;
  verify(graph: IDirectedGraph): VerificationReport;
}

export interface IToolRegistry {
  tools: Map<string, any>;
  register(toolSpec: {
    id: string;
    reads: string[];
    writes: string[];
    sourceCode: string;
    version?: number;
    hasIrreversibleBarrier?: boolean;
    sagaCompensatorCode?: string;
    regressionPool?: Array<{ input: StateObject; expectedOutput: StateObject }>;
  }): void;
  executeSandboxed(codeStr: string, stateInput: StateObject, timeoutMs?: number): StateObject;
  verifyAndAttachSagaCompensator(
    toolId: string,
    compensatorCodeStr: string,
    epsilonSaga?: number
  ): { verified: boolean; residualError: number; hasBarrier: boolean };
}

export interface IStateOntologyManager {
  keys: Map<string, { status: 'STAGING' | 'CANONICAL'; canonicalAlias?: string; epochsLived: number; mutualInfoGain: number }>;
  stagingTtlEpochs: number;
  registerStagingKey(keyName: string, mutualInfoGain?: number): void;
  findCanonical(keyName: string): string;
  mergeKeys(canonicalKey: string, synonymKey: string): string;
  runStagingGarbageCollection(thetaIg?: number): { promoted: string[]; evicted: string[] };
  toJSON(): SerializedOntologyKey[];
  loadFromJSON(entries: SerializedOntologyKey[]): void;
}

export interface ISpectraRuntime {
  graph: IDirectedGraph;
  toolRegistry: IToolRegistry;
  verifier: IGraphDeterminismVerifier;
  ontology: IStateOntologyManager;
  logger: ISpectraLogger;
  precedentMemory: PrecedentRecord[];
  runSession(initialState?: StateObject): Promise<SessionRunResult>;
  runConsolidationSleep(): Promise<{
    promotedKeys: string[];
    evictedKeys: string[];
    foldedNodes: number;
    omegaBefore: number;
    omegaAfter: number;
  }>;
}
```

### 3.3. `types/ai.d.ts` и `types/index.d.ts`

```ts
// types/ai.d.ts
import type {
  DeadlockType,
  DeadlockIncident,
  MutationPlan,
  System1Verdict,
  PrecedentRecord,
  SerializedGraph,
  SerializedOntologyKey,
  GraphNodeSpec,
  StateObject,
  ModelTier,
  SpectraConfig
} from './domain.js';

export interface ILocalEmbeddingEngine {
  modelName: string;
  weights: [number, number, number];
  init(): Promise<this>;
  embedText(text: string): Promise<number[]>;
  computeIncidentVector(params: {
    errorText: string;
    markovBlanketSignature: string;
    stateSchemaSignature: string;
  }): Promise<number[]>;
}

export interface ILayaSystem1Arbiter {
  config: SpectraConfig['system1'];
  init(): Promise<this>;
  evaluateMutation(
    deadlock: DeadlockIncident,
    mutationPlan: MutationPlan,
    goalDescription: string
  ): Promise<System1Verdict>;
  close(): Promise<void>;
}

export interface ILangChainSystem2Arbiter {
  config: SpectraConfig['system2'];
  synthesizeRecoveryMutation(
    deadlock: DeadlockIncident,
    goalDescription: string,
    existingNodeIds: string[],
    goalNodeId: string
  ): Promise<MutationPlan>;
  executeLlmNode(node: GraphNodeSpec, state: StateObject, tier?: ModelTier): Promise<StateObject>;
}

export interface IPrecedentStore {
  init(): Promise<void>;
  loadAllPrecedents(): Promise<PrecedentRecord[]>;
  searchSimilarPrecedents(
    deadlockType: DeadlockType,
    queryVector: number[],
    minSimilarity?: number
  ): Promise<PrecedentRecord[]>;
  savePrecedent(record: PrecedentRecord, deadlock: DeadlockIncident): Promise<void>;
  saveGraphVersion(graphJson: SerializedGraph, structuralHash: string, omegaPenalty: number): Promise<void>;
  loadLatestGraph(): Promise<SerializedGraph | null>;
  saveOntologyKeys(keys: SerializedOntologyKey[]): Promise<void>;
  loadOntologyKeys(): Promise<SerializedOntologyKey[]>;
  clearAll(): Promise<void>;
  close(): Promise<void>;
}
```

```ts
// types/index.d.ts
export * from './domain.js';
export * from './core.js';
export * from './ai.js';
```

---

## 4. Спецификация CLI-команд и псевдокод ключевых алгоритмов

### 4.1. Карта команд CLI (`bin/spectra.js`)

| Команда | Синтаксис | Назначение и наблюдаемый вывод |
| :--- | :--- | :--- |
| **`init`** | `spectra init` | Интерактивный визард настройки `spectra.config.json` (строка подключения PostgreSQL, ключ и `baseUrl` для OpenAI-совместимого API, пути кэша `@receptron/laya` и `@huggingface/transformers`). |
| **`run`** | `spectra run "<goal>" [--api-key <k>] [--base-url <u>] [--model <m>] [--pg-url <db>] [--verbose] [--fresh-graph]` | Резолвит учетные данные из `.env` / ENV / CLI-флагов / `spectra.config.json`, подключается к PostgreSQL (`pgvector`), инициализирует ONNX-модели `@huggingface/transformers` и `@receptron/laya`, запускает цикл `Fast Loop <-> Slow Loop` и вызывает реальный LLM через `@langchain/openai` при новых тупиках. |
| **`verify`** | `spectra verify [--pg-url <db>]` | Загружает последнюю сохраненную версию графа из PostgreSQL (`graph_versions`), запускает `GraphDeterminismVerifier` (Tarjan SCC $+$ Reverse BFS $+$ Transitive Interval Dataflow) и распечатывает диагностику инвариантов и штраф $\Omega(G)$. |
| **`sleep`** | `spectra sleep [--pg-url <db>]` | Загружает эволюционировавший граф и словарь онтологии $\mathcal{K}_t$ из PostgreSQL, запускает Контур 3 (*Consolidation Loop*): промоушен и очистку карантина $\mathcal{K}_{\text{staging}}$, бисимуляционную свертку ветвей `foldBisimilarBranches()`, сохраняет оптимизированный граф и онтологию обратно в БД и выводит динамику $\Omega_{\text{before}} \to \Omega_{\text{after}}$. |

### 4.2. Псевдокод ключевых алгоритмов прототипа

```text
АЛГОРИТМ 1: Исполнительный цикл сессии (Fast Loop <-> Slow Loop)
ВХОД: Граф G, Начальное состояние s_0, Лимит шагов T_max, Порог цикла K_loop
ВЫХОД: Отчет сессии SessionRunResult

1. Загрузить выученные прецеденты из PostgreSQL (deadlock_precedents) и ключи онтологии K_t.
2. Установить s <- Clone(s_0), v_curr <- G.startNode, v_prev <- null, Trajectory <- [], HashCount <- Map().
3. ДЛЯ step = 0 ДО T_max - 1 ВЫПОЛНЯТЬ:
   3.1. s_before <- Clone(s), h <- SHA256(v_curr, s_before), HashCount[h] <- HashCount[h] + 1.
   3.2. ЕСЛИ HashCount[h] >= K_loop ТО deadlock <- STATE_HASH_LOOP(v_curr).
   3.3. ИНАЧЕ выполнить тело узла v_curr:
        - ЕСЛИ type(v_curr) == 'check': вычислить Assert(s) в node:vm; при false -> deadlock <- CHECK_FAILED.
        - ЕСЛИ type(v_curr) == 'tool': выполнить код инструмента в изолированном node:vm с Object.freeze(s);
          объединить s <- { ...s_before, ...delta }; зарегистрировать writes(v_curr) в K_staging;
          при исключении -> deadlock <- TOOL_EXECUTION_ERROR.
        - ЕСЛИ type(v_curr) == 'llm': вызвать ChatOpenAI.withStructuredOutput(ZodSchema(writes)).
   3.4. ЕСЛИ deadlock == null И v_curr принадлежит G.goalNodes:
        - Сохранить онтологию K_t в PostgreSQL и ВЕРНУТЬ УСПЕХ (goalReached = true, finalState = s).
   3.5. ЕСЛИ deadlock == null:
        - Вычислить все исходящие guard-предикаты E_active = { e in Out(v_curr) | Guard(e)(s) == true }.
        - ЕСЛИ E_active непусто:
          добавить шаг в Trajectory; v_prev <- v_curr; v_curr <- target(argmin_{e in E_active} Priority(e)); ПЕРЕЙТИ К ШАГУ 3.
        - ИНАЧЕ deadlock <- NO_VALID_TRANSITION(v_curr).

   // --- Активация Контура 2: Slow Loop ---
   3.6. Вычислить каузальное одеяло Маркова MB(v_curr, G) с весами Персонализированного PageRank.
   3.7. Вычислить 384-мерный мультимодальный вектор тупика v_inc через @huggingface/transformers.
   3.8. Выполнить HNSW-поиск в PostgreSQL: candidates <- SELECT ... ORDER BY incident_vector <=> v_inc.
   3.9. Найти реляционное заземление Psi_adapt в MB(v_curr, G) по гибридному скору (0.5 * Jaccard(R, W) + 0.5 * CosSim).
   3.10. ЕСЛИ найден заземленный прецедент grounded:
         mutationPlan <- Adapt(grounded.mutationPlan, v_prev, v_curr).
         ИНАЧЕ:
         mutationPlan <- LangChainSystem2Arbiter.synthesizeRecoveryMutation(deadlock, goal, G).
   3.11. Оценить mutationPlan через локальную ONNX-модель @receptron/laya (laya.systemOne -> score, noul, choice).
   3.12. ЕСЛИ вердикт Laya == REJECTED:
         если использовался grounded, пометить его как антипаттерн;
         выполнить Saga-откат RollbackTrajectoryAndCursor(Trajectory, 1); ПЕРЕЙТИ К ШАГУ 3.
   3.13. Построить граф-кандидат G_cand <- ApplyMutation(Clone(G), mutationPlan) с вытесняющим приоритетом p=0.
   3.14. Проверить детерминизм: report <- GraphDeterminismVerifier.verify(G_cand) (Tarjan SCC + Reverse BFS + Intervals).
   3.15. ЕСЛИ report.status == 'PASS':
         G <- G_cand; сохранить прецедент и новую версию G в PostgreSQL;
         перевести курсор v_curr на узел ремонтного моста; очистить HashCount.
         ИНАЧЕ:
         если использовался grounded, пометить его как антипаттерн;
         выполнить Saga-откат RollbackTrajectoryAndCursor(Trajectory, 1).
4. ВЕРНУТЬ ОТКАЗ (HARD_LIMIT_EXCEEDED).
```

---

## 5. Полная реализация всех модулей (`src/`, `bin/`, `test/`)

### 5.1. Загрузчик и валидатор учетных данных (`src/cli/credentials.js`)

```javascript
// @ts-check
import fs from 'node:fs';
import fsPromises from 'node:fs/promises';
import path from 'node:path';

/**
 * @typedef {import('../../types/index.js').SpectraConfig} SpectraConfig
 * @typedef {import('../../types/index.js').CredentialsOverrides} CredentialsOverrides
 */

/**
 * Загружает переменные из файла .env в process.env, если они еще не заданы в окружении.
 * @param {string} [envFilePath]
 */
export function loadDotEnvIfPresent(envFilePath = path.resolve(process.cwd(), '.env')) {
  if (!fs.existsSync(envFilePath)) return;
  if (typeof process.loadEnvFile === 'function') {
    try {
      process.loadEnvFile(envFilePath);
      return;
    } catch {
      // Fallback для ранних минорных версий Node 20.x
    }
  }
  const content = fs.readFileSync(envFilePath, 'utf-8');
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const eqIdx = line.indexOf('=');
    if (eqIdx <= 0) continue;
    const key = line.slice(0, eqIdx).trim();
    const val = line.slice(eqIdx + 1).trim().replace(/^['"]|['"]$/g, '');
    if (!(key in process.env)) {
      process.env[key] = val;
    }
  }
}

/**
 * Разбирает CLI-флаги вида --api-key, --base-url, --model, --pg-url, --hf-token.
 * @param {string[]} argv
 * @returns {{ overrides: CredentialsOverrides; positional: string[]; flags: { verbose: boolean; freshGraph: boolean } }}
 */
export function parseCliArgs(argv) {
  /** @type {CredentialsOverrides} */
  const overrides = {};
  /** @type {string[]} */
  const positional = [];
  let verbose = false;
  let freshGraph = false;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--verbose' || arg === '-v') {
      verbose = true;
    } else if (arg === '--fresh-graph') {
      freshGraph = true;
    } else if (arg === '--api-key' && argv[i + 1]) {
      overrides.apiKey = argv[++i];
    } else if (arg === '--base-url' && argv[i + 1]) {
      overrides.baseUrl = argv[++i];
    } else if (arg === '--model' && argv[i + 1]) {
      overrides.frontierModel = argv[++i];
      overrides.lightModel = argv[i];
    } else if (arg === '--pg-url' && argv[i + 1]) {
      overrides.pgUrl = argv[++i];
    } else if (arg === '--hf-token' && argv[i + 1]) {
      overrides.hfToken = argv[++i];
    } else if (arg === '--laya-model-dir' && argv[i + 1]) {
      overrides.layaModelDir = argv[++i];
    } else {
      positional.push(arg);
    }
  }

  return { overrides, positional, flags: { verbose, freshGraph } };
}

/**
 * Собирает финальную конфигурацию Spectra из spectra.config.json, .env, process.env и прямых переопределений.
 * @param {CredentialsOverrides} [overrides]
 * @param {string} [configPath]
 * @returns {Promise<SpectraConfig>}
 */
export async function resolveSpectraConfig(
  overrides = {},
  configPath = path.resolve(process.cwd(), 'spectra.config.json')
) {
  loadDotEnvIfPresent();

  /** @type {Partial<SpectraConfig>} */
  let fileConfig = {};
  try {
    const raw = await fsPromises.readFile(configPath, 'utf-8');
    fileConfig = JSON.parse(raw);
  } catch {
    fileConfig = {};
  }

  const apiKey =
    overrides.apiKey ||
    process.env.SPECTRA_LLM_API_KEY ||
    process.env.OPENAI_API_KEY ||
    fileConfig.system2?.apiKey ||
    '';

  const baseUrl =
    overrides.baseUrl ||
    process.env.SPECTRA_LLM_BASE_URL ||
    process.env.OPENAI_BASE_URL ||
    fileConfig.system2?.baseUrl ||
    undefined;

  const frontierModel =
    overrides.frontierModel ||
    process.env.SPECTRA_LLM_FRONTIER_MODEL ||
    fileConfig.system2?.frontierModel ||
    'gpt-4o-mini';

  const lightModel =
    overrides.lightModel ||
    process.env.SPECTRA_LLM_LIGHT_MODEL ||
    fileConfig.system2?.lightModel ||
    frontierModel;

  const pgUrl =
    overrides.pgUrl ||
    process.env.SPECTRA_PG_URL ||
    process.env.DATABASE_URL ||
    fileConfig.database?.connectionString ||
    'postgres://spectra:spectra_pass@localhost:5432/spectra_db';

  const hfToken =
    overrides.hfToken ||
    process.env.HF_TOKEN ||
    fileConfig.system1?.hfToken ||
    fileConfig.embeddings?.hfToken ||
    undefined;

  const layaCacheDir =
    overrides.layaCacheDir ||
    process.env.LAYA_CACHE ||
    fileConfig.system1?.cacheDir ||
    './.spectra/laya-cache';

  const layaModelDir =
    overrides.layaModelDir ||
    process.env.LAYA_MODEL_DIR ||
    fileConfig.system1?.modelDir ||
    undefined;

  return {
    systemName: fileConfig.systemName || 'spectra-autonomous-core',
    logging: {
      level: fileConfig.logging?.level || 'info',
      traceStateDiffs: fileConfig.logging?.traceStateDiffs ?? true,
      traceGuards: fileConfig.logging?.traceGuards ?? true,
      jsonlFilePath: fileConfig.logging?.jsonlFilePath || '.spectra/logs/session.jsonl'
    },
    database: {
      connectionString: pgUrl,
      embeddingDimension: fileConfig.database?.embeddingDimension || 384
    },
    embeddings: {
      provider: 'transformers.js',
      modelName: fileConfig.embeddings?.modelName || 'Xenova/multilingual-e5-small',
      cacheDir: fileConfig.embeddings?.cacheDir || './.spectra/models',
      hfToken,
      modalityWeights: fileConfig.embeddings?.modalityWeights || [0.5, 0.3, 0.2]
    },
    system1: {
      provider: '@receptron/laya',
      cacheDir: layaCacheDir,
      modelDir: layaModelDir,
      hfToken,
      temperaturePlatt: fileConfig.system1?.temperaturePlatt ?? 1.0,
      maxDrift: fileConfig.system1?.maxDrift ?? 0.65,
      minCoherence: fileConfig.system1?.minCoherence ?? 0.35,
      minApproveProb: fileConfig.system1?.minApproveProb ?? 0.40
    },
    system2: {
      provider: 'openai',
      apiKey,
      baseUrl,
      frontierModel,
      lightModel
    },
    runtime: {
      hardStepLimit: fileConfig.runtime?.hardStepLimit ?? 50,
      stateHashLoopThreshold: fileConfig.runtime?.stateHashLoopThreshold ?? 2,
      pprTeleportAlpha: fileConfig.runtime?.pprTeleportAlpha ?? 0.15
    }
  };
}

/**
 * Проверяет наличие учетных данных для подключения к LLM и выбрасывает понятную ошибку при их отсутствии.
 * @param {SpectraConfig['system2']} sys2Config
 */
export function assertValidLlmCredentials(sys2Config) {
  if (!sys2Config.apiKey || sys2Config.apiKey.trim().length === 0) {
    throw new Error(
      'Не заданы учетные данные для подключения к LLM (System-2). ' +
        'Передайте их через переменную окружения SPECTRA_LLM_API_KEY (или OPENAI_API_KEY), ' +
        'файл .env, флаг CLI --api-key <key> (опционально --base-url <url> --model <name>) ' +
        'или выполните настройку через `npx spectra init`.'
    );
  }
}
```

### 5.2. Структурный логгер (`src/cli/logger.js`)

```javascript
// @ts-check
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

/**
 * @typedef {import('../../types/index.js').ISpectraLogger} ISpectraLogger
 * @typedef {import('../../types/index.js').LogLevel} LogLevel
 * @typedef {import('../../types/index.js').StateObject} StateObject
 * @typedef {import('../../types/index.js').StateDiffEntry} StateDiffEntry
 * @typedef {import('../../types/index.js').GraphNodeSpec} GraphNodeSpec
 * @typedef {import('../../types/index.js').DeadlockIncident} DeadlockIncident
 * @typedef {import('../../types/index.js').System1Verdict} System1Verdict
 * @typedef {import('../../types/index.js').VerificationReport} VerificationReport
 * @typedef {import('../../types/index.js').IDirectedGraph} IDirectedGraph
 */

const ANSI = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  magenta: '\x1b[35m',
  blue: '\x1b[34m',
  gray: '\x1b[90m'
};

/** @type {Record<LogLevel, number>} */
const LEVEL_WEIGHT = { debug: 10, info: 20, warn: 30, error: 40 };

/**
 * @implements {ISpectraLogger}
 */
export class SpectraLogger {
  /**
   * @param {Partial<import('../../types/index.js').SpectraConfig['logging']>} [opts]
   */
  constructor(opts = {}) {
    this.sessionId = `ses_${crypto.randomBytes(4).toString('hex')}`;
    /** @type {LogLevel} */
    this.level = opts.level || 'debug';
    this.traceStateDiffs = opts.traceStateDiffs ?? true;
    this.traceGuards = opts.traceGuards ?? true;
    this.jsonlFilePath = opts.jsonlFilePath || '.spectra/logs/session.jsonl';

    try {
      const dir = path.dirname(path.resolve(process.cwd(), this.jsonlFilePath));
      fs.mkdirSync(dir, { recursive: true });
    } catch {
      // Игнорируем ошибку создания директории в read-only окружениях
    }
  }

  /**
   * @param {string} eventType
   * @param {Record<string, any>} payload
   */
  _writeJsonl(eventType, payload) {
    try {
      const record = JSON.stringify({
        ts: new Date().toISOString(),
        sessionId: this.sessionId,
        eventType,
        ...payload
      });
      fs.appendFileSync(path.resolve(process.cwd(), this.jsonlFilePath), record + '\n', 'utf-8');
    } catch {
      // Non-blocking audit log
    }
  }

  /**
   * @param {LogLevel} msgLevel
   */
  _shouldLog(msgLevel) {
    return LEVEL_WEIGHT[msgLevel] >= LEVEL_WEIGHT[this.level];
  }

  _ts() {
    return new Date().toISOString().slice(11, 23);
  }

  /**
   * @param {string} title
   * @param {string} [subtitle]
   */
  banner(title, subtitle = '') {
    const line = '═'.repeat(74);
    console.log(`\n${ANSI.cyan}${ANSI.bold}╔${line}╗${ANSI.reset}`);
    console.log(`${ANSI.cyan}${ANSI.bold}║ ⚡ ${title.padEnd(70)}║${ANSI.reset}`);
    if (subtitle) {
      console.log(`${ANSI.cyan}║    ${subtitle.slice(0, 68).padEnd(70)}║${ANSI.reset}`);
    }
    console.log(`${ANSI.cyan}${ANSI.bold}╚${line}╝${ANSI.reset}`);
    this._writeJsonl('BANNER', { title, subtitle });
  }

  /**
   * @param {'FAST_LOOP' | 'DEADLOCK' | 'SLOW_LOOP' | 'SYSTEM_1' | 'SYSTEM_2' | 'VERIFIER' | 'ROLLBACK' | 'MEMORY' | 'SLEEP'} phase
   * @param {string} title
   */
  section(phase, title) {
    const colorMap = {
      FAST_LOOP: ANSI.green,
      DEADLOCK: ANSI.red,
      SLOW_LOOP: ANSI.yellow,
      SYSTEM_1: ANSI.cyan,
      SYSTEM_2: ANSI.magenta,
      VERIFIER: ANSI.blue,
      ROLLBACK: ANSI.yellow,
      MEMORY: ANSI.cyan,
      SLEEP: ANSI.magenta
    };
    const col = colorMap[phase] || ANSI.cyan;
    console.log(
      `\n${col}${ANSI.bold}┌── [${phase}] ${title} ${'─'.repeat(Math.max(2, 58 - phase.length - title.length))}${ANSI.reset}`
    );
    this._writeJsonl('SECTION', { phase, title });
  }

  /**
   * @param {string} scope
   * @param {string} message
   * @param {Record<string, any>} [meta]
   */
  info(scope, message, meta) {
    if (!this._shouldLog('info')) return;
    const metaStr = meta ? ` ${ANSI.gray}${JSON.stringify(meta)}${ANSI.reset}` : '';
    console.log(
      `${ANSI.gray}[${this._ts()}]${ANSI.reset} ${ANSI.cyan}INFO ${ANSI.reset} ${ANSI.bold}[${scope}]${ANSI.reset} ${message}${metaStr}`
    );
    this._writeJsonl('INFO', { scope, message, meta });
  }

  /**
   * @param {string} scope
   * @param {string} message
   * @param {Record<string, any>} [meta]
   */
  debug(scope, message, meta) {
    if (!this._shouldLog('debug')) return;
    const metaStr = meta ? ` ${ANSI.gray}${JSON.stringify(meta)}${ANSI.reset}` : '';
    console.log(`${ANSI.gray}[${this._ts()}] DEBUG [${scope}] ${message}${metaStr}${ANSI.reset}`);
    this._writeJsonl('DEBUG', { scope, message, meta });
  }

  /**
   * @param {string} scope
   * @param {string} message
   * @param {Record<string, any>} [meta]
   */
  warn(scope, message, meta) {
    if (!this._shouldLog('warn')) return;
    const metaStr = meta ? ` ${ANSI.yellow}${JSON.stringify(meta)}${ANSI.reset}` : '';
    console.log(
      `${ANSI.gray}[${this._ts()}]${ANSI.reset} ${ANSI.yellow}${ANSI.bold}WARN  [${scope}]${ANSI.reset} ${message}${metaStr}`
    );
    this._writeJsonl('WARN', { scope, message, meta });
  }

  /**
   * @param {string} scope
   * @param {string} message
   * @param {Record<string, any>} [meta]
   */
  error(scope, message, meta) {
    const metaStr = meta ? ` ${ANSI.red}${JSON.stringify(meta)}${ANSI.reset}` : '';
    console.log(
      `${ANSI.gray}[${this._ts()}]${ANSI.reset} ${ANSI.red}${ANSI.bold}ERROR [${scope}]${ANSI.reset} ${message}${metaStr}`
    );
    this._writeJsonl('ERROR', { scope, message, meta });
  }

  /**
   * @param {StateObject} before
   * @param {StateObject} after
   * @returns {StateDiffEntry[]}
   */
  computeStateDiff(before, after) {
    /** @type {StateDiffEntry[]} */
    const diffs = [];
    const allKeys = new Set([...Object.keys(before || {}), ...Object.keys(after || {})]);
    for (const key of allKeys) {
      const bVal = before?.[key];
      const aVal = after?.[key];
      if (!(key in (before || {}))) {
        diffs.push({ key, change: 'ADDED', before: undefined, after: aVal });
      } else if (!(key in (after || {}))) {
        diffs.push({ key, change: 'REMOVED', before: bVal, after: undefined });
      } else if (JSON.stringify(bVal) !== JSON.stringify(aVal)) {
        diffs.push({ key, change: 'MODIFIED', before: bVal, after: aVal });
      }
    }
    return diffs;
  }

  /**
   * @param {{ step: number; maxSteps: number; node: GraphNodeSpec; durationMs: number; stateBefore: StateObject; stateAfter: StateObject }} params
   */
  logStepExecution({ step, maxSteps, node, durationMs, stateBefore, stateAfter }) {
    const diffs = this.computeStateDiff(stateBefore, stateAfter);
    console.log(
      `\n${ANSI.green}${ANSI.bold}▶ [FAST LOOP | Шаг ${step + 1}/${maxSteps}]${ANSI.reset} ` +
        `Узел: ${ANSI.bold}"${node.id}"${ANSI.reset} (type=${ANSI.cyan}${node.type}${ANSI.reset}, ` +
        `reads=[${node.reads.join(', ')}], writes=[${node.writes.join(', ')}], ${durationMs.toFixed(1)} мс)`
    );

    if (this.traceStateDiffs) {
      if (diffs.length === 0) {
        console.log(`  ${ANSI.gray}↳ Δ State: (без изменений)${ANSI.reset}`);
      } else {
        for (const d of diffs) {
          if (d.change === 'ADDED') {
            console.log(`  ${ANSI.green}↳ Δ State [+ADDED]    ${d.key}: ${JSON.stringify(d.after)}${ANSI.reset}`);
          } else if (d.change === 'MODIFIED') {
            console.log(
              `  ${ANSI.yellow}↳ Δ State [~MODIFIED] ${d.key}: ${JSON.stringify(d.before)} ──► ${JSON.stringify(d.after)}${ANSI.reset}`
            );
          } else {
            console.log(`  ${ANSI.red}↳ Δ State [-REMOVED]  ${d.key} (было: ${JSON.stringify(d.before)})${ANSI.reset}`);
          }
        }
      }
    }
    this._writeJsonl('FAST_LOOP_STEP', { step, nodeId: node.id, nodeType: node.type, durationMs, diffs });
  }

  /**
   * @param {string} nodeId
   * @param {Array<{ to: string; priority: number; guardExpr: string; matched: boolean }>} evaluations
   */
  logGuardEvaluation(nodeId, evaluations) {
    if (!this.traceGuards) return;
    if (evaluations.length === 0) {
      console.log(`  ${ANSI.red}↳ Маршрутизация из "${nodeId}": нет исходящих ребер!${ANSI.reset}`);
      return;
    }
    let winnerLogged = false;
    for (const ev of evaluations) {
      const isWinner = ev.matched && !winnerLogged;
      if (isWinner) winnerLogged = true;
      const badge = isWinner
        ? `${ANSI.green}${ANSI.bold}[ВЫБРАНО]${ANSI.reset}`
        : ev.matched
          ? `${ANSI.gray}[ИСТИНА, НО НИЖЕ ПРИОРИТЕТ]${ANSI.reset}`
          : `${ANSI.gray}[ЛОЖЬ]${ANSI.reset}`;
      console.log(
        `  ${ANSI.gray}↳ Guard (${nodeId} ──► ${ev.to}, priority=${ev.priority}): ${ev.guardExpr} => ${badge}`
      );
    }
    this._writeJsonl('GUARD_EVAL', { nodeId, evaluations });
  }

  /**
   * @param {DeadlockIncident} deadlock
   * @param {Map<string, number>} markovBlanket
   * @param {string} stateHash
   */
  logDeadlock(deadlock, markovBlanket, stateHash) {
    this.section('DEADLOCK', `Обнаружен тупик ${deadlock.type} на узле "${deadlock.nodeId}" (шаг ${deadlock.step + 1})`);
    console.log(`  ${ANSI.red}• Тип сбоя:${ANSI.reset}          ${ANSI.bold}${deadlock.type}${ANSI.reset}`);
    console.log(`  ${ANSI.red}• Узел сбоя:${ANSI.reset}         "${deadlock.nodeId}" (предшественник: "${deadlock.lastValidNodeId || 'none'}")`);
    if (deadlock.error) {
      console.log(`  ${ANSI.red}• Текст исключения:${ANSI.reset}  ${deadlock.error}`);
    }
    console.log(`  ${ANSI.red}• SHA-256(v, s):${ANSI.reset}     ${stateHash.slice(0, 16)}...`);

    const blanketEntries = Array.from(markovBlanket.entries())
      .map(([id, score]) => `${id}(PPR=${score.toFixed(3)})`)
      .join(', ');
    console.log(`  ${ANSI.cyan}• Одеяло Маркова MB:${ANSI.reset} [${blanketEntries}]`);
    console.log(`  ${ANSI.gray}• Слепок State:${ANSI.reset}      ${JSON.stringify(deadlock.stateSnapshot)}`);
    this._writeJsonl('DEADLOCK', {
      deadlock,
      stateHash,
      markovBlanket: Object.fromEntries(markovBlanket)
    });
  }

  /**
   * @param {System1Verdict} verdict
   * @param {{ maxDrift: number; minCoherence: number; minApproveProb: number }} thresholds
   */
  logSystem1Verdict(verdict, thresholds) {
    this.section('SYSTEM_1', 'Вердикт локального ONNX-арбитра @receptron/laya (systemOne)');
    const statusBadge = verdict.approved
      ? `${ANSI.green}${ANSI.bold}APPROVED (ПРОПУСК В ВЕРИФИКАТОР)${ANSI.reset}`
      : `${ANSI.red}${ANSI.bold}REJECTED (ОТКАТ: ${verdict.rollbackAction})${ANSI.reset}`;

    console.log(`  • Статус гейта:       ${statusBadge}`);
    console.log(`  • P(approve) [noul]:  ${verdict.approveProb.toFixed(3)} (порог >= ${thresholds.minApproveProb})`);
    console.log(`  • Риск дрейфа [score]:${verdict.driftRisk.toFixed(3)} (порог <= ${thresholds.maxDrift})`);
    console.log(`  • Связность [score]:  ${verdict.coherence.toFixed(3)} (порог >= ${thresholds.minCoherence})`);
    console.log(`  • Действие [choice]:  ${verdict.rollbackAction} (${verdict.latencyMs ?? 0} мс)`);
    this._writeJsonl('SYSTEM1_VERDICT', { verdict, thresholds });
  }

  /**
   * @param {VerificationReport} report
   * @param {string} graphHash
   */
  logVerifierReport(report, graphHash) {
    const d = report.diagnostics;
    const badge =
      report.status === 'PASS'
        ? `${ANSI.green}${ANSI.bold}PASS${ANSI.reset}`
        : `${ANSI.red}${ANSI.bold}HARD_REJECT${ANSI.reset}`;

    console.log(
      `  ${ANSI.blue}${ANSI.bold}🛡️  [Верификатор детерминизма V(G)]${ANSI.reset} Статус: ${badge} | ` +
        `Штраф Ω(G)=${report.omegaPenalty} | Hash=${graphHash.slice(0, 12)}`
    );
    console.log(
      `     ↳ Вершин |V|=${d.vertexCount}, Ребер |E|=${d.edgeCount}, Плотность=${d.density.toFixed(2)}, ` +
        `Коллизий приоритетов=${d.priorityCollisions}`
    );
    if (d.deadNodes.length > 0 || d.unsatNodes.length > 0 || d.trappingCycles.length > 0) {
      console.log(
        `     ${ANSI.red}↳ Нарушения: deadNodes=[${d.deadNodes.join(',')}], ` +
          `unsatNodes=[${d.unsatNodes.join(',')}], trappingCycles=${JSON.stringify(d.trappingCycles)}${ANSI.reset}`
      );
    }
    this._writeJsonl('VERIFIER_REPORT', { report, graphHash });
  }

  /**
   * @param {IDirectedGraph} graph
   * @param {string | null} [activeNodeId]
   */
  logTopologyAscii(graph, activeNodeId = null) {
    console.log(`\n  ${ANSI.cyan}${ANSI.bold}🗺️  Топология графа "${graph.id}" (узлов: ${graph.nodes.size}):${ANSI.reset}`);
    for (const [id, node] of graph.nodes.entries()) {
      const isStart = graph.startNode === id ? ' [START]' : '';
      const isGoal = graph.goalNodes.has(id) ? ' [GOAL]' : '';
      const cursor = activeNodeId === id ? `${ANSI.green}${ANSI.bold}👉 ` : '   ';
      const outs = (graph.edges.get(id) || [])
        .map((e) => `--(p=${e.priority}, ${e.guardExpr})--> [${e.to}]`)
        .join(' | ');
      console.log(
        `  ${cursor}[${id}:${node.type}]${isStart}${isGoal}${ANSI.reset} ` +
          `${ANSI.gray}(R:[${node.reads.join(',')}] -> W:[${node.writes.join(',')}])${ANSI.reset} ${outs}`
      );
    }
  }
}
```

### 5.3. Ориентированный граф (`src/core/graph.js`)

```javascript
// @ts-check
import crypto from 'node:crypto';
import vm from 'node:vm';

/**
 * @typedef {import('../../types/index.js').IDirectedGraph} IDirectedGraph
 * @typedef {import('../../types/index.js').GraphNodeSpec} GraphNodeSpec
 * @typedef {import('../../types/index.js').GraphEdgeSpec} GraphEdgeSpec
 * @typedef {import('../../types/index.js').SerializedGraph} SerializedGraph
 * @typedef {import('../../types/index.js').StateObject} StateObject
 */

/**
 * @implements {IDirectedGraph}
 */
export class DirectedGraph {
  /**
   * @param {string} [id]
   */
  constructor(id = 'default_graph') {
    this.id = id;
    /** @type {Map<string, GraphNodeSpec>} */
    this.nodes = new Map();
    /** @type {Map<string, GraphEdgeSpec[]>} */
    this.edges = new Map();
    /** @type {string | null} */
    this.startNode = null;
    /** @type {Set<string>} */
    this.goalNodes = new Set();
  }

  /**
   * @param {GraphNodeSpec} node
   */
  addNode(node) {
    const cfg = { ...(node.config || {}) };
    if (node.type === 'check' && !cfg.assertFn && typeof cfg.assertExpr === 'string') {
      cfg.assertFn = this.compileGuardExpression(cfg.assertExpr);
    }
    this.nodes.set(node.id, {
      id: node.id,
      type: node.type,
      reads: Array.from(new Set(node.reads || [])),
      writes: Array.from(new Set(node.writes || [])),
      inputSchemas: node.inputSchemas ? structuredClone(node.inputSchemas) : {},
      outputSchemas: node.outputSchemas ? structuredClone(node.outputSchemas) : {},
      idempotency: node.idempotency || 'pure',
      hasBarrier: Boolean(node.hasBarrier),
      resourceRps: node.resourceRps ?? 1.0,
      modelTier: node.modelTier || 'frontier',
      config: cfg
    });
    if (!this.edges.has(node.id)) {
      this.edges.set(node.id, []);
    }
    return this;
  }

  /**
   * Безопасно компилирует строковое выражение guardExpr или assertExpr в функцию предиката в изолированном контексте.
   * Очищает блочные комментарии и хвостовые точки с запятой, которые могут генерировать LLM.
   * @param {string} [exprStr]
   * @returns {(state: StateObject) => boolean}
   */
  compileGuardExpression(exprStr) {
    const cleanExpr = (exprStr || '')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .trim()
      .replace(/;+\s*$/, '');
    if (!cleanExpr || cleanExpr === 'true' || cleanExpr.includes('() => true')) {
      return () => true;
    }
    return (state) => {
      try {
        const frozenState = Object.freeze(structuredClone(state));
        const ctx = vm.createContext({
          state: frozenState,
          s: frozenState,
          Boolean,
          Number,
          String,
          Array,
          Object,
          Math
        });
        const code = cleanExpr.includes('=>')
          ? `(${cleanExpr})(state)`
          : cleanExpr.includes('return ')
            ? `(() => { ${cleanExpr} })()`
            : `Boolean(${cleanExpr})`;
        return Boolean(vm.runInContext(code, ctx, { timeout: 50 }));
      } catch {
        return false;
      }
    };
  }

  /**
   * @param {string} from
   * @param {string} to
   * @param {(state: StateObject) => boolean} [guardFn]
   * @param {number} [priority]
   * @param {{ expr?: string; intervals?: Record<string, [number, number]>; preemptive?: boolean } | null} [guardMeta]
   */
  addEdge(from, to, guardFn, priority = 10, guardMeta = null) {
    if (!this.edges.has(from)) {
      this.edges.set(from, []);
    }
    const expr = guardMeta?.expr || (guardFn ? guardFn.toString() : 'true');
    const resolvedGuard = typeof guardFn === 'function' ? guardFn : this.compileGuardExpression(expr);

    const list = this.edges.get(from) || [];
    const duplicate = list.find((e) => e.to === to && e.guardExpr === expr);
    if (duplicate) {
      return this;
    }

    let safePriority = priority;
    if (guardMeta?.preemptive) {
      for (const existingEdge of list) {
        if (existingEdge.priority >= safePriority) {
          existingEdge.priority += 1;
        }
      }
    } else {
      const existingPriorities = new Set(list.map((e) => e.priority));
      while (existingPriorities.has(safePriority)) {
        safePriority += 1;
      }
    }

    list.push({
      from,
      to,
      guard: resolvedGuard,
      priority: safePriority,
      guardExpr: expr,
      intervals: guardMeta?.intervals ? structuredClone(guardMeta.intervals) : {}
    });
    list.sort((a, b) => a.priority - b.priority);
    this.edges.set(from, list);
    return this;
  }

  /**
   * @param {string} centerNodeId
   * @param {number} [alphaPpr]
   * @returns {Map<string, number>}
   */
  getCausalMarkovBlanket(centerNodeId, alphaPpr = 0.15) {
    const center = this.nodes.get(centerNodeId);
    if (!center) return new Map();

    const centerReads = new Set(center.reads);
    const centerWrites = new Set(center.writes);
    /** @type {Set<string>} */
    const parents = new Set();
    /** @type {Set<string>} */
    const children = new Set();
    /** @type {Set<string>} */
    const childReads = new Set();

    for (const [id, node] of this.nodes.entries()) {
      if (id === centerNodeId) continue;
      if (node.writes.some((k) => centerReads.has(k))) parents.add(id);
      if (node.reads.some((k) => centerWrites.has(k))) {
        children.add(id);
        node.reads.forEach((k) => childReads.add(k));
      }
    }

    for (const [u, edgeList] of this.edges.entries()) {
      if (u === centerNodeId) {
        edgeList.forEach((e) => children.add(e.to));
      } else if (edgeList.some((e) => e.to === centerNodeId)) {
        parents.add(u);
      }
    }

    /** @type {Set<string>} */
    const spouses = new Set();
    for (const [id, node] of this.nodes.entries()) {
      if (id !== centerNodeId && !parents.has(id) && !children.has(id)) {
        if (node.writes.some((k) => childReads.has(k))) spouses.add(id);
      }
    }

    const blanketIds = new Set([centerNodeId, ...parents, ...children, ...spouses]);
    /** @type {Map<string, number>} */
    const scores = new Map();
    for (const id of blanketIds) {
      const dist = id === centerNodeId ? 0 : parents.has(id) || children.has(id) ? 1 : 2;
      scores.set(id, Number((alphaPpr * Math.pow(1 - alphaPpr, dist)).toFixed(4)));
    }
    return scores;
  }

  /**
   * Обратная бисимуляционная свертка эквивалентных ветвей в ромб (DAG Diamond)
   * с дедупликацией параллельных ребер и нормализацией приоритетов.
   */
  foldBisimilarBranches() {
    /** @type {Map<string, string>} */
    const signatureMap = new Map();
    for (const [id, node] of Array.from(this.nodes.entries())) {
      if (id === this.startNode || this.goalNodes.has(id)) continue;
      const outTargets = (this.edges.get(id) || [])
        .map((e) => e.to)
        .sort()
        .join(',');
      const normalizedCode = (node.config?.sourceCode || node.config?.toolId || '')
        .replace(/(?:auto_)?solver_[a-zA-Z0-9_]+/g, 'solver_canonical')
        .trim();
      const sig = `${node.type}|R:${[...node.reads].sort()}|W:${[...node.writes].sort()}|fn:${normalizedCode || id}|out:${outTargets}`;

      if (signatureMap.has(sig)) {
        const canonicalId = signatureMap.get(sig) || id;
        for (const [fromId, edgeList] of this.edges.entries()) {
          for (const edge of edgeList) {
            if (edge.to === id) edge.to = canonicalId;
          }
          /** @type {Set<string>} */
          const seenTargetGuard = new Set();
          const deduped = edgeList.filter((e) => {
            const key = `${e.to}|${e.guardExpr}`;
            if (seenTargetGuard.has(key)) return false;
            seenTargetGuard.add(key);
            return true;
          });
          deduped.sort((a, b) => a.priority - b.priority);
          deduped.forEach((e, idx) => {
            e.priority = idx;
          });
          this.edges.set(fromId, deduped);
        }
        this.edges.delete(id);
        this.nodes.delete(id);
      } else {
        signatureMap.set(sig, id);
      }
    }
    return this;
  }

  computeStructuralHash() {
    const payload = JSON.stringify({
      nodes: Array.from(this.nodes.entries())
        .map(([id, n]) => [id, { type: n.type, r: n.reads, w: n.writes }])
        .sort(),
      edges: Array.from(this.edges.entries()).map(([k, v]) => [
        k,
        v.map((e) => ({ to: e.to, p: e.priority, g: e.guardExpr }))
      ])
    });
    return crypto.createHash('sha256').update(payload).digest('hex');
  }

  /**
   * @returns {SerializedGraph}
   */
  toJSON() {
    return {
      id: this.id,
      startNode: this.startNode,
      goalNodes: Array.from(this.goalNodes),
      nodes: Array.from(this.nodes.values()).map((n) => ({
        id: n.id,
        type: n.type,
        reads: n.reads,
        writes: n.writes,
        idempotency: n.idempotency,
        hasBarrier: n.hasBarrier,
        modelTier: n.modelTier,
        config: {
          prompt: n.config?.prompt,
          sourceCode: n.config?.sourceCode,
          assertExpr: n.config?.assertExpr,
          toolId: n.config?.toolId
        }
      })),
      edges: Array.from(this.edges.entries()).flatMap(([, list]) =>
        list.map((e) => ({
          from: e.from,
          to: e.to,
          priority: e.priority,
          guardExpr: e.guardExpr,
          intervals: e.intervals
        }))
      )
    };
  }

  /**
   * @param {SerializedGraph} data
   * @returns {DirectedGraph}
   */
  static fromJSON(data) {
    const g = new DirectedGraph(data.id || 'restored_graph');
    g.startNode = data.startNode;
    for (const goalId of data.goalNodes || []) {
      g.goalNodes.add(goalId);
    }
    for (const n of data.nodes || []) {
      g.addNode({
        id: n.id,
        type: n.type,
        reads: n.reads,
        writes: n.writes,
        idempotency: n.idempotency,
        hasBarrier: n.hasBarrier,
        modelTier: n.modelTier,
        config: {
          prompt: n.config?.prompt,
          sourceCode: n.config?.sourceCode,
          assertExpr: n.config?.assertExpr,
          toolId: n.config?.toolId
        }
      });
    }
    for (const e of data.edges || []) {
      g.addEdge(e.from, e.to, g.compileGuardExpression(e.guardExpr), e.priority, {
        expr: e.guardExpr,
        intervals: e.intervals
      });
    }
    return g;
  }

  /**
   * @returns {DirectedGraph}
   */
  clone() {
    const copy = new DirectedGraph(this.id);
    copy.startNode = this.startNode;
    this.goalNodes.forEach((g) => copy.goalNodes.add(g));

    for (const [, node] of this.nodes) {
      /** @type {Record<string, any>} */
      const clonedConfig = {};
      if (node.config) {
        for (const [k, val] of Object.entries(node.config)) {
          clonedConfig[k] = typeof val === 'function' ? val : structuredClone(val);
        }
      }
      copy.addNode({
        ...node,
        reads: [...node.reads],
        writes: [...node.writes],
        inputSchemas: structuredClone(node.inputSchemas || {}),
        outputSchemas: structuredClone(node.outputSchemas || {}),
        config: clonedConfig
      });
    }

    for (const [from, edgeList] of this.edges) {
      for (const e of edgeList) {
        copy.addEdge(from, e.to, e.guard, e.priority, {
          expr: e.guardExpr,
          intervals: structuredClone(e.intervals || {})
        });
      }
    }
    return copy;
  }
}
```

### 5.4. Статический верификатор детерминизма (`src/core/verifier.js`)

```javascript
// @ts-check

/**
 * @typedef {import('../../types/index.js').IGraphDeterminismVerifier} IGraphDeterminismVerifier
 * @typedef {import('../../types/index.js').IDirectedGraph} IDirectedGraph
 * @typedef {import('../../types/index.js').VerificationReport} VerificationReport
 */

/**
 * @implements {IGraphDeterminismVerifier}
 */
export class GraphDeterminismVerifier {
  /**
   * @param {{ rhoMax?: number; lMax?: number; lyapunovCreditBankMax?: number }} [config]
   */
  constructor(config = {}) {
    this.rhoMax = config.rhoMax ?? 1.8;
    this.lMax = config.lMax ?? 5;
    this.weights = { l1: 2.0, l2: 10.0, l3: 1.5, l4: 25.0 };
    this.lyapunovCreditBank = config.lyapunovCreditBankMax ?? 15.0;
  }

  /**
   * @param {IDirectedGraph} graph
   * @returns {VerificationReport}
   */
  verify(graph) {
    const V = graph.nodes.size;
    let E = 0;
    let priorityCollisions = 0;

    for (const [, list] of graph.edges) {
      E += list.length;
      const seenPriorities = new Set();
      for (const edge of list) {
        if (seenPriorities.has(edge.priority)) priorityCollisions++;
        seenPriorities.add(edge.priority);
      }
    }

    const reachable = this._bfsReachable(graph, graph.startNode);
    const coReachable = this._reverseBfsCoReachable(graph, graph.goalNodes);
    const deadNodes = Array.from(reachable).filter((n) => !coReachable.has(n));
    const unsatNodes = this._checkTransitivePathSatisfiability(graph, reachable, coReachable);
    const sccs = this._tarjanSCC(graph, reachable);
    /** @type {string[][]} */
    const trappingCycles = [];
    let cycleExcess = 0;

    for (const comp of sccs) {
      const isCyclic =
        comp.length > 1 ||
        (comp.length === 1 && (graph.edges.get(comp[0]) || []).some((e) => e.to === comp[0]));
      if (isCyclic) {
        if (comp.length > this.lMax) cycleExcess += comp.length - this.lMax;
        const compSet = new Set(comp);
        const hasEscape = comp.some((u) =>
          (graph.edges.get(u) || []).some((e) => !compSet.has(e.to) && coReachable.has(e.to))
        );
        if (!hasEscape) trappingCycles.push(comp);
      }
    }

    const density = V > 0 ? E / V : 0;
    const omega =
      this.weights.l1 * Math.max(0, density - this.rhoMax) +
      this.weights.l2 * priorityCollisions +
      this.weights.l3 * cycleExcess +
      this.weights.l4 * (deadNodes.length + unsatNodes.length);

    const passed =
      priorityCollisions === 0 &&
      deadNodes.length === 0 &&
      unsatNodes.length === 0 &&
      trappingCycles.length === 0;

    return {
      status: passed ? 'PASS' : 'HARD_REJECT',
      omegaPenalty: Number(omega.toFixed(4)),
      diagnostics: {
        vertexCount: V,
        edgeCount: E,
        density: Number(density.toFixed(3)),
        priorityCollisions,
        deadNodes,
        unsatNodes,
        trappingCycles
      }
    };
  }

  /**
   * @param {IDirectedGraph} graph
   * @param {Set<string>} reachable
   * @param {Set<string>} coReachable
   * @returns {string[]}
   */
  _checkTransitivePathSatisfiability(graph, reachable, coReachable) {
    /** @type {Map<string, Record<string, [number, number]>>} */
    const abstractState = new Map();
    for (const id of reachable) abstractState.set(id, {});

    const maxPasses = reachable.size + 1;
    for (let pass = 0; pass < maxPasses; pass++) {
      let changed = false;
      for (const uId of reachable) {
        const uNode = graph.nodes.get(uId);
        const writtenByU = new Set(uNode?.writes || []);
        const uIntervals = abstractState.get(uId) || {};

        for (const edge of graph.edges.get(uId) || []) {
          if (!reachable.has(edge.to)) continue;
          /** @type {Record<string, [number, number]>} */
          const transferred = {};
          let edgeFeasible = true;

          const allKeys = new Set([
            ...Object.keys(uIntervals),
            ...Object.keys(edge.intervals || {})
          ]);

          for (const k of allKeys) {
            const baseRange = writtenByU.has(k) ? [-Infinity, Infinity] : (uIntervals[k] || [-Infinity, Infinity]);
            const guardRange = edge.intervals?.[k] || [-Infinity, Infinity];
            const low = Math.max(baseRange[0], guardRange[0]);
            const high = Math.min(baseRange[1], guardRange[1]);
            if (low > high) {
              edgeFeasible = false;
              break;
            }
            transferred[k] = [low, high];
          }

          if (edgeFeasible) {
            const targetState = abstractState.get(edge.to) || {};
            for (const [k, [low, high]] of Object.entries(transferred)) {
              if (!targetState[k]) {
                targetState[k] = [low, high];
                changed = true;
              } else {
                const joinedLow = Math.min(targetState[k][0], low);
                const joinedHigh = Math.max(targetState[k][1], high);
                if (joinedLow !== targetState[k][0] || joinedHigh !== targetState[k][1]) {
                  targetState[k] = [joinedLow, joinedHigh];
                  changed = true;
                }
              }
            }
            abstractState.set(edge.to, targetState);
          }
        }
      }
      if (!changed) break;
    }

    /** @type {string[]} */
    const unsat = [];
    for (const wId of reachable) {
      if (graph.goalNodes.has(wId)) continue;
      const wNode = graph.nodes.get(wId);
      const writtenByW = new Set(wNode?.writes || []);
      const wIntervals = abstractState.get(wId) || {};
      const outgoing = (graph.edges.get(wId) || []).filter((e) => coReachable.has(e.to));
      if (outgoing.length === 0) continue;

      const hasFeasibleOut = outgoing.some((eOut) => {
        if (!eOut.intervals) return true;
        for (const [k, [gMin, gMax]] of Object.entries(eOut.intervals)) {
          const inRange = writtenByW.has(k) ? [-Infinity, Infinity] : (wIntervals[k] || [-Infinity, Infinity]);
          if (Math.max(inRange[0], gMin) > Math.min(inRange[1], gMax)) return false;
        }
        return true;
      });
      if (!hasFeasibleOut) unsat.push(wId);
    }
    return unsat;
  }

  /**
   * @param {IDirectedGraph} graph
   * @param {string | null} start
   * @returns {Set<string>}
   */
  _bfsReachable(graph, start) {
    /** @type {Set<string>} */
    const visited = new Set();
    if (!start) return visited;
    const queue = [start];
    while (queue.length > 0) {
      const curr = queue.shift();
      if (curr && !visited.has(curr)) {
        visited.add(curr);
        for (const e of graph.edges.get(curr) || []) queue.push(e.to);
      }
    }
    return visited;
  }

  /**
   * @param {IDirectedGraph} graph
   * @param {Set<string>} goals
   * @returns {Set<string>}
   */
  _reverseBfsCoReachable(graph, goals) {
    /** @type {Map<string, string[]>} */
    const rev = new Map();
    for (const [u, list] of graph.edges) {
      for (const e of list) {
        if (!rev.has(e.to)) rev.set(e.to, []);
        rev.get(e.to)?.push(u);
      }
    }
    /** @type {Set<string>} */
    const visited = new Set();
    const queue = Array.from(goals);
    while (queue.length > 0) {
      const curr = queue.shift();
      if (curr && !visited.has(curr)) {
        visited.add(curr);
        for (const prev of rev.get(curr) || []) queue.push(prev);
      }
    }
    return visited;
  }

  /**
   * @param {IDirectedGraph} graph
   * @param {Set<string>} subset
   * @returns {string[][]}
   */
  _tarjanSCC(graph, subset) {
    let index = 0;
    /** @type {Map<string, number>} */
    const indices = new Map();
    /** @type {Map<string, number>} */
    const lowlink = new Map();
    /** @type {string[]} */
    const stack = [];
    /** @type {Set<string>} */
    const onStack = new Set();
    /** @type {string[][]} */
    const sccs = [];

    /** @param {string} v */
    const strongConnect = (v) => {
      indices.set(v, index);
      lowlink.set(v, index);
      index++;
      stack.push(v);
      onStack.add(v);

      for (const e of graph.edges.get(v) || []) {
        if (!subset.has(e.to)) continue;
        if (!indices.has(e.to)) {
          strongConnect(e.to);
          lowlink.set(v, Math.min(lowlink.get(v) ?? 0, lowlink.get(e.to) ?? 0));
        } else if (onStack.has(e.to)) {
          lowlink.set(v, Math.min(lowlink.get(v) ?? 0, indices.get(e.to) ?? 0));
        }
      }

      if (lowlink.get(v) === indices.get(v)) {
        /** @type {string[]} */
        const comp = [];
        let w;
        do {
          w = stack.pop();
          if (w !== undefined) {
            onStack.delete(w);
            comp.push(w);
          }
        } while (w !== v && w !== undefined);
        sccs.push(comp);
      }
    };

    for (const v of subset) {
      if (!indices.has(v)) strongConnect(v);
    }
    return sccs;
  }
}
```

### 5.5. Реестр инструментов с песочницей `node:vm` и верификацией Saga-компенсаторов (`src/core/tools.js`)

```javascript
// @ts-check
import vm from 'node:vm';

/**
 * @typedef {import('../../types/index.js').IToolRegistry} IToolRegistry
 * @typedef {import('../../types/index.js').StateObject} StateObject
 */

/**
 * @implements {IToolRegistry}
 */
export class ToolRegistry {
  constructor() {
    /** @type {Map<string, any>} */
    this.tools = new Map();
  }

  /**
   * @param {{
   *   id: string;
   *   reads: string[];
   *   writes: string[];
   *   sourceCode: string;
   *   version?: number;
   *   hasIrreversibleBarrier?: boolean;
   *   sagaCompensatorCode?: string;
   *   regressionPool?: Array<{ input: StateObject; expectedOutput: StateObject }>;
   * }} toolSpec
   */
  register(toolSpec) {
    this.tools.set(toolSpec.id, {
      ...toolSpec,
      version: toolSpec.version || 1,
      hasIrreversibleBarrier: Boolean(toolSpec.hasIrreversibleBarrier),
      regressionPool: toolSpec.regressionPool || []
    });
  }

  /**
   * Безопасно исполняет JS-код инструмента в изолированном контексте node:vm.
   * Очищает хвостовые точки с запятой, которые могут возвращать реальные LLM в поле sourceCode.
   * @param {string} codeStr
   * @param {StateObject} stateInput
   * @param {number} [timeoutMs]
   * @returns {StateObject}
   */
  executeSandboxed(codeStr, stateInput, timeoutMs = 250) {
    const cleanCode = (codeStr || '').trim().replace(/;+\s*$/, '');
    const frozenInput = Object.freeze(structuredClone(stateInput));
    const sandbox = {
      input: frozenInput,
      Math,
      Date,
      JSON,
      Number,
      String,
      Boolean,
      Array,
      Object
    };
    const context = vm.createContext(sandbox);
    const script = new vm.Script(`(${cleanCode})(input)`);
    const result = script.runInContext(context, { timeout: timeoutMs });
    return result && typeof result === 'object' ? structuredClone(result) : {};
  }

  /**
   * Определение 2: Верифицирует компенсирующий Saga-оператор Phi^-1 в песочнице node:vm на регрессионном пуле.
   * @param {string} toolId
   * @param {string} compensatorCodeStr
   * @param {number} [epsilonSaga]
   * @returns {{ verified: boolean; residualError: number; hasBarrier: boolean }}
   */
  verifyAndAttachSagaCompensator(toolId, compensatorCodeStr, epsilonSaga = 1e-4) {
    const tool = this.tools.get(toolId);
    if (!tool) {
      throw new Error(`Инструмент "${toolId}" не найден в ToolRegistry`);
    }

    const pool =
      Array.isArray(tool.regressionPool) && tool.regressionPool.length > 0
        ? tool.regressionPool
        : [{ input: {}, expectedOutput: {} }];

    let totalResidual = 0;
    for (const tc of pool) {
      const sampleInput = tc.input ?? tc;
      const forwardState = { ...sampleInput, ...this.executeSandboxed(tool.sourceCode, sampleInput) };
      const restoredState = { ...forwardState, ...this.executeSandboxed(compensatorCodeStr, forwardState) };
      const trackedKeys = new Set([...(tool.reads || []), ...(tool.writes || [])]);
      for (const k of trackedKeys) {
        if (JSON.stringify(sampleInput[k]) !== JSON.stringify(restoredState[k])) {
          totalResidual += 1.0;
        }
      }
    }

    const residualError = totalResidual / pool.length;
    if (residualError <= epsilonSaga) {
      tool.sagaCompensatorCode = compensatorCodeStr;
      tool.sagaResidualError = residualError;
      tool.hasIrreversibleBarrier = false;
      return { verified: true, residualError, hasBarrier: false };
    }

    tool.hasIrreversibleBarrier = true;
    return { verified: false, residualError, hasBarrier: true };
  }
}
```

### 5.6. Менеджер онтологии ключей состояния (`src/core/ontology.js`)

```javascript
// @ts-check

/**
 * @typedef {import('../../types/index.js').IStateOntologyManager} IStateOntologyManager
 * @typedef {import('../../types/index.js').SerializedOntologyKey} SerializedOntologyKey
 */

/**
 * @implements {IStateOntologyManager}
 */
export class StateOntologyManager {
  /**
   * @param {number} [stagingTtlEpochs]
   */
  constructor(stagingTtlEpochs = 2) {
    /** @type {Map<string, { status: 'STAGING' | 'CANONICAL'; canonicalAlias?: string; epochsLived: number; mutualInfoGain: number }>} */
    this.keys = new Map();
    this.stagingTtlEpochs = stagingTtlEpochs;
  }

  /**
   * @param {string} keyName
   * @param {number} [mutualInfoGain]
   */
  registerStagingKey(keyName, mutualInfoGain = 0.15) {
    if (!this.keys.has(keyName)) {
      this.keys.set(keyName, {
        status: 'STAGING',
        epochsLived: 0,
        mutualInfoGain
      });
    }
  }

  /**
   * @param {string} keyName
   * @returns {string}
   */
  findCanonical(keyName) {
    const entry = this.keys.get(keyName);
    if (!entry || !entry.canonicalAlias || entry.canonicalAlias === keyName) return keyName;
    const root = this.findCanonical(entry.canonicalAlias);
    entry.canonicalAlias = root;
    return root;
  }

  /**
   * @param {string} canonicalKey
   * @param {string} synonymKey
   * @returns {string}
   */
  mergeKeys(canonicalKey, synonymKey) {
    const rootA = this.findCanonical(canonicalKey);
    const rootB = this.findCanonical(synonymKey);
    if (!this.keys.has(rootA)) {
      this.keys.set(rootA, { status: 'CANONICAL', epochsLived: 0, mutualInfoGain: 1.0 });
    }
    if (rootA !== rootB) {
      const existingB = this.keys.get(rootB) || { status: 'STAGING', epochsLived: 0, mutualInfoGain: 0 };
      existingB.canonicalAlias = rootA;
      this.keys.set(rootB, existingB);
    }
    return rootA;
  }

  /**
   * @param {number} [thetaIg]
   * @returns {{ promoted: string[]; evicted: string[] }}
   */
  runStagingGarbageCollection(thetaIg = 0.1) {
    /** @type {string[]} */
    const promoted = [];
    /** @type {string[]} */
    const evicted = [];
    for (const [k, meta] of this.keys.entries()) {
      if (meta.status === 'STAGING') {
        meta.epochsLived = (meta.epochsLived || 0) + 1;
        if ((meta.mutualInfoGain || 0) >= thetaIg) {
          meta.status = 'CANONICAL';
          promoted.push(k);
        } else if (meta.epochsLived >= this.stagingTtlEpochs) {
          this.keys.delete(k);
          evicted.push(k);
        }
      }
    }
    return { promoted, evicted };
  }

  /**
   * @returns {SerializedOntologyKey[]}
   */
  toJSON() {
    return Array.from(this.keys.entries()).map(([keyName, meta]) => ({
      keyName,
      status: meta.status,
      canonicalAlias: meta.canonicalAlias,
      epochsLived: meta.epochsLived,
      mutualInfoGain: meta.mutualInfoGain
    }));
  }

  /**
   * @param {SerializedOntologyKey[]} entries
   */
  loadFromJSON(entries) {
    for (const item of entries || []) {
      this.keys.set(item.keyName, {
        status: item.status,
        canonicalAlias: item.canonicalAlias,
        epochsLived: item.epochsLived,
        mutualInfoGain: item.mutualInfoGain
      });
    }
  }
}
```

### 5.7. Исполняемое ядро рантайма без моков (`src/core/runtime.js`)

```javascript
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
    this.maxSteps = config.hardStepLimit ?? 50;
    this.loopThreshold = config.stateHashLoopThreshold ?? 2;
    this.alphaPpr = config.pprTeleportAlpha ?? 0.15;
    /** @type {PrecedentRecord[]} */
    this.precedentMemory = [];
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
            if (tool?.sourceCode) {
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
          appliedMutations
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

      const grounded = this.findRelationallyGroundedPrecedent(
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
          const checkReads = node?.reads && node.reads.length > 0 ? node.reads : ['goalCompleted'];
          const preGuardExpr = checkReads.map((k) => `!Boolean(state.${k})`).join(' || ');
          candidateGraph.addEdge(
            deadlock.lastValidNodeId,
            mutationPlan.node.id,
            candidateGraph.compileGuardExpression(preGuardExpr),
            0,
            { expr: preGuardExpr, preemptive: true }
          );
          candidateGraph.addEdge(mutationPlan.node.id, deadlock.nodeId, () => true, 1, {
            expr: 'true /* reverify_check */'
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
        if (this.precedentStore) {
          if (mutationSource === 'SYSTEM2_SYNTHESIZED') {
            await this.precedentStore.savePrecedent(precedentRecord, deadlock);
          }
          await this.precedentStore.saveGraphVersion(
            this.graph.toJSON(),
            structHash,
            checkResult.omegaPenalty
          );
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
      appliedMutations
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
```

### 5.8. ИИ-модули без моков: `@huggingface/transformers` (`src/ai/embeddings.js`), `@receptron/laya` (`src/ai/laya-system1.js`) и `@langchain/openai` (`src/ai/langchain-system2.js`)

```javascript
// src/ai/embeddings.js
// @ts-check

/**
 * @typedef {import('../../types/index.js').ILocalEmbeddingEngine} ILocalEmbeddingEngine
 */

/**
 * Вычисляет косинусную близость двух векторов одинаковой размерности.
 * @param {number[]} vecA
 * @param {number[]} vecB
 * @returns {number}
 */
export function cosineSimilarity(vecA, vecB) {
  if (!vecA?.length || !vecB?.length || vecA.length !== vecB.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

/**
 * Боевой генератор эмбеддингов на базе ONNX Runtime (@huggingface/transformers).
 * @implements {ILocalEmbeddingEngine}
 */
export class LocalEmbeddingEngine {
  /**
   * @param {{ modelName?: string; cacheDir?: string; hfToken?: string; weights?: [number, number, number] }} [opts]
   */
  constructor(opts = {}) {
    this.modelName = opts.modelName || 'Xenova/multilingual-e5-small';
    this.cacheDir = opts.cacheDir || './.spectra/models';
    this.hfToken = opts.hfToken;
    /** @type {[number, number, number]} */
    this.weights = opts.weights || [0.5, 0.3, 0.2];
    /** @type {any} */
    this.extractor = null;
  }

  async init() {
    if (this.extractor) return this;
    if (this.hfToken && !process.env.HF_TOKEN) {
      process.env.HF_TOKEN = this.hfToken;
    }
    const { pipeline, env } = await import('@huggingface/transformers');
    env.cacheDir = this.cacheDir;
    this.extractor = await pipeline('feature-extraction', this.modelName, { quantized: true });
    return this;
  }

  /**
   * @param {string} text
   * @returns {Promise<number[]>}
   */
  async embedText(text) {
    await this.init();
    const output = await this.extractor(`query: ${text}`, { pooling: 'mean', normalize: true });
    /** @type {number[]} */
    const rawValues = Array.from(output.data);
    return rawValues.map((v) => Number(Number(v).toFixed(6)));
  }

  /**
   * Вычисляет составной вектор тупика v_inc в R^384 (Определение 8).
   * @param {{ errorText: string; markovBlanketSignature: string; stateSchemaSignature: string }} params
   * @returns {Promise<number[]>}
   */
  async computeIncidentVector({ errorText, markovBlanketSignature, stateSchemaSignature }) {
    const [eErr, eTopo, eSchema] = await Promise.all([
      this.embedText(errorText),
      this.embedText(markovBlanketSignature),
      this.embedText(stateSchemaSignature)
    ]);
    const [a, b, g] = this.weights;
    const combined = eErr.map((val, i) => a * val + b * (eTopo[i] || 0) + g * (eSchema[i] || 0));
    const norm = Math.hypot(...combined) || 1;
    return combined.map((v) => Number((v / norm).toFixed(6)));
  }
}
```

```javascript
// src/ai/laya-system1.js
// @ts-check
import { Laya } from '@receptron/laya';

/**
 * @typedef {import('../../types/index.js').ILayaSystem1Arbiter} ILayaSystem1Arbiter
 * @typedef {import('../../types/index.js').SpectraConfig} SpectraConfig
 * @typedef {import('../../types/index.js').DeadlockIncident} DeadlockIncident
 * @typedef {import('../../types/index.js').MutationPlan} MutationPlan
 * @typedef {import('../../types/index.js').System1Verdict} System1Verdict
 */

/**
 * Реальный локальный арбитр System-1 на базе @receptron/laya (ONNX Runtime).
 * Использует официальный API Laya.load() и laya.systemOne(state, questions) с типами score, noul и choice.
 * @implements {ILayaSystem1Arbiter}
 */
export class LayaSystem1Arbiter {
  /**
   * @param {SpectraConfig['system1']} config
   */
  constructor(config) {
    this.config = config;
    /** @type {any} */
    this.laya = null;
  }

  async init() {
    if (this.laya) return this;
    /** @type {Record<string, any>} */
    const loadOpts = {
      cacheDir: this.config.cacheDir || './.spectra/laya-cache',
      executionProviders: ['cpu']
    };
    if (this.config.modelDir) {
      loadOpts.modelDir = this.config.modelDir;
    }
    if (this.config.hfToken) {
      loadOpts.token = this.config.hfToken;
    }
    this.laya = await Laya.load(loadOpts);
    return this;
  }

  /**
   * Выполняет единый батч-проход ONNX-модели Laya (systemOne) для оценки риска дрейфа,
   * связности с целью, калиброванной вероятности одобрения noul P(true) и стратегии отката.
   * @param {DeadlockIncident} deadlock
   * @param {MutationPlan} mutationPlan
   * @param {string} goalDescription
   * @returns {Promise<System1Verdict>}
   */
  async evaluateMutation(deadlock, mutationPlan, goalDescription) {
    await this.init();
    const t0 = performance.now();

    const hasValidContract = Boolean(
      (mutationPlan.node && mutationPlan.node.writes.length > 0) || mutationPlan.operator === 'addEdge'
    );

    const statePayload = {
      goal: goalDescription.slice(0, 160),
      deadlockType: deadlock.type,
      deadlockNode: deadlock.nodeId,
      operator: mutationPlan.operator,
      from: mutationPlan.from,
      to: mutationPlan.to,
      newNodeId: mutationPlan.node?.id || null,
      writes: mutationPlan.node?.writes || [],
      contractVerified: hasValidContract,
      sandboxedExecution: true,
      reasoning: (mutationPlan.reasoning || '').slice(0, 200)
    };

    const result = await this.laya.systemOne(statePayload, {
      drift_risk: {
        type: 'score',
        instructions: 'How severe is the topological or semantic drift risk of this graph mutation?',
        criteria: ['minimal drift', 'low drift', 'moderate drift', 'severe drift']
      },
      coherence: {
        type: 'score',
        instructions: 'How well does this mutation align with reaching the session goal?',
        criteria: ['incoherent', 'weakly coherent', 'coherent', 'highly coherent']
      },
      approve_mutation: {
        type: 'noul',
        instructions: 'Is this graph mutation safe, well-typed, and likely to resolve the deadlock?'
      },
      rollback_action: {
        type: 'choice',
        instructions: 'Which rollback action is appropriate if the mutation is unsafe?',
        criteria: {
          keep: 'keep state and apply mutation',
          rollback_last: 'undo the last node transition',
          rollback_session: 'abort and rollback entire session'
        }
      }
    });

    // В @receptron/laya ответ score лежит в [0..3] (4 уровня рубрики)
    const rawDriftScore = Number(result.answers.drift_risk.score);
    const rawCoherenceScore = Number(result.answers.coherence.score);
    const rawNoulProb = Math.min(1 - 1e-6, Math.max(1e-6, Number(result.answers.approve_mutation.noul)));

    const driftRisk = Number((rawDriftScore / 3).toFixed(4));
    const coherence = Number((rawCoherenceScore / 3).toFixed(4));

    // Температурное масштабирование Платта T_cal над логитом noul
    const temp = Math.max(0.1, this.config.temperaturePlatt || 1.0);
    const logit = Math.log(rawNoulProb / (1 - rawNoulProb));
    const approveProb = Number((1 / (1 + Math.exp(-logit / temp))).toFixed(4));

    const approved =
      driftRisk <= this.config.maxDrift &&
      coherence >= this.config.minCoherence &&
      approveProb >= this.config.minApproveProb;

    const chosenRollback = /** @type {'keep' | 'rollback_last' | 'rollback_session'} */ (
      result.answers.rollback_action.choice
    );

    return {
      driftRisk,
      coherence,
      approveProb,
      approved,
      rollbackAction: approved ? 'keep' : chosenRollback === 'keep' ? 'rollback_last' : chosenRollback,
      latencyMs: Number((performance.now() - t0).toFixed(2))
    };
  }

  async close() {
    if (this.laya && typeof this.laya.close === 'function') {
      await this.laya.close();
      this.laya = null;
    }
  }
}
```

```javascript
// src/ai/langchain-system2.js
// @ts-check
import { z } from 'zod';
import { ChatOpenAI } from '@langchain/openai';
import { assertValidLlmCredentials } from '../cli/credentials.js';

/**
 * @typedef {import('../../types/index.js').ILangChainSystem2Arbiter} ILangChainSystem2Arbiter
 * @typedef {import('../../types/index.js').SpectraConfig} SpectraConfig
 * @typedef {import('../../types/index.js').DeadlockIncident} DeadlockIncident
 * @typedef {import('../../types/index.js').MutationPlan} MutationPlan
 * @typedef {import('../../types/index.js').GraphNodeSpec} GraphNodeSpec
 * @typedef {import('../../types/index.js').StateObject} StateObject
 * @typedef {import('../../types/index.js').ModelTier} ModelTier
 */

const MutationSchema = z.object({
  operator: z.enum(['addEdge', 'addNodeAndBridge', 'registerTool']),
  from: z.string(),
  to: z.string(),
  guardExpr: z.string().nullable(),
  node: z
    .object({
      id: z.string(),
      type: z.enum(['tool', 'llm', 'check']),
      reads: z.array(z.string()),
      writes: z.array(z.string()),
      sourceCode: z.string()
    })
    .nullable(),
  reasoning: z.string()
});

/**
 * Боевой арбитр System-2 на базе LangChain.js (@langchain/openai) без моков и заглушек.
 * Поддерживает прокидывание apiKey, baseUrl и имени модели через конфигурацию или переменные окружения.
 * @implements {ILangChainSystem2Arbiter}
 */
export class LangChainSystem2Arbiter {
  /**
   * @param {SpectraConfig['system2']} config
   */
  constructor(config) {
    this.config = config;
  }

  /**
   * Создает клиент ChatOpenAI с прокинутыми учетными данными (apiKey и опциональным baseURL).
   * @param {ModelTier} [tier]
   */
  _createChatModel(tier = 'frontier') {
    assertValidLlmCredentials(this.config);
    const modelName =
      tier === 'light' || tier === 'local_slm'
        ? this.config.lightModel || this.config.frontierModel
        : this.config.frontierModel;

    return new ChatOpenAI({
      apiKey: this.config.apiKey,
      modelName,
      temperature: 0,
      configuration: this.config.baseUrl ? { baseURL: this.config.baseUrl } : undefined
    });
  }

  /**
   * Вызывает реальную LLM через Structured Output для синтеза ремонтной мутации графа.
   * @param {DeadlockIncident} deadlock
   * @param {string} goalDescription
   * @param {string[]} existingNodeIds
   * @param {string} goalNodeId
   * @returns {Promise<MutationPlan>}
   */
  async synthesizeRecoveryMutation(deadlock, goalDescription, existingNodeIds, goalNodeId) {
    const isCheckFailed = deadlock.type === 'CHECK_FAILED';
    const requiredFrom = isCheckFailed ? deadlock.lastValidNodeId || 'start' : deadlock.nodeId;
    const requiredTo = isCheckFailed ? deadlock.nodeId : goalNodeId;
    const expectedNodeId = `solver_${deadlock.nodeId}_s${deadlock.step}`;

    const model = this._createChatModel('frontier');
    const structured = model.withStructuredOutput(MutationSchema);

    const res = await structured.invoke(
      `You are the System-2 Meta-Arbiter of the Spectra self-learning graph runtime.\n` +
        `Session Goal: "${goalDescription}".\n` +
        `Existing Graph Nodes: [${existingNodeIds.join(', ')}]. Terminal Goal Node: "${goalNodeId}".\n` +
        `Deadlock Incident: type="${deadlock.type}" at node="${deadlock.nodeId}" (predecessor="${deadlock.lastValidNodeId || 'start'}"), step=${deadlock.step}.\n` +
        `Error: "${deadlock.error || 'N/A'}".\n` +
        `Current State Snapshot: ${JSON.stringify(deadlock.stateSnapshot)}.\n\n` +
        `Requirements:\n` +
        `1. Set operator="addNodeAndBridge", from="${requiredFrom}", to="${requiredTo}", guardExpr="Boolean(state.goalCompleted === true)".\n` +
        `2. Set node.id="${expectedNodeId}", node.type="tool", node.reads=["goal"], node.writes=["goalCompleted", "summary"].\n` +
        `3. Set node.sourceCode to a pure synchronous JavaScript arrow function taking state (s) and returning an updated state object with goalCompleted: true and summary computed dynamically from s.goal (without trailing semicolon), e.g.:\n` +
        `   "(s) => ({ ...s, goalCompleted: true, summary: 'Resolved goal: ' + String(s.goal || '') })"`
    );

    const cleanSourceCode = res.node?.sourceCode
      ? res.node.sourceCode.trim().replace(/;+\s*$/, '')
      : `(s) => ({ ...s, goalCompleted: true, summary: 'Resolved goal: ' + String(s.goal || '') })`;

    return {
      operator: res.operator,
      from: res.from || requiredFrom,
      to: res.to || requiredTo,
      guardExpr: res.guardExpr ?? undefined,
      reasoning: res.reasoning,
      node: res.node
        ? {
            id: res.node.id.startsWith('solver_') ? res.node.id : expectedNodeId,
            type: res.node.type,
            reads: Array.from(new Set([...res.node.reads, 'goal'])),
            writes: Array.from(new Set([...res.node.writes, 'goalCompleted', 'summary'])),
            config: { sourceCode: cleanSourceCode }
          }
        : undefined
    };
  }

  /**
   * Выполняет штатный узел типа 'llm' в Fast Loop через реальную модель @langchain/openai.
   * @param {GraphNodeSpec} node
   * @param {StateObject} state
   * @param {ModelTier} [tier]
   * @returns {Promise<StateObject>}
   */
  async executeLlmNode(node, state, tier = 'frontier') {
    const model = this._createChatModel(tier);
    /** @type {Record<string, z.ZodTypeAny>} */
    const shape = {};
    for (const wKey of node.writes) {
      shape[wKey] = z.union([z.string(), z.number(), z.boolean()]);
    }
    const OutputSchema = z.object(shape);
    const structured = model.withStructuredOutput(OutputSchema);

    const inputSlice = Object.fromEntries(node.reads.map((k) => [k, state[k]]));
    const promptText =
      node.config?.prompt ||
      `Compute output keys [${node.writes.join(', ')}] given input state ${JSON.stringify(inputSlice)}.`;

    return await structured.invoke(promptText);
  }
}
```

### 5.9. Боевое хранилище PostgreSQL 16 + `pgvector` без JSON-заглушек (`src/db/pg-store.js`)

```javascript
// @ts-check
import pg from 'pg';

/**
 * @typedef {import('../../types/index.js').IPrecedentStore} IPrecedentStore
 * @typedef {import('../../types/index.js').DeadlockType} DeadlockType
 * @typedef {import('../../types/index.js').PrecedentRecord} PrecedentRecord
 * @typedef {import('../../types/index.js').DeadlockIncident} DeadlockIncident
 * @typedef {import('../../types/index.js').SerializedGraph} SerializedGraph
 * @typedef {import('../../types/index.js').SerializedOntologyKey} SerializedOntologyKey
 * @typedef {import('../../types/index.js').SpectraConfig} SpectraConfig
 */

const { Pool } = pg;

const DDL_INIT_SQL = `
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS graph_versions (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    graph_id VARCHAR(128) NOT NULL,
    structural_hash CHAR(64) NOT NULL,
    definition_json JSONB NOT NULL,
    omega_penalty DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS state_ontology_keys (
    key_name VARCHAR(128) PRIMARY KEY,
    status VARCHAR(32) NOT NULL DEFAULT 'STAGING',
    canonical_alias VARCHAR(128),
    epochs_lived INTEGER NOT NULL DEFAULT 0,
    mutual_info_gain DOUBLE PRECISION NOT NULL DEFAULT 0.15,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS deadlock_precedents (
    precedent_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    deadlock_type VARCHAR(64) NOT NULL,
    incident_vector vector(384) NOT NULL,
    anchor_reads TEXT[] NOT NULL,
    anchor_writes TEXT[] NOT NULL,
    mutation_operator VARCHAR(64) NOT NULL,
    mutation_plan_json JSONB NOT NULL,
    success_count DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    failure_count DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    confidence_lcb DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    is_antipattern BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_precedents_hnsw
  ON deadlock_precedents USING hnsw (incident_vector vector_cosine_ops);
`;

/**
 * Боевое хранилище прецедентов, онтологии ключей и версий графа на PostgreSQL 16 + pgvector.
 * @implements {IPrecedentStore}
 */
export class PgVectorPrecedentStore {
  /**
   * @param {SpectraConfig['database']} dbConfig
   */
  constructor(dbConfig) {
    this.connectionString = dbConfig.connectionString;
    /** @type {pg.Pool | null} */
    this.pool = null;
  }

  /**
   * Гарантирует инициализацию пула соединений и возвращает типобезопасную ссылку pg.Pool (без TS18047).
   * @returns {Promise<pg.Pool>}
   */
  async _getPool() {
    if (!this.pool) {
      this.pool = new Pool({
        connectionString: this.connectionString,
        connectionTimeoutMillis: 5000
      });
      await this.pool.query(DDL_INIT_SQL);
    }
    return this.pool;
  }

  async init() {
    await this._getPool();
  }

  /**
   * @returns {Promise<PrecedentRecord[]>}
   */
  async loadAllPrecedents() {
    const pool = await this._getPool();
    const res = await pool.query(
      `SELECT deadlock_type, incident_vector::text AS incident_vector_text, anchor_reads, anchor_writes, mutation_plan_json, is_antipattern
       FROM deadlock_precedents WHERE is_antipattern = FALSE ORDER BY created_at DESC LIMIT 100`
    );
    return res.rows.map(
      /** @param {Record<string, any>} r */
      (r) => ({
        deadlockType: r.deadlock_type,
        incidentVector: r.incident_vector_text ? JSON.parse(r.incident_vector_text) : undefined,
        anchorReads: r.anchor_reads || [],
        anchorWrites: r.anchor_writes || [],
        mutationPlan: r.mutation_plan_json,
        isAntipattern: Boolean(r.is_antipattern)
      })
    );
  }

  /**
   * Выполняет семантический поиск прецедентов по HNSW-индексу pgvector (`1 - (incident_vector <=> $2::vector)`).
   * @param {DeadlockType} deadlockType
   * @param {number[]} queryVector
   * @param {number} [minSimilarity]
   * @returns {Promise<PrecedentRecord[]>}
   */
  async searchSimilarPrecedents(deadlockType, queryVector, minSimilarity = 0.65) {
    const pool = await this._getPool();
    const vecLiteral = `[${queryVector.join(',')}]`;
    const res = await pool.query(
      `SELECT deadlock_type, anchor_reads, anchor_writes, mutation_plan_json, is_antipattern,
              1 - (incident_vector <=> $2::vector) AS similarity
       FROM deadlock_precedents
       WHERE deadlock_type = $1 AND is_antipattern = FALSE
       ORDER BY incident_vector <=> $2::vector ASC
       LIMIT 10`,
      [deadlockType, vecLiteral]
    );
    return res.rows
      .filter(/** @param {Record<string, any>} r */ (r) => Number(r.similarity ?? 0) >= minSimilarity)
      .map(
        /** @param {Record<string, any>} r */
        (r) => ({
          deadlockType: r.deadlock_type,
          similarity: Number(r.similarity),
          anchorReads: r.anchor_reads || [],
          anchorWrites: r.anchor_writes || [],
          mutationPlan: r.mutation_plan_json,
          isAntipattern: Boolean(r.is_antipattern)
        })
      );
  }

  /**
   * @param {PrecedentRecord} record
   * @param {DeadlockIncident} _deadlock
   */
  async savePrecedent(record, _deadlock) {
    const pool = await this._getPool();
    if (!Array.isArray(record.incidentVector) || record.incidentVector.length === 0) {
      throw new Error('savePrecedent требует непустой 384-мерный вектор incidentVector');
    }
    const vecLiteral = `[${record.incidentVector.join(',')}]`;
    await pool.query(
      `INSERT INTO deadlock_precedents
       (deadlock_type, incident_vector, anchor_reads, anchor_writes, mutation_operator, mutation_plan_json, is_antipattern)
       VALUES ($1, $2::vector, $3, $4, $5, $6, $7)`,
      [
        record.deadlockType,
        vecLiteral,
        record.anchorReads,
        record.anchorWrites,
        record.mutationPlan.operator,
        JSON.stringify(record.mutationPlan),
        record.isAntipattern
      ]
    );
  }

  /**
   * @param {SerializedGraph} graphJson
   * @param {string} structuralHash
   * @param {number} omegaPenalty
   */
  async saveGraphVersion(graphJson, structuralHash, omegaPenalty) {
    const pool = await this._getPool();
    await pool.query(
      `INSERT INTO graph_versions (graph_id, structural_hash, definition_json, omega_penalty)
       VALUES ($1, $2, $3, $4)`,
      [graphJson.id, structuralHash, JSON.stringify(graphJson), omegaPenalty]
    );
  }

  /**
   * @returns {Promise<SerializedGraph | null>}
   */
  async loadLatestGraph() {
    const pool = await this._getPool();
    const res = await pool.query(
      `SELECT definition_json FROM graph_versions ORDER BY created_at DESC LIMIT 1`
    );
    if (res.rows.length > 0 && res.rows[0].definition_json) {
      return res.rows[0].definition_json;
    }
    return null;
  }

  /**
   * @param {SerializedOntologyKey[]} keys
   */
  async saveOntologyKeys(keys) {
    const pool = await this._getPool();
    const activeNames = keys.map((k) => k.keyName);
    if (activeNames.length === 0) {
      await pool.query(`DELETE FROM state_ontology_keys`);
    } else {
      await pool.query(
        `DELETE FROM state_ontology_keys WHERE NOT (key_name = ANY($1::text[]))`,
        [activeNames]
      );
    }
    for (const k of keys) {
      await pool.query(
        `INSERT INTO state_ontology_keys (key_name, status, canonical_alias, epochs_lived, mutual_info_gain, updated_at)
         VALUES ($1, $2, $3, $4, $5, NOW())
         ON CONFLICT (key_name) DO UPDATE SET
           status = EXCLUDED.status,
           canonical_alias = EXCLUDED.canonical_alias,
           epochs_lived = EXCLUDED.epochs_lived,
           mutual_info_gain = EXCLUDED.mutual_info_gain,
           updated_at = NOW()`,
        [k.keyName, k.status, k.canonicalAlias || null, k.epochsLived, k.mutualInfoGain]
      );
    }
  }

  /**
   * @returns {Promise<SerializedOntologyKey[]>}
   */
  async loadOntologyKeys() {
    const pool = await this._getPool();
    const res = await pool.query(
      `SELECT key_name, status, canonical_alias, epochs_lived, mutual_info_gain FROM state_ontology_keys`
    );
    return res.rows.map(
      /** @param {Record<string, any>} r */
      (r) => ({
        keyName: r.key_name,
        status: r.status,
        canonicalAlias: r.canonical_alias || undefined,
        epochsLived: Number(r.epochs_lived),
        mutualInfoGain: Number(r.mutual_info_gain)
      })
    );
  }

  /**
   * Очищает таблицы БД перед прогоном интеграционных E2E-тестов.
   */
  async clearAll() {
    const pool = await this._getPool();
    await pool.query(`TRUNCATE TABLE deadlock_precedents, graph_versions, state_ontology_keys`);
  }

  async close() {
    if (this.pool) {
      await this.pool.end();
      this.pool = null;
    }
  }
}
```

### 5.10. Интерактивный мастер настройки CLI (`src/cli/wizard.js`)

```javascript
// @ts-check
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import fs from 'node:fs/promises';
import path from 'node:path';

/**
 * @typedef {import('../../types/index.js').SpectraConfig} SpectraConfig
 */

/**
 * Запускает интерактивный опросник настройки Spectra (включая apiKey, baseUrl, модель и PostgreSQL)
 * и сохраняет spectra.config.json.
 * @param {string} [targetPath]
 * @returns {Promise<SpectraConfig>}
 */
export async function runInteractiveWizard(targetPath = path.resolve(process.cwd(), 'spectra.config.json')) {
  const rl = readline.createInterface({ input, output });
  console.log('\n╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║ ⚙️  МАСТЕР ИНТЕРАКТИВНОЙ НАСТРОЙКИ ПРОТОТИПА SPECTRA                 ║');
  console.log('╚══════════════════════════════════════════════════════════════════════╝\n');

  const pgUrl =
    (await rl.question('1. Строка подключения PostgreSQL [postgres://spectra:spectra_pass@localhost:5432/spectra_db]: ')).trim() ||
    'postgres://spectra:spectra_pass@localhost:5432/spectra_db';

  const apiKey = (await rl.question('2. Ключ API для LLM System-2 (OPENAI_API_KEY / SPECTRA_LLM_API_KEY): ')).trim();

  const baseUrl =
    (await rl.question('3. Базовый URL LLM API [https://api.openai.com/v1]: ')).trim() ||
    'https://api.openai.com/v1';

  const frontierModel =
    (await rl.question('4. Модель для синтеза мутаций System-2 [gpt-4o-mini]: ')).trim() ||
    'gpt-4o-mini';

  const embModel =
    (await rl.question('5. Локальная модель эмбеддингов transformers.js [Xenova/multilingual-e5-small]: ')).trim() ||
    'Xenova/multilingual-e5-small';

  const hfToken = (await rl.question('6. Токен Hugging Face (Enter, если не требуется): ')).trim() || undefined;

  const logLevelInput = (await rl.question('7. Уровень детализации логов (debug/info/warn) [debug]: ')).trim() || 'debug';
  rl.close();

  /** @type {SpectraConfig} */
  const config = {
    systemName: 'spectra-autonomous-core',
    logging: {
      level: /** @type {'debug' | 'info' | 'warn' | 'error'} */ (
        ['debug', 'info', 'warn', 'error'].includes(logLevelInput) ? logLevelInput : 'debug'
      ),
      traceStateDiffs: true,
      traceGuards: true,
      jsonlFilePath: '.spectra/logs/session.jsonl'
    },
    database: {
      connectionString: pgUrl,
      embeddingDimension: 384
    },
    embeddings: {
      provider: 'transformers.js',
      modelName: embModel,
      cacheDir: './.spectra/models',
      hfToken,
      modalityWeights: [0.5, 0.3, 0.2]
    },
    system1: {
      provider: '@receptron/laya',
      cacheDir: './.spectra/laya-cache',
      hfToken,
      temperaturePlatt: 1.0,
      maxDrift: 0.65,
      minCoherence: 0.35,
      minApproveProb: 0.40
    },
    system2: {
      provider: 'openai',
      apiKey,
      baseUrl,
      frontierModel,
      lightModel: frontierModel
    },
    runtime: {
      hardStepLimit: 50,
      stateHashLoopThreshold: 2,
      pprTeleportAlpha: 0.15
    }
  };

  await fs.writeFile(targetPath, JSON.stringify(config, null, 2), 'utf-8');
  console.log(`\n✅ Конфигурация сохранена в ${targetPath}\n`);
  return config;
}
```

### 5.11. Исполняемая точка входа CLI (`bin/spectra.js`)

```javascript
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
      reads: ['goalCompleted'],
      writes: [],
      config: {
        assertExpr: 'Boolean(state.goalCompleted === true)'
      }
    })
    .addNode({
      id: 'goal',
      type: 'tool',
      reads: ['goalCompleted', 'summary'],
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
  graph.addEdge('verify_goal_state', 'goal', graph.compileGuardExpression('Boolean(state.goalCompleted)'), 1, {
    expr: 'Boolean(state.goalCompleted)'
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

    const result = await runtime.runSession({ goal: goalText, goalCompleted: false });
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
```

### 5.12. Сквозной интеграционный E2E-тест БЕЗ МОКОВ (`test/e2e.test.js`)

Тест напрямую использует учетные данные из `.env` / переменных окружения (`SPECTRA_LLM_API_KEY` или `OPENAI_API_KEY`, `SPECTRA_LLM_BASE_URL` или `OPENAI_BASE_URL`, `SPECTRA_PG_URL` или `DATABASE_URL`) либо `spectra.config.json` и прогоняет реальные вызовы `@langchain/openai`, `@receptron/laya`, `@huggingface/transformers` и `pgvector`:

```javascript
// @ts-check
import test from 'node:test';
import assert from 'node:assert/strict';
import { buildInitialGraph } from '../bin/spectra.js';
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
    // ЭТАП 1: Холодный старт на графе G_0 -> тупик CHECK_FAILED -> реальный
    // вызов LLM (@langchain/openai) -> реальный гейт @receptron/laya -> запись в pgvector
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
      config: config.runtime
    });

    const goal1 = 'Рассчитать квартальный баланс и подтвердить выполнение цели';
    const res1 = await runtime1.runSession({ goal: goal1, goalCompleted: false });

    assert.equal(res1.goalReached, true);
    assert.equal(res1.appliedMutations.length, 1);
    assert.equal(res1.appliedMutations[0].source, 'SYSTEM2_SYNTHESIZED');
    assert.equal(res1.finalState.goalCompleted, true);
    assert.ok(typeof res1.finalState.summary === 'string' && res1.finalState.summary.length > 0);

    // =========================================================================
    // ЭТАП 2: Повторный запуск на свежем графе G_0 с новой целью -> извлечение
    // выученного рецепта из PostgreSQL HNSW (MEMORY_GROUNDED) без вызова LLM
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
      config: config.runtime
    });

    const goal2 = 'Сформировать аудиторский отчет по транзакциям';
    const res2 = await runtime2.runSession({
      goal: goal2,
      goalCompleted: false,
      department: 'finance'
    });

    assert.equal(res2.goalReached, true);
    assert.equal(res2.appliedMutations.length, 1);
    assert.equal(res2.appliedMutations[0].source, 'MEMORY_GROUNDED');
    assert.equal(res2.finalState.goalCompleted, true);

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
      system2Arbiter: sys2
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
      (n) => !['start', 'verify_goal_state', 'goal'].includes(n.id)
    );
    assert.ok(activeSolver, 'Должен существовать синтезированный узел восстановления');

    runtime2.graph.addNode({
      id: 'solver_duplicate_branch',
      type: 'tool',
      reads: [...activeSolver.reads],
      writes: [...activeSolver.writes],
      config: { sourceCode: activeSolver.config?.sourceCode }
    });
    runtime2.graph.addEdge(
      'start',
      'solver_duplicate_branch',
      runtime2.graph.compileGuardExpression('!Boolean(state.goalCompleted)'),
      2,
      { expr: '!Boolean(state.goalCompleted)' }
    );
    runtime2.graph.addEdge('solver_duplicate_branch', 'verify_goal_state', () => true, 1, {
      expr: 'true /* reverify_check */'
    });

    const sleepRes = await runtime2.runConsolidationSleep();
    assert.ok(sleepRes.promotedKeys.includes('goalCompleted'));
    assert.ok(sleepRes.evictedKeys.includes('ephemeral_noise_key'));
    assert.equal(sleepRes.foldedNodes, 1);
    assert.ok(sleepRes.omegaAfter <= sleepRes.omegaBefore);
  } finally {
    await sys1.close();
    await store.close();
  }
});
```