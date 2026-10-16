# Инженерный план реализации прототипа SPECTRA

Настоящий документ описывает пошаговую архитектуру и реализацию рабочего прототипа самообучающейся графовой ИИ-системы **Spectra** на чистом JavaScript (Pure ESM) с использованием **100% рукописных (handwritten) `.d.ts` файлов деклараций** (без автогенерации через `tsc`), интерактивного CLI (`spectra init` и `spectra run "<goal>"`), локальных эмбеддингов на `@huggingface/transformers`, арбитра System-1 на `@receptron/laya`, арбитра System-2 на LangChain.js и персистентного слоя PostgreSQL 16 + `pgvector`.

---

## 1. Технологический стек и архитектура репозитория

### 1.1. Роли компонентов стека

| Компонент | Пакет / Технология | Роль в трехконтурной архитектуре Spectra |
| :--- | :--- | :--- |
| **Исполнительная среда** | **Node.js 20+** (`"type": "module"`) | Прямой запуск ESM `.js`-модулей без сборки и транспиляции, изолированная песочница `node:vm` для динамических тулов, нативный `node:readline/promises` для интерактивного CLI. |
| **Контракты и типизация** | **Самописные `.d.ts` (Handwritten Declarations) + JSDoc** | **Contract-First подход:** файлы `.d.ts` **никогда не генерируются автоматически** (`declaration: false`, `noEmit: true`). Они пишутся разработчиком вручную в директории `types/`, хранятся в Git как эталонные контракты архитектуры, а `tsc --noEmit` проверяет, что `.js`-реализация строго соответствует этим рукописным `.d.ts` интерфейсам. |
| **Локальные эмбеддинги** | **`@huggingface/transformers`** (`transformers.js`) | Локальная CPU-генерация эмбеддингов ($\mathbb{R}^{384}$, модель `Xenova/multilingual-e5-small`) для мультимодального вектора тупика $\mathbf{v}_{\text{inc}}$ и вектора цели $\mathbf{v}_{\text{intent}}$. |
| **Арбитр System-1** | **`@receptron/laya`** (ONNX Runtime) | Быстрый локальный гейткипер ($\sim 120\text{ мс}$ на CPU): оценивает риск дрейфа $r_{\text{drift}}$, связность с целью $s_{\text{coh}}$, калиброванную вероятность одобрения $P_{\text{approve}}$ и выбирает рецепт из памяти прецедентов без вызова LLM. |
| **Арбитр System-2 и `llm`-узлы** | **LangChain.js** (`@langchain/openai`, `zod`) | Синтез планов мутаций графа при новых тупиках через `.withStructuredOutput(ZodSchema)`, первичная сборка графа под цель `<goal>` и выполнение генеративных `llm`-узлов с каскадом моделей. |
| **Память и журнал версий** | **PostgreSQL 16 + `pgvector`** (`pg`) | Хранение иммутабельных версий графа, HNSW-индекс прецедентов тупиков `vector(384)`, библиотека `MacroNode`, реестр инструментов и статистика бандита Томпсона. |
| **CLI-интерфейс** | **`bin/spectra.js`** | Интерактивный визард настройки `spectra init`, запуск цикла решения задачи `spectra run "<goal>"`, верификация графа `spectra verify` и запуск фазы консолидации `spectra sleep`. |

### 1.2. Файловая структура проекта

Все файлы `.d.ts` в папке `types/` являются **рукописными исходниками** (не артефактами сборки) и коммитятся в репозиторий наравне с `.js`-кодом:

```text
spectra-prototype/
├── bin/
│   └── spectra.js                  # Исполняемая точка входа CLI (#!/usr/bin/env node)
├── types/                          # 100% САМОПИСНЫЕ (HANDWRITTEN) КОНТРАКТЫ ТИПОВ
│   ├── index.d.ts                  # Корневой экспорт всех рукописных типов пакета
│   ├── domain.d.ts                 # Доменные структуры: State, Graph, Deadlock, Mutation, Config
│   ├── core.d.ts                   # Рукописные контракты классов ядра (DirectedGraph, Verifier, Runtime)
│   └── ai.d.ts                     # Рукописные контракты ИИ-модулей (Transformers, Laya, LangChain)
├── sql/
│   └── 001_init_pgvector.sql       # DDL-миграция PostgreSQL + pgvector (vector(384))
├── src/
│   ├── cli/
│   │   ├── wizard.js               # Интерактивный визард настройки `spectra init`
│   │   └── runner.js               # Обработчик команды `spectra run "<goal>"` и телеметрия
│   ├── core/
│   │   ├── graph.js                # DirectedGraph (Markov Blanket, MetaInit, Pushout, Mitosis)
│   │   ├── verifier.js             # GraphDeterminismVerifier (Tarjan SCC, Interval Dataflow, Lyapunov)
│   │   ├── tools.js                # ToolRegistry (песочница node:vm, CEGIS-патч, Saga-откат)
│   │   ├── ontology.js             # StateOntologyManager (Union-Find сжатие алиасов и GC)
│   │   └── runtime.js              # SpectraRuntime (Fast Loop <-> Slow Loop <-> Rollback)
│   ├── ai/
│   │   ├── embeddings.js           # Обертка над @huggingface/transformers + адаптер SO(d)
│   │   ├── laya-system1.js         # Обертка над @receptron/laya (System-1 Gate & Choice)
│   │   └── langchain-system2.js    # Обертка над LangChain.js (System-2 Synthesizer & Goal Bootstrap)
│   └── db/
│       └── pg-store.js             # Клиент PostgreSQL + pgvector (Experience Replay & Graph Journal)
├── docker-compose.yml              # PostgreSQL 16 с расширением pgvector
├── package.json                    # ESM-манифест с полем "types": "./types/index.d.ts"
├── tsconfig.json                   # Строгий режим проверки JS по самописным .d.ts (declaration: false)
└── spectra.config.json             # Генерируется интерактивно через `spectra init`
```

---

## 2. Манифест проекта и инфраструктура (`package.json`, `tsconfig.json`, `docker-compose.yml`)

### 2.1. `package.json` (Pure ESM + привязка к самописным `types/index.d.ts`)

```json
{
  "name": "spectra-autonomous-cli",
  "version": "0.1.0",
  "description": "Continual Self-Learning Graph AI Runtime with System-1/System-2 Arbitrage",
  "type": "module",
  "main": "./src/core/runtime.js",
  "types": "./types/index.d.ts",
  "bin": {
    "spectra": "./bin/spectra.js"
  },
  "scripts": {
    "typecheck": "tsc --noEmit",
    "db:up": "docker compose up -d",
    "init": "node ./bin/spectra.js init",
    "start": "node ./bin/spectra.js"
  },
  "dependencies": {
    "@huggingface/transformers": "^3.1.0",
    "@langchain/core": "^0.3.15",
    "@langchain/openai": "^0.3.12",
    "@receptron/laya": "^0.2.0",
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

### 2.2. `tsconfig.json` (Запрет генерации `.d.ts` и валидация по рукописным контрактам)

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
    "baseUrl": ".",
    "paths": {
      "#types": ["./types/index.d.ts"],
      "#types/*": ["./types/*"]
    },
    "types": ["node"]
  },
  "include": ["bin/**/*.js", "src/**/*.js", "types/**/*.d.ts"]
}
```

### 2.3. `docker-compose.yml` (PostgreSQL 16 + `pgvector`)

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    container_name: spectra_pgvector
    restart: unless-stopped
    environment:
      POSTGRES_USER: spectra
      POSTGRES_PASSWORD: spectra_password
      POSTGRES_DB: spectra_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
      - ./sql/001_init_pgvector.sql:/docker-entrypoint-initdb.d/001_init_pgvector.sql:ro

volumes:
  pgdata:
```

---

## 3. Самописные определения типов TypeScript (`types/*.d.ts`)

Все файлы в папке `types/` создаются **вручную** (Handwritten First).

### 3.1. Доменные контракты (`types/domain.d.ts`)

```typescript
export type NodeType = 'llm' | 'tool' | 'branch' | 'join' | 'check' | 'macro' | 'fused';
export type IdempotencyClass = 'pure' | 'idempotent' | 'non_idempotent';
export type ModelTier = 'local_slm' | 'light' | 'frontier';
export type DeadlockType =
  | 'NO_VALID_TRANSITION'
  | 'STATE_HASH_LOOP'
  | 'CHECK_FAILED'
  | 'TOOL_EXECUTION_ERROR'
  | 'HARD_LIMIT_EXCEEDED';

export type StateObject = Record<string, any>;

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

export interface DeadlockIncident {
  type: DeadlockType;
  nodeId: string;
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

export interface System1Verdict {
  driftRisk: number;
  coherence: number;
  approveProb: number;
  approved: boolean;
  rollbackAction: 'keep' | 'rollback_last' | 'rollback_session';
  selectedPrecedentId?: string | null;
}

export interface VerificationReport {
  status: 'PASS' | 'HARD_REJECT';
  omegaPenalty: number;
  diagnostics: {
    priorityCollisions: number;
    deadNodes: string[];
    unsatNodes: string[];
    trappingCycles: string[][];
  };
}

export interface SessionRunResult {
  goalReached: boolean;
  reason?: string;
  steps: number;
  rollbacksCount: number;
  finalState: StateObject;
  omegaPenalty?: number;
  appliedMutations: Array<{
    deadlock: DeadlockIncident;
    mutationPlan: MutationPlan;
    omegaPenalty: number;
  }>;
}

export interface SpectraConfig {
  systemName: string;
  database: {
    connectionString: string;
    embeddingDimension: number;
  };
  embeddings: {
    provider: 'transformers.js';
    modelName: string;
    cacheDir: string;
    modalityWeights: [number, number, number];
  };
  system1: {
    provider: '@receptron/laya';
    temperaturePlatt: number;
    maxDrift: number;
    minCoherence: number;
    minApproveProb: number;
  };
  system2: {
    provider: 'openai';
    apiKey: string;
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

### 3.2. Рукописные контракты классов ядра (`types/core.d.ts`)

```typescript
import type {
  GraphNodeSpec,
  GraphEdgeSpec,
  StateObject,
  VerificationReport,
  DeadlockType,
  MutationPlan,
  DeadlockIncident,
  System1Verdict,
  SessionRunResult
} from './domain.js';

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
    guardMeta?: { expr?: string; intervals?: Record<string, [number, number]> } | null
  ): this;
  getCausalMarkovBlanket(centerNodeId: string, alphaPpr?: number): Map<string, number>;
  computeStructuralHash(): string;
  clone(): IDirectedGraph;
}

export interface IGraphDeterminismVerifier {
  rhoMax: number;
  lMax: number;
  lyapunovCreditBank: number;

  verify(graph: IDirectedGraph): VerificationReport;
  verifyAmortizedLyapunovStability(
    prevEnergy: number,
    nextEnergy: number,
    deltaMin?: number,
    maxBank?: number
  ): {
    accepted: boolean;
    mode: 'STRICT_DISSIPATION' | 'PROVISIONAL_INVESTMENT' | 'REJECTED_BY_LYAPUNOV';
    deltaV: number;
    bank: number;
  };
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
    regressionPool?: Array<{ input: StateObject; expectedOutput: StateObject }>;
  }): void;
  executeSandboxed(codeStr: string, stateInput: StateObject, timeoutMs?: number): StateObject;
}

export interface ISpectraRuntime {
  graph: IDirectedGraph;
  toolRegistry: IToolRegistry;
  verifier: IGraphDeterminismVerifier;
  ontology: any;
  runSession(initialState?: StateObject): Promise<SessionRunResult>;
}
```

### 3.3. Рукописные контракты ИИ-модулей (`types/ai.d.ts`)

```typescript
import type {
  DeadlockIncident,
  MutationPlan,
  System1Verdict,
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
  init(): Promise<void>;
  evaluateMutation(
    deadlock: DeadlockIncident,
    mutationPlan: MutationPlan,
    goalDescription: string
  ): Promise<System1Verdict>;
}

export interface ILangChainSystem2Arbiter {
  config: SpectraConfig['system2'];
  executeLlmNode(
    promptTemplate: string,
    state: StateObject,
    writeKeys: string[],
    tier?: ModelTier
  ): Promise<StateObject>;
  synthesizeRecoveryMutation(
    deadlock: DeadlockIncident,
    goalDescription: string,
    existingNodeIds: string[],
    goalNodeId: string
  ): Promise<MutationPlan>;
}
```

### 3.4. Корневой файл типов (`types/index.d.ts`)

```typescript
export * from './domain.js';
export * from './core.js';
export * from './ai.js';
```

---

## 4. Интерактивный CLI и запуск цикла `spectra run "<goal>"`

### 4.1. Интерактивный визард настройки (`src/cli/wizard.js`)

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
 * Запускает интерактивный опросник настройки Spectra и сохраняет spectra.config.json.
 * @param {string} [configPath]
 * @returns {Promise<SpectraConfig>}
 */
export async function runInteractiveWizard(configPath = 'spectra.config.json') {
  const rl = readline.createInterface({ input, output });
  console.log('\n============================================================');
  console.log('⚡ SPECTRA: Интерактивный мастер настройки самообучающейся системы');
  console.log('============================================================\n');

  try {
    const pgUri =
      (
        await rl.question(
          '1. PostgreSQL + pgvector URI [postgresql://spectra:spectra_password@localhost:5432/spectra_db]: '
        )
      ).trim() || 'postgresql://spectra:spectra_password@localhost:5432/spectra_db';

    const embModel =
      (
        await rl.question(
          '2. Локальная модель эмбеддингов transformers.js [Xenova/multilingual-e5-small]: '
        )
      ).trim() || 'Xenova/multilingual-e5-small';

    const maxDriftStr = (
      await rl.question('3. Порог риска дрейфа System-1 @receptron/laya (0.05..0.50) [0.35]: ')
    ).trim();
    const maxDrift = Number.isFinite(Number(maxDriftStr)) && maxDriftStr !== '' ? Number(maxDriftStr) : 0.35;

    const openaiKey = (
      await rl.question('4. OpenAI API Key для System-2 LangChain (Enter для локального/offline режима): ')
    ).trim();

    const frontierModel =
      (await rl.question('5. Старшая модель System-2 (frontier) [gpt-4o]: ')).trim() || 'gpt-4o';

    const stepLimitStr = (
      await rl.question('6. Жесткий бюджет шагов сессии Fast Loop T_max [50]: ')
    ).trim();
    const hardStepLimit = Number.parseInt(stepLimitStr, 10) || 50;

    /** @type {SpectraConfig} */
    const config = {
      systemName: 'spectra-autonomous-core',
      database: {
        connectionString: pgUri,
        embeddingDimension: 384
      },
      embeddings: {
        provider: 'transformers.js',
        modelName: embModel,
        cacheDir: './.cache/transformers',
        modalityWeights: [0.5, 0.3, 0.2]
      },
      system1: {
        provider: '@receptron/laya',
        temperaturePlatt: 1.0,
        maxDrift,
        minCoherence: 0.65,
        minApproveProb: 0.70
      },
      system2: {
        provider: 'openai',
        apiKey: openaiKey,
        frontierModel,
        lightModel: 'gpt-4o-mini'
      },
      runtime: {
        hardStepLimit,
        stateHashLoopThreshold: 2,
        pprTeleportAlpha: 0.15
      }
    };

    const fullPath = path.resolve(process.cwd(), configPath);
    await fs.writeFile(fullPath, JSON.stringify(config, null, 2), 'utf-8');
    console.log(`\n✅ Конфигурация сохранена в ${fullPath}`);
    console.log('Теперь запустите цикл выполнения командой:');
    console.log('  spectra run "Собрать аналитический отчет и проверить аномалии"\n');
    return config;
  } finally {
    rl.close();
  }
}
```

### 4.2. Исполняемый бинарник `bin/spectra.js` и запуск `spectra run "<goal>"`

```javascript
#!/usr/bin/env node
// @ts-check
import fs from 'node:fs/promises';
import path from 'node:path';
import { runInteractiveWizard } from '../src/cli/wizard.js';
import { LocalEmbeddingEngine } from '../src/ai/embeddings.js';
import { LayaSystem1Arbiter } from '../src/ai/laya-system1.js';
import { LangChainSystem2Arbiter } from '../src/ai/langchain-system2.js';
import {
  DirectedGraph,
  GraphDeterminismVerifier,
  ToolRegistry,
  StateOntologyManager,
  SpectraRuntime
} from '../src/core/runtime.js';

/**
 * @typedef {import('../types/index.js').SpectraConfig} SpectraConfig
 */

/**
 * @returns {Promise<SpectraConfig>}
 */
async function loadOrInitConfig() {
  const cfgPath = path.resolve(process.cwd(), 'spectra.config.json');
  try {
    const raw = await fs.readFile(cfgPath, 'utf-8');
    return JSON.parse(raw);
  } catch {
    console.log('⚠️ Файл spectra.config.json не найден. Запускаю интерактивную настройку...');
    return await runInteractiveWizard(cfgPath);
  }
}

/**
 * @param {string} goalText
 */
async function handleRunCommand(goalText) {
  const config = await loadOrInitConfig();
  console.log(`\n🚀 [SPECTRA RUN] Цель: "${goalText}"`);

  // 1. Инициализация локальных эмбеддингов (transformers.js) и арбитров System-1 / System-2
  const embedder = new LocalEmbeddingEngine({
    modelName: config.embeddings.modelName,
    cacheDir: config.embeddings.cacheDir,
    weights: config.embeddings.modalityWeights
  });
  console.log(`🧠 [Embeddings] Векторизация цели через ${config.embeddings.modelName}...`);
  const intentVec = await embedder.embedText(goalText);
  console.log(`   ↳ Вектор намерения v_intent построен (dim = ${intentVec.length})`);

  const sys1 = new LayaSystem1Arbiter(config.system1);
  const sys2 = new LangChainSystem2Arbiter(config.system2);
  const tools = new ToolRegistry();
  const verifier = new GraphDeterminismVerifier();
  const ontology = new StateOntologyManager();

  // 2. Построение стартового графа G_0
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
        assertFn: (s) => Boolean(s.goalCompleted === true)
      }
    })
    .addNode({
      id: 'goal',
      type: 'tool',
      reads: ['goalCompleted', 'summary'],
      writes: [],
      config: {
        handler: async (s) => s
      }
    });

  graph.startNode = 'start';
  graph.goalNodes.add('goal');
  graph.addEdge('start', 'verify_goal_state', () => true, 1);
  graph.addEdge('verify_goal_state', 'goal', (s) => Boolean(s.goalCompleted), 1);

  // 3. Создание рантайма с подключением System-1 (@receptron/laya) и System-2 (LangChain)
  const runtime = new SpectraRuntime({
    graph,
    toolRegistry: tools,
    verifier,
    ontology,
    config: config.runtime,
    system1Evaluator: async (deadlock, mutationPlan) => {
      const verdict = await sys1.evaluateMutation(deadlock, mutationPlan, goalText);
      console.log(
        `🛡️  [System-1 Laya Gate] P(approve)=${verdict.approveProb}, drift=${verdict.driftRisk}, coherence=${verdict.coherence}`
      );
      return verdict;
    },
    system2Synthesizer: async (deadlock, state, currentGraph) => {
      console.log(`⚡ [Deadlock: ${deadlock.type}] на узле "${deadlock.nodeId}". Вызов System-2...`);
      const blanket = currentGraph.getCausalMarkovBlanket(deadlock.nodeId);
      const vInc = await embedder.computeIncidentVector({
        errorText: `${deadlock.type}: ${deadlock.error || ''}`,
        markovBlanketSignature: Array.from(blanket.keys()).join('->'),
        stateSchemaSignature: Object.keys(state).sort().join(',')
      });
      console.log(`   ↳ Сформирован мультимодальный вектор тупика v_inc (dim=${vInc.length})`);

      const plan = await sys2.synthesizeRecoveryMutation(
        { ...deadlock, step: 1, stateSnapshot: state },
        goalText,
        Array.from(currentGraph.nodes.keys()),
        'goal'
      );
      if (plan.node?.config?.sourceCode) {
        tools.register({
          id: plan.node.id,
          reads: plan.node.reads,
          writes: plan.node.writes,
          sourceCode: plan.node.config.sourceCode
        });
      }
      console.log(`🔧 [System-2 Plan] Оператор: ${plan.operator} (${plan.from} -> ${plan.to})`);
      return plan;
    }
  });

  // 4. Исполнение цикла (Fast Loop <-> Slow Loop)
  const result = await runtime.runSession({ goal: goalText, goalCompleted: false });
  console.log('\n============================================================');
  console.log(`🏁 Результат выполнения: ${result.goalReached ? 'ЦЕЛЬ ДОСТИГНУТА' : 'ОСТАНОВКА'}`);
  console.log(`   • Шагов выполнено: ${result.steps}`);
  console.log(`   • Мутаций применено в Slow Loop: ${result.appliedMutations.length}`);
  console.log(`   • Штраф сложности графа Omega(G): ${result.omegaPenalty ?? 0}`);
  console.log(`   • Финальное состояние State:`, JSON.stringify(result.finalState, null, 2));
  console.log('============================================================\n');
}

async function main() {
  const [, , cmd, ...rest] = process.argv;
  if (cmd === 'init') {
    await runInteractiveWizard();
  } else if (cmd === 'run') {
    const goal = rest.join(' ').trim();
    if (!goal) {
      console.error('❌ Укажите цель запуска: spectra run "<ваша цель>"');
      process.exit(1);
    }
    await handleRunCommand(goal);
  } else {
    console.log('Использование CLI Spectra:');
    console.log('  spectra init                — Интерактивная настройка конфигурации');
    console.log('  spectra run "<goal>"        — Запуск адаптивного цикла под цель');
  }
}

main().catch((err) => {
  console.error('Критическая ошибка Spectra CLI:', err);
  process.exit(1);
});
```

---

## 5. Дорожная карта реализации по этапам

1. **Этап 1 (Дни 1–3):** Инициализация репозитория, создание папки `types/` с рукописными `.d.ts` контрактами, настройка `tsconfig.json` (`noEmit: true`, `declaration: false`) и написание модулей Fast/Slow Loop.
2. **Этап 2 (Дни 4–7):** Поднятие PostgreSQL 16 с `pgvector` через `docker compose up -d`, миграция `sql/001_init_pgvector.sql` и подключение `@huggingface/transformers` с `@receptron/laya`.
3. **Неделя 2:** Интеграция LangChain.js для System-2, оформление CLI-бинарника в `bin/spectra.js` и запуск сквозных тестов команды `spectra run`.