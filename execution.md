# РОЛЬ И ЗАДАЧА
Ты — Principal Systems Architect и Lead Node.js / AI Engineer. Твоя задача — полностью реализовать рабочий прототип самообучающейся графовой когнитивной системы **Spectra** строго по инженерному плану `implementation_plan.md` и теоретической базе из `spectra_master_specification.md` (развитие концепции `idea.md`).

Ты должен создать все файлы проекта без сокращений, без `TODO`, без заглушек и без единого мока (Zero Mocks Policy), пройти статическую проверку типов `npm run typecheck` (`tsc --noEmit`) и успешно выполнить боевой E2E-тест `npm test`.

---

# 1. ГЛАВНАЯ ИДЕЯ СИСТЕМЫ SPECTRA (КОНЦЕПТУАЛЬНЫЙ ЯКОРЬ)
При написании каждого модуля строго соблюдай фундаментальную идею **Spectra**:

1. **Отказ от хаотичного агентного цикла в пользу графового автомата (`DirectedGraph`):**
   - Нейросеть не управляет циклом исполнения напрямую через свободный текст — она изолирована внутри типизированных узлов графа (`llm`, `tool`, `branch`, `join`, `check`).
   - Все данные передаются только через единый сериализуемый объект состояния `State` с явными контрактами чтения `R(v)` и записи `W(v)`.
   - Переходы между узлами определяются детерминированными предикатами `guard(state)` со строгим линейным порядком приоритетов ребер `Π(e)`.

2. **Трехконтурная архитектура непрерывного самообучения (Continual Self-Learning):**
   - **Контур 1 — Fast Loop (Исполнительный уровень):** Быстрое и детерминированное движение по графу. На каждом шаге рантайм контролирует 5 инвариантов тупика (`NO_VALID_TRANSITION`, `CHECK_FAILED`, `TOOL_EXECUTION_ERROR`, `STATE_HASH_LOOP` по хэшу пары `(nodeId, state)`, `HARD_LIMIT_EXCEEDED`).
   - **Контур 2 — Slow Loop (Мета-уровень восстановления и адаптации):** Активируется ТОЛЬКО при возникновении тупика в Fast Loop:
     a) **Извлечение опыта (Experience Replay):** Формируется мультимодальный вектор инцидента `v_inc` (ошибка + локальная топология одеяла Маркова `MB(v_t, G)` + сигнатура схемы `State`). Выполняется гибридный поиск в PostgreSQL + `pgvector` (косинусное сходство HNSW + реляционное заземление `Ψ_adapt` по коэффициенту Жаккара контрактов `reads/writes`).
     b) **Синтез мутации (System-2):** Если в памяти нет подходящего прецедента, вызывается генеративный арбитр System-2 (`LangChain.js` + `ChatOpenAI.withStructuredOutput(Zod)`), который синтезирует план мутации графа (`addNodeAndBridge` и др.) и при необходимости — чистый JS-код нового инструмента для изолированной песочницы `node:vm` (`ToolRegistry`).
     c) **Дуальный гейткипинг перед коммитом мутации:**
        - **System-1 Арбитр (`@receptron/laya` на ONNX Runtime):** За один прямой проход оценивает риск топологического дрейфа (`score`), связность с целью (`score`), вероятность одобрения `noul` `P(true)` и область отката (`keep`, `rollback_last`, `rollback_session`).
        - **Верификатор детерминированности (`GraphDeterminismVerifier`):** Алгоритмически проверяет граф-кандидат через поиск компонент сильной связности Тарьяна (`Tarjan SCC` — блокировка циклов без прогресса и без лимита повторов), обратный обход в ширину (`Reverse BFS` — достижимость `goalNodes`), отсутствие коллизий приоритетов ребер и транзитивную совместимость интервалов `guard`-предикатов. При нарушении инвариантов выдается мгновенный `HARD_REJECT`.
     d) **Откат как операция первого класса (`Rollback & Saga`):** Все версии графа пишутся в append-only журнал `graph_versions` в PostgreSQL. При отклонении мутации рантайм откатывает состояние `State` (выполняя компенсирующие Saga-операторы `Φ_v^-1` для неидемпотентных узлов) и возвращает курсор вершины.
   - **Контур 3 — Consolidation Loop («Фаза сна» / Фоновый рефакторинг):** Предотвращает деградацию графа в «спагетти-сеть» после множества рантайм-мутаций:
     - Вычисляет итоговую полезность сессии `U_total = U_base - Ω(G)` (с учетом топологического штрафа верификатора `Ω(G)`).
     - Калибрует байесовское доверие к прецедентам в `pgvector` и навсегда блокирует токсичные мутации (`is_antipattern = TRUE`).
     - Выполняет бисимуляционную свертку эквивалентных параллельных веток (`foldBisimilarBranches`), продвигает информативные ключи из карантина онтологии `K_staging -> K_canonical` и вычищает шумовые ключи по TTL.

---

# 2. ЖЕСТКИЕ ТЕХНИЧЕСКИЕ ОГРАНИЧЕНИЯ СТЕКА
1. **Runtime & Modules:** Node.js >= 20.0.0, чистые ESM-модули (`"type": "module"`, обязательные расширения `.js` в локальных `import`).
2. **Самописные `.d.ts` контракты (Contract-First):**
   - Файлы `types/*.d.ts` пишутся ТОЛЬКО вручную.
   - В `tsconfig.json` обязательно: `"allowJs": true`, `"checkJs": true`, `"strict": true`, `"noEmit": true`, `"declaration": false`, `"emitDeclarationOnly": false`.
   - Все `.js` классы обязаны иметь JSDoc-аннотации `@implements {...}` и строгую типизацию параметров, чтобы `npm run typecheck` (`tsc --noEmit`) проходил с 0 ошибок.
3. **Запрет на моки (Zero Mocks Policy):**
   - Никаких фейковых эмбеддингов, стабов LLM или in-memory подмен БД в коде и тестах.
   - Эмбеддинги считаются локально через `@huggingface/transformers` (`Xenova/multilingual-e5-small`, `384` измерения).
   - System-1 работает через реальный пакет `@receptron/laya` (`Laya.load()` и `laya.systemOne()`).
   - System-2 работает через `@langchain/openai` (`ChatOpenAI` + `zod`).
   - Хранилище работает поверх реального PostgreSQL 16 с расширением `pgvector`.
   - Все учетные данные (`SPECTRA_LLM_API_KEY` / `OPENAI_API_KEY`, `SPECTRA_LLM_BASE_URL`, `SPECTRA_LLM_FRONTIER_MODEL`, `SPECTRA_PG_URL`, `HF_TOKEN`) прокидываются через единый резолвер `src/cli/credentials.js` (из CLI-флагов, переменных окружения, `.env` или `spectra.config.json`).
4. **Полная наблюдаемость (`SpectraLogger`):**
   - Каждое действие системы в рантайме должно логироваться через `src/cli/logger.js`: ASCII-карта графа с указателем текущего узла (`👉`), пошаговый дифф ключей `State` (`+ADDED`, `~MODIFIED`, `-REMOVED`), таблица проверки `guard`-ребер с приоритетами, веса одеяла Маркова (PPR), метрики `@receptron/laya`, диагностика верификатора Тарьяна и параллельная запись в `.spectra/logs/session.jsonl`.

---

# 3. ПОШАГОВЫЙ ПЛАН ВЫПОЛНЕНИЯ (ПОРЯДОК СОЗДАНИЯ ФАЙЛОВ)

Выполняй реализацию строго по следующим 6 этапам из `implementation_plan.md`:

### Этап 1: Инфраструктура и конфигурация
Создай корневые файлы проекта:
- `package.json` (скрипты `start`, `init`, `typecheck`, `test`, `db:up`, `db:migrate` и `bin: { "spectra": "./bin/spectra.js" }`)
- `tsconfig.json`
- `.env.example`
- `docker-compose.yml` (образ `pgvector/pgvector:pg16` с healthcheck и монтированием `./sql/001_init_pgvector.sql`)
- `sql/001_init_pgvector.sql` (расширение `vector`, таблицы `graph_versions`, `deadlock_precedents` с HNSW-индексом `vector_cosine_ops` размерности `384`, `state_ontology_keys`)

### Этап 2: Рукописные контракты типов (`types/`)
Создай самописные файлы деклараций:
- `types/domain.d.ts` (`NodeType`, `IdempotencyKind`, `DeadlockType`, `GraphState`, `IntervalBound`, `GraphEdge`, `GraphNode`, `VerificationDiagnostics`, `VerificationResult`, `DeadlockIncident`, `MutationPlan`, `LayaSystem1Verdict`, `PrecedentRecord`, `SpectraConfig`)
- `types/core.d.ts` (`ISpectraLogger`, `IDirectedGraph`, `IGraphDeterminismVerifier`, `IToolRegistry`, `IStateOntologyManager`, `ISpectraRuntime`)
- `types/ai.d.ts` (`ILocalEmbeddingEngine`, `ILayaSystem1Arbiter`, `ILangChainSystem2Arbiter`)
- `types/index.d.ts`

### Этап 3: Детерминированное ядро, песочница и онтология (`src/core/`)
Реализуй модули ядра:
- `src/core/graph.js` (`DirectedGraph`: гибридный `clone()` без падения на функциях, `compileGuardExpression()` в `node:vm` с очисткой блочных комментариев и хвостовых `;`, `getStructuralHash()`, `getCausalMarkovBlanket()`, `foldBisimilarBranches()`, `toJSON()` / `fromJSON()`)
- `src/core/verifier.js` (`GraphDeterminismVerifier`: поиск SCC по Тарьяну, обратный BFS достижимости целей, проверка коллизий приоритетов, прямая абстрактная интерпретация интервалов `_checkTransitivePathSatisfiability()` и расчет штрафа `Ω(G)`)
- `src/core/tools.js` (`ToolRegistry`: изолированное выполнение в `node:vm` через `executeSandboxed()` с заморозкой копии входа, очисткой хвостовых `;` и проверкой Saga-компенсаторов `verifyAndAttachSagaCompensator()`)
- `src/core/ontology.js` (`StateOntologyManager`: сжатие путей алиасов `findCanonical()`, регистрация в `K_staging`, продвижение при `IG >= 0.05` и сборка мусора по TTL)
- `src/core/runtime.js` (`SpectraRuntime`: цикл `Fast Loop <-> Slow Loop`, вставка предваряющего моста перед `check`-узлом при `CHECK_FAILED` с преемптивным приоритетом `0` и отрицанием проверяемых ключей, гибридный поиск и реляционное заземление прецедентов, маркировка отклоненных прецедентов как `isAntipattern = true`, откат `rollbackTrajectoryAndCursor` с очисткой хэшей циклов и метод `runConsolidationSleep()`)

### Этап 4: Слой ИИ и векторное хранилище (`src/ai/` и `src/db/`)
Реализуй боевые адаптеры без моков:
- `src/ai/embeddings.js` (`LocalEmbeddingEngine` на `@huggingface/transformers`, модель `Xenova/multilingual-e5-small`, расчет нормированных `embedText()` и мультимодального `computeIncidentVector()`)
- `src/ai/laya-system1.js` (`LayaSystem1Arbiter` на `@receptron/laya`: `Laya.load()` и батч-вызов `laya.systemOne()` с вопросами типов `score`, `noul`, `choice`)
- `src/ai/langchain-system2.js` (`LangChainSystem2Arbiter` на `@langchain/openai` + `zod` с `.nullable()` полями для совместимости со `strict: true` в OpenAI Structured Outputs, методы `synthesizeMutation()` и `executeLlmNode()`)
- `src/db/pg-store.js` (`PgVectorPrecedentStore` с типобезопасным приватным методом `_getPool()`, векторным поиском `1 - (incident_vector <=> $2::vector) AS similarity`, сохранением версий графа и синхронизацией ключей онтологии)

### Этап 5: Логирование, прокидывание учетных данных и интерактивный CLI (`src/cli/` и `bin/`)
Реализуй CLI-интерфейс:
- `src/cli/logger.js` (`SpectraLogger` с ANSI-форматированием и записью в `.spectra/logs/session.jsonl`)
- `src/cli/credentials.js` (`loadDotEnvFile()`, `parseCliFlags()`, `resolveSpectraConfig()`, `assertValidLlmCredentials()`)
- `src/cli/wizard.js` (`runInteractiveWizard()` на `node:readline/promises`)
- `bin/spectra.js` (команды `spectra init`, `spectra run "<goal>"` с поддержкой `--fresh-graph`, `--api-key`, `--base-url`, `--model`, `--pg-url`, `--verbose`, а также `spectra verify` и `spectra sleep`, плюс кроссплатформенная проверка точки входа через `realpathSync`)

### Этап 6: Финальная верификация и сквозное тестирование (`test/e2e.test.js`)
1. Создай файл `test/e2e.test.js` на встроенном раннере `node:test` и `node:assert/strict`.
2. Выполни `npm run typecheck` (`tsc --noEmit`) и убедись, что ошибок типизации `0`.
3. Подними базу данных (`npm run db:up`) и запусти `npm test`, проверив все 4 этапа самообучения в живом рантайме:
   - **Сессия 1:** Возникновение тупика `CHECK_FAILED` на стартовом графе `G_0` -> синтез мутации и кода инструмента реальной LLM (`SYSTEM2_SYNTHESIZED`) -> одобрение `@receptron/laya` и `GraphDeterminismVerifier` -> сохранение прецедента в `pgvector` -> достижение `goal`.
   - **Сессия 2 (Самообучение на новом `goal` и чистом графе `G_0`):** Возникновение аналогичного тупика -> мгновенное извлечение рецепта из `pgvector` (`MEMORY_GROUNDED`) без вызова тяжелой LLM (`llmCalls === 0`) -> успешное достижение новой цели.
   - **Тест Saga-компенсатора:** Проверка обратимости неидемпотентного инструмента (`barrier === false`) в `node:vm`.
   - **Фаза «Сна» (`runConsolidationSleep`):** Проверка бисимуляционной свертки дублирующей ветви (`foldedNodes === 1`), продвижения информативного ключа в `K_canonical`, удаления шумового ключа из `K_staging` и снижения топологического штрафа `Ω(G*) < Ω(G)`. 