CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Ensure user role spectra exists if initialized by superuser
DO $$
BEGIN
  IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'spectra') THEN
    CREATE ROLE spectra WITH SUPERUSER LOGIN PASSWORD 'spectra_password';
  END IF;
END
$$;

-- 1. Журнал иммутабельных версий графа с поддержкой 3-Way Pushout и энергетического фильтра Ляпунова
CREATE TABLE IF NOT EXISTS graph_versions (
    version_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    graph_id VARCHAR(128) NOT NULL,
    parent_version_id UUID REFERENCES graph_versions(version_id),
    secondary_parent_version_id UUID REFERENCES graph_versions(version_id),
    structural_hash CHAR(64) NOT NULL,
    definition_json JSONB NOT NULL,
    omega_penalty DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    lyapunov_energy DOUBLE PRECISION,
    lyapunov_credit_bank DOUBLE PRECISION NOT NULL DEFAULT 15.0,
    release_stage VARCHAR(32) NOT NULL DEFAULT 'SHADOW',
    traffic_allocation DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_graph_versions_lookup ON graph_versions(graph_id, release_stage, created_at DESC);

-- 2. Эволюционирующий словарь онтологии ключей состояния K_t со сжатием путей алиасов и TTL карантина
CREATE TABLE IF NOT EXISTS state_ontology_keys (
    key_name VARCHAR(128) PRIMARY KEY,
    json_schema JSONB NOT NULL,
    semantic_vector vector(384) NOT NULL,
    mutual_info_gain DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    status VARCHAR(32) NOT NULL DEFAULT 'STAGING',
    canonical_alias VARCHAR(128) REFERENCES state_ontology_keys(key_name),
    staging_epochs_lived INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Реестр выученных морфизмов приведения типов M_coerce для подтипирования Лисков в MetaInit
CREATE TABLE IF NOT EXISTS schema_coercion_morphisms (
    morphism_id VARCHAR(128) PRIMARY KEY,
    source_schema_hash CHAR(64) NOT NULL,
    target_schema_hash CHAR(64) NOT NULL,
    coercer_function_ast JSONB NOT NULL,
    empirical_error_rate DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Хранилище прецедентов дэдлоков с эмбеддингами v_inc (HNSW) и LCB-оценкой доверия
CREATE TABLE IF NOT EXISTS deadlock_precedents (
    precedent_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    deadlock_type VARCHAR(64) NOT NULL,
    incident_vector vector(384) NOT NULL,
    anchor_reads TEXT[] NOT NULL,
    anchor_writes TEXT[] NOT NULL,
    mutation_operator VARCHAR(64) NOT NULL,
    mutation_plan_json JSONB NOT NULL,
    entry_guard_predicate TEXT,
    success_count DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    failure_count DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    raw_executions INTEGER NOT NULL DEFAULT 0,
    avg_delta_u_dr_lcb DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    retro_audit_correction_sum DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    conformal_quantile DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    confidence_lcb DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    local_density DOUBLE PRECISION NOT NULL DEFAULT 0.5,
    is_permanent_antipattern BOOLEAN NOT NULL DEFAULT FALSE,
    last_applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_precedents_hnsw ON deadlock_precedents USING hnsw (incident_vector vector_cosine_ops) WITH (m = 16, ef_construction = 128);

-- 5. Библиотека иерархических макронод (MacroNodes уровней 1..L_max) с JSON-схемами контрактов
CREATE TABLE IF NOT EXISTS macro_patterns (
    macro_id VARCHAR(128) PRIMARY KEY,
    version INTEGER NOT NULL DEFAULT 1,
    hierarchy_level INTEGER NOT NULL DEFAULT 1,
    intent_vector vector(384) NOT NULL,
    contract_reads TEXT[] NOT NULL,
    contract_writes TEXT[] NOT NULL,
    input_schemas_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_schemas_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    subgraph_json JSONB NOT NULL,
    expected_cost DOUBLE PRECISION NOT NULL DEFAULT 5.0,
    confidence_lcb DOUBLE PRECISION NOT NULL DEFAULT 0.8,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_macro_intent_hnsw ON macro_patterns USING hnsw (intent_vector vector_cosine_ops);

-- 6. Реестр инструментов с регрессионным пулом S_test (CEGIS), квотами и Saga-компенсаторами
CREATE TABLE IF NOT EXISTS registered_tools (
    tool_id VARCHAR(128) PRIMARY KEY,
    version INTEGER NOT NULL DEFAULT 1,
    description TEXT NOT NULL,
    intent_vector vector(384) NOT NULL,
    source_code TEXT NOT NULL,
    saga_compensator_code TEXT,
    rate_limit_rps INTEGER NOT NULL DEFAULT 5,
    has_irreversible_barrier BOOLEAN NOT NULL DEFAULT FALSE,
    regression_test_pool JSONB NOT NULL DEFAULT '[]'::jsonb,
    quarantine_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_tool_intent_hnsw ON registered_tools USING hnsw (intent_vector vector_cosine_ops);

-- 7. Многорукие бандиты для динамического выбора мета-операторов с регуляризацией Лагранжа
CREATE TABLE IF NOT EXISTS meta_operator_bandit (
    operator_name VARCHAR(64) PRIMARY KEY,
    alpha_success DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    beta_failure DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    lagrangian_multiplier_lambda DOUBLE PRECISION NOT NULL DEFAULT 0.0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
