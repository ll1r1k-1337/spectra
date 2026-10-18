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
