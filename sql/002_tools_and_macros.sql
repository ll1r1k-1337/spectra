-- sql/002_tools_and_macros.sql
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS registered_tools (
  tool_id VARCHAR(128) PRIMARY KEY,
  reads TEXT[] NOT NULL,
  writes TEXT[] NOT NULL,
  source_code TEXT NOT NULL,
  version INTEGER NOT NULL DEFAULT 1,
  has_irreversible_barrier BOOLEAN NOT NULL DEFAULT FALSE,
  saga_compensator_code TEXT,
  regression_pool JSONB NOT NULL DEFAULT '[]'::jsonb,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS macro_patterns (
  macro_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name VARCHAR(128) NOT NULL UNIQUE,
  source_node_ids TEXT[] NOT NULL,
  target_node_id VARCHAR(128),
  pattern_json JSONB NOT NULL,
  support INTEGER NOT NULL DEFAULT 1,
  success_rate DOUBLE PRECISION NOT NULL DEFAULT 0.5,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_registered_tools_updated
ON registered_tools (updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_macro_patterns_name
ON macro_patterns (name);

CREATE INDEX IF NOT EXISTS idx_macro_patterns_support
ON macro_patterns (support DESC);

CREATE INDEX IF NOT EXISTS idx_macro_patterns_source_nodes
ON macro_patterns USING gin (source_node_ids);
