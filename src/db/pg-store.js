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
 * @typedef {import('../../types/index.js').RegisteredTool} RegisteredTool
 * @typedef {import('../../types/index.js').MacroPattern} MacroPattern
 */

const { Pool } = pg;

/**
 * Wilson lower bound for a Bernoulli success rate.
 * @param {number} successes
 * @param {number} failures
 * @param {number} [z]
 * @returns {number}
 */
function wilsonLowerBound(successes, failures, z = 1.96) {
  const n = successes + failures;
  if (!Number.isFinite(n) || n <= 0) return 0.5;
  const phat = Math.min(1, Math.max(0, successes / n));
  const denom = 1 + (z * z) / n;
  const center = phat + (z * z) / (2 * n);
  const margin = z * Math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n);
  return Math.min(1, Math.max(0, (center - margin) / denom));
}

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
              success_count, failure_count, confidence_lcb,
              1 - (incident_vector <=> $2::vector) AS similarity
       FROM deadlock_precedents
       WHERE deadlock_type = $1 AND is_antipattern = FALSE
       ORDER BY incident_vector <=> $2::vector ASC
       LIMIT 100`,
      [deadlockType, vecLiteral]
    );
    return res.rows
      .filter(/** @param {Record<string, any>} r */ (r) => Number(r.similarity ?? 0) >= minSimilarity)
      .map(
        /** @param {Record<string, any>} r */
        (r) => {
          const successCount = Number(r.success_count ?? 0);
          const failureCount = Number(r.failure_count ?? 0);
          const wilson = wilsonLowerBound(successCount, failureCount);
          const reliability = Math.max(0.05, Number(r.confidence_lcb ?? wilson));
          const similarity = Number(r.similarity);
          const coldStartBonus = 0.05 / (1 + successCount + failureCount);
          return {
            deadlockType: r.deadlock_type,
            similarity,
            score: similarity * (0.5 + 0.5 * reliability) + coldStartBonus,
            successCount,
            failureCount,
            confidenceLcb: reliability,
            anchorReads: r.anchor_reads || [],
            anchorWrites: r.anchor_writes || [],
            mutationPlan: r.mutation_plan_json,
            isAntipattern: Boolean(r.is_antipattern)
          };
        }
      )
      .sort((a, b) => b.score - a.score)
      .slice(0, 10);
  }

  /**
   * Recomputes LCB confidence from success/failure counts and flags antipatterns.
   * @returns {Promise<number>}
   */
  async calibratePrecedents() {
    const pool = await this._getPool();
    const res = await pool.query(
      `SELECT precedent_id, success_count, failure_count FROM deadlock_precedents`
    );
    for (const r of res.rows) {
      const successCount = Number(r.success_count ?? 0);
      const failureCount = Number(r.failure_count ?? 0);
      const confidenceLcb = wilsonLowerBound(successCount + 1, failureCount + 1);
      const total = successCount + failureCount;
      const isAntipattern = failureCount >= 3 && (total === 0 || successCount / total < 0.3);
      await pool.query(
        `UPDATE deadlock_precedents
         SET confidence_lcb = $2, is_antipattern = $3
         WHERE precedent_id = $1`,
        [r.precedent_id, confidenceLcb, isAntipattern]
      );
    }
    return res.rows.length;
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
   * @returns {Promise<RegisteredTool[]>}
   */
  async loadRegisteredTools() {
    const pool = await this._getPool();
    const res = await pool.query(
      `SELECT tool_id, reads, writes, source_code, version, has_irreversible_barrier,
              saga_compensator_code, regression_pool
       FROM registered_tools ORDER BY tool_id`
    );
    return res.rows.map(
      /** @param {Record<string, any>} r */
      (r) => ({
        id: r.tool_id,
        reads: r.reads || [],
        writes: r.writes || [],
        sourceCode: r.source_code,
        version: Number(r.version ?? 1),
        hasIrreversibleBarrier: Boolean(r.has_irreversible_barrier),
        sagaCompensatorCode: r.saga_compensator_code || undefined,
        regressionPool: r.regression_pool || []
      })
    );
  }

  /**
   * @param {RegisteredTool} tool
   */
  async saveRegisteredTool(tool) {
    const pool = await this._getPool();
    await pool.query(
      `INSERT INTO registered_tools
       (tool_id, reads, writes, source_code, version, has_irreversible_barrier,
        saga_compensator_code, regression_pool, updated_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, NOW())
       ON CONFLICT (tool_id) DO UPDATE SET
         reads = EXCLUDED.reads,
         writes = EXCLUDED.writes,
         source_code = EXCLUDED.source_code,
         version = EXCLUDED.version,
         has_irreversible_barrier = EXCLUDED.has_irreversible_barrier,
         saga_compensator_code = EXCLUDED.saga_compensator_code,
         regression_pool = EXCLUDED.regression_pool,
         updated_at = NOW()`,
      [
        tool.id,
        tool.reads,
        tool.writes,
        tool.sourceCode,
        tool.version ?? 1,
        Boolean(tool.hasIrreversibleBarrier),
        tool.sagaCompensatorCode || null,
        JSON.stringify(tool.regressionPool || [])
      ]
    );
  }

  /**
   * @param {MacroPattern} pattern
   */
  async saveMacroPattern(pattern) {
    const pool = await this._getPool();
    await pool.query(
      `INSERT INTO macro_patterns
       (macro_id, name, source_node_ids, target_node_id, pattern_json, support, success_rate)
       VALUES (COALESCE($1::uuid, uuid_generate_v4()), $2, $3, $4, $5, $6, $7)
       ON CONFLICT (name) DO UPDATE SET
         source_node_ids = EXCLUDED.source_node_ids,
         target_node_id = EXCLUDED.target_node_id,
         pattern_json = EXCLUDED.pattern_json,
         support = EXCLUDED.support,
         success_rate = EXCLUDED.success_rate`,
      [
        pattern.id || null,
        pattern.name,
        pattern.sourceNodeIds,
        pattern.targetNodeId || null,
        JSON.stringify(pattern.patternJson),
        pattern.support ?? 1,
        pattern.successRate ?? 0.5
      ]
    );
  }

  /**
   * @returns {Promise<Record<string, any>>}
   */
  async getStats() {
    const pool = await this._getPool();
    const res = await pool.query(
      `SELECT
         (SELECT COUNT(*) FROM graph_versions)::int AS graph_versions,
         (SELECT COUNT(*) FROM deadlock_precedents)::int AS deadlock_precedents,
         (SELECT COUNT(*) FROM registered_tools)::int AS registered_tools,
         (SELECT COUNT(*) FROM macro_patterns)::int AS macro_patterns,
         (SELECT COUNT(*) FROM state_ontology_keys)::int AS state_ontology_keys`
    );
    return res.rows[0];
  }

  /**
   * @returns {Promise<Record<string, number>>}
   */
  async pruneMemory() {
    const pool = await this._getPool();
    const res = await pool.query(
      `WITH deleted_precedents AS (
         DELETE FROM deadlock_precedents
         WHERE is_antipattern = TRUE AND failure_count >= 3
         RETURNING 1
       ), deleted_graph_versions AS (
         DELETE FROM graph_versions
         WHERE version_id NOT IN (
           SELECT version_id FROM graph_versions ORDER BY created_at DESC LIMIT 20
         )
         RETURNING 1
       )
       SELECT
         (SELECT COUNT(*) FROM deleted_precedents)::int AS deleted_precedents,
         (SELECT COUNT(*) FROM deleted_graph_versions)::int AS deleted_graph_versions`
    );
    return res.rows[0];
  }

  /**
   * Очищает таблицы БД перед прогоном интеграционных E2E-тестов.
   */
  async clearAll() {
    const pool = await this._getPool();
    await pool.query(
      `TRUNCATE TABLE deadlock_precedents, graph_versions, state_ontology_keys, registered_tools, macro_patterns`
    );
  }

  async close() {
    if (this.pool) {
      await this.pool.end();
      this.pool = null;
    }
  }
}
