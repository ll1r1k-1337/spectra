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
