// @ts-check
import pg from 'pg';
import { logger } from '../utils/logger.js';

/**
 * @typedef {import('../../types/domain.js').SpectraConfig} SpectraConfig
 */

export class PgVectorStore {
  /**
   * @param {string | pg.PoolConfig} [config]
   */
  constructor(config) {
    if (typeof config === 'string') {
      this.poolConfig = { connectionString: config };
      this.connectionString = config;
    } else if (config) {
      this.poolConfig = config;
      this.connectionString = config.connectionString || '';
    } else {
      this.connectionString = 'postgresql://spectra:spectra_password@localhost:5432/spectra_db';
      this.poolConfig = { connectionString: this.connectionString };
    }

    /** @type {pg.Pool | null} */
    this.pool = null;
    this.isConnected = false;
    /** @type {Promise<this> | null} */
    this.initPromise = null;
  }

  /**
   * Initializes PostgreSQL pool connection gracefully.
   * @returns {Promise<this>}
   */
  async init() {
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = (async () => {
      try {
        this.pool = new pg.Pool(this.poolConfig);
        // Error handler to prevent unhandled error event crashing Node.js process
        this.pool.on('error', (err) => {
          this.isConnected = false;
          logger.warn('PgVector', `PostgreSQL pool connection error: ${err.message}`);
        });

        const client = await this.pool.connect();
        client.release();
        this.isConnected = true;
        logger.info('PgVector', 'Connected to PostgreSQL database');
      } catch (err) {
        this.isConnected = false;
        logger.warn('PgVector', `Failed to connect to PostgreSQL: ${err instanceof Error ? err.message : String(err)} (running in disconnected mode)`);
      }
      return this;
    })();

    return this.initPromise;
  }

  /**
   * Formats a float array into pgvector literal syntax `[0.1,0.2,...]`.
   * @param {number[]} vector
   * @returns {string}
   */
  static formatVector(vector) {
    return `[${vector.join(',')}]`;
  }

  /**
   * Saves an immutable graph version definition.
   * @param {object} versionData
   * @param {string} versionData.graphId
   * @param {string} [versionData.parentVersionId]
   * @param {string} [versionData.secondaryParentVersionId]
   * @param {string} versionData.structuralHash
   * @param {any} versionData.definitionJson
   * @param {number} [versionData.omegaPenalty]
   * @param {number} [versionData.lyapunovEnergy]
   * @param {number} [versionData.lyapunovCreditBank]
   * @param {string} [versionData.releaseStage]
   * @param {number} [versionData.trafficAllocation]
   * @returns {Promise<any | null>}
   */
  async saveGraphVersion(versionData) {
    if (!this.isConnected || !this.pool) return null;
    try {
      logger.debug('PgVector', 'Executing query: saveGraphVersion', {
        graphId: versionData.graphId,
        structuralHash: versionData.structuralHash
      });
      const query = `
        INSERT INTO graph_versions (
          graph_id, parent_version_id, secondary_parent_version_id,
          structural_hash, definition_json, omega_penalty,
          lyapunov_energy, lyapunov_credit_bank, release_stage, traffic_allocation
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING *
      `;
      const res = await this.pool.query(query, [
        versionData.graphId,
        versionData.parentVersionId || null,
        versionData.secondaryParentVersionId || null,
        versionData.structuralHash,
        JSON.stringify(versionData.definitionJson),
        versionData.omegaPenalty ?? 0.0,
        versionData.lyapunovEnergy ?? null,
        versionData.lyapunovCreditBank ?? 15.0,
        versionData.releaseStage || 'SHADOW',
        versionData.trafficAllocation ?? 0.0
      ]);
      return res.rows[0] || null;
    } catch (err) {
      logger.error('PgVector', `saveGraphVersion query failed: ${err instanceof Error ? err.message : String(err)}`, { error: err });
      return null;
    }
  }

  /**
   * Retrieves latest released or created graph version for graphId.
   * @param {string} graphId
   * @returns {Promise<any | null>}
   */
  async getLatestGraphVersion(graphId) {
    if (!this.isConnected || !this.pool) return null;
    try {
      logger.debug('PgVector', 'Executing query: getLatestGraphVersion', { graphId });
      const query = `
        SELECT * FROM graph_versions
        WHERE graph_id = $1
        ORDER BY created_at DESC
        LIMIT 1
      `;
      const res = await this.pool.query(query, [graphId]);
      return res.rows[0] || null;
    } catch (err) {
      logger.error('PgVector', `getLatestGraphVersion query failed: ${err instanceof Error ? err.message : String(err)}`, { error: err });
      return null;
    }
  }

  /**
   * Stores a deadlock incident and associated recovery mutation.
   * @param {object} precedentData
   * @param {string} precedentData.deadlockType
   * @param {number[]} precedentData.incidentVector
   * @param {string[]} [precedentData.anchorReads]
   * @param {string[]} [precedentData.anchorWrites]
   * @param {string} precedentData.mutationOperator
   * @param {any} precedentData.mutationPlan
   * @param {string} [precedentData.entryGuardPredicate]
   * @param {number} [precedentData.successCount]
   * @param {number} [precedentData.failureCount]
   * @param {boolean} [precedentData.isPermanentAntipattern]
   * @returns {Promise<any | null>}
   */
  async saveDeadlockPrecedent(precedentData) {
    if (!this.isConnected || !this.pool) return null;
    try {
      logger.debug('PgVector', 'Executing query: saveDeadlockPrecedent', {
        deadlockType: precedentData.deadlockType,
        mutationOperator: precedentData.mutationOperator
      });
      const vecStr = PgVectorStore.formatVector(precedentData.incidentVector);
      const query = `
        INSERT INTO deadlock_precedents (
          deadlock_type, incident_vector, anchor_reads, anchor_writes,
          mutation_operator, mutation_plan_json, entry_guard_predicate,
          success_count, failure_count, is_permanent_antipattern
        ) VALUES ($1, $2::vector, $3, $4, $5, $6, $7, $8, $9, $10)
        RETURNING *
      `;
      const res = await this.pool.query(query, [
        precedentData.deadlockType,
        vecStr,
        precedentData.anchorReads || [],
        precedentData.anchorWrites || [],
        precedentData.mutationOperator,
        JSON.stringify(precedentData.mutationPlan),
        precedentData.entryGuardPredicate || null,
        precedentData.successCount ?? 1.0,
        precedentData.failureCount ?? 0.0,
        Boolean(precedentData.isPermanentAntipattern)
      ]);
      return res.rows[0] || null;
    } catch (err) {
      logger.error('PgVector', `saveDeadlockPrecedent query failed: ${err instanceof Error ? err.message : String(err)}`, { error: err });
      return null;
    }
  }

  /**
   * Finds nearest deadlock precedents using cosine distance (<=>).
   * @param {number[]} vector
   * @param {string} deadlockType
   * @param {number} [limit]
   * @returns {Promise<any[]>}
   */
  async findSimilarPrecedents(vector, deadlockType, limit = 5) {
    if (!this.isConnected || !this.pool) return [];
    try {
      logger.debug('PgVector', 'Executing query: findSimilarPrecedents', { deadlockType, limit });
      const vecStr = PgVectorStore.formatVector(vector);
      const query = `
        SELECT precedent_id, deadlock_type, anchor_reads, anchor_writes,
               mutation_operator, mutation_plan_json, entry_guard_predicate,
               success_count, failure_count, avg_delta_u_dr_lcb,
               (incident_vector <=> $1::vector) AS distance
        FROM deadlock_precedents
        WHERE deadlock_type = $2 AND is_permanent_antipattern = FALSE AND success_count > failure_count
        ORDER BY distance ASC
        LIMIT $3
      `;
      const res = await this.pool.query(query, [vecStr, deadlockType, limit]);
      const results = res.rows.map((r) => ({
        precedentId: r.precedent_id,
        deadlockType: r.deadlock_type,
        operator: r.mutation_operator,
        cosineDistance: r.distance
      }));
      logger.info('PgVector', `Nearest neighbor search found ${res.rows.length} precedent(s) for "${deadlockType}"`, {
        deadlockType,
        results
      });
      return res.rows;
    } catch (err) {
      logger.error('PgVector', `findSimilarPrecedents query failed: ${err instanceof Error ? err.message : String(err)}`, { error: err });
      return [];
    }
  }

  /**
   * Updates the success or failure count for a precedent.
   * @param {string} precedentId
   * @param {boolean} wasSuccess
   * @returns {Promise<any | null>}
   */
  async updatePrecedentOutcome(precedentId, wasSuccess) {
    if (!this.isConnected || !this.pool) return null;
    try {
      logger.debug('PgVector', 'Executing query: updatePrecedentOutcome', { precedentId, wasSuccess });
      const query = wasSuccess
        ? `UPDATE deadlock_precedents
           SET success_count = success_count + 1,
               raw_executions = raw_executions + 1,
               last_applied_at = NOW()
           WHERE precedent_id = $1
           RETURNING *`
        : `UPDATE deadlock_precedents
           SET failure_count = failure_count + 1,
               raw_executions = raw_executions + 1,
               last_applied_at = NOW()
           WHERE precedent_id = $1
           RETURNING *`;
      const res = await this.pool.query(query, [precedentId]);
      return res.rows[0] || null;
    } catch (err) {
      logger.error('PgVector', `updatePrecedentOutcome query failed: ${err instanceof Error ? err.message : String(err)}`, { error: err });
      return null;
    }
  }

  /**
   * Marks a precedent as permanent antipattern.
   * @param {string} precedentId
   * @returns {Promise<any | null>}
   */
  async markPrecedentAntipattern(precedentId) {
    if (!this.isConnected || !this.pool) return null;
    try {
      logger.debug('PgVector', 'Executing query: markPrecedentAntipattern', { precedentId });
      const query = `
        UPDATE deadlock_precedents
        SET is_permanent_antipattern = TRUE,
            failure_count = failure_count + 1,
            raw_executions = raw_executions + 1,
            last_applied_at = NOW()
        WHERE precedent_id = $1
        RETURNING *
      `;
      const res = await this.pool.query(query, [precedentId]);
      return res.rows[0] || null;
    } catch (err) {
      logger.error('PgVector', `markPrecedentAntipattern query failed: ${err instanceof Error ? err.message : String(err)}`, { error: err });
      return null;
    }
  }

  /**
   * Saves or updates a macro node pattern.
   * @param {object} macroData
   * @param {string} macroData.macroId
   * @param {number} [macroData.version]
   * @param {number} [macroData.hierarchyLevel]
   * @param {number[]} macroData.intentVector
   * @param {string[]} [macroData.contractReads]
   * @param {string[]} [macroData.contractWrites]
   * @param {any} [macroData.subgraphDefinitionJson]
   * @param {boolean} [macroData.verifiedPure]
   * @param {number} [macroData.successRate]
   * @param {number} [macroData.executionCount]
   * @returns {Promise<any | null>}
   */
  async saveMacroPattern(macroData) {
    if (!this.isConnected || !this.pool) return null;
    try {
      const vecStr = PgVectorStore.formatVector(macroData.intentVector);
      const query = `
        INSERT INTO macro_patterns (
          macro_id, version, hierarchy_level, intent_vector,
          contract_reads, contract_writes, subgraph_definition_json,
          verified_pure, success_rate, execution_count
        ) VALUES ($1, $2, $3, $4::vector, $5, $6, $7, $8, $9, $10)
        ON CONFLICT (macro_id) DO UPDATE SET
          version = EXCLUDED.version,
          hierarchy_level = EXCLUDED.hierarchy_level,
          intent_vector = EXCLUDED.intent_vector,
          contract_reads = EXCLUDED.contract_reads,
          contract_writes = EXCLUDED.contract_writes,
          subgraph_definition_json = EXCLUDED.subgraph_definition_json,
          verified_pure = EXCLUDED.verified_pure,
          success_rate = EXCLUDED.success_rate,
          execution_count = EXCLUDED.execution_count
        RETURNING *
      `;
      const res = await this.pool.query(query, [
        macroData.macroId,
        macroData.version ?? 1,
        macroData.hierarchyLevel ?? 1,
        vecStr,
        macroData.contractReads || [],
        macroData.contractWrites || [],
        JSON.stringify(macroData.subgraphDefinitionJson || {}),
        macroData.verifiedPure ?? true,
        macroData.successRate ?? 1.0,
        macroData.executionCount ?? 1
      ]);
      return res.rows[0] || null;
    } catch (err) {
      logger.error('PgVector', `saveMacroPattern query failed: ${err instanceof Error ? err.message : String(err)}`, { error: err });
      return null;
    }
  }

  /**
   * Searches macro patterns using cosine similarity on intent_vector.
   * @param {number[]} intentVector
   * @param {number} [limit]
   * @returns {Promise<any[]>}
   */
  async searchMacroPatterns(intentVector, limit = 5) {
    if (!this.isConnected || !this.pool) return [];
    try {
      logger.debug('PgVector', 'Executing query: searchMacroPatterns', { limit });
      const vecStr = PgVectorStore.formatVector(intentVector);
      const query = `
        SELECT macro_id, version, hierarchy_level, contract_reads, contract_writes,
               subgraph_definition_json, verified_pure, success_rate, execution_count,
               (intent_vector <=> $1::vector) AS distance
        FROM macro_patterns
        ORDER BY distance ASC
        LIMIT $2
      `;
      const res = await this.pool.query(query, [vecStr, limit]);
      const results = res.rows.map((r) => ({
        macroId: r.macro_id,
        cosineDistance: r.distance
      }));
      logger.info('PgVector', `Nearest neighbor search found ${res.rows.length} macro pattern(s)`, { results });
      return res.rows;
    } catch (err) {
      logger.error('PgVector', `searchMacroPatterns query failed: ${err instanceof Error ? err.message : String(err)}`, { error: err });
      return [];
    }
  }

  /**
   * Updates multi-armed bandit posterior beta distribution for meta-operators.
   * @param {string} operatorName
   * @param {boolean} isSuccess
   * @param {number} [lambda]
   * @returns {Promise<any | null>}
   */
  async updateBanditPosterior(operatorName, isSuccess, lambda = 0.0) {
    if (!this.isConnected || !this.pool) return null;
    try {
      logger.debug('PgVector', 'Executing query: updateBanditPosterior', { operatorName, isSuccess, lambda });
      const alphaInc = isSuccess ? 1.0 : 0.0;
      const betaInc = isSuccess ? 0.0 : 1.0;
      const query = `
        INSERT INTO meta_operator_bandit (
          operator_name, alpha_success, beta_failure, lagrangian_multiplier_lambda, updated_at
        ) VALUES ($1, 1.0 + $2, 1.0 + $3, $4, NOW())
        ON CONFLICT (operator_name) DO UPDATE SET
          alpha_success = meta_operator_bandit.alpha_success + $2,
          beta_failure = meta_operator_bandit.beta_failure + $3,
          lagrangian_multiplier_lambda = $4,
          updated_at = NOW()
        RETURNING *
      `;
      const res = await this.pool.query(query, [operatorName, alphaInc, betaInc, lambda]);
      return res.rows[0] || null;
    } catch (err) {
      logger.error('PgVector', `updateBanditPosterior query failed: ${err instanceof Error ? err.message : String(err)}`, { error: err });
      return null;
    }
  }

  /**
   * Retrieves bandit posterior distributions for a given operator.
   * @param {string} operatorName
   * @returns {Promise<any | null>}
   */
  async getBanditPosterior(operatorName) {
    if (!this.isConnected || !this.pool) return null;
    try {
      logger.debug('PgVector', 'Executing query: getBanditPosterior', { operatorName });
      const query = `
        SELECT * FROM meta_operator_bandit
        WHERE operator_name = $1
      `;
      const res = await this.pool.query(query, [operatorName]);
      return res.rows[0] || null;
    } catch (err) {
      logger.error('PgVector', `getBanditPosterior query failed: ${err instanceof Error ? err.message : String(err)}`, { error: err });
      return null;
    }
  }

  /**
   * Closes database pool cleanly.
   * @returns {Promise<void>}
   */
  async close() {
    if (this.pool) {
      try {
        await this.pool.end();
      } catch {
        // Pool end error ignored
      }
      this.isConnected = false;
      this.pool = null;
      this.initPromise = null;
    }
  }
}
