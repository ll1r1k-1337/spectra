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
