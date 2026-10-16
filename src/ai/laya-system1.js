// @ts-check
import { logger } from '../utils/logger.js';

/**
 * @typedef {import('../../types/ai.js').ILayaSystem1Arbiter} ILayaSystem1Arbiter
 * @typedef {import('../../types/domain.js').DeadlockIncident} DeadlockIncident
 * @typedef {import('../../types/domain.js').MutationPlan} MutationPlan
 * @typedef {import('../../types/domain.js').System1Verdict} System1Verdict
 * @typedef {import('../../types/domain.js').SpectraConfig} SpectraConfig
 */

/**
 * @implements {ILayaSystem1Arbiter}
 */
export class LayaSystem1Arbiter {
  /**
   * @param {Partial<SpectraConfig['system1']> & Record<string, any>} [config]
   */
  constructor(config = {}) {
    /** @type {SpectraConfig['system1']} */
    this.config = {
      provider: 'laya-onnx',
      confidenceThreshold: config.confidenceThreshold ?? 0.7,
      driftTolerance: config.driftTolerance ?? 0.35
    };
    this.maxDrift = config.maxDrift ?? this.config.driftTolerance;
    this.minCoherence = config.minCoherence ?? 0.65;
    this.minApproveProb = config.minApproveProb ?? this.config.confidenceThreshold;
    this.temperaturePlatt = config.temperaturePlatt ?? 1.0;
    /** @type {any} */
    this.laya = null;
    this.initialized = false;
  }

  /**
   * Initializes Laya ONNX runtime or marks offline for heuristic fallback.
   * @returns {Promise<void>}
   */
  async init() {
    if (this.initialized) {
      return;
    }
    try {
      const { Laya } = await import('@receptron/laya');
      this.laya = await Laya.load();
    } catch {
      this.laya = null;
    }
    this.initialized = true;
  }

  /**
   * Evaluates proposed mutation against current deadlock and system goal.
   * @param {DeadlockIncident} deadlock
   * @param {MutationPlan} mutationPlan
   * @param {string} goalDescription
   * @returns {Promise<System1Verdict>}
   */
  async evaluateMutation(deadlock, mutationPlan, goalDescription) {
    if (!this.initialized) {
      await this.init();
    }

    logger.debug('Laya', 'Extracted input features for System-1 evaluation', {
      deadlockType: deadlock.type,
      nodeId: deadlock.nodeId,
      mutationOperator: mutationPlan.operator,
      goal: goalDescription,
      reads: Object.keys(deadlock.stateSnapshot || {}),
      reasoning: mutationPlan.reasoning
    });

    if (this.laya) {
      try {
        logger.info('Laya', 'Evaluating mutation using ONNX model path');
        const result = await this.laya.systemOne(
          { deadlock, mutationPlan, goal: goalDescription },
          {
            drift: {
              type: 'score',
              instructions: 'Assess potential hallucination or behavioral drift risk of the mutation',
              criteria: ['minimal', 'moderate', 'high', 'critical']
            },
            coherence: {
              type: 'score',
              instructions: 'Assess causal and structural coherence with the target goal',
              criteria: ['none', 'low', 'moderate', 'high']
            },
            approve: {
              type: 'noul',
              instructions: 'Approve this mutation plan for execution?'
            }
          }
        );

        const driftRisk = Number((result.drift.score / 3).toFixed(4));
        const coherence = Number((result.coherence.score / 3).toFixed(4));
        const approveProb = Number(result.approve.noul.toFixed(4));
        const approved =
          driftRisk <= this.maxDrift &&
          coherence >= this.minCoherence &&
          approveProb >= this.minApproveProb;

        /** @type {'keep' | 'rollback_last' | 'rollback_session'} */
        const rollbackAction = approved ? 'keep' : driftRisk > 0.65 ? 'rollback_session' : 'rollback_last';

        logger.info('Laya', `System-1 evaluation (ONNX): P(approve)=${approveProb}, driftRisk=${driftRisk}, coherence=${coherence}, approved=${approved}`);
        logger.debug('Laya', 'Calibrated System-1 ONNX verdict details', {
          verdict: { driftRisk, coherence, approveProb, approved, rollbackAction }
        });

        return {
          driftRisk,
          coherence,
          approveProb,
          approved,
          rollbackAction,
          selectedPrecedentId: null
        };
      } catch (err) {
        logger.warn('Laya', `ONNX evaluation failed: ${err instanceof Error ? err.message : String(err)}; falling back to heuristic evaluation`);
      }
    }

    return this.evaluateHeuristic(deadlock, mutationPlan, goalDescription);
  }

  /**
   * Deterministic heuristic scoring based on configured thresholds.
   * @param {DeadlockIncident} deadlock
   * @param {MutationPlan} mutationPlan
   * @param {string} _goalDescription
   * @returns {System1Verdict}
   */
  evaluateHeuristic(deadlock, mutationPlan, _goalDescription) {
    logger.info('Laya', 'Evaluating mutation using heuristic fallback path (ONNX runtime offline)');
    let driftRisk = 0.12;
    let coherence = 0.88;

    if (mutationPlan.operator === 'patchToolAST') {
      driftRisk += 0.12;
    } else if (mutationPlan.operator === 'splitNode') {
      driftRisk += 0.08;
    } else if (mutationPlan.operator === 'addNodeAndBridge') {
      driftRisk += 0.04;
    }

    if (mutationPlan.reasoning && mutationPlan.reasoning.length > 10) {
      coherence += 0.05;
    }

    if (mutationPlan.to === 'goal' || mutationPlan.from === deadlock.nodeId) {
      coherence += 0.04;
    }

    driftRisk = Math.min(0.95, Math.max(0.05, Number(driftRisk.toFixed(4))));
    coherence = Math.min(0.99, Math.max(0.05, Number(coherence.toFixed(4))));

    const temp = this.temperaturePlatt || 1.0;
    const logit = (coherence - driftRisk) / temp;
    const approveProb = Math.min(0.99, Math.max(0.01, Number((1 / (1 + Math.exp(-2.5 * logit))).toFixed(4))));

    const approved =
      driftRisk <= this.maxDrift &&
      coherence >= this.minCoherence &&
      approveProb >= this.minApproveProb;

    /** @type {'keep' | 'rollback_last' | 'rollback_session'} */
    const rollbackAction = approved ? 'keep' : driftRisk > 0.65 ? 'rollback_session' : 'rollback_last';

    logger.info('Laya', `System-1 evaluation (heuristic): P(approve)=${approveProb}, driftRisk=${driftRisk}, coherence=${coherence}, approved=${approved}`);
    logger.debug('Laya', 'Calibrated System-1 heuristic verdict details', {
      verdict: { driftRisk, coherence, approveProb, approved, rollbackAction }
    });

    return {
      driftRisk,
      coherence,
      approveProb,
      approved,
      rollbackAction,
      selectedPrecedentId: null
    };
  }
}
