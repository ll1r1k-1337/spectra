import test from 'node:test';
import assert from 'node:assert/strict';
import {
  applyRetroactiveAuditCorrection,
  computeWeightedConformalDoublyRobustUtility,
  updateLagrangianResourcePrices,
  selectModelTierForNode,
  updateCusumDiscountFactor,
  computePairedCrnSpsaStep
} from '../src/core/runtime.js';
import { LocalEmbeddingEngine } from '../src/ai/embeddings.js';
import { LayaSystem1Arbiter } from '../src/ai/laya-system1.js';
import { LangChainSystem2Arbiter } from '../src/ai/langchain-system2.js';

test('Mathematical routines: Retroactive audit and Lagrangian pricing', () => {
  // Retroactive audit
  const retro = applyRetroactiveAuditCorrection(10.0, 0.5, 100.0, 1.0, 0.6);
  assert.equal(retro, 10.0 + 0.5 * 100.0 * 0.4);

  // Lagrangian resource pricing
  const nextDuals = updateLagrangianResourcePrices(
    { tokens: 0.0, steps: 0.1 },
    { tokens: 1500, steps: 40 },
    { tokens: 1000, steps: 50 },
    0.05
  );
  assert.equal(nextDuals.tokens, 0.05 * 500); // Increased because 1500 > 1000
  assert.equal(nextDuals.steps, Math.max(0, 0.1 + 0.05 * (40 - 50))); // Decreased
});

test('Mathematical routines: Model tier cascading and CUSUM discount', () => {
  const tierStats = {
    frontier: { passRate: 0.98 },
    light: { lcbPassRate: 0.97 },
    local_slm: { lcbPassRate: 0.90 }
  };
  const tier = selectModelTierForNode(tierStats, 0.02, 1.0);
  assert.equal(tier, 'light'); // light (0.97) >= frontier - 0.02 (0.96)

  // CUSUM
  const cusumRes = updateCusumDiscountFactor(0.0, true, 0.85, 0.05, 3.5, 0.4);
  assert.equal(typeof cusumRes.discountFactor, 'number');
  assert.ok(cusumRes.discountFactor > 0);
});

test('Mathematical routines: Paired CRN-SPSA Step', () => {
  const currPsi = { threshold: 0.5 };
  const momentum = { threshold: 0.0 };
  const plusOutcomes = [10.0, 12.0];
  const minusOutcomes = [8.0, 9.0];
  const delta = { threshold: 1.0 };

  const spsaRes = computePairedCrnSpsaStep(
    currPsi,
    momentum,
    plusOutcomes,
    minusOutcomes,
    delta,
    0.05,
    0.05,
    0.9,
    { threshold: [0.1, 0.9] }
  );

  assert.ok(spsaRes.nextPsi.threshold > 0.5);
});

test('AI Modules: LocalEmbeddingEngine fallback/fast embedding', async () => {
  const embedder = new LocalEmbeddingEngine({
    modelName: 'Xenova/multilingual-e5-small',
    cacheDir: './.cache/transformers',
    weights: [0.5, 0.3, 0.2]
  });

  const v1 = await embedder.embedText('System anomaly detected');
  assert.equal(v1.length, 384);

  const vInc = await embedder.computeIncidentVector({
    errorText: 'CHECK_FAILED',
    markovBlanketSignature: 'node_a->node_b',
    stateSchemaSignature: 'val_a,val_b'
  });
  assert.equal(vInc.length, 384);
});

test('AI Modules: LayaSystem1Arbiter and LangChainSystem2Arbiter offline synthesis', async () => {
  const sys1 = new LayaSystem1Arbiter({
    maxDrift: 0.35,
    minCoherence: 0.65,
    minApproveProb: 0.70
  });

  const sys2 = new LangChainSystem2Arbiter({
    frontierModel: 'gpt-4o',
    lightModel: 'gpt-4o-mini',
    baseURL: 'https://custom-llm-proxy.internal/v1'
  });
  assert.equal(sys2.config.baseURL, 'https://custom-llm-proxy.internal/v1');

  const deadlock = {
    type: 'NO_VALID_TRANSITION',
    nodeId: 'worker_node',
    step: 2,
    stateSnapshot: { val: 42 }
  };

  const plan = await sys2.synthesizeRecoveryMutation(
    deadlock,
    'Compute target value',
    ['start', 'worker_node', 'goal'],
    'goal'
  );

  assert.ok(plan.operator);
  assert.equal(plan.from, 'worker_node');

  const verdict = await sys1.evaluateMutation(deadlock, plan, 'Compute target value');
  assert.equal(typeof verdict.approved, 'boolean');
  assert.equal(typeof verdict.approveProb, 'number');
});
