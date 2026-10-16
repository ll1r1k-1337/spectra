import test from 'node:test';
import assert from 'node:assert/strict';
import {
  DirectedGraph,
  GraphDeterminismVerifier,
  ToolRegistry,
  StateOntologyManager,
  SpectraRuntime
} from '../src/core/runtime.js';

test('DirectedGraph builds nodes, edges and computes structural hash', () => {
  const g = new DirectedGraph('test_graph');
  g.addNode({ id: 'start', type: 'tool', reads: [], writes: ['x'] });
  g.addNode({ id: 'goal', type: 'check', reads: ['x'], writes: [] });
  g.addEdge('start', 'goal', () => true, 1);
  g.startNode = 'start';
  g.goalNodes.add('goal');

  assert.equal(g.nodes.size, 2);
  assert.equal(g.edges.get('start')?.length, 1);
  const hash = g.computeStructuralHash();
  assert.equal(typeof hash, 'string');
  assert.equal(hash.length, 64);
});

test('GraphDeterminismVerifier accepts deterministic connected graph', () => {
  const verifier = new GraphDeterminismVerifier();
  const g = new DirectedGraph('valid_graph');
  g.addNode({ id: 'start', type: 'tool', reads: [], writes: ['x'] });
  g.addNode({ id: 'goal', type: 'check', reads: ['x'], writes: [] });
  g.addEdge('start', 'goal', (s) => Boolean(s.x), 1);
  g.startNode = 'start';
  g.goalNodes.add('goal');

  const report = verifier.verify(g);
  assert.equal(report.status, 'PASS');
  assert.equal(report.diagnostics.deadNodes.length, 0);
  assert.equal(report.diagnostics.priorityCollisions, 0);
});

test('ToolRegistry executes in node:vm sandbox and tests CEGIS patching', () => {
  const registry = new ToolRegistry();
  registry.register({
    id: 'add_one',
    reads: ['val'],
    writes: ['val'],
    sourceCode: 'function(input) { return { ...input, val: (input.val || 0) + 1 }; }'
  });

  const out = registry.executeSandboxed(registry.tools.get('add_one').sourceCode, { val: 5 });
  assert.equal(out.val, 6);

  // CEGIS patch
  const patchRes = registry.patchToolWithCEGIS(
    'add_one',
    'function(input) { return { ...input, val: (input.val || 0) + 1 }; }',
    { val: 10 }
  );
  assert.equal(patchRes.accepted, true);
});

test('StateOntologyManager path compression and alias resolution', () => {
  const ontology = new StateOntologyManager();
  ontology.keys.set('user_id', { jsonSchema: {}, status: 'CANONICAL' });
  ontology.keys.set('uid', { jsonSchema: {}, status: 'CANONICAL', canonicalAlias: 'user_id' });
  ontology.keys.set('account_uid', { jsonSchema: {}, status: 'CANONICAL', canonicalAlias: 'uid' });

  assert.equal(ontology.findCanonical('account_uid'), 'user_id');
});

test('SpectraRuntime resolves deadlock via Slow Loop and reaches goal', async () => {
  const g = new DirectedGraph('runtime_graph');
  g.addNode({
    id: 'start',
    type: 'tool',
    reads: [],
    writes: ['initialized'],
    config: {
      handler: async (s) => ({ ...s, initialized: true })
    }
  });
  g.addNode({
    id: 'deadlock_node',
    type: 'tool',
    reads: ['initialized'],
    writes: [],
    config: {
      handler: async (s) => s
    }
  });
  g.addNode({
    id: 'goal',
    type: 'check',
    reads: ['done'],
    writes: [],
    config: {
      assertFn: (s) => Boolean(s.done)
    }
  });

  g.startNode = 'start';
  g.goalNodes.add('goal');
  g.addEdge('start', 'deadlock_node', () => true, 1);
  // No outgoing edge from deadlock_node -> will trigger NO_VALID_TRANSITION

  const runtime = new SpectraRuntime({
    graph: g,
    system1Evaluator: async () => ({
      driftRisk: 0.1,
      coherence: 0.9,
      approveProb: 0.95,
      approved: true,
      rollbackAction: 'keep'
    }),
    system2Synthesizer: async (deadlock) => ({
      operator: 'addNodeAndBridge',
      from: deadlock.nodeId,
      to: 'goal',
      node: {
        id: 'fix_node',
        type: 'tool',
        reads: [],
        writes: ['done', 'goalCompleted'],
        config: {
          handler: async (s) => ({ ...s, done: true, goalCompleted: true })
        }
      }
    })
  });

  const res = await runtime.runSession({ done: false, goalCompleted: false });
  assert.equal(res.goalReached, true);
  assert.equal(res.appliedMutations.length, 1);
});

test('Goal Contract Invariant: Premature goal node trigger routes to INVALID_GOAL_STATE and Slow Loop recovery', async () => {
  const g = new DirectedGraph();
  g.addNode({
    id: 'start',
    type: 'tool',
    reads: [],
    writes: [],
    config: { handler: async (s) => s }
  });
  g.addNode({
    id: 'goal',
    type: 'tool',
    reads: ['goalCompleted'],
    writes: [],
    config: { handler: async (s) => s }
  });

  g.startNode = 'start';
  g.goalNodes.add('goal');
  // Premature unconditional edge directly to goal node without goalCompleted
  g.addEdge('start', 'goal', () => true, 1);

  let synthesized = false;
  const runtime = new SpectraRuntime({
    graph: g,
    system1Evaluator: async () => ({
      driftRisk: 0.1,
      coherence: 0.9,
      approveProb: 0.95,
      approved: true,
      rollbackAction: 'keep'
    }),
    system2Synthesizer: async (deadlock) => {
      assert.equal(deadlock.type, 'INVALID_GOAL_STATE');
      synthesized = true;
      return {
        operator: 'addNodeAndBridge',
        from: 'start',
        to: 'goal',
        node: {
          id: 'real_work_node',
          type: 'tool',
          reads: [],
          writes: ['goalCompleted'],
          config: {
            handler: async (s) => ({ ...s, goalCompleted: true })
          }
        }
      };
    }
  });

  const res = await runtime.runSession({ goalCompleted: false });
  assert.equal(synthesized, true);
  assert.equal(res.goalReached, true);
  assert.equal(res.finalState.goalCompleted, true);
});

test('ToolRegistry: Sandboxed tool code has access to fs and path', async () => {
  const tools = new ToolRegistry();
  const res = tools.executeSandboxed(
    `((state) => {
      const p = path.join('foo', 'bar');
      const hasFs = typeof fs.writeFileSync === 'function';
      return { ...state, p, hasFs, goalCompleted: true };
    })`,
    { goalCompleted: false }
  );

  assert.equal(res.p, 'foo/bar');
  assert.equal(res.hasFs, true);
  assert.equal(res.goalCompleted, true);
});
