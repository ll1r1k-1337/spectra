import test from 'node:test';
import assert from 'node:assert/strict';
import { DirectedGraph } from '../src/core/graph.js';
import { GraphDeterminismVerifier } from '../src/core/verifier.js';

test('DirectedGraph.metaInitialize applies Liskov subtyping and injects coercers', () => {
  const skillLib = [
    {
      id: 'step_1',
      reads: ['raw_input'],
      writes: ['count_str'],
      inputSchemas: { raw_input: { type: 'string' } },
      outputSchemas: { count_str: { type: 'string' } },
      expectedCost: 2,
      confidenceLcb: 0.9,
      similarityToIntent: 0.85
    },
    {
      id: 'step_2',
      reads: ['count_num'],
      writes: ['result'],
      inputSchemas: { count_num: { type: 'number', minimum: 0, maximum: 100 } },
      outputSchemas: { result: { type: 'string' } },
      expectedCost: 3,
      confidenceLcb: 0.95,
      similarityToIntent: 0.9
    }
  ];

  // count_str is 'string', but step_2 expects 'number'
  // When metaInitialize connects them or adapts them:
  const g = DirectedGraph.metaInitialize(
    new Array(384).fill(0.1),
    ['raw_input'],
    ['count_str'],
    skillLib,
    50,
    { raw_input: { type: 'string' } }
  );

  assert.equal(g.startNode, 'start');
  assert.ok(g.nodes.has('step_1'));
  assert.ok(g.goalNodes.has('goal'));
});

test('DirectedGraph.tryPushoutOrBifurcateMerge performs disjoint Pushout', () => {
  const base = new DirectedGraph('base');
  base.addNode({ id: 'start', type: 'tool', reads: [], writes: [] });
  base.addNode({ id: 'goal', type: 'tool', reads: [], writes: [] });
  base.addEdge('start', 'goal', 1);

  const branchA = {
    touchedNodes: ['node_a'],
    modifiedReads: ['x'],
    modifiedWrites: ['y'],
    applyTo(g) {
      g.addNode({ id: 'node_a', type: 'tool', reads: ['x'], writes: ['y'] });
    }
  };

  const branchB = {
    touchedNodes: ['node_b'],
    modifiedReads: ['z'],
    modifiedWrites: ['w'],
    applyTo(g) {
      g.addNode({ id: 'node_b', type: 'tool', reads: ['z'], writes: ['w'] });
    }
  };

  const res = DirectedGraph.tryPushoutOrBifurcateMerge(base, branchA, branchB);
  assert.equal(res.merged, true);
  assert.equal(res.mode, 'PUSHOUT');
  assert.ok(res.graph.nodes.has('node_a'));
  assert.ok(res.graph.nodes.has('node_b'));
});

test('DirectedGraph.foldBisimilarBranches collapses duplicate diamond branches', () => {
  const g = new DirectedGraph('diamond');
  g.addNode({ id: 'start', type: 'tool', reads: [], writes: [] });
  g.addNode({ id: 'worker_1', type: 'tool', reads: ['a'], writes: ['b'], config: { toolId: 'compute' } });
  g.addNode({ id: 'worker_2', type: 'tool', reads: ['a'], writes: ['b'], config: { toolId: 'compute' } });
  g.addNode({ id: 'goal', type: 'check', reads: ['b'], writes: [] });

  g.startNode = 'start';
  g.goalNodes.add('goal');
  g.addEdge('start', 'worker_1', 1);
  g.addEdge('start', 'worker_2', 2);
  g.addEdge('worker_1', 'goal', 1);
  g.addEdge('worker_2', 'goal', 1);

  assert.equal(g.nodes.size, 4);
  g.foldBisimilarBranches();
  assert.equal(g.nodes.size, 3); // worker_1 and worker_2 folded
});

test('DirectedGraph.splitOverloadedLlmNode performs entropy-driven mitosis', () => {
  const g = new DirectedGraph('mitosis_test');
  g.addNode({
    id: 'heavy_llm',
    type: 'llm',
    reads: ['context'],
    writes: ['summary', 'sentiment']
  });
  g.addNode({ id: 'start', type: 'tool', reads: [], writes: ['context'] });
  g.addNode({ id: 'goal', type: 'tool', reads: ['summary', 'sentiment'], writes: [] });
  g.startNode = 'start';
  g.goalNodes.add('goal');
  g.addEdge('start', 'heavy_llm', 1);
  g.addEdge('heavy_llm', 'goal', 1);

  const splitOk = g.splitOverloadedLlmNode('heavy_llm', ['summary'], ['sentiment']);
  assert.equal(splitOk, true);
  assert.ok(!g.nodes.has('heavy_llm'));
  assert.ok(g.nodes.has('heavy_llm_mitosis_1'));
  assert.ok(g.nodes.has('heavy_llm_mitosis_2'));
});

test('GraphDeterminismVerifier detects priority collisions and invalid SCC trapping cycles', () => {
  const verifier = new GraphDeterminismVerifier();

  // Graph with priority collision
  const g1 = new DirectedGraph('collision');
  g1.addNode({ id: 'start', type: 'tool', reads: [], writes: [] });
  g1.addNode({ id: 'a', type: 'tool', reads: [], writes: [] });
  g1.addNode({ id: 'b', type: 'tool', reads: [], writes: [] });
  g1.startNode = 'start';
  g1.goalNodes.add('a');
  g1.goalNodes.add('b');
  g1.addEdge('start', 'a', 1);
  g1.addEdge('start', 'b', 1); // Collision: priority 1 twice

  const report1 = verifier.verify(g1);
  assert.equal(report1.status, 'HARD_REJECT');
  assert.equal(report1.diagnostics.priorityCollisions, 1);

  // Graph with trapping cycle
  const g2 = new DirectedGraph('trap');
  g2.addNode({ id: 'start', type: 'tool', reads: [], writes: [] });
  g2.addNode({ id: 'c1', type: 'tool', reads: [], writes: [] });
  g2.addNode({ id: 'c2', type: 'tool', reads: [], writes: [] });
  g2.addNode({ id: 'goal', type: 'tool', reads: [], writes: [] });
  g2.startNode = 'start';
  g2.goalNodes.add('goal');
  g2.addEdge('start', 'c1', 1);
  g2.addEdge('c1', 'c2', 1);
  g2.addEdge('c2', 'c1', 1); // Cycle c1 <-> c2 with no escape to goal

  const report2 = verifier.verify(g2);
  assert.equal(report2.status, 'HARD_REJECT');
  assert.ok(report2.diagnostics.trappingCycles.length > 0);
});

test('GraphDeterminismVerifier Invariant 2a detects transitive interval contradictions', () => {
  const verifier = new GraphDeterminismVerifier();
  const g = new DirectedGraph('interval_test');
  g.addNode({ id: 'start', type: 'tool', reads: [], writes: [] });
  g.addNode({ id: 'intermediate', type: 'tool', reads: [], writes: [] }); // Does not modify x
  g.addNode({ id: 'goal', type: 'tool', reads: [], writes: [] });
  g.startNode = 'start';
  g.goalNodes.add('goal');

  // Edge 1 requires x in [10, 20]
  g.addEdge('start', 'intermediate', 1, () => true, { expr: 'x in [10, 20]', intervals: { x: [10, 20] } });
  // Edge 2 requires x in [0, 5] -> Contradicts [10, 20] because intermediate doesn't write x!
  g.addEdge('intermediate', 'goal', 1, () => true, { expr: 'x in [0, 5]', intervals: { x: [0, 5] } });

  const report = verifier.verify(g);
  assert.equal(report.status, 'HARD_REJECT');
  assert.ok(report.diagnostics.unsatNodes.includes('intermediate'));
});
