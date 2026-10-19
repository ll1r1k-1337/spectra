// @ts-check
import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { scanWorkspace, persistArtifactsToDisk } from '../bin/spectra.js';
import { AcceptanceGateValidator, GOAL_PREDICATE_EXPR } from '../src/core/acceptance-gate.js';
import { DirectedGraph } from '../src/core/graph.js';
import { SpectraRuntime } from '../src/core/runtime.js';
import { GraphDeterminismVerifier } from '../src/core/verifier.js';
import { ToolRegistry } from '../src/core/tools.js';
import { StateOntologyManager } from '../src/core/ontology.js';
import { SpectraLogger } from '../src/cli/logger.js';

/** @returns {Promise<string>} */
async function tempDir() {
  return fs.mkdtemp(path.join(os.tmpdir(), 'spectra-unit-'));
}

test('scan_workspace -> patches -> disk cycle', async () => {
  const dir = await tempDir();
  try {
    await fs.writeFile(path.join(dir, 'app.js'), 'function add(a, b) { return a + b; }\n', 'utf8');
    await fs.writeFile(path.join(dir, 'index.html'), '<script src="app.js"></script>\n', 'utf8');

    const scanned = await scanWorkspace(dir);
    const app = scanned.workspaceFiles.find((f) => f.path === 'app.js');
    assert.ok(app);
    assert.match(String(app.content), /function add/);

    const out = await persistArtifactsToDisk(
      {
        artifacts: [],
        patches: [{ file: 'app.js', search: 'return a + b;', replace: 'return a + b + 1;' }]
      },
      dir
    );

    assert.deepEqual(out.syntaxErrors, []);
    assert.deepEqual(out.persistedFiles, ['app.js']);
    assert.equal(out.artifacts.length, 1);
    assert.match(out.artifacts[0].content, /return a \+ b \+ 1;/);
    assert.match(await fs.readFile(path.join(dir, 'app.js'), 'utf8'), /return a \+ b \+ 1;/);
    assert.deepEqual(out.patches, []);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});

test('acceptance gate rejects missing HTML references and broken output', () => {
  const validator = new AcceptanceGateValidator();

  const ok = validator.validateSync({
    artifacts: [
      { name: 'index.html', content: '<script src="main.js"></script>' },
      { name: 'main.js', content: 'console.log(1);' }
    ],
    persistedFiles: ['index.html', 'main.js'],
    syntaxErrors: []
  });
  assert.equal(ok.passed, true);

  const brokenRef = validator.validateSync({
    artifacts: [{ name: 'index.html', content: '<script src="missing.js"></script>' }],
    persistedFiles: ['index.html'],
    syntaxErrors: []
  });
  assert.equal(brokenRef.passed, false);
  assert.ok(brokenRef.errors.some((e) => e.includes('missing.js')));

  const existingRef = validator.validateSync({
    artifacts: [{ name: 'index.html', content: '<script src="main.js"></script>' }],
    persistedFiles: ['index.html'],
    workspaceFiles: [{ path: 'main.js', size: 10, ext: '.js', content: 'console.log(1);' }],
    syntaxErrors: []
  });
  assert.equal(existingRef.passed, true);

  const empty = validator.validateSync({ artifacts: [], persistedFiles: [], syntaxErrors: [] });
  assert.equal(empty.passed, false);

  const syntaxBroken = validator.validateSync({
    artifacts: [{ name: 'main.js', content: 'function {' }],
    persistedFiles: ['main.js'],
    syntaxErrors: ['main.js: Unexpected token']
  });
  assert.equal(syntaxBroken.passed, false);

  const predicate = new DirectedGraph('gate').compileGuardExpression(GOAL_PREDICATE_EXPR);
  assert.equal(
    predicate({
      artifacts: [
        { name: 'index.html', content: '<script src="main.js"></script>' },
        { name: 'main.js', content: 'console.log(1);' }
      ],
      persistedFiles: ['index.html', 'main.js'],
      syntaxErrors: []
    }),
    true
  );
  assert.equal(
    predicate({
      artifacts: [{ name: 'index.html', content: '<script src="missing.js"></script>' }],
      persistedFiles: ['index.html'],
      syntaxErrors: []
    }),
    false
  );
});

test('precedents are committed only on goal completion (deferred commit)', async () => {
  const dir = await tempDir();
  let precedentSaves = 0;
  let graphSaves = 0;

  const graph = new DirectedGraph('deferred_commit');
  graph.addNode({
    id: 'start',
    type: 'tool',
    reads: [],
    writes: ['x'],
    config: { sourceCode: '(s) => ({ ...s, x: 1 })' }
  });
  graph.addNode({ id: 'blocked', type: 'check', reads: ['x'], writes: [], config: { assertExpr: 'false' } });
  graph.addNode({ id: 'goal', type: 'tool', reads: [], writes: [], config: { sourceCode: '(s) => s' } });
  graph.startNode = 'start';
  graph.goalNodes.add('goal');
  graph.addEdge('start', 'blocked', () => true, 1, { expr: 'true' });
  graph.addEdge('blocked', 'goal', () => true, 1, { expr: 'true' });

  const store = {
    loadAllPrecedents: async () => [],
    loadOntologyKeys: async () => [],
    searchSimilarPrecedents: async () => [],
    loadRegisteredTools: async () => [],
    saveOntologyKeys: async () => {},
    savePrecedent: async () => {
      precedentSaves++;
    },
    saveGraphVersion: async () => {
      graphSaves++;
    }
  };
  const embedder = { computeIncidentVector: async () => new Array(384).fill(0) };
  const system1 = {
    config: { maxDrift: 1, minCoherence: 0, minApproveProb: 0 },
    evaluateMutation: async () => ({
      approved: false,
      driftRisk: 0,
      coherence: 1,
      approveProb: 0,
      rollbackAction: 'keep'
    })
  };
  const system2 = {
    config: { frontierModel: 'fake' },
    synthesizeRecoveryMutation: async () => ({
      operator: 'addEdge',
      from: 'start',
      to: 'goal',
      guardExpr: 'true',
      reasoning: 'test-only rejected mutation'
    })
  };

  try {
    const runtime = new SpectraRuntime({
      graph,
      toolRegistry: new ToolRegistry(),
      verifier: new GraphDeterminismVerifier(),
      ontology: new StateOntologyManager(),
      logger: new SpectraLogger({
        level: 'error',
        traceStateDiffs: false,
        traceGuards: false,
        jsonlFilePath: path.join(dir, 'session.jsonl')
      }),
      // @ts-expect-error test double
      precedentStore: store,
      // @ts-expect-error test double
      embedder,
      // @ts-expect-error test double
      system1Arbiter: system1,
      // @ts-expect-error test double
      system2Arbiter: system2,
      config: { hardStepLimit: 2 }
    });

    const result = await runtime.runSession({ goal: 'never reached' });
    assert.equal(result.goalReached, false);
    assert.equal(precedentSaves, 0);
    assert.equal(graphSaves, 0);
  } finally {
    await fs.rm(dir, { recursive: true, force: true });
  }
});
