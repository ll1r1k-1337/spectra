#!/usr/bin/env node
// @ts-check
import fs from 'node:fs/promises';
import path from 'node:path';
import { runInteractiveWizard } from '../src/cli/wizard.js';
import { runSessionGoal, buildInitialGraph } from '../src/cli/runner.js';
import { DirectedGraph, GraphDeterminismVerifier } from '../src/core/runtime.js';
import { logger } from '../src/utils/logger.js';

function printHelp() {
  console.log(`
Spectra Autonomous CLI: Continual Self-Learning Graph AI Runtime

Usage:
  spectra init                 Run interactive setup wizard
  spectra run "<goal>" [flags] Execute autonomous goal loop
  spectra verify [graphPath]   Verify graph determinism and Lyapunov bounds
  spectra --help, -h           Display this help message

Flags:
  --debug, -d, --verbose, -v   Enable structured debug logging output

Examples:
  spectra init
  spectra run "Scrape and summarize the latest tech news" --debug
  spectra verify ./graph.json -v
`);
}

/**
 * Handles `spectra verify [graphPath]` command.
 * @param {string} [graphPath]
 * @param {boolean} [isDebug]
 */
async function handleVerifyCommand(graphPath, isDebug = false) {
  if (isDebug) {
    logger.setLevel('debug');
  }
  logger.debug('Verifier', `Initiating verification${graphPath ? ` for graph file: ${graphPath}` : ' for initial graph'}`);
  const verifier = new GraphDeterminismVerifier();
  let graph;

  if (graphPath) {
    const fullPath = path.resolve(process.cwd(), graphPath);
    try {
      const content = await fs.readFile(fullPath, 'utf-8');
      const data = JSON.parse(content);
      graph = new DirectedGraph(data.id || 'imported_graph');
      if (Array.isArray(data.nodes)) {
        for (const n of data.nodes) {
          graph.addNode(n);
        }
      }
      if (Array.isArray(data.edges)) {
        for (const e of data.edges) {
          graph.addEdge(e.from, e.to, e.priority ?? 1, () => true, e.guardExpr || '() => true');
        }
      }
      graph.startNode = data.startNode || null;
      if (Array.isArray(data.goalNodes)) {
        for (const g of data.goalNodes) graph.goalNodes.add(g);
      }
    } catch (err) {
      logger.error('Verifier', `Failed to load graph file "${graphPath}": ${err instanceof Error ? err.message : String(err)}`);
      console.error(`❌ Failed to load graph file "${graphPath}":`, err instanceof Error ? err.message : err);
      process.exit(1);
    }
  } else {
    graph = buildInitialGraph();
  }

  const report = verifier.verify(graph);
  logger.debug('Verifier', 'Verification report computed', { report });
  console.log('\n============================================================');
  console.log(`🔍 [SPECTRA VERIFIER] Status: ${report.status}`);
  console.log('============================================================');
  console.log(`   • Omega Penalty:       ${report.omegaPenalty}`);
  console.log(`   • Priority Collisions: ${report.diagnostics.priorityCollisions}`);
  console.log(`   • Dead Nodes:          ${report.diagnostics.deadNodes.length > 0 ? report.diagnostics.deadNodes.join(', ') : 'none'}`);
  console.log(`   • Unsat Nodes:         ${report.diagnostics.unsatNodes.length > 0 ? report.diagnostics.unsatNodes.join(', ') : 'none'}`);
  console.log(`   • Trapping Cycles:     ${report.diagnostics.trappingCycles.length}`);
  console.log('============================================================\n');
}

async function main() {
  const rawArgs = process.argv.slice(2);
  const debugFlags = new Set(['--debug', '-d', '--verbose', '-v']);
  const isDebug = rawArgs.some((arg) => debugFlags.has(arg));
  const args = rawArgs.filter((arg) => !debugFlags.has(arg));
  const command = args[0];

  if (isDebug) {
    logger.setLevel('debug');
  }

  switch (command) {
    case 'init': {
      await runInteractiveWizard();
      break;
    }
    case 'run': {
      const goal = args.slice(1).join(' ').trim();
      if (!goal) {
        console.error('❌ Error: Goal argument is required. Example: spectra run "Analyze market trends"');
        process.exit(1);
      }
      await runSessionGoal(goal, undefined, { debug: isDebug, verbose: isDebug });
      break;
    }
    case 'verify': {
      const graphPath = args[1];
      await handleVerifyCommand(graphPath, isDebug);
      break;
    }
    case '--help':
    case '-h':
    case 'help':
    case undefined: {
      printHelp();
      break;
    }
    default: {
      console.error(`❌ Unknown command: "${command}"\n`);
      printHelp();
      process.exit(1);
    }
  }
}

main().catch((err) => {
  console.error('Unhandled CLI Error:', err);
  process.exit(1);
});
