import test from 'node:test';
import assert from 'node:assert/strict';
import { Logger, logger, LOG_LEVELS, LEVEL_NAMES } from '../src/utils/logger.js';

test('Logger: default level and programmatic setLevel / getLevel', () => {
  const l = new Logger('info');
  assert.equal(l.getLevel(), 'info');

  l.setLevel('debug');
  assert.equal(l.getLevel(), 'debug');

  l.setLevel('warn');
  assert.equal(l.getLevel(), 'warn');

  l.setLevel('error');
  assert.equal(l.getLevel(), 'error');

  l.setLevel('silent');
  assert.equal(l.getLevel(), 'silent');

  // Numeric levels
  l.setLevel(0);
  assert.equal(l.getLevel(), 'debug');
  l.setLevel(3);
  assert.equal(l.getLevel(), 'error');
});

test('Logger: level filtering suppresses lower level messages', () => {
  const l = new Logger('warn');
  /** @type {string[]} */
  const capturedStdout = [];
  /** @type {string[]} */
  const capturedStderr = [];

  const origStdout = process.stdout.write.bind(process.stdout);
  const origStderr = process.stderr.write.bind(process.stderr);

  // @ts-ignore
  process.stdout.write = (chunk) => {
    capturedStdout.push(chunk.toString());
    return true;
  };
  // @ts-ignore
  process.stderr.write = (chunk) => {
    capturedStderr.push(chunk.toString());
    return true;
  };

  try {
    l.debug('TestTag', 'debug message');
    l.info('TestTag', 'info message');
    l.warn('TestTag', 'warn message');
    l.error('TestTag', 'error message');
  } finally {
    process.stdout.write = origStdout;
    process.stderr.write = origStderr;
  }

  assert.equal(capturedStdout.length, 0); // debug & info suppressed
  assert.equal(capturedStderr.length, 2); // warn & error captured
  assert.ok(capturedStderr[0].includes('[WARN]'));
  assert.ok(capturedStderr[0].includes('[TestTag]'));
  assert.ok(capturedStderr[0].includes('warn message'));
  assert.ok(capturedStderr[1].includes('[ERROR]'));
  assert.ok(capturedStderr[1].includes('error message'));
});

test('Logger: child logger binds and prefixes tags correctly', () => {
  const l = new Logger('debug');
  /** @type {string[]} */
  const captured = [];
  const origStdout = process.stdout.write.bind(process.stdout);
  // @ts-ignore
  process.stdout.write = (chunk) => {
    captured.push(chunk.toString());
    return true;
  };

  try {
    const child = l.child('FastLoop');
    child.info('Loop step executed', { step: 1 });
    const subChild = child.child('SPSAStep');
    subChild.debug('Perturbation applied', { delta: 0.05 });
  } finally {
    process.stdout.write = origStdout;
  }

  assert.equal(captured.length, 2);
  assert.ok(captured[0].includes('[INFO]'));
  assert.ok(captured[0].includes('[FastLoop]'));
  assert.ok(captured[0].includes('Loop step executed'));
  assert.ok(captured[0].includes('step: 1'));

  assert.ok(captured[1].includes('[DEBUG]'));
  assert.ok(captured[1].includes('[FastLoop:SPSAStep]'));
  assert.ok(captured[1].includes('Perturbation applied'));
  assert.ok(captured[1].includes('delta: 0.05'));
});

test('Logger: silent level suppresses all output', () => {
  const l = new Logger('silent');
  /** @type {string[]} */
  const captured = [];
  const origStdout = process.stdout.write.bind(process.stdout);
  const origStderr = process.stderr.write.bind(process.stderr);
  // @ts-ignore
  process.stdout.write = (chunk) => { captured.push(chunk.toString()); return true; };
  // @ts-ignore
  process.stderr.write = (chunk) => { captured.push(chunk.toString()); return true; };

  try {
    l.debug('Tag', 'msg');
    l.info('Tag', 'msg');
    l.warn('Tag', 'msg');
    l.error('Tag', 'msg');
  } finally {
    process.stdout.write = origStdout;
    process.stderr.write = origStderr;
  }

  assert.equal(captured.length, 0);
});

test('Logger: singleton instance exists and exports valid methods', () => {
  assert.equal(typeof logger.debug, 'function');
  assert.equal(typeof logger.info, 'function');
  assert.equal(typeof logger.warn, 'function');
  assert.equal(typeof logger.error, 'function');
  assert.equal(typeof logger.child, 'function');
  assert.equal(typeof logger.setLevel, 'function');
  assert.equal(typeof logger.getLevel, 'function');
});
