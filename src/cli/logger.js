// @ts-check
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';

/**
 * @typedef {import('../../types/index.js').ISpectraLogger} ISpectraLogger
 * @typedef {import('../../types/index.js').LogLevel} LogLevel
 * @typedef {import('../../types/index.js').StateObject} StateObject
 * @typedef {import('../../types/index.js').StateDiffEntry} StateDiffEntry
 * @typedef {import('../../types/index.js').GraphNodeSpec} GraphNodeSpec
 * @typedef {import('../../types/index.js').DeadlockIncident} DeadlockIncident
 * @typedef {import('../../types/index.js').System1Verdict} System1Verdict
 * @typedef {import('../../types/index.js').VerificationReport} VerificationReport
 * @typedef {import('../../types/index.js').IDirectedGraph} IDirectedGraph
 */

const ANSI = {
  reset: '\x1b[0m',
  bold: '\x1b[1m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  magenta: '\x1b[35m',
  blue: '\x1b[34m',
  gray: '\x1b[90m'
};

/** @type {Record<LogLevel, number>} */
const LEVEL_WEIGHT = { debug: 10, info: 20, warn: 30, error: 40 };

/**
 * @implements {ISpectraLogger}
 */
export class SpectraLogger {
  /**
   * @param {Partial<import('../../types/index.js').SpectraConfig['logging']>} [opts]
   */
  constructor(opts = {}) {
    this.sessionId = `ses_${crypto.randomBytes(4).toString('hex')}`;
    /** @type {LogLevel} */
    this.level = opts.level || 'debug';
    this.traceStateDiffs = opts.traceStateDiffs ?? true;
    this.traceGuards = opts.traceGuards ?? true;
    this.jsonlFilePath = opts.jsonlFilePath || '.spectra/logs/session.jsonl';

    try {
      const dir = path.dirname(path.resolve(process.cwd(), this.jsonlFilePath));
      fs.mkdirSync(dir, { recursive: true });
    } catch {
      // Игнорируем ошибку создания директории в read-only окружениях
    }
  }

  /**
   * @param {string} eventType
   * @param {Record<string, any>} payload
   */
  _writeJsonl(eventType, payload) {
    try {
      const record = JSON.stringify({
        ts: new Date().toISOString(),
        sessionId: this.sessionId,
        eventType,
        ...payload
      });
      fs.appendFileSync(path.resolve(process.cwd(), this.jsonlFilePath), record + '\n', 'utf-8');
    } catch {
      // Non-blocking audit log
    }
  }

  /**
   * @param {LogLevel} msgLevel
   */
  _shouldLog(msgLevel) {
    return LEVEL_WEIGHT[msgLevel] >= LEVEL_WEIGHT[this.level];
  }

  _ts() {
    return new Date().toISOString().slice(11, 23);
  }

  /**
   * @param {string} title
   * @param {string} [subtitle]
   */
  banner(title, subtitle = '') {
    const line = '═'.repeat(74);
    console.log(`\n${ANSI.cyan}${ANSI.bold}╔${line}╗${ANSI.reset}`);
    console.log(`${ANSI.cyan}${ANSI.bold}║ ⚡ ${title.padEnd(70)}║${ANSI.reset}`);
    if (subtitle) {
      console.log(`${ANSI.cyan}║    ${subtitle.slice(0, 68).padEnd(70)}║${ANSI.reset}`);
    }
    console.log(`${ANSI.cyan}${ANSI.bold}╚${line}╝${ANSI.reset}`);
    this._writeJsonl('BANNER', { title, subtitle });
  }

  /**
   * @param {'FAST_LOOP' | 'DEADLOCK' | 'SLOW_LOOP' | 'SYSTEM_1' | 'SYSTEM_2' | 'VERIFIER' | 'ROLLBACK' | 'MEMORY' | 'SLEEP'} phase
   * @param {string} title
   */
  section(phase, title) {
    const colorMap = {
      FAST_LOOP: ANSI.green,
      DEADLOCK: ANSI.red,
      SLOW_LOOP: ANSI.yellow,
      SYSTEM_1: ANSI.cyan,
      SYSTEM_2: ANSI.magenta,
      VERIFIER: ANSI.blue,
      ROLLBACK: ANSI.yellow,
      MEMORY: ANSI.cyan,
      SLEEP: ANSI.magenta
    };
    const col = colorMap[phase] || ANSI.cyan;
    console.log(
      `\n${col}${ANSI.bold}┌── [${phase}] ${title} ${'─'.repeat(Math.max(2, 58 - phase.length - title.length))}${ANSI.reset}`
    );
    this._writeJsonl('SECTION', { phase, title });
  }

  /**
   * @param {string} scope
   * @param {string} message
   * @param {Record<string, any>} [meta]
   */
  info(scope, message, meta) {
    if (!this._shouldLog('info')) return;
    const metaStr = meta ? ` ${ANSI.gray}${JSON.stringify(meta)}${ANSI.reset}` : '';
    console.log(
      `${ANSI.gray}[${this._ts()}]${ANSI.reset} ${ANSI.cyan}INFO ${ANSI.reset} ${ANSI.bold}[${scope}]${ANSI.reset} ${message}${metaStr}`
    );
    this._writeJsonl('INFO', { scope, message, meta });
  }

  /**
   * @param {string} scope
   * @param {string} message
   * @param {Record<string, any>} [meta]
   */
  debug(scope, message, meta) {
    if (!this._shouldLog('debug')) return;
    const metaStr = meta ? ` ${ANSI.gray}${JSON.stringify(meta)}${ANSI.reset}` : '';
    console.log(`${ANSI.gray}[${this._ts()}] DEBUG [${scope}] ${message}${metaStr}${ANSI.reset}`);
    this._writeJsonl('DEBUG', { scope, message, meta });
  }

  /**
   * @param {string} scope
   * @param {string} message
   * @param {Record<string, any>} [meta]
   */
  warn(scope, message, meta) {
    if (!this._shouldLog('warn')) return;
    const metaStr = meta ? ` ${ANSI.yellow}${JSON.stringify(meta)}${ANSI.reset}` : '';
    console.log(
      `${ANSI.gray}[${this._ts()}]${ANSI.reset} ${ANSI.yellow}${ANSI.bold}WARN  [${scope}]${ANSI.reset} ${message}${metaStr}`
    );
    this._writeJsonl('WARN', { scope, message, meta });
  }

  /**
   * @param {string} scope
   * @param {string} message
   * @param {Record<string, any>} [meta]
   */
  error(scope, message, meta) {
    const metaStr = meta ? ` ${ANSI.red}${JSON.stringify(meta)}${ANSI.reset}` : '';
    console.log(
      `${ANSI.gray}[${this._ts()}]${ANSI.reset} ${ANSI.red}${ANSI.bold}ERROR [${scope}]${ANSI.reset} ${message}${metaStr}`
    );
    this._writeJsonl('ERROR', { scope, message, meta });
  }

  /**
   * @param {StateObject} before
   * @param {StateObject} after
   * @returns {StateDiffEntry[]}
   */
  computeStateDiff(before, after) {
    /** @type {StateDiffEntry[]} */
    const diffs = [];
    const allKeys = new Set([...Object.keys(before || {}), ...Object.keys(after || {})]);
    for (const key of allKeys) {
      const bVal = before?.[key];
      const aVal = after?.[key];
      if (!(key in (before || {}))) {
        diffs.push({ key, change: 'ADDED', before: undefined, after: aVal });
      } else if (!(key in (after || {}))) {
        diffs.push({ key, change: 'REMOVED', before: bVal, after: undefined });
      } else if (JSON.stringify(bVal) !== JSON.stringify(aVal)) {
        diffs.push({ key, change: 'MODIFIED', before: bVal, after: aVal });
      }
    }
    return diffs;
  }

  /**
   * @param {{ step: number; maxSteps: number; node: GraphNodeSpec; durationMs: number; stateBefore: StateObject; stateAfter: StateObject }} params
   */
  logStepExecution({ step, maxSteps, node, durationMs, stateBefore, stateAfter }) {
    const diffs = this.computeStateDiff(stateBefore, stateAfter);
    console.log(
      `\n${ANSI.green}${ANSI.bold}▶ [FAST LOOP | Шаг ${step + 1}/${maxSteps}]${ANSI.reset} ` +
        `Узел: ${ANSI.bold}"${node.id}"${ANSI.reset} (type=${ANSI.cyan}${node.type}${ANSI.reset}, ` +
        `reads=[${node.reads.join(', ')}], writes=[${node.writes.join(', ')}], ${durationMs.toFixed(1)} мс)`
    );

    if (this.traceStateDiffs) {
      if (diffs.length === 0) {
        console.log(`  ${ANSI.gray}↳ Δ State: (без изменений)${ANSI.reset}`);
      } else {
        for (const d of diffs) {
          if (d.change === 'ADDED') {
            console.log(`  ${ANSI.green}↳ Δ State [+ADDED]    ${d.key}: ${JSON.stringify(d.after)}${ANSI.reset}`);
          } else if (d.change === 'MODIFIED') {
            console.log(
              `  ${ANSI.yellow}↳ Δ State [~MODIFIED] ${d.key}: ${JSON.stringify(d.before)} ──► ${JSON.stringify(d.after)}${ANSI.reset}`
            );
          } else {
            console.log(`  ${ANSI.red}↳ Δ State [-REMOVED]  ${d.key} (было: ${JSON.stringify(d.before)})${ANSI.reset}`);
          }
        }
      }
    }
    this._writeJsonl('FAST_LOOP_STEP', { step, nodeId: node.id, nodeType: node.type, durationMs, diffs });
  }

  /**
   * @param {string} nodeId
   * @param {Array<{ to: string; priority: number; guardExpr: string; matched: boolean }>} evaluations
   */
  logGuardEvaluation(nodeId, evaluations) {
    if (!this.traceGuards) return;
    if (evaluations.length === 0) {
      console.log(`  ${ANSI.red}↳ Маршрутизация из "${nodeId}": нет исходящих ребер!${ANSI.reset}`);
      return;
    }
    let winnerLogged = false;
    for (const ev of evaluations) {
      const isWinner = ev.matched && !winnerLogged;
      if (isWinner) winnerLogged = true;
      const badge = isWinner
        ? `${ANSI.green}${ANSI.bold}[ВЫБРАНО]${ANSI.reset}`
        : ev.matched
          ? `${ANSI.gray}[ИСТИНА, НО НИЖЕ ПРИОРИТЕТ]${ANSI.reset}`
          : `${ANSI.gray}[ЛОЖЬ]${ANSI.reset}`;
      console.log(
        `  ${ANSI.gray}↳ Guard (${nodeId} ──► ${ev.to}, priority=${ev.priority}): ${ev.guardExpr} => ${badge}`
      );
    }
    this._writeJsonl('GUARD_EVAL', { nodeId, evaluations });
  }

  /**
   * @param {DeadlockIncident} deadlock
   * @param {Map<string, number>} markovBlanket
   * @param {string} stateHash
   */
  logDeadlock(deadlock, markovBlanket, stateHash) {
    this.section('DEADLOCK', `Обнаружен тупик ${deadlock.type} на узле "${deadlock.nodeId}" (шаг ${deadlock.step + 1})`);
    console.log(`  ${ANSI.red}• Тип сбоя:${ANSI.reset}          ${ANSI.bold}${deadlock.type}${ANSI.reset}`);
    console.log(`  ${ANSI.red}• Узел сбоя:${ANSI.reset}         "${deadlock.nodeId}" (предшественник: "${deadlock.lastValidNodeId || 'none'}")`);
    if (deadlock.error) {
      console.log(`  ${ANSI.red}• Текст исключения:${ANSI.reset}  ${deadlock.error}`);
    }
    console.log(`  ${ANSI.red}• SHA-256(v, s):${ANSI.reset}     ${stateHash.slice(0, 16)}...`);

    const blanketEntries = Array.from(markovBlanket.entries())
      .map(([id, score]) => `${id}(PPR=${score.toFixed(3)})`)
      .join(', ');
    console.log(`  ${ANSI.cyan}• Одеяло Маркова MB:${ANSI.reset} [${blanketEntries}]`);
    console.log(`  ${ANSI.gray}• Слепок State:${ANSI.reset}      ${JSON.stringify(deadlock.stateSnapshot)}`);
    this._writeJsonl('DEADLOCK', {
      deadlock,
      stateHash,
      markovBlanket: Object.fromEntries(markovBlanket)
    });
  }

  /**
   * @param {System1Verdict} verdict
   * @param {{ maxDrift: number; minCoherence: number; minApproveProb: number }} thresholds
   */
  logSystem1Verdict(verdict, thresholds) {
    this.section('SYSTEM_1', 'Вердикт локального ONNX-арбитра @receptron/laya (systemOne)');
    const statusBadge = verdict.approved
      ? `${ANSI.green}${ANSI.bold}APPROVED (ПРОПУСК В ВЕРИФИКАТОР)${ANSI.reset}`
      : `${ANSI.red}${ANSI.bold}REJECTED (ОТКАТ: ${verdict.rollbackAction})${ANSI.reset}`;

    console.log(`  • Статус гейта:       ${statusBadge}`);
    console.log(`  • P(approve) [noul]:  ${verdict.approveProb.toFixed(3)} (порог >= ${thresholds.minApproveProb})`);
    console.log(`  • Риск дрейфа [score]:${verdict.driftRisk.toFixed(3)} (порог <= ${thresholds.maxDrift})`);
    console.log(`  • Связность [score]:  ${verdict.coherence.toFixed(3)} (порог >= ${thresholds.minCoherence})`);
    console.log(`  • Действие [choice]:  ${verdict.rollbackAction} (${verdict.latencyMs ?? 0} мс)`);
    this._writeJsonl('SYSTEM1_VERDICT', { verdict, thresholds });
  }

  /**
   * @param {VerificationReport} report
   * @param {string} graphHash
   */
  logVerifierReport(report, graphHash) {
    const d = report.diagnostics;
    const badge =
      report.status === 'PASS'
        ? `${ANSI.green}${ANSI.bold}PASS${ANSI.reset}`
        : `${ANSI.red}${ANSI.bold}HARD_REJECT${ANSI.reset}`;

    console.log(
      `  ${ANSI.blue}${ANSI.bold}🛡️  [Верификатор детерминизма V(G)]${ANSI.reset} Статус: ${badge} | ` +
        `Штраф Ω(G)=${report.omegaPenalty} | Hash=${graphHash.slice(0, 12)}`
    );
    console.log(
      `     ↳ Вершин |V|=${d.vertexCount}, Ребер |E|=${d.edgeCount}, Плотность=${d.density.toFixed(2)}, ` +
        `Коллизий приоритетов=${d.priorityCollisions}`
    );
    if (d.deadNodes.length > 0 || d.unsatNodes.length > 0 || d.trappingCycles.length > 0) {
      console.log(
        `     ${ANSI.red}↳ Нарушения: deadNodes=[${d.deadNodes.join(',')}], ` +
          `unsatNodes=[${d.unsatNodes.join(',')}], trappingCycles=${JSON.stringify(d.trappingCycles)}${ANSI.reset}`
      );
    }
    this._writeJsonl('VERIFIER_REPORT', { report, graphHash });
  }

  /**
   * @param {IDirectedGraph} graph
   * @param {string | null} [activeNodeId]
   */
  logTopologyAscii(graph, activeNodeId = null) {
    console.log(`\n  ${ANSI.cyan}${ANSI.bold}🗺️  Топология графа "${graph.id}" (узлов: ${graph.nodes.size}):${ANSI.reset}`);
    for (const [id, node] of graph.nodes.entries()) {
      const isStart = graph.startNode === id ? ' [START]' : '';
      const isGoal = graph.goalNodes.has(id) ? ' [GOAL]' : '';
      const cursor = activeNodeId === id ? `${ANSI.green}${ANSI.bold}👉 ` : '   ';
      const outs = (graph.edges.get(id) || [])
        .map((e) => `--(p=${e.priority}, ${e.guardExpr})--> [${e.to}]`)
        .join(' | ');
      console.log(
        `  ${cursor}[${id}:${node.type}]${isStart}${isGoal}${ANSI.reset} ` +
          `${ANSI.gray}(R:[${node.reads.join(',')}] -> W:[${node.writes.join(',')}])${ANSI.reset} ${outs}`
      );
    }
  }
}
