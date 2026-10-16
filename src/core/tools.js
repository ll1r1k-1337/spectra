import vm from 'node:vm';
import fs from 'node:fs';
import path from 'node:path';
import { logger } from '../utils/logger.js';

/**
 * @typedef {import('#types').StateObject} StateObject
 * @typedef {import('#types').IToolRegistry} IToolRegistry
 */

/**
 * ToolRegistry manages registered executable tools with isolated node:vm sandbox execution,
 * CEGIS automated patching against counterexamples and regression pools,
 * and automated verification of Saga compensators.
 *
 * @implements {IToolRegistry}
 */
export class ToolRegistry {
  constructor() {
    /** @type {Map<string, any>} */
    this.tools = new Map();
  }

  /**
   * Register a tool specification.
   *
   * @param {{
   *   id: string,
   *   reads: string[],
   *   writes: string[],
   *   sourceCode: string,
   *   version?: number,
   *   hasIrreversibleBarrier?: boolean,
   *   regressionPool?: Array<{ input: StateObject, expectedOutput: StateObject }>
   * }} toolSpec
   */
  register(toolSpec) {
    this.tools.set(toolSpec.id, {
      ...toolSpec,
      version: toolSpec.version || 1,
      hasIrreversibleBarrier: Boolean(toolSpec.hasIrreversibleBarrier),
      regressionPool: toolSpec.regressionPool || []
    });
  }

  /**
   * Execute code string in an isolated node:vm sandbox.
   *
   * @param {string} codeStr
   * @param {StateObject} stateInput
   * @param {number} [timeoutMs=250]
   * @returns {StateObject}
   */
  executeSandboxed(codeStr, stateInput, timeoutMs = 250) {
    const startTime = performance.now();
    logger.debug('ToolVM', `Executing sandboxed script (timeout: ${timeoutMs}ms)`);
    try {
      /** @type {{ input: StateObject, output: StateObject | null, Math: Math, Date: DateConstructor, JSON: JSON, fs: typeof fs, path: typeof path, console: typeof console }} */
      const sandbox = {
        input: structuredClone(stateInput),
        output: null,
        Math,
        Date,
        JSON,
        fs,
        path,
        console
      };
      const context = vm.createContext(sandbox);
      const script = new vm.Script(`output = (${codeStr})(input);`);
      script.runInContext(context, { timeout: timeoutMs });
      const durationMs = performance.now() - startTime;
      logger.debug('ToolVM', `Sandboxed script completed in ${durationMs.toFixed(2)}ms`);
      return sandbox.output ?? {};
    } catch (err) {
      const durationMs = performance.now() - startTime;
      logger.error('ToolVM', `Sandboxed script execution failed after ${durationMs.toFixed(2)}ms: ${err instanceof Error ? err.message : String(err)}`, {
        durationMs,
        error: err
      });
      throw err;
    }
  }

  /**
   * Definition 2: Verify whether candidate compensator code restores original state
   * within tolerance epsilonSaga on regression pool.
   *
   * @param {string} toolId
   * @param {string} compensatorCodeStr
   * @param {number} [epsilonSaga=1e-4]
   * @returns {{ verified: boolean, residualError?: number, hasBarrier: boolean }}
   */
  verifyAndAttachSagaCompensator(toolId, compensatorCodeStr, epsilonSaga = 1e-4) {
    const tool = this.tools.get(toolId);
    if (!tool) throw new Error(`Tool ${toolId} not found`);

    logger.info('ToolVM', `Verifying Saga compensator for tool "${toolId}"`);
    let totalResidual = 0;
    const pool = tool.regressionPool.length > 0 ? tool.regressionPool : [{ input: {}, expectedOutput: {} }];

    for (const tc of pool) {
      try {
        const inputState = tc.input ?? tc;
        const forwardState = this.executeSandboxed(tool.sourceCode, inputState);
        const restoredState = this.executeSandboxed(compensatorCodeStr, forwardState);
        const keys = new Set([...(tool.reads || []), ...(tool.writes || [])]);
        for (const k of keys) {
          if (JSON.stringify(inputState[k]) !== JSON.stringify(restoredState?.[k])) {
            totalResidual += 1.0;
          }
        }
      } catch {
        tool.hasIrreversibleBarrier = true;
        logger.warn('ToolVM', `Saga compensator execution threw error for tool "${toolId}"; marking irreversible barrier`);
        return { verified: false, hasBarrier: true };
      }
    }

    const avgError = totalResidual / pool.length;
    if (avgError <= epsilonSaga) {
      tool.sagaCompensatorCode = compensatorCodeStr;
      tool.sagaResidualError = avgError;
      tool.hasIrreversibleBarrier = false;
      logger.info('ToolVM', `Saga compensator verified for tool "${toolId}" (residual: ${avgError.toFixed(6)})`);
      return { verified: true, residualError: avgError, hasBarrier: false };
    }

    tool.hasIrreversibleBarrier = true;
    logger.warn('ToolVM', `Saga compensator residual error ${avgError.toFixed(6)} exceeded tolerance ${epsilonSaga} for tool "${toolId}"; marking irreversible barrier`);
    return { verified: false, residualError: avgError, hasBarrier: true };
  }

  /**
   * CEGIS (Counterexample-Guided Inductive Synthesis) tool patching with regression verification.
   *
   * @param {string} toolId
   * @param {string} candidateCodeStr
   * @param {StateObject} failingInputState
   * @returns {{ accepted: boolean, newVersion?: number, output?: StateObject, reason?: string }}
   */
  patchToolWithCEGIS(toolId, candidateCodeStr, failingInputState) {
    const tool = this.tools.get(toolId);
    if (!tool) throw new Error(`Tool ${toolId} not found`);

    logger.info('ToolVM', `CEGIS patch verification initiated for tool "${toolId}"`, {
      toolId,
      currentVersion: tool.version,
      regressionPoolSize: tool.regressionPool.length
    });

    for (let i = 0; i < tool.regressionPool.length; i++) {
      const testCase = tool.regressionPool[i];
      logger.debug('ToolVM', `CEGIS regression check iteration ${i + 1}/${tool.regressionPool.length}`);
      const outNew = this.executeSandboxed(candidateCodeStr, testCase.input);
      if (JSON.stringify(outNew) !== JSON.stringify(testCase.expectedOutput)) {
        logger.warn('ToolVM', `CEGIS regression mismatch on test case ${i + 1} for tool "${toolId}"`, {
          input: testCase.input,
          expected: testCase.expectedOutput,
          actual: outNew
        });
        return { accepted: false, reason: 'REGRESSION_MISMATCH' };
      }
    }

    try {
      const fixedOut = this.executeSandboxed(candidateCodeStr, failingInputState);
      tool.regressionPool.push({ input: structuredClone(failingInputState), expectedOutput: fixedOut });
      tool.sourceCode = candidateCodeStr;
      tool.version += 1;
      logger.info('ToolVM', `CEGIS patch accepted for tool "${toolId}" (updated to version ${tool.version})`, {
        newVersion: tool.version,
        output: fixedOut
      });
      return { accepted: true, newVersion: tool.version, output: fixedOut };
    } catch (err) {
      logger.error('ToolVM', `CEGIS counterexample execution failed for tool "${toolId}": ${err instanceof Error ? err.message : String(err)}`, {
        error: err
      });
      return { accepted: false, reason: `COUNTEREXAMPLE_FAILED: ${err instanceof Error ? err.message : String(err)}` };
    }
  }
}
