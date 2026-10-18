// @ts-check
import vm from 'node:vm';

/**
 * @typedef {import('../../types/index.js').IToolRegistry} IToolRegistry
 * @typedef {import('../../types/index.js').StateObject} StateObject
 */

/**
 * @implements {IToolRegistry}
 */
export class ToolRegistry {
  constructor() {
    /** @type {Map<string, any>} */
    this.tools = new Map();
  }

  /**
   * @param {{
   *   id: string;
   *   reads: string[];
   *   writes: string[];
   *   sourceCode: string;
   *   version?: number;
   *   hasIrreversibleBarrier?: boolean;
   *   sagaCompensatorCode?: string;
   *   regressionPool?: Array<{ input: StateObject; expectedOutput: StateObject }>;
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
   * Безопасно исполняет JS-код инструмента в изолированном контексте node:vm.
   * Очищает хвостовые точки с запятой, которые могут возвращать реальные LLM в поле sourceCode.
   * @param {string} codeStr
   * @param {StateObject} stateInput
   * @param {number} [timeoutMs]
   * @returns {StateObject}
   */
  executeSandboxed(codeStr, stateInput, timeoutMs = 250) {
    const cleanCode = (codeStr || '').trim().replace(/;+\s*$/, '');
    const frozenInput = Object.freeze(structuredClone(stateInput));
    const sandbox = {
      input: frozenInput,
      Math,
      Date,
      JSON,
      Number,
      String,
      Boolean,
      Array,
      Object
    };
    const context = vm.createContext(sandbox);
    const script = new vm.Script(`(${cleanCode})(input)`);
    const result = script.runInContext(context, { timeout: timeoutMs });
    return result && typeof result === 'object' ? structuredClone(result) : {};
  }

  /**
   * Определение 2: Верифицирует компенсирующий Saga-оператор Phi^-1 в песочнице node:vm на регрессионном пуле.
   * @param {string} toolId
   * @param {string} compensatorCodeStr
   * @param {number} [epsilonSaga]
   * @returns {{ verified: boolean; residualError: number; hasBarrier: boolean }}
   */
  verifyAndAttachSagaCompensator(toolId, compensatorCodeStr, epsilonSaga = 1e-4) {
    const tool = this.tools.get(toolId);
    if (!tool) {
      throw new Error(`Инструмент "${toolId}" не найден в ToolRegistry`);
    }

    const pool =
      Array.isArray(tool.regressionPool) && tool.regressionPool.length > 0
        ? tool.regressionPool
        : [{ input: {}, expectedOutput: {} }];

    let totalResidual = 0;
    for (const tc of pool) {
      const sampleInput = tc.input ?? tc;
      const forwardState = { ...sampleInput, ...this.executeSandboxed(tool.sourceCode, sampleInput) };
      const restoredState = { ...forwardState, ...this.executeSandboxed(compensatorCodeStr, forwardState) };
      const trackedKeys = new Set([...(tool.reads || []), ...(tool.writes || [])]);
      for (const k of trackedKeys) {
        if (JSON.stringify(sampleInput[k]) !== JSON.stringify(restoredState[k])) {
          totalResidual += 1.0;
        }
      }
    }

    const residualError = totalResidual / pool.length;
    if (residualError <= epsilonSaga) {
      tool.sagaCompensatorCode = compensatorCodeStr;
      tool.sagaResidualError = residualError;
      tool.hasIrreversibleBarrier = false;
      return { verified: true, residualError, hasBarrier: false };
    }

    tool.hasIrreversibleBarrier = true;
    return { verified: false, residualError, hasBarrier: true };
  }
}
