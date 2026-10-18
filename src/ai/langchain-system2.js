// src/ai/langchain-system2.js
// @ts-check
import { z } from 'zod';
import { ChatOpenAI } from '@langchain/openai';
import { assertValidLlmCredentials } from '../cli/credentials.js';

/**
 * @typedef {import('../../types/index.js').ILangChainSystem2Arbiter} ILangChainSystem2Arbiter
 * @typedef {import('../../types/index.js').SpectraConfig} SpectraConfig
 * @typedef {import('../../types/index.js').DeadlockIncident} DeadlockIncident
 * @typedef {import('../../types/index.js').MutationPlan} MutationPlan
 * @typedef {import('../../types/index.js').GraphNodeSpec} GraphNodeSpec
 * @typedef {import('../../types/index.js').StateObject} StateObject
 * @typedef {import('../../types/index.js').ModelTier} ModelTier
 */

const MutationSchema = z.object({
  operator: z.enum(['addEdge', 'addNodeAndBridge', 'registerTool']),
  from: z.string(),
  to: z.string(),
  guardExpr: z.string().nullable(),
  node: z
    .object({
      id: z.string(),
      type: z.enum(['tool', 'llm', 'check']),
      reads: z.array(z.string()),
      writes: z.array(z.string()),
      sourceCode: z.string()
    })
    .nullable(),
  reasoning: z.string()
});

/**
 * Боевой арбитр System-2 на базе LangChain.js (@langchain/openai) без моков и заглушек.
 * Поддерживает прокидывание apiKey, baseUrl и имени модели через конфигурацию или переменные окружения.
 * @implements {ILangChainSystem2Arbiter}
 */
export class LangChainSystem2Arbiter {
  /**
   * @param {SpectraConfig['system2']} config
   */
  constructor(config) {
    this.config = config;
  }

  /**
   * Создает клиент ChatOpenAI с прокинутыми учетными данными (apiKey и опциональным baseURL).
   * @param {ModelTier} [tier]
   */
  _createChatModel(tier = 'frontier') {
    assertValidLlmCredentials(this.config);
    const modelName =
      tier === 'light' || tier === 'local_slm'
        ? this.config.lightModel || this.config.frontierModel
        : this.config.frontierModel;

    return new ChatOpenAI({
      apiKey: this.config.apiKey,
      modelName,
      temperature: 0,
      configuration: this.config.baseUrl ? { baseURL: this.config.baseUrl } : undefined
    });
  }

  /**
   * Вызывает реальную LLM через Structured Output для синтеза ремонтной мутации графа.
   * @param {DeadlockIncident} deadlock
   * @param {string} goalDescription
   * @param {string[]} existingNodeIds
   * @param {string} goalNodeId
   * @returns {Promise<MutationPlan>}
   */
  async synthesizeRecoveryMutation(deadlock, goalDescription, existingNodeIds, goalNodeId) {
    const isCheckFailed = deadlock.type === 'CHECK_FAILED';
    const requiredFrom = isCheckFailed ? deadlock.lastValidNodeId || 'start' : deadlock.nodeId;
    const requiredTo = isCheckFailed ? deadlock.nodeId : goalNodeId;
    const expectedNodeId = `solver_${deadlock.nodeId}_s${deadlock.step}`;

    const model = this._createChatModel('frontier');
    const structured = model.withStructuredOutput(MutationSchema);

    const res = await structured.invoke(
      `You are the System-2 Meta-Arbiter of the Spectra self-learning graph runtime.\n` +
        `Session Goal: "${goalDescription}".\n` +
        `Existing Graph Nodes: [${existingNodeIds.join(', ')}]. Terminal Goal Node: "${goalNodeId}".\n` +
        `Deadlock Incident: type="${deadlock.type}" at node="${deadlock.nodeId}" (predecessor="${deadlock.lastValidNodeId || 'start'}"), step=${deadlock.step}.\n` +
        `Error: "${deadlock.error || 'N/A'}".\n` +
        `Current State Snapshot: ${JSON.stringify(deadlock.stateSnapshot)}.\n\n` +
        `Requirements:\n` +
        `1. Set operator="addNodeAndBridge", from="${requiredFrom}", to="${requiredTo}", guardExpr="Boolean(Array.isArray(state.artifacts) && state.artifacts.length > 0)".\n` +
        `2. Set node.id="${expectedNodeId}", node.type="llm", node.reads=["goal"], node.writes=["artifacts", "summary"].\n` +
        `3. Set node.sourceCode to the PROMPT (a plain instruction string, not code) telling the node to actually produce the deliverable: return "artifacts" as an array of {name, content} with real generated content (source code / files) that satisfies the goal, plus a short "summary". Do NOT set goalCompleted. Example:\n` +
        `   "Produce the real deliverable that satisfies the goal. Return artifacts as [{name, content}] with actual generated code/files, and a short summary of what was produced."\n`
    );

    const cleanSourceCode = res.node?.sourceCode
      ? res.node.sourceCode.trim().replace(/;+\s*$/, '')
      : "Produce the real deliverable that satisfies the goal. Return artifacts as [{name, content}] with actual generated code/files, and a summary of what was produced.";

    return {
      operator: res.operator,
      from: res.from || requiredFrom,
      to: res.to || requiredTo,
      guardExpr: res.guardExpr ?? undefined,
      reasoning: res.reasoning,
      node: res.node
        ? {
            id: res.node.id.startsWith('solver_') ? res.node.id : expectedNodeId,
            type: res.node.type,
            reads: Array.from(new Set([...res.node.reads, 'goal'])),
            writes: Array.from(new Set([...res.node.writes, 'artifacts', 'summary'])),
            config: { prompt: cleanSourceCode, sourceCode: cleanSourceCode }
          }
        : undefined
    };
  }

  /**
   * Выполняет штатный узел типа 'llm' в Fast Loop через реальную модель @langchain/openai.
   * @param {GraphNodeSpec} node
   * @param {StateObject} state
   * @param {ModelTier} [tier]
   * @returns {Promise<StateObject>}
   */
  async executeLlmNode(node, state, tier = 'frontier') {
    const model = this._createChatModel(tier);
    /** @type {Record<string, z.ZodTypeAny>} */
    const shape = {};
    for (const wKey of node.writes) {
      shape[wKey] = wKey === 'artifacts'
        ? z.array(z.object({ name: z.string(), content: z.string() }))
        : z.union([z.string(), z.number(), z.boolean()]);
    }
    const OutputSchema = z.object(shape);
    const structured = model.withStructuredOutput(OutputSchema);

    const inputSlice = Object.fromEntries(node.reads.map((k) => [k, state[k]]));
    const promptText =
      node.config?.prompt ||
      `Compute output keys [${node.writes.join(', ')}] given input state ${JSON.stringify(inputSlice)}.`;

    return await structured.invoke(promptText);
  }
}
