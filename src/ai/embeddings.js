// src/ai/embeddings.js
// @ts-check

/**
 * @typedef {import('../../types/index.js').ILocalEmbeddingEngine} ILocalEmbeddingEngine
 */

/**
 * Вычисляет косинусную близость двух векторов одинаковой размерности.
 * @param {number[]} vecA
 * @param {number[]} vecB
 * @returns {number}
 */
export function cosineSimilarity(vecA, vecB) {
  if (!vecA?.length || !vecB?.length || vecA.length !== vecB.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom > 0 ? dot / denom : 0;
}

/**
 * Боевой генератор эмбеддингов на базе ONNX Runtime (@huggingface/transformers).
 * @implements {ILocalEmbeddingEngine}
 */
export class LocalEmbeddingEngine {
  /**
   * @param {{ modelName?: string; cacheDir?: string; hfToken?: string; weights?: [number, number, number] }} [opts]
   */
  constructor(opts = {}) {
    this.modelName = opts.modelName || 'Xenova/multilingual-e5-small';
    this.cacheDir = opts.cacheDir || './.spectra/models';
    this.hfToken = opts.hfToken;
    /** @type {[number, number, number]} */
    this.weights = opts.weights || [0.5, 0.3, 0.2];
    /** @type {any} */
    this.extractor = null;
  }

  async init() {
    if (this.extractor) return this;
    if (this.hfToken && !process.env.HF_TOKEN) {
      process.env.HF_TOKEN = this.hfToken;
    }
    const { pipeline, env } = await import('@huggingface/transformers');
    env.cacheDir = this.cacheDir;
    this.extractor = await pipeline('feature-extraction', this.modelName, { dtype: 'q8' });
    return this;
  }

  /**
   * @param {string} text
   * @returns {Promise<number[]>}
   */
  async embedText(text) {
    await this.init();
    const output = await this.extractor(`query: ${text}`, { pooling: 'mean', normalize: true });
    /** @type {number[]} */
    const rawValues = Array.from(output.data);
    return rawValues.map((v) => Number(Number(v).toFixed(6)));
  }

  /**
   * Вычисляет составной вектор тупика v_inc в R^384 (Определение 8).
   * @param {{ errorText: string; markovBlanketSignature: string; stateSchemaSignature: string }} params
   * @returns {Promise<number[]>}
   */
  async computeIncidentVector({ errorText, markovBlanketSignature, stateSchemaSignature }) {
    const [eErr, eTopo, eSchema] = await Promise.all([
      this.embedText(errorText),
      this.embedText(markovBlanketSignature),
      this.embedText(stateSchemaSignature)
    ]);
    const [a, b, g] = this.weights;
    const combined = eErr.map((val, i) => a * val + b * (eTopo[i] || 0) + g * (eSchema[i] || 0));
    const norm = Math.hypot(...combined) || 1;
    return combined.map((v) => Number((v / norm).toFixed(6)));
  }
}
