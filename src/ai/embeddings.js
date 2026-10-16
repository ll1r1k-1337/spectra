// @ts-check
import crypto from 'node:crypto';

/**
 * @typedef {import('../../types/ai.js').ILocalEmbeddingEngine} ILocalEmbeddingEngine
 */

/**
 * Generates a deterministic 384-dimensional L2-normalized float array for fallback scenarios.
 * @param {string} text
 * @param {number} [dim]
 * @returns {number[]}
 */
export function generateDeterministicVector(text, dim = 384) {
  const hash = crypto.createHash('sha256').update(String(text)).digest();
  let seed = hash.readUInt32LE(0) ^ hash.readUInt32LE(4) ^ hash.readUInt32LE(8);
  const prng = () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  const vector = new Array(dim);
  let sumSq = 0;
  for (let i = 0; i < dim; i++) {
    const val = prng() * 2 - 1;
    vector[i] = val;
    sumSq += val * val;
  }
  const norm = Math.sqrt(sumSq) || 1;
  for (let i = 0; i < dim; i++) {
    vector[i] = vector[i] / norm;
  }
  return vector;
}

/**
 * @implements {ILocalEmbeddingEngine}
 */
export class LocalEmbeddingEngine {
  /**
   * @param {object} [options]
   * @param {string} [options.modelName]
   * @param {string} [options.cacheDir]
   * @param {[number, number, number]} [options.weights]
   */
  constructor(options = {}) {
    this.modelName = options.modelName || 'Xenova/multilingual-e5-small';
    this.cacheDir = options.cacheDir || './.cache/transformers';
    this.weights = options.weights || [0.5, 0.3, 0.2];
    /** @type {any} */
    this.pipelineInstance = null;
    this.initialized = false;
  }

  /**
   * Initializes local embedding model or falls back gracefully to deterministic hashing.
   * @returns {Promise<this>}
   */
  async init() {
    if (this.initialized) {
      return this;
    }
    try {
      const { pipeline } = await import('@huggingface/transformers');
      this.pipelineInstance = await pipeline('feature-extraction', this.modelName, {
        cache_dir: this.cacheDir
      });
    } catch {
      // Graceful fallback when pipeline cannot be loaded (offline/sandbox/missing native libs)
      this.pipelineInstance = null;
    }
    this.initialized = true;
    return this;
  }

  /**
   * Embeds single text query into 384-dimensional normalized vector.
   * @param {string} text
   * @returns {Promise<number[]>}
   */
  async embedText(text) {
    if (!this.initialized) {
      await this.init();
    }

    if (this.pipelineInstance) {
      try {
        const output = await this.pipelineInstance(text, { pooling: 'mean', normalize: true });
        const raw = Array.from(output.data);
        if (raw.length === 384) {
          return raw;
        }
        if (raw.length > 384) {
          const sliced = raw.slice(0, 384);
          const norm = Math.hypot(...sliced) || 1;
          return sliced.map((x) => x / norm);
        }
      } catch {
        // Fallback to deterministic generation if pipeline execution fails
      }
    }

    return generateDeterministicVector(text, 384);
  }

  /**
   * Computes multi-modal incident vector v_inc by combining error, markov blanket, and schema.
   * Weighted by [w0, w1, w2] and normalized.
   * @param {object} params
   * @param {string} params.errorText
   * @param {string} params.markovBlanketSignature
   * @param {string} params.stateSchemaSignature
   * @returns {Promise<number[]>}
   */
  async computeIncidentVector({ errorText, markovBlanketSignature, stateSchemaSignature }) {
    const [w0, w1, w2] = this.weights;
    const [vErr, vMarkov, vSchema] = await Promise.all([
      this.embedText(errorText || ''),
      this.embedText(markovBlanketSignature || ''),
      this.embedText(stateSchemaSignature || '')
    ]);

    const combined = new Array(384);
    let sumSq = 0;
    for (let i = 0; i < 384; i++) {
      const val = w0 * vErr[i] + w1 * vMarkov[i] + w2 * vSchema[i];
      combined[i] = val;
      sumSq += val * val;
    }

    const norm = Math.sqrt(sumSq) || 1;
    for (let i = 0; i < 384; i++) {
      combined[i] = combined[i] / norm;
    }

    return combined;
  }
}
