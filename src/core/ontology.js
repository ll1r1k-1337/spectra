/**
 * StateOntologyManager maintains the evolving state space K_t,
 * alias resolution with Union-Find Tarjan path compression,
 * and staging garbage collection based on mutual information gain.
 */
export class StateOntologyManager {
  /**
   * @param {number} [stagingTtlEpochs=5]
   */
  constructor(stagingTtlEpochs = 5) {
    /** @type {Map<string, { canonicalAlias?: string, status?: 'STAGING' | 'CANONICAL', mutualInfoGain?: number, epochsLived?: number }>} */
    this.keys = new Map();
    /** @type {number} */
    this.stagingTtlEpochs = stagingTtlEpochs;
  }

  /**
   * Find the canonical key name using Tarjan path compression (Union-Find).
   *
   * @param {string} keyName
   * @returns {string}
   */
  findCanonical(keyName) {
    const entry = this.keys.get(keyName);
    if (!entry || !entry.canonicalAlias || entry.canonicalAlias === keyName) return keyName;
    const root = this.findCanonical(entry.canonicalAlias);
    entry.canonicalAlias = root;
    return root;
  }

  /**
   * Merge two keys by pointing synonym root to canonical root.
   *
   * @param {string} canonicalKey
   * @param {string} synonymKey
   * @returns {string} canonical root
   */
  mergeKeys(canonicalKey, synonymKey) {
    const rootA = this.findCanonical(canonicalKey);
    const rootB = this.findCanonical(synonymKey);
    if (rootA !== rootB) {
      const entryB = this.keys.get(rootB);
      if (entryB) {
        entryB.canonicalAlias = rootA;
      } else {
        this.keys.set(rootB, { canonicalAlias: rootA });
      }
    }
    return rootA;
  }

  /**
   * Run garbage collection for staging keys. Promotes to CANONICAL if mutualInfoGain >= thetaIg,
   * evicts if lived >= stagingTtlEpochs.
   *
   * @param {number} [thetaIg=0.10]
   * @returns {string[]} evicted key names
   */
  runStagingGarbageCollection(thetaIg = 0.10) {
    /** @type {string[]} */
    const evicted = [];
    for (const [k, meta] of this.keys.entries()) {
      if (meta.status === 'STAGING') {
        meta.epochsLived = (meta.epochsLived || 0) + 1;
        if ((meta.mutualInfoGain || 0) >= thetaIg) {
          meta.status = 'CANONICAL';
        } else if (meta.epochsLived >= this.stagingTtlEpochs) {
          this.keys.delete(k);
          evicted.push(k);
        }
      }
    }
    return evicted;
  }
}
