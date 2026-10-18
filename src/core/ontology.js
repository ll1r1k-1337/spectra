// @ts-check

/**
 * @typedef {import('../../types/index.js').IStateOntologyManager} IStateOntologyManager
 * @typedef {import('../../types/index.js').SerializedOntologyKey} SerializedOntologyKey
 */

/**
 * @implements {IStateOntologyManager}
 */
export class StateOntologyManager {
  /**
   * @param {number} [stagingTtlEpochs]
   */
  constructor(stagingTtlEpochs = 2) {
    /** @type {Map<string, { status: 'STAGING' | 'CANONICAL'; canonicalAlias?: string; epochsLived: number; mutualInfoGain: number }>} */
    this.keys = new Map();
    this.stagingTtlEpochs = stagingTtlEpochs;
  }

  /**
   * @param {string} keyName
   * @param {number} [mutualInfoGain]
   */
  registerStagingKey(keyName, mutualInfoGain = 0.15) {
    if (!this.keys.has(keyName)) {
      this.keys.set(keyName, {
        status: 'STAGING',
        epochsLived: 0,
        mutualInfoGain
      });
    }
  }

  /**
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
   * @param {string} canonicalKey
   * @param {string} synonymKey
   * @returns {string}
   */
  mergeKeys(canonicalKey, synonymKey) {
    const rootA = this.findCanonical(canonicalKey);
    const rootB = this.findCanonical(synonymKey);
    if (!this.keys.has(rootA)) {
      this.keys.set(rootA, { status: 'CANONICAL', epochsLived: 0, mutualInfoGain: 1.0 });
    }
    if (rootA !== rootB) {
      const existingB = this.keys.get(rootB) || { status: 'STAGING', epochsLived: 0, mutualInfoGain: 0 };
      existingB.canonicalAlias = rootA;
      this.keys.set(rootB, existingB);
    }
    return rootA;
  }

  /**
   * @param {number} [thetaIg]
   * @returns {{ promoted: string[]; evicted: string[] }}
   */
  runStagingGarbageCollection(thetaIg = 0.1) {
    /** @type {string[]} */
    const promoted = [];
    /** @type {string[]} */
    const evicted = [];
    for (const [k, meta] of this.keys.entries()) {
      if (meta.status === 'STAGING') {
        meta.epochsLived = (meta.epochsLived || 0) + 1;
        if ((meta.mutualInfoGain || 0) >= thetaIg) {
          meta.status = 'CANONICAL';
          promoted.push(k);
        } else if (meta.epochsLived >= this.stagingTtlEpochs) {
          this.keys.delete(k);
          evicted.push(k);
        }
      }
    }
    return { promoted, evicted };
  }

  /**
   * @returns {SerializedOntologyKey[]}
   */
  toJSON() {
    return Array.from(this.keys.entries()).map(([keyName, meta]) => ({
      keyName,
      status: meta.status,
      canonicalAlias: meta.canonicalAlias,
      epochsLived: meta.epochsLived,
      mutualInfoGain: meta.mutualInfoGain
    }));
  }

  /**
   * @param {SerializedOntologyKey[]} entries
   */
  loadFromJSON(entries) {
    for (const item of entries || []) {
      this.keys.set(item.keyName, {
        status: item.status,
        canonicalAlias: item.canonicalAlias,
        epochsLived: item.epochsLived,
        mutualInfoGain: item.mutualInfoGain
      });
    }
  }
}
