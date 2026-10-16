import crypto from 'node:crypto';

/**
 * @typedef {import('#types').GraphNodeSpec} GraphNodeSpec
 * @typedef {import('#types').GraphEdgeSpec} GraphEdgeSpec
 * @typedef {import('#types').StateObject} StateObject
 * @typedef {import('#types').JsonSchemaProperty} JsonSchemaProperty
 * @typedef {import('#types').IDirectedGraph} IDirectedGraph
 */

/**
 * DirectedGraph represents an executable computation graph with Liskov-typed MetaInit,
 * Causal Markov Blanket computation, UCB-CMI read contract deflation,
 * Pushout/Guarded bifurcation merge, Bisimulation diamond folding, and Entropy-driven mitosis.
 *
 * @implements {IDirectedGraph}
 */
export class DirectedGraph {
  /** @type {Map<string, { id: string, transformKey: string, coercerFn: (val: any) => any }>} */
  static schemaMorphismRegistry = new Map();

  /**
   * @param {string} [id='default_graph']
   */
  constructor(id = 'default_graph') {
    /** @type {string} */
    this.id = id;
    /** @type {Map<string, GraphNodeSpec>} */
    this.nodes = new Map();
    /** @type {Map<string, GraphEdgeSpec[]>} */
    this.edges = new Map();
    /** @type {string | null} */
    this.startNode = null;
    /** @type {Set<string>} */
    this.goalNodes = new Set();
  }

  /**
   * Add a node specification to the graph.
   * @param {GraphNodeSpec} node
   * @returns {this}
   */
  addNode(node) {
    this.nodes.set(node.id, {
      id: node.id,
      type: node.type,
      reads: Array.from(new Set(node.reads || [])),
      writes: Array.from(new Set(node.writes || [])),
      inputSchemas: node.inputSchemas || {},
      outputSchemas: node.outputSchemas || {},
      idempotency: node.idempotency || 'pure',
      hasBarrier: Boolean(node.hasBarrier),
      resourceRps: node.resourceRps ?? 1.0,
      modelTier: node.modelTier || 'frontier',
      config: node.config || {}
    });
    if (!this.edges.has(node.id)) {
      this.edges.set(node.id, []);
    }
    return this;
  }

  /**
   * Add an edge between nodes. Supports both argument conventions:
   * (from, to, priority, guard, guardExpr, intervals) or (from, to, guardFn, priority, guardMeta).
   *
   * @param {string} from
   * @param {string} to
   * @param {number | ((state: StateObject) => boolean)} [arg3=10]
   * @param {((state: StateObject) => boolean) | number} [arg4]
   * @param {string | { expr?: string, intervals?: Record<string, [number, number]> } | null} [arg5=null]
   * @param {Record<string, [number, number]> | null} [arg6=null]
   * @returns {this}
   */
  addEdge(from, to, arg3 = 10, arg4 = () => true, arg5 = null, arg6 = null) {
    if (!this.edges.has(from)) {
      this.edges.set(from, []);
    }

    let priority = 10;
    /** @type {(state: StateObject) => boolean} */
    let guardFn = () => true;
    let guardExpr = '';
    /** @type {Record<string, [number, number]>} */
    let intervals = {};

    if (typeof arg3 === 'function') {
      guardFn = arg3;
      priority = typeof arg4 === 'number' ? arg4 : 10;
      if (arg5 && typeof arg5 === 'object') {
        guardExpr = arg5.expr || guardFn.toString();
        intervals = arg5.intervals || {};
      } else {
        guardExpr = typeof arg5 === 'string' ? arg5 : guardFn.toString();
        intervals = arg6 || {};
      }
    } else {
      priority = typeof arg3 === 'number' ? arg3 : 10;
      guardFn = typeof arg4 === 'function' ? arg4 : () => true;
      if (arg5 && typeof arg5 === 'object') {
        guardExpr = arg5.expr || guardFn.toString();
        intervals = arg5.intervals || {};
      } else {
        guardExpr = typeof arg5 === 'string' ? arg5 : guardFn.toString();
        intervals = arg6 || {};
      }
    }

    const edgeList = /** @type {GraphEdgeSpec[]} */ (this.edges.get(from));
    edgeList.push({
      from,
      to,
      guard: guardFn,
      priority,
      guardExpr,
      intervals
    });
    edgeList.sort((a, b) => a.priority - b.priority);
    return this;
  }

  /**
   * Definition 8 & 11: Compute Causal Markov Blanket MB(v_t, G)
   * with Personalized PageRank weights along dataflow dependencies.
   *
   * @param {string} centerNodeId
   * @param {number} [alphaPpr=0.15]
   * @returns {Map<string, number>}
   */
  getCausalMarkovBlanket(centerNodeId, alphaPpr = 0.15) {
    const center = this.nodes.get(centerNodeId);
    if (!center) return new Map();

    const centerReads = new Set(center.reads);
    const centerWrites = new Set(center.writes);
    const parents = new Set();
    const children = new Set();
    const childReads = new Set();

    for (const [id, node] of this.nodes.entries()) {
      if (id === centerNodeId) continue;
      if (node.writes.some((k) => centerReads.has(k))) parents.add(id);
      if (node.reads.some((k) => centerWrites.has(k))) {
        children.add(id);
        node.reads.forEach((k) => childReads.add(k));
      }
    }

    const spouses = new Set();
    for (const [id, node] of this.nodes.entries()) {
      if (id !== centerNodeId && !parents.has(id) && !children.has(id)) {
        if (node.writes.some((k) => childReads.has(k))) spouses.add(id);
      }
    }

    const blanketIds = new Set([centerNodeId, ...parents, ...children, ...spouses]);
    /** @type {Map<string, number>} */
    const scores = new Map();
    for (const id of blanketIds) {
      const dist = id === centerNodeId ? 0 : (parents.has(id) || children.has(id) ? 1 : 2);
      scores.set(id, alphaPpr * Math.pow(1 - alphaPpr, dist));
    }
    return scores;
  }

  /**
   * Definition 2a: Counterfactually safe causal deflation of read contracts R(v)
   * via upper confidence bound CMI_UCB and zero counterfactual sensitivity.
   *
   * @param {Map<string, Map<string, { totalRuns?: number, minorityRuns?: number, empiricalCmi?: number, counterfactualSensitivity?: number }>>} cmiStatsMatrix
   * @param {number} [epsilonCmi=0.02]
   * @param {number} [cCmi=0.15]
   * @returns {number} prunedCount
   */
  deflateSpuriousReadContracts(cmiStatsMatrix, epsilonCmi = 0.02, cCmi = 0.15) {
    let prunedCount = 0;
    for (const [nodeId, node] of this.nodes.entries()) {
      const nodeStats = cmiStatsMatrix.get(nodeId);
      if (!nodeStats || node.reads.length <= 1) continue;
      const filteredReads = node.reads.filter((k) => {
        const stat = nodeStats.get(k);
        if (!stat) return true;
        const nTotal = Math.max(2, stat.totalRuns ?? 40);
        const nEff = Math.max(0, stat.minorityRuns ?? 0);
        const cmiUcb = (stat.empiricalCmi ?? 0) + cCmi * Math.sqrt(Math.log(nTotal) / (nEff + 1));
        const cfSens = stat.counterfactualSensitivity ?? 1.0;
        return !(cmiUcb < epsilonCmi && cfSens === 0);
      });
      if (filteredReads.length > 0 && filteredReads.length < node.reads.length) {
        prunedCount += node.reads.length - filteredReads.length;
        node.reads = filteredReads;
      }
    }
    return prunedCount;
  }

  /**
   * Definition 12b: Entropy-driven mitosis (splitting) of cognitively overloaded LLM node.
   *
   * @param {string} nodeId
   * @param {string[]} partitionW1
   * @param {string[]} partitionW2
   * @returns {boolean}
   */
  splitOverloadedLlmNode(nodeId, partitionW1, partitionW2) {
    const node = this.nodes.get(nodeId);
    if (!node || node.type !== 'llm' || partitionW1.length === 0 || partitionW2.length === 0) {
      return false;
    }
    const node1Id = `${nodeId}_mitosis_1`;
    const node2Id = `${nodeId}_mitosis_2`;

    this.addNode({
      id: node1Id,
      type: 'llm',
      reads: [...node.reads],
      writes: [...partitionW1],
      modelTier: 'light',
      config: { ...node.config, role: 'subtask_part_1' }
    });
    this.addNode({
      id: node2Id,
      type: 'llm',
      reads: [...node.reads, ...partitionW1],
      writes: [...partitionW2],
      modelTier: 'light',
      config: { ...node.config, role: 'subtask_part_2' }
    });

    for (const [, edgeList] of this.edges.entries()) {
      for (const edge of edgeList) {
        if (edge.to === nodeId) edge.to = node1Id;
      }
    }
    const outgoing = this.edges.get(nodeId) || [];
    this.edges.set(node2Id, outgoing.map((e) => ({ ...e, from: node2Id })));
    this.edges.delete(nodeId);
    this.nodes.delete(nodeId);
    this.addEdge(node1Id, node2Id, 1, () => true);

    if (this.startNode === nodeId) this.startNode = node1Id;
    if (this.goalNodes.has(nodeId)) {
      this.goalNodes.delete(nodeId);
      this.goalNodes.add(node2Id);
    }
    return true;
  }

  /**
   * Definition 1a: Liskov JSON Schema subtype check.
   *
   * @param {JsonSchemaProperty | undefined} outSchema
   * @param {JsonSchemaProperty | undefined} inSchema
   * @returns {boolean}
   */
  static isJsonSchemaSubtype(outSchema, inSchema) {
    if (!outSchema || !inSchema) return true;
    if (outSchema.type !== inSchema.type) {
      return outSchema.type === 'integer' && inSchema.type === 'number';
    }
    if (outSchema.nullable === true && inSchema.nullable === false) return false;
    if (typeof inSchema.minimum === 'number' && (outSchema.minimum ?? -Infinity) < inSchema.minimum) {
      return false;
    }
    if (typeof inSchema.maximum === 'number' && (outSchema.maximum ?? Infinity) > inSchema.maximum) {
      return false;
    }
    return true;
  }

  /**
   * Definition 1a: Resolve or synthesize schema coercion morphism.
   *
   * @param {string} key
   * @param {JsonSchemaProperty | undefined} outSchema
   * @param {JsonSchemaProperty | undefined} inSchema
   * @returns {{ id: string, transformKey: string, coercerFn: (val: any) => any }}
   */
  static resolveOrSynthesizeSchemaMorphism(key, outSchema, inSchema) {
    const hashPair = crypto
      .createHash('sha256')
      .update(JSON.stringify({ outSchema, inSchema }))
      .digest('hex');
    if (DirectedGraph.schemaMorphismRegistry.has(hashPair)) {
      return /** @type {{ id: string, transformKey: string, coercerFn: (val: any) => any }} */ (
        DirectedGraph.schemaMorphismRegistry.get(hashPair)
      );
    }
    const minBound = typeof inSchema?.minimum === 'number' ? inSchema.minimum : -Infinity;
    const maxBound = typeof inSchema?.maximum === 'number' ? inSchema.maximum : Infinity;
    const morphism = {
      id: `morphism_${hashPair.slice(0, 8)}`,
      transformKey: key,
      coercerFn: (/** @type {any} */ val) => {
        const num = typeof val === 'number' ? val : Number(val ?? 0);
        const safe = Number.isNaN(num) ? (Number.isFinite(minBound) ? minBound : 0) : num;
        return Math.min(maxBound, Math.max(minBound, safe));
      }
    };
    DirectedGraph.schemaMorphismRegistry.set(hashPair, morphism);
    return morphism;
  }

  /**
   * Definition 1a: Type-safe cross-task meta-synthesis of starting graph G_0^*
   * with Liskov JSON-schema structural subtyping and morphism synthesis.
   *
   * @param {number[]} intentVector
   * @param {string[]} initKeys
   * @param {string[]} goalKeys
   * @param {Array<{ id: string, reads: string[], writes: string[], type?: any, inputSchemas?: Record<string, JsonSchemaProperty>, outputSchemas?: Record<string, JsonSchemaProperty>, expectedCost?: number, similarityToIntent?: number, confidenceLcb?: number }>} skillLibrary
   * @param {number} [costBudgetMax=40.0]
   * @param {Record<string, JsonSchemaProperty>} [initSchemas={}]
   * @returns {DirectedGraph}
   */
  static metaInitialize(intentVector, initKeys, goalKeys, skillLibrary, costBudgetMax = 40.0, initSchemas = {}) {
    const g = new DirectedGraph('meta_synthesized_graph');
    const available = new Set(initKeys);
    /** @type {Record<string, JsonSchemaProperty>} */
    const activeSchemas = { ...initSchemas };
    /** @type {Array<typeof skillLibrary[0]>} */
    const selectedSkills = [];
    let totalExpectedCost = 0;

    g.addNode({ id: 'start', type: 'tool', reads: [], writes: Array.from(initKeys), outputSchemas: activeSchemas });
    g.startNode = 'start';

    let progress = true;
    while (progress && !goalKeys.every((k) => available.has(k))) {
      progress = false;
      /** @type {typeof skillLibrary[0] | null} */
      let bestSkill = null;
      let bestScore = Infinity;

      for (const skill of skillLibrary) {
        const canRun = skill.reads.every((r) => available.has(r));
        const producesNew = skill.writes.some((w) => !available.has(w));
        if (canRun && producesNew && (totalExpectedCost + (skill.expectedCost || 5)) <= costBudgetMax) {
          const sim = Math.max(0.05, skill.similarityToIntent || 0.8);
          const score = (skill.expectedCost || 5) / ((skill.confidenceLcb || 0.8) * sim);
          if (score < bestScore) {
            bestScore = score;
            bestSkill = skill;
          }
        }
      }

      if (bestSkill) {
        selectedSkills.push(bestSkill);
        totalExpectedCost += bestSkill.expectedCost || 5;
        for (const w of bestSkill.writes) {
          available.add(w);
          if (bestSkill.outputSchemas?.[w]) activeSchemas[w] = bestSkill.outputSchemas[w];
        }
        progress = true;
      }
    }

    let prevId = 'start';
    for (const s of selectedSkills) {
      const mismatchedKeys = s.reads.filter(
        (k) => !DirectedGraph.isJsonSchemaSubtype(activeSchemas[k], s.inputSchemas?.[k])
      );
      if (mismatchedKeys.length > 0) {
        const coerceId = `coerce_before_${s.id}`;
        const morphisms = mismatchedKeys.map((k) =>
          DirectedGraph.resolveOrSynthesizeSchemaMorphism(k, activeSchemas[k], s.inputSchemas?.[k])
        );
        g.addNode({
          id: coerceId,
          type: 'tool',
          reads: mismatchedKeys,
          writes: mismatchedKeys,
          outputSchemas: s.inputSchemas || {},
          config: { morphisms: morphisms.map((m) => m.id) }
        });
        g.addEdge(prevId, coerceId, 1, () => true);
        prevId = coerceId;
      }
      g.addNode({
        id: s.id,
        type: s.type || 'macro',
        reads: s.reads,
        writes: s.writes,
        inputSchemas: s.inputSchemas,
        outputSchemas: s.outputSchemas
      });
      g.addEdge(prevId, s.id, 1, () => true);
      prevId = s.id;
    }
    g.addNode({ id: 'goal', type: 'check', reads: goalKeys, writes: [] });
    g.addEdge(prevId, 'goal', 1, () => true);
    g.goalNodes.add('goal');
    return g;
  }

  /**
   * Definition 1b: Algebraic Pushout, 3-Way guarded bifurcation, and bisimulation folding.
   *
   * @param {DirectedGraph} baseGraph
   * @param {any} branchA
   * @param {any} branchB
   * @returns {{ merged: boolean, mode?: string, graph?: DirectedGraph, reason?: string }}
   */
  static tryPushoutOrBifurcateMerge(baseGraph, branchA, branchB) {
    const readsA = new Set(branchA.modifiedReads || []);
    const writesA = new Set(branchA.modifiedWrites || []);
    const readsB = new Set(branchB.modifiedReads || []);
    const writesB = new Set(branchB.modifiedWrites || []);

    const touchedA = branchA.touchedNodes || [];
    const touchedB = branchB.touchedNodes || [];

    const disjointNodes = !touchedA.some((/** @type {string} */ n) => touchedB.includes(n));
    const noDataConflict =
      ![...writesA].some((k) => readsB.has(k) || writesB.has(k)) &&
      ![...writesB].some((k) => readsA.has(k) || writesA.has(k));

    if (disjointNodes && noDataConflict) {
      const merged = baseGraph.clone();
      if (typeof branchA.applyTo === 'function') branchA.applyTo(merged);
      if (typeof branchB.applyTo === 'function') branchB.applyTo(merged);
      return { merged: true, mode: 'PUSHOUT', graph: merged };
    }

    const sharedHotspots = touchedA.filter((/** @type {string} */ n) => touchedB.includes(n));
    const entryDisjoint = typeof branchA.entryGuardDisjointFrom === 'function'
      ? branchA.entryGuardDisjointFrom(branchB)
      : true;

    if (sharedHotspots.length === 1 && entryDisjoint) {
      const merged = baseGraph.clone();
      const hotspot = sharedHotspots[0];
      if (typeof branchA.applyGuardedBranch === 'function') branchA.applyGuardedBranch(merged, hotspot, 1);
      if (typeof branchB.applyGuardedBranch === 'function') branchB.applyGuardedBranch(merged, hotspot, 2);
      merged.foldBisimilarBranches();
      return { merged: true, mode: 'GUARDED_BIFURCATION_WITH_BISIM_FOLD', graph: merged };
    }

    return { merged: false, reason: 'REQUIRES_RELATIONAL_REBASE' };
  }

  /**
   * Definition 1b p.3: Reverse bisimulation branch folding into DAG diamonds.
   *
   * @returns {this}
   */
  foldBisimilarBranches() {
    const signatureMap = new Map();
    for (const [id, node] of this.nodes.entries()) {
      if (id === this.startNode || this.goalNodes.has(id)) continue;
      const outTargets = (this.edges.get(id) || []).map((e) => e.to).sort().join(',');
      const sig = `${node.type}|R:${[...node.reads].sort()}|W:${[...node.writes].sort()}|fn:${node.config?.toolId || node.config?.promptHash || id}|out:${outTargets}`;

      if (signatureMap.has(sig)) {
        const canonicalId = signatureMap.get(sig);
        for (const [, edgeList] of this.edges.entries()) {
          for (const edge of edgeList) {
            if (edge.to === id) edge.to = canonicalId;
          }
        }
        this.edges.delete(id);
        this.nodes.delete(id);
      } else {
        signatureMap.set(sig, id);
      }
    }
    return this;
  }

  /**
   * Compute deterministic canonical structural hash of the graph.
   *
   * @returns {string} SHA-256 hex
   */
  computeStructuralHash() {
    const payload = JSON.stringify({
      nodes: Array.from(this.nodes.entries()).sort(),
      edges: Array.from(this.edges.entries()).map(([k, v]) => [
        k,
        v.map((e) => ({ to: e.to, p: e.priority, g: e.guardExpr }))
      ])
    });
    return crypto.createHash('sha256').update(payload).digest('hex');
  }

  /**
   * Deep clone the graph topology and node configurations.
   *
   * @returns {DirectedGraph}
   */
  clone() {
    const copy = new DirectedGraph(this.id);
    copy.startNode = this.startNode;
    this.goalNodes.forEach((g) => copy.goalNodes.add(g));
    for (const [, node] of this.nodes) {
      /** @type {GraphNodeSpec} */
      const clonedNode = {
        ...node,
        reads: [...node.reads],
        writes: [...node.writes],
        inputSchemas: node.inputSchemas ? { ...node.inputSchemas } : {},
        outputSchemas: node.outputSchemas ? { ...node.outputSchemas } : {},
        config: node.config ? { ...node.config } : {}
      };
      copy.addNode(clonedNode);
    }
    for (const [from, edgeList] of this.edges) {
      for (const e of edgeList) {
        copy.addEdge(from, e.to, e.priority, e.guard, e.guardExpr, e.intervals ? { ...e.intervals } : {});
      }
    }
    return copy;
  }
}
