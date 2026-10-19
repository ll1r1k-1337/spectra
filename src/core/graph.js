// @ts-check
import crypto from 'node:crypto';
import vm from 'node:vm';

/**
 * @typedef {import('../../types/index.js').IDirectedGraph} IDirectedGraph
 * @typedef {import('../../types/index.js').GraphNodeSpec} GraphNodeSpec
 * @typedef {import('../../types/index.js').GraphEdgeSpec} GraphEdgeSpec
 * @typedef {import('../../types/index.js').SerializedGraph} SerializedGraph
 * @typedef {import('../../types/index.js').StateObject} StateObject
 */

/**
 * @implements {IDirectedGraph}
 */
export class DirectedGraph {
  /**
   * @param {string} [id]
   */
  constructor(id = 'default_graph') {
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
   * @param {GraphNodeSpec} node
   */
  addNode(node) {
    const cfg = { ...(node.config || {}) };
    if (node.type === 'check' && !cfg.assertFn && typeof cfg.assertExpr === 'string') {
      cfg.assertFn = this.compileGuardExpression(cfg.assertExpr);
    }
    this.nodes.set(node.id, {
      id: node.id,
      type: node.type,
      reads: Array.from(new Set(node.reads || [])),
      writes: Array.from(new Set(node.writes || [])),
      inputSchemas: node.inputSchemas ? structuredClone(node.inputSchemas) : {},
      outputSchemas: node.outputSchemas ? structuredClone(node.outputSchemas) : {},
      idempotency: node.idempotency || 'pure',
      hasBarrier: Boolean(node.hasBarrier),
      resourceRps: node.resourceRps ?? 1.0,
      modelTier: node.modelTier || 'frontier',
      config: cfg
    });
    if (!this.edges.has(node.id)) {
      this.edges.set(node.id, []);
    }
    return this;
  }

  /**
   * Безопасно компилирует строковое выражение guardExpr или assertExpr в функцию предиката в изолированном контексте.
   * Очищает блочные комментарии и хвостовые точки с запятой, которые могут генерировать LLM.
   * @param {string} [exprStr]
   * @returns {(state: StateObject) => boolean}
   */
  compileGuardExpression(exprStr) {
    const cleanExpr = (exprStr || '')
      .replace(/\/\*[\s\S]*?\*\//g, '')
      .trim()
      .replace(/;+\s*$/, '');
    if (!cleanExpr || cleanExpr === 'true' || cleanExpr.includes('() => true')) {
      return () => true;
    }
    return (state) => {
      try {
        const frozenState = Object.freeze(structuredClone(state));
        const ctx = vm.createContext({
          state: frozenState,
          s: frozenState,
          Boolean,
          Number,
          String,
          Array,
          Object,
          Math
        });
        const code = cleanExpr.includes('=>')
          ? `(${cleanExpr})(state)`
          : cleanExpr.includes('return ')
            ? `(() => { ${cleanExpr} })()`
            : `Boolean(${cleanExpr})`;
        return Boolean(vm.runInContext(code, ctx, { timeout: 50 }));
      } catch {
        return false;
      }
    };
  }

  /**
   * @param {string} from
   * @param {string} to
   * @param {(state: StateObject) => boolean} [guardFn]
   * @param {number} [priority]
   * @param {{ expr?: string; intervals?: Record<string, [number, number]>; preemptive?: boolean } | null} [guardMeta]
   */
  addEdge(from, to, guardFn, priority = 10, guardMeta = null) {
    if (!this.edges.has(from)) {
      this.edges.set(from, []);
    }
    const expr = guardMeta?.expr || (guardFn ? guardFn.toString() : 'true');
    const resolvedGuard = typeof guardFn === 'function' ? guardFn : this.compileGuardExpression(expr);

    const list = this.edges.get(from) || [];
    const duplicate = list.find((e) => e.to === to && e.guardExpr === expr);
    if (duplicate) {
      return this;
    }

    let safePriority = priority;
    if (guardMeta?.preemptive) {
      for (const existingEdge of list) {
        if (existingEdge.priority >= safePriority) {
          existingEdge.priority += 1;
        }
      }
    } else {
      const existingPriorities = new Set(list.map((e) => e.priority));
      while (existingPriorities.has(safePriority)) {
        safePriority += 1;
      }
    }

    list.push({
      from,
      to,
      guard: resolvedGuard,
      priority: safePriority,
      guardExpr: expr,
      intervals: guardMeta?.intervals ? structuredClone(guardMeta.intervals) : {}
    });
    list.sort((a, b) => a.priority - b.priority);
    this.edges.set(from, list);
    return this;
  }

  /**
   * @param {string} centerNodeId
   * @param {number} [alphaPpr]
   * @returns {Map<string, number>}
   */
  getCausalMarkovBlanket(centerNodeId, alphaPpr = 0.15) {
    const center = this.nodes.get(centerNodeId);
    if (!center) return new Map();

    const centerReads = new Set(center.reads);
    const centerWrites = new Set(center.writes);
    /** @type {Set<string>} */
    const parents = new Set();
    /** @type {Set<string>} */
    const children = new Set();
    /** @type {Set<string>} */
    const childReads = new Set();

    for (const [id, node] of this.nodes.entries()) {
      if (id === centerNodeId) continue;
      if (node.writes.some((k) => centerReads.has(k))) parents.add(id);
      if (node.reads.some((k) => centerWrites.has(k))) {
        children.add(id);
        node.reads.forEach((k) => childReads.add(k));
      }
    }

    for (const [u, edgeList] of this.edges.entries()) {
      if (u === centerNodeId) {
        edgeList.forEach((e) => children.add(e.to));
      } else if (edgeList.some((e) => e.to === centerNodeId)) {
        parents.add(u);
      }
    }

    /** @type {Set<string>} */
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
      const dist = id === centerNodeId ? 0 : parents.has(id) || children.has(id) ? 1 : 2;
      scores.set(id, Number((alphaPpr * Math.pow(1 - alphaPpr, dist)).toFixed(4)));
    }
    return scores;
  }

  /**
   * Обратная бисимуляционная свертка эквивалентных ветвей в ромб (DAG Diamond)
   * с дедупликацией параллельных ребер и нормализацией приоритетов.
   */
  foldBisimilarBranches() {
    /** @type {Map<string, string>} */
    const signatureMap = new Map();
    for (const [id, node] of Array.from(this.nodes.entries())) {
      if (id === this.startNode || this.goalNodes.has(id)) continue;
      const outTargets = (this.edges.get(id) || [])
        .map((e) => e.to)
        .sort()
        .join(',');
      const normalizedCode = (node.config?.sourceCode || node.config?.toolId || '')
        .replace(/(?:auto_)?solver_[a-zA-Z0-9_]+/g, 'solver_canonical')
        .trim();
      const sig = `${node.type}|R:${[...node.reads].sort()}|W:${[...node.writes].sort()}|fn:${normalizedCode || id}|out:${outTargets}`;

      if (signatureMap.has(sig)) {
        const canonicalId = signatureMap.get(sig) || id;
        for (const [fromId, edgeList] of this.edges.entries()) {
          for (const edge of edgeList) {
            if (edge.to === id) edge.to = canonicalId;
          }
          /** @type {Set<string>} */
          const seenTargetGuard = new Set();
          const deduped = edgeList.filter((e) => {
            const key = `${e.to}|${e.guardExpr}`;
            if (seenTargetGuard.has(key)) return false;
            seenTargetGuard.add(key);
            return true;
          });
          deduped.sort((a, b) => a.priority - b.priority);
          deduped.forEach((e, idx) => {
            e.priority = idx;
          });
          this.edges.set(fromId, deduped);
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
   * Fuses maximal linear chains of tool nodes into single 'fused' nodes.
   * @returns {number}
   */
  fuseLinearToolChains() {
    /** @type {Map<string, Array<{ from: string, edge: GraphEdgeSpec }>>} */
    const incoming = new Map();
    for (const [from, list] of this.edges.entries()) {
      for (const edge of list) {
        if (!incoming.has(edge.to)) incoming.set(edge.to, []);
        const bucket = incoming.get(edge.to);
        if (bucket) bucket.push({ from, edge });
      }
    }

    /** @param {string} id */
    const isFusableTool = (id) => {
      const node = this.nodes.get(id);
      if (!node || node.type !== 'tool') return false;
      if (id === this.startNode || this.goalNodes.has(id)) return false;
      if (typeof node.config?.sourceCode !== 'string') return false;
      return (this.edges.get(id) || []).length === 1;
    };

    /** @type {string[][]} */
    const chains = [];
    /** @type {Set<string>} */
    const used = new Set();
    for (const id of Array.from(this.nodes.keys()).sort()) {
      if (!isFusableTool(id) || used.has(id)) continue;
      const preds = incoming.get(id) || [];
      if (preds.length === 1 && isFusableTool(preds[0].from)) {
        const predOut = this.edges.get(preds[0].from) || [];
        if (predOut.length === 1 && predOut[0].to === id) continue;
      }

      const chain = [id];
      let current = id;
      while (true) {
        const out = this.edges.get(current) || [];
        if (out.length !== 1) break;
        const next = out[0].to;
        const nextIn = incoming.get(next) || [];
        if (!isFusableTool(next) || nextIn.length !== 1 || nextIn[0].from !== current) break;
        if (chain.includes(next)) break;
        chain.push(next);
        current = next;
      }
      if (chain.length >= 2 && !chain.some((cid) => used.has(cid))) {
        chains.push(chain);
        chain.forEach((cid) => used.add(cid));
      }
    }

    let fusedCount = 0;
    for (const chain of chains) {
      const first = chain[0];
      const last = chain[chain.length - 1];
      const firstNode = this.nodes.get(first);
      const lastNode = this.nodes.get(last);
      if (!firstNode || !lastNode) continue;

      let valid = true;
      for (let i = 0; i < chain.length - 1; i++) {
        const out = this.edges.get(chain[i]) || [];
        if (out.length !== 1 || out[0].to !== chain[i + 1]) {
          valid = false;
          break;
        }
      }
      const lastOut = this.edges.get(last) || [];
      if (!valid || lastOut.length !== 1) continue;

      const fusedId = `fused_${first}_${last}`;
      if (this.nodes.has(fusedId)) continue;

      const incomingEdges = incoming.get(first) || [];
      const successorEdge = lastOut[0];
      const reads = Array.from(new Set(firstNode.reads));
      const writes = Array.from(new Set(lastNode.writes));

      for (const cid of chain) {
        this.nodes.delete(cid);
        this.edges.delete(cid);
      }
      for (const cid of chain) {
        for (const [from, list] of this.edges.entries()) {
          if (chain.includes(from)) continue;
          this.edges.set(
            from,
            list.filter((e) => e.to !== cid)
          );
        }
      }

      this.addNode({
        id: fusedId,
        type: 'fused',
        reads,
        writes,
        config: { chain: [...chain] }
      });

      for (const { from, edge } of incomingEdges) {
        if (chain.includes(from)) continue;
        this.addEdge(from, fusedId, edge.guard, edge.priority, {
          expr: edge.guardExpr,
          intervals: edge.intervals
        });
      }
      this.addEdge(fusedId, successorEdge.to, successorEdge.guard, successorEdge.priority, {
        expr: successorEdge.guardExpr,
        intervals: successorEdge.intervals
      });
      fusedCount++;
    }
    return fusedCount;
  }

  /**
   * Removes non-goal nodes unreachable from start or unable to reach a goal.
   * @returns {number}
   */
  pruneDeadBranches() {
    if (this.goalNodes.size === 0 || !this.startNode) return 0;

    /** @type {Set<string>} */
    const reachable = new Set();
    /** @type {string[]} */
    const stack = [this.startNode];
    while (stack.length > 0) {
      const id = stack.pop();
      if (!id || reachable.has(id)) continue;
      reachable.add(id);
      for (const edge of this.edges.get(id) || []) stack.push(edge.to);
    }

    /** @type {Map<string, string[]>} */
    const reverse = new Map();
    for (const [from, list] of this.edges.entries()) {
      for (const edge of list) {
        if (!reverse.has(edge.to)) reverse.set(edge.to, []);
        const bucket = reverse.get(edge.to);
        if (bucket) bucket.push(from);
      }
    }

    /** @type {Set<string>} */
    const canReachGoal = new Set();
    /** @type {string[]} */
    const queue = Array.from(this.goalNodes);
    while (queue.length > 0) {
      const id = queue.pop();
      if (!id || canReachGoal.has(id)) continue;
      canReachGoal.add(id);
      for (const from of reverse.get(id) || []) queue.push(from);
    }

    /** @type {string[]} */
    const dead = [];
    for (const id of this.nodes.keys()) {
      if (id === this.startNode || this.goalNodes.has(id)) continue;
      if (!reachable.has(id) || !canReachGoal.has(id)) dead.push(id);
    }

    for (const id of dead) {
      this.nodes.delete(id);
      this.edges.delete(id);
      for (const [from, list] of this.edges.entries()) {
        if (list.some((e) => e.to === id)) {
          this.edges.set(
            from,
            list.filter((e) => e.to !== id)
          );
        }
      }
    }
    return dead.length;
  }

  computeStructuralHash() {
    const payload = JSON.stringify({
      nodes: Array.from(this.nodes.entries())
        .map(([id, n]) => [id, { type: n.type, r: n.reads, w: n.writes }])
        .sort(),
      edges: Array.from(this.edges.entries()).map(([k, v]) => [
        k,
        v.map((e) => ({ to: e.to, p: e.priority, g: e.guardExpr }))
      ])
    });
    return crypto.createHash('sha256').update(payload).digest('hex');
  }

  /**
   * @returns {SerializedGraph}
   */
  toJSON() {
    return {
      id: this.id,
      startNode: this.startNode,
      goalNodes: Array.from(this.goalNodes),
      nodes: Array.from(this.nodes.values()).map((n) => ({
        id: n.id,
        type: n.type,
        reads: n.reads,
        writes: n.writes,
        idempotency: n.idempotency,
        hasBarrier: n.hasBarrier,
        modelTier: n.modelTier,
        config: {
          prompt: n.config?.prompt,
          sourceCode: n.config?.sourceCode,
          assertExpr: n.config?.assertExpr,
          toolId: n.config?.toolId,
          chain: n.config?.chain
        }
      })),
      edges: Array.from(this.edges.entries()).flatMap(([, list]) =>
        list.map((e) => ({
          from: e.from,
          to: e.to,
          priority: e.priority,
          guardExpr: e.guardExpr,
          intervals: e.intervals
        }))
      )
    };
  }

  /**
   * @param {SerializedGraph} data
   * @returns {DirectedGraph}
   */
  static fromJSON(data) {
    const g = new DirectedGraph(data.id || 'restored_graph');
    g.startNode = data.startNode;
    for (const goalId of data.goalNodes || []) {
      g.goalNodes.add(goalId);
    }
    for (const n of data.nodes || []) {
      g.addNode({
        id: n.id,
        type: n.type,
        reads: n.reads,
        writes: n.writes,
        idempotency: n.idempotency,
        hasBarrier: n.hasBarrier,
        modelTier: n.modelTier,
        config: {
          prompt: n.config?.prompt,
          sourceCode: n.config?.sourceCode,
          assertExpr: n.config?.assertExpr,
          toolId: n.config?.toolId,
          chain: n.config?.chain
        }
      });
    }
    for (const e of data.edges || []) {
      g.addEdge(e.from, e.to, g.compileGuardExpression(e.guardExpr), e.priority, {
        expr: e.guardExpr,
        intervals: e.intervals
      });
    }
    return g;
  }

  /**
   * @returns {DirectedGraph}
   */
  clone() {
    const copy = new DirectedGraph(this.id);
    copy.startNode = this.startNode;
    this.goalNodes.forEach((g) => copy.goalNodes.add(g));

    for (const [, node] of this.nodes) {
      /** @type {Record<string, any>} */
      const clonedConfig = {};
      if (node.config) {
        for (const [k, val] of Object.entries(node.config)) {
          clonedConfig[k] = typeof val === 'function' ? val : structuredClone(val);
        }
      }
      copy.addNode({
        ...node,
        reads: [...node.reads],
        writes: [...node.writes],
        inputSchemas: structuredClone(node.inputSchemas || {}),
        outputSchemas: structuredClone(node.outputSchemas || {}),
        config: clonedConfig
      });
    }

    for (const [from, edgeList] of this.edges) {
      for (const e of edgeList) {
        copy.addEdge(from, e.to, e.guard, e.priority, {
          expr: e.guardExpr,
          intervals: structuredClone(e.intervals || {})
        });
      }
    }
    return copy;
  }
}
