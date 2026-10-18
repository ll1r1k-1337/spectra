// @ts-check

/**
 * @typedef {import('../../types/index.js').IGraphDeterminismVerifier} IGraphDeterminismVerifier
 * @typedef {import('../../types/index.js').IDirectedGraph} IDirectedGraph
 * @typedef {import('../../types/index.js').VerificationReport} VerificationReport
 */

/**
 * @implements {IGraphDeterminismVerifier}
 */
export class GraphDeterminismVerifier {
  /**
   * @param {{ rhoMax?: number; lMax?: number; lyapunovCreditBankMax?: number }} [config]
   */
  constructor(config = {}) {
    this.rhoMax = config.rhoMax ?? 1.8;
    this.lMax = config.lMax ?? 5;
    this.weights = { l1: 2.0, l2: 10.0, l3: 1.5, l4: 25.0 };
    this.lyapunovCreditBank = config.lyapunovCreditBankMax ?? 15.0;
  }

  /**
   * @param {IDirectedGraph} graph
   * @returns {VerificationReport}
   */
  verify(graph) {
    const V = graph.nodes.size;
    let E = 0;
    let priorityCollisions = 0;

    for (const [, list] of graph.edges) {
      E += list.length;
      const seenPriorities = new Set();
      for (const edge of list) {
        if (seenPriorities.has(edge.priority)) priorityCollisions++;
        seenPriorities.add(edge.priority);
      }
    }

    const reachable = this._bfsReachable(graph, graph.startNode);
    const coReachable = this._reverseBfsCoReachable(graph, graph.goalNodes);
    const deadNodes = Array.from(reachable).filter((n) => !coReachable.has(n));
    const unsatNodes = this._checkTransitivePathSatisfiability(graph, reachable, coReachable);
    const sccs = this._tarjanSCC(graph, reachable);
    /** @type {string[][]} */
    const trappingCycles = [];
    let cycleExcess = 0;

    for (const comp of sccs) {
      const isCyclic =
        comp.length > 1 ||
        (comp.length === 1 && (graph.edges.get(comp[0]) || []).some((e) => e.to === comp[0]));
      if (isCyclic) {
        if (comp.length > this.lMax) cycleExcess += comp.length - this.lMax;
        const compSet = new Set(comp);
        const hasEscape = comp.some((u) =>
          (graph.edges.get(u) || []).some((e) => !compSet.has(e.to) && coReachable.has(e.to))
        );
        if (!hasEscape) trappingCycles.push(comp);
      }
    }

    const density = V > 0 ? E / V : 0;
    const omega =
      this.weights.l1 * Math.max(0, density - this.rhoMax) +
      this.weights.l2 * priorityCollisions +
      this.weights.l3 * cycleExcess +
      this.weights.l4 * (deadNodes.length + unsatNodes.length);

    const passed =
      priorityCollisions === 0 &&
      deadNodes.length === 0 &&
      unsatNodes.length === 0 &&
      trappingCycles.length === 0;

    return {
      status: passed ? 'PASS' : 'HARD_REJECT',
      omegaPenalty: Number(omega.toFixed(4)),
      diagnostics: {
        vertexCount: V,
        edgeCount: E,
        density: Number(density.toFixed(3)),
        priorityCollisions,
        deadNodes,
        unsatNodes,
        trappingCycles
      }
    };
  }

  /**
   * @param {IDirectedGraph} graph
   * @param {Set<string>} reachable
   * @param {Set<string>} coReachable
   * @returns {string[]}
   */
  _checkTransitivePathSatisfiability(graph, reachable, coReachable) {
    /** @type {Map<string, Record<string, [number, number]>>} */
    const abstractState = new Map();
    for (const id of reachable) abstractState.set(id, {});

    const maxPasses = reachable.size + 1;
    for (let pass = 0; pass < maxPasses; pass++) {
      let changed = false;
      for (const uId of reachable) {
        const uNode = graph.nodes.get(uId);
        const writtenByU = new Set(uNode?.writes || []);
        const uIntervals = abstractState.get(uId) || {};

        for (const edge of graph.edges.get(uId) || []) {
          if (!reachable.has(edge.to)) continue;
          /** @type {Record<string, [number, number]>} */
          const transferred = {};
          let edgeFeasible = true;

          const allKeys = new Set([
            ...Object.keys(uIntervals),
            ...Object.keys(edge.intervals || {})
          ]);

          for (const k of allKeys) {
            const baseRange = writtenByU.has(k) ? [-Infinity, Infinity] : (uIntervals[k] || [-Infinity, Infinity]);
            const guardRange = edge.intervals?.[k] || [-Infinity, Infinity];
            const low = Math.max(baseRange[0], guardRange[0]);
            const high = Math.min(baseRange[1], guardRange[1]);
            if (low > high) {
              edgeFeasible = false;
              break;
            }
            transferred[k] = [low, high];
          }

          if (edgeFeasible) {
            const targetState = abstractState.get(edge.to) || {};
            for (const [k, [low, high]] of Object.entries(transferred)) {
              if (!targetState[k]) {
                targetState[k] = [low, high];
                changed = true;
              } else {
                const joinedLow = Math.min(targetState[k][0], low);
                const joinedHigh = Math.max(targetState[k][1], high);
                if (joinedLow !== targetState[k][0] || joinedHigh !== targetState[k][1]) {
                  targetState[k] = [joinedLow, joinedHigh];
                  changed = true;
                }
              }
            }
            abstractState.set(edge.to, targetState);
          }
        }
      }
      if (!changed) break;
    }

    /** @type {string[]} */
    const unsat = [];
    for (const wId of reachable) {
      if (graph.goalNodes.has(wId)) continue;
      const wNode = graph.nodes.get(wId);
      const writtenByW = new Set(wNode?.writes || []);
      const wIntervals = abstractState.get(wId) || {};
      const outgoing = (graph.edges.get(wId) || []).filter((e) => coReachable.has(e.to));
      if (outgoing.length === 0) continue;

      const hasFeasibleOut = outgoing.some((eOut) => {
        if (!eOut.intervals) return true;
        for (const [k, [gMin, gMax]] of Object.entries(eOut.intervals)) {
          const inRange = writtenByW.has(k) ? [-Infinity, Infinity] : (wIntervals[k] || [-Infinity, Infinity]);
          if (Math.max(inRange[0], gMin) > Math.min(inRange[1], gMax)) return false;
        }
        return true;
      });
      if (!hasFeasibleOut) unsat.push(wId);
    }
    return unsat;
  }

  /**
   * @param {IDirectedGraph} graph
   * @param {string | null} start
   * @returns {Set<string>}
   */
  _bfsReachable(graph, start) {
    /** @type {Set<string>} */
    const visited = new Set();
    if (!start) return visited;
    const queue = [start];
    while (queue.length > 0) {
      const curr = queue.shift();
      if (curr && !visited.has(curr)) {
        visited.add(curr);
        for (const e of graph.edges.get(curr) || []) queue.push(e.to);
      }
    }
    return visited;
  }

  /**
   * @param {IDirectedGraph} graph
   * @param {Set<string>} goals
   * @returns {Set<string>}
   */
  _reverseBfsCoReachable(graph, goals) {
    /** @type {Map<string, string[]>} */
    const rev = new Map();
    for (const [u, list] of graph.edges) {
      for (const e of list) {
        if (!rev.has(e.to)) rev.set(e.to, []);
        rev.get(e.to)?.push(u);
      }
    }
    /** @type {Set<string>} */
    const visited = new Set();
    const queue = Array.from(goals);
    while (queue.length > 0) {
      const curr = queue.shift();
      if (curr && !visited.has(curr)) {
        visited.add(curr);
        for (const prev of rev.get(curr) || []) queue.push(prev);
      }
    }
    return visited;
  }

  /**
   * @param {IDirectedGraph} graph
   * @param {Set<string>} subset
   * @returns {string[][]}
   */
  _tarjanSCC(graph, subset) {
    let index = 0;
    /** @type {Map<string, number>} */
    const indices = new Map();
    /** @type {Map<string, number>} */
    const lowlink = new Map();
    /** @type {string[]} */
    const stack = [];
    /** @type {Set<string>} */
    const onStack = new Set();
    /** @type {string[][]} */
    const sccs = [];

    /** @param {string} v */
    const strongConnect = (v) => {
      indices.set(v, index);
      lowlink.set(v, index);
      index++;
      stack.push(v);
      onStack.add(v);

      for (const e of graph.edges.get(v) || []) {
        if (!subset.has(e.to)) continue;
        if (!indices.has(e.to)) {
          strongConnect(e.to);
          lowlink.set(v, Math.min(lowlink.get(v) ?? 0, lowlink.get(e.to) ?? 0));
        } else if (onStack.has(e.to)) {
          lowlink.set(v, Math.min(lowlink.get(v) ?? 0, indices.get(e.to) ?? 0));
        }
      }

      if (lowlink.get(v) === indices.get(v)) {
        /** @type {string[]} */
        const comp = [];
        let w;
        do {
          w = stack.pop();
          if (w !== undefined) {
            onStack.delete(w);
            comp.push(w);
          }
        } while (w !== v && w !== undefined);
        sccs.push(comp);
      }
    };

    for (const v of subset) {
      if (!indices.has(v)) strongConnect(v);
    }
    return sccs;
  }
}
