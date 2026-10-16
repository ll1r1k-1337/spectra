import type {
  GraphNodeSpec,
  GraphEdgeSpec,
  StateObject,
  VerificationReport,
  SessionRunResult
} from './domain.js';

export interface IDirectedGraph {
  id: string;
  nodes: Map<string, GraphNodeSpec>;
  edges: Map<string, GraphEdgeSpec[]>;
  startNode: string | null;
  goalNodes: Set<string>;

  addNode(node: GraphNodeSpec): this;
  addEdge(
    from: string,
    to: string,
    priority?: number,
    guard?: (state: StateObject) => boolean,
    guardExpr?: string,
    intervals?: Record<string, [number, number]>
  ): this;
  getCausalMarkovBlanket(centerNodeId: string, alphaPpr?: number): Map<string, number>;
  computeStructuralHash(): string;
  clone(): IDirectedGraph;
}

export interface IGraphDeterminismVerifier {
  rhoMax: number;
  lMax: number;
  lyapunovCreditBank: number;

  verify(graph: IDirectedGraph): VerificationReport;
  verifyAmortizedLyapunovStability(
    prevEnergy: number,
    nextEnergy: number,
    deltaMin?: number,
    maxBank?: number
  ): {
    accepted: boolean;
    mode: 'STRICT_DISSIPATION' | 'PROVISIONAL_INVESTMENT' | 'REJECTED_BY_LYAPUNOV';
    deltaV: number;
    bank: number;
  };
}

export interface IToolRegistry {
  tools: Map<string, any>;
  register(toolSpec: {
    id: string;
    reads: string[];
    writes: string[];
    sourceCode: string;
    version?: number;
    hasIrreversibleBarrier?: boolean;
    regressionPool?: Array<{ input: StateObject; expectedOutput: StateObject }>;
  }): void;
  executeSandboxed(codeStr: string, stateInput: StateObject, timeoutMs?: number): StateObject;
}

export interface ISpectraRuntime {
  graph: IDirectedGraph;
  toolRegistry: IToolRegistry;
  verifier: IGraphDeterminismVerifier;
  ontology: any;
  runSession(initialState?: StateObject): Promise<SessionRunResult>;
}
