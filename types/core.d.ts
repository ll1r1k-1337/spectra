import type {
  GraphNodeSpec,
  GraphEdgeSpec,
  SerializedGraph,
  SerializedOntologyKey,
  StateObject,
  StateDiffEntry,
  VerificationReport,
  MutationPlan,
  DeadlockIncident,
  System1Verdict,
  SessionRunResult,
  LogLevel,
  PrecedentRecord
} from './domain.js';

export interface ISpectraLogger {
  sessionId: string;
  level: LogLevel;
  traceStateDiffs: boolean;
  traceGuards: boolean;

  banner(title: string, subtitle?: string): void;
  section(
    phase: 'FAST_LOOP' | 'DEADLOCK' | 'SLOW_LOOP' | 'SYSTEM_1' | 'SYSTEM_2' | 'VERIFIER' | 'ROLLBACK' | 'MEMORY' | 'SLEEP',
    title: string
  ): void;
  info(scope: string, message: string, meta?: Record<string, any>): void;
  debug(scope: string, message: string, meta?: Record<string, any>): void;
  warn(scope: string, message: string, meta?: Record<string, any>): void;
  error(scope: string, message: string, meta?: Record<string, any>): void;

  logStepExecution(params: {
    step: number;
    maxSteps: number;
    node: GraphNodeSpec;
    durationMs: number;
    stateBefore: StateObject;
    stateAfter: StateObject;
  }): void;

  logGuardEvaluation(
    nodeId: string,
    evaluations: Array<{ to: string; priority: number; guardExpr: string; matched: boolean }>
  ): void;

  logDeadlock(deadlock: DeadlockIncident, markovBlanket: Map<string, number>, stateHash: string): void;
  logSystem1Verdict(
    verdict: System1Verdict,
    thresholds: { maxDrift: number; minCoherence: number; minApproveProb: number }
  ): void;
  logVerifierReport(report: VerificationReport, graphHash: string): void;
  logTopologyAscii(graph: IDirectedGraph, activeNodeId?: string | null): void;
  computeStateDiff(before: StateObject, after: StateObject): StateDiffEntry[];
}

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
    guardFn?: (state: StateObject) => boolean,
    priority?: number,
    guardMeta?: { expr?: string; intervals?: Record<string, [number, number]>; preemptive?: boolean } | null
  ): this;
  compileGuardExpression(exprStr?: string): (state: StateObject) => boolean;
  getCausalMarkovBlanket(centerNodeId: string, alphaPpr?: number): Map<string, number>;
  foldBisimilarBranches(): this;
  computeStructuralHash(): string;
  toJSON(): SerializedGraph;
  clone(): IDirectedGraph;
}

export interface IGraphDeterminismVerifier {
  rhoMax: number;
  lMax: number;
  lyapunovCreditBank: number;
  verify(graph: IDirectedGraph): VerificationReport;
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
    sagaCompensatorCode?: string;
    regressionPool?: Array<{ input: StateObject; expectedOutput: StateObject }>;
  }): void;
  executeSandboxed(codeStr: string, stateInput: StateObject, timeoutMs?: number): StateObject;
  verifyAndAttachSagaCompensator(
    toolId: string,
    compensatorCodeStr: string,
    epsilonSaga?: number
  ): { verified: boolean; residualError: number; hasBarrier: boolean };
}

export interface IStateOntologyManager {
  keys: Map<string, { status: 'STAGING' | 'CANONICAL'; canonicalAlias?: string; epochsLived: number; mutualInfoGain: number }>;
  stagingTtlEpochs: number;
  registerStagingKey(keyName: string, mutualInfoGain?: number): void;
  findCanonical(keyName: string): string;
  mergeKeys(canonicalKey: string, synonymKey: string): string;
  runStagingGarbageCollection(thetaIg?: number): { promoted: string[]; evicted: string[] };
  toJSON(): SerializedOntologyKey[];
  loadFromJSON(entries: SerializedOntologyKey[]): void;
}

export interface ISpectraRuntime {
  graph: IDirectedGraph;
  toolRegistry: IToolRegistry;
  verifier: IGraphDeterminismVerifier;
  ontology: IStateOntologyManager;
  logger: ISpectraLogger;
  precedentMemory: PrecedentRecord[];
  runSession(initialState?: StateObject): Promise<SessionRunResult>;
  runConsolidationSleep(): Promise<{
    promotedKeys: string[];
    evictedKeys: string[];
    foldedNodes: number;
    omegaBefore: number;
    omegaAfter: number;
  }>;
}
