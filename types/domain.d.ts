export type NodeType = 'llm' | 'tool' | 'branch' | 'join' | 'check' | 'macro' | 'fused';
export type IdempotencyClass = 'pure' | 'idempotent' | 'non_idempotent';
export type ModelTier = 'local_slm' | 'light' | 'frontier';
export type LogLevel = 'debug' | 'info' | 'warn' | 'error';

export type DeadlockType =
  | 'NO_VALID_TRANSITION'
  | 'STATE_HASH_LOOP'
  | 'CHECK_FAILED'
  | 'TOOL_EXECUTION_ERROR'
  | 'HARD_LIMIT_EXCEEDED';

export interface StateObject {
  [key: string]: any;
  goal?: string;
  artifacts?: Artifact[];
  patches?: ArtifactPatch[];
  workspaceFiles?: WorkspaceFile[];
  persistedFiles?: string[];
  syntaxErrors?: string[];
}

export interface StateDiffEntry {
  key: string;
  change: 'ADDED' | 'MODIFIED' | 'REMOVED';
  before: any;
  after: any;
}

export interface Artifact {
  name: string;
  content: string;
}

export interface ArtifactPatch {
  file: string;
  search: string;
  replace: string;
}

export interface WorkspaceFile {
  path: string;
  size: number;
  ext: string;
  content?: string;
  contentTruncated?: boolean;
}

export interface RegisteredTool {
  id: string;
  reads: string[];
  writes: string[];
  sourceCode: string;
  version?: number;
  hasIrreversibleBarrier?: boolean;
  sagaCompensatorCode?: string;
  regressionPool?: Array<{ input: StateObject; expectedOutput: StateObject }>;
}

export interface MacroPattern {
  id?: string;
  name: string;
  sourceNodeIds: string[];
  targetNodeId?: string;
  patternJson: Record<string, any>;
  support?: number;
  successRate?: number;
}

export interface JsonSchemaProperty {
  type: 'string' | 'number' | 'integer' | 'boolean' | 'object' | 'array';
  nullable?: boolean;
  minimum?: number;
  maximum?: number;
}

export interface GraphNodeSpec {
  id: string;
  type: NodeType;
  reads: string[];
  writes: string[];
  inputSchemas?: Record<string, JsonSchemaProperty>;
  outputSchemas?: Record<string, JsonSchemaProperty>;
  idempotency?: IdempotencyClass;
  hasBarrier?: boolean;
  resourceRps?: number;
  modelTier?: ModelTier;
  config?: {
    prompt?: string;
    toolId?: string;
    sourceCode?: string;
    assertExpr?: string;
    assertFn?: (state: StateObject) => boolean;
    handler?: (state: StateObject) => Promise<StateObject> | StateObject;
    llmHandler?: (state: StateObject, tier: ModelTier) => Promise<StateObject>;
    [key: string]: any;
  };
}

export interface GraphEdgeSpec {
  from: string;
  to: string;
  priority: number;
  guard: (state: StateObject) => boolean;
  guardExpr: string;
  intervals?: Record<string, [number, number]>;
}

export interface SerializedGraph {
  id: string;
  startNode: string | null;
  goalNodes: string[];
  nodes: Array<{
    id: string;
    type: NodeType;
    reads: string[];
    writes: string[];
    idempotency?: IdempotencyClass;
    hasBarrier?: boolean;
    modelTier?: ModelTier;
    config?: {
      prompt?: string;
      sourceCode?: string;
      assertExpr?: string;
      toolId?: string;
      chain?: string[];
    };
  }>;
  edges: Array<{
    from: string;
    to: string;
    priority: number;
    guardExpr: string;
    intervals?: Record<string, [number, number]>;
  }>;
}

export interface SerializedOntologyKey {
  keyName: string;
  status: 'STAGING' | 'CANONICAL';
  canonicalAlias?: string;
  epochsLived: number;
  mutualInfoGain: number;
}

export interface DeadlockIncident {
  type: DeadlockType;
  nodeId: string;
  lastValidNodeId?: string | null;
  step: number;
  error?: string;
  stateSnapshot: StateObject;
}

export interface MutationPlan {
  operator: 'addEdge' | 'addNodeAndBridge' | 'patchToolAST' | 'registerTool' | 'splitNode';
  from: string;
  to: string;
  priority?: number;
  guardFn?: (state: StateObject) => boolean;
  guardExpr?: string;
  intervals?: Record<string, [number, number]>;
  node?: GraphNodeSpec;
  toolCode?: string;
  reasoning?: string;
}

export interface PrecedentRecord {
  precedentId?: string;
  deadlockType: DeadlockType;
  incidentVector?: number[];
  similarity?: number;
  anchorReads: string[];
  anchorWrites: string[];
  mutationPlan: MutationPlan;
  successCount?: number;
  failureCount?: number;
  confidenceLcb?: number;
  isAntipattern: boolean;
}

export interface System1Verdict {
  driftRisk: number;
  coherence: number;
  approveProb: number;
  approved: boolean;
  rollbackAction: 'keep' | 'rollback_last' | 'rollback_session';
  selectedPrecedentId?: string | null;
  latencyMs?: number;
}

export interface VerificationReport {
  status: 'PASS' | 'HARD_REJECT';
  omegaPenalty: number;
  diagnostics: {
    vertexCount: number;
    edgeCount: number;
    density: number;
    priorityCollisions: number;
    deadNodes: string[];
    unsatNodes: string[];
    trappingCycles: string[][];
  };
}

export interface SessionRunResult {
  sessionId: string;
  goalReached: boolean;
  reason?: string;
  steps: number;
  rollbacksCount: number;
  finalState: StateObject;
  omegaPenalty?: number;
  durationMs: number;
  llmCalls: number;
  appliedMutations: Array<{
    deadlock: DeadlockIncident;
    mutationPlan: MutationPlan;
    omegaPenalty: number;
    source: 'MEMORY_GROUNDED' | 'SYSTEM2_SYNTHESIZED';
  }>;
}

export interface CredentialsOverrides {
  apiKey?: string;
  baseUrl?: string;
  frontierModel?: string;
  lightModel?: string;
  pgUrl?: string;
  hfToken?: string;
  layaCacheDir?: string;
  layaModelDir?: string;
}

export interface SpectraConfig {
  systemName: string;
  logging: {
    level: LogLevel;
    traceStateDiffs: boolean;
    traceGuards: boolean;
    jsonlFilePath: string;
  };
  database: {
    connectionString: string;
    embeddingDimension: number;
  };
  embeddings: {
    provider: 'transformers.js';
    modelName: string;
    cacheDir: string;
    hfToken?: string;
    modalityWeights: [number, number, number];
  };
  system1: {
    provider: '@receptron/laya';
    cacheDir?: string;
    modelDir?: string;
    hfToken?: string;
    temperaturePlatt: number;
    maxDrift: number;
    minCoherence: number;
    minApproveProb: number;
  };
  system2: {
    provider: 'openai';
    apiKey: string;
    baseUrl?: string;
    frontierModel: string;
    lightModel: string;
  };
  runtime: {
    hardStepLimit: number;
    stateHashLoopThreshold: number;
    pprTeleportAlpha: number;
  };
}
