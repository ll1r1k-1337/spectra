export type NodeType = 'llm' | 'tool' | 'branch' | 'join' | 'check' | 'macro' | 'fused';
export type IdempotencyClass = 'pure' | 'idempotent' | 'non_idempotent';
export type ModelTier = 'local_slm' | 'light' | 'frontier';
export type DeadlockType =
  | 'NO_VALID_TRANSITION'
  | 'STATE_HASH_LOOP'
  | 'CHECK_FAILED'
  | 'TOOL_EXECUTION_ERROR'
  | 'HARD_LIMIT_EXCEEDED'
  | 'INVALID_GOAL_STATE';

export type StateObject = Record<string, any>;

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

export interface DeadlockIncident {
  type: DeadlockType;
  nodeId: string;
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

export interface System1Verdict {
  driftRisk: number;
  coherence: number;
  approveProb: number;
  approved: boolean;
  rollbackAction: 'keep' | 'rollback_last' | 'rollback_session';
  selectedPrecedentId?: string | null;
}

export interface VerificationReport {
  status: 'PASS' | 'HARD_REJECT';
  omegaPenalty: number;
  diagnostics: {
    priorityCollisions: number;
    deadNodes: string[];
    unsatNodes: string[];
    trappingCycles: string[][];
  };
}

export interface SessionRunResult {
  goalReached: boolean;
  reason?: string;
  steps: number;
  rollbacksCount: number;
  finalState: StateObject;
  omegaPenalty?: number;
  appliedMutations: Array<{
    deadlock: DeadlockIncident;
    mutationPlan: MutationPlan;
    omegaPenalty: number;
  }>;
}

export interface SpectraConfig {
  systemName: string;
  logLevel?: 'debug' | 'info' | 'warn' | 'error' | 'silent';
  database: {
    connectionString: string;
    embeddingDimension: number;
  };
  embeddings: {
    provider: 'transformers.js';
    modelName: string;
    cacheDir: string;
    modalityWeights: [number, number, number];
  };
  system1: {
    provider: 'laya-onnx';
    confidenceThreshold: number;
    driftTolerance: number;
  };
  system2: {
    provider: 'langchain';
    openaiApiKey?: string;
    baseURL?: string;
    modelName: string;
    lightModelName: string;
    temperature: number;
    maxTokens: number;
  };
  runtime: {
    maxSteps: number;
    rhoMax: number;
    lMax: number;
    initialLyapunovBank: number;
  };
}
