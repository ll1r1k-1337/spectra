// types/ai.d.ts
import type {
  DeadlockType,
  DeadlockIncident,
  MutationPlan,
  System1Verdict,
  PrecedentRecord,
  SerializedGraph,
  SerializedOntologyKey,
  GraphNodeSpec,
  StateObject,
  ModelTier,
  SpectraConfig
} from './domain.js';

export interface ILocalEmbeddingEngine {
  modelName: string;
  weights: [number, number, number];
  init(): Promise<this>;
  embedText(text: string): Promise<number[]>;
  computeIncidentVector(params: {
    errorText: string;
    markovBlanketSignature: string;
    stateSchemaSignature: string;
  }): Promise<number[]>;
}

export interface ILayaSystem1Arbiter {
  config: SpectraConfig['system1'];
  init(): Promise<this>;
  evaluateMutation(
    deadlock: DeadlockIncident,
    mutationPlan: MutationPlan,
    goalDescription: string
  ): Promise<System1Verdict>;
  close(): Promise<void>;
}

export interface ILangChainSystem2Arbiter {
  config: SpectraConfig['system2'];
  synthesizeRecoveryMutation(
    deadlock: DeadlockIncident,
    goalDescription: string,
    existingNodeIds: string[],
    goalNodeId: string
  ): Promise<MutationPlan>;
  executeLlmNode(node: GraphNodeSpec, state: StateObject, tier?: ModelTier): Promise<StateObject>;
}

export interface IPrecedentStore {
  init(): Promise<void>;
  loadAllPrecedents(): Promise<PrecedentRecord[]>;
  searchSimilarPrecedents(
    deadlockType: DeadlockType,
    queryVector: number[],
    minSimilarity?: number
  ): Promise<PrecedentRecord[]>;
  savePrecedent(record: PrecedentRecord, deadlock: DeadlockIncident): Promise<void>;
  saveGraphVersion(graphJson: SerializedGraph, structuralHash: string, omegaPenalty: number): Promise<void>;
  loadLatestGraph(): Promise<SerializedGraph | null>;
  saveOntologyKeys(keys: SerializedOntologyKey[]): Promise<void>;
  loadOntologyKeys(): Promise<SerializedOntologyKey[]>;
  clearAll(): Promise<void>;
  close(): Promise<void>;
}
