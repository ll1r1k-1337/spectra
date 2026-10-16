import type {
  DeadlockIncident,
  MutationPlan,
  System1Verdict,
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
  init(): Promise<void>;
  evaluateMutation(
    deadlock: DeadlockIncident,
    mutationPlan: MutationPlan,
    goalDescription: string
  ): Promise<System1Verdict>;
}

export interface ILangChainSystem2Arbiter {
  config: SpectraConfig['system2'];
  executeLlmNode(
    promptTemplate: string,
    state: StateObject,
    writeKeys: string[],
    tier?: ModelTier
  ): Promise<StateObject>;
  synthesizeRecoveryMutation(
    deadlock: DeadlockIncident,
    goalDescription: string,
    existingNodeIds: string[],
    goalNodeId: string
  ): Promise<MutationPlan>;
}
