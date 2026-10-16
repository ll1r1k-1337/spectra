// @ts-check
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import fs from 'node:fs/promises';
import path from 'node:path';

/**
 * @typedef {import('../../types/index.js').SpectraConfig} SpectraConfig
 */

/**
 * Runs interactive CLI setup wizard and writes configuration to disk.
 * @param {string} [configPath]
 * @returns {Promise<SpectraConfig>}
 */
export async function runInteractiveWizard(configPath = 'spectra.config.json') {
  const rl = readline.createInterface({ input, output });
  console.log('\n============================================================');
  console.log('⚡ SPECTRA: Interactive Configuration Wizard');
  console.log('============================================================\n');

  try {
    const pgUri =
      (
        await rl.question(
          '1. PostgreSQL + pgvector URI [postgresql://spectra:spectra_password@localhost:5432/spectra_db]: '
        )
      ).trim() || 'postgresql://spectra:spectra_password@localhost:5432/spectra_db';

    const embModel =
      (
        await rl.question(
          '2. Local transformers.js embedding model [Xenova/multilingual-e5-small]: '
        )
      ).trim() || 'Xenova/multilingual-e5-small';

    const maxDriftStr = (
      await rl.question('3. System-1 drift risk threshold (0.05..0.50) [0.35]: ')
    ).trim();
    const maxDrift = Number.isFinite(Number(maxDriftStr)) && maxDriftStr !== '' ? Number(maxDriftStr) : 0.35;

    const apiKey = (await rl.question('4. OpenAI API Key for System-2 LangChain (Enter to skip): ')).trim();

    const baseURL = (
      await rl.question('5. OpenAI Base URL (e.g. https://api.openai.com/v1, OpenRouter, vLLM, or Enter for default): ')
    ).trim();

    const frontierModel =
      (await rl.question('6. Frontier LLM model [gpt-4o]: ')).trim() || 'gpt-4o';

    const lightModel =
      (await rl.question('7. Light LLM model [gpt-4o-mini]: ')).trim() || 'gpt-4o-mini';

    /** @type {SpectraConfig} */
    const config = {
      systemName: 'Spectra Autonomous Runtime',
      database: {
        connectionString: pgUri,
        embeddingDimension: 384
      },
      embeddings: {
        provider: 'transformers.js',
        modelName: embModel,
        cacheDir: './.cache/transformers',
        modalityWeights: [0.5, 0.3, 0.2]
      },
      system1: {
        provider: 'laya-onnx',
        confidenceThreshold: 0.7,
        driftTolerance: maxDrift
      },
      system2: {
        provider: 'langchain',
        openaiApiKey: apiKey || undefined,
        baseURL: baseURL || undefined,
        modelName: frontierModel,
        lightModelName: lightModel,
        temperature: 0.2,
        maxTokens: 2048
      },
      runtime: {
        maxSteps: 30,
        rhoMax: 1,
        lMax: 5,
        initialLyapunovBank: 15.0
      }
    };

    const targetFile = path.resolve(process.cwd(), configPath);
    await fs.mkdir(path.dirname(targetFile), { recursive: true });
    await fs.writeFile(targetFile, JSON.stringify(config, null, 2), 'utf-8');
    console.log(`\n✅ Configuration successfully written to ${targetFile}\n`);
    return config;
  } finally {
    rl.close();
  }
}
