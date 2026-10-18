// @ts-check
import readline from 'node:readline/promises';
import { stdin as input, stdout as output } from 'node:process';
import fs from 'node:fs/promises';
import path from 'node:path';

/**
 * @typedef {import('../../types/index.js').SpectraConfig} SpectraConfig
 */

/**
 * Запускает интерактивный опросник настройки Spectra (включая apiKey, baseUrl, модель и PostgreSQL)
 * и сохраняет spectra.config.json.
 * @param {string} [targetPath]
 * @returns {Promise<SpectraConfig>}
 */
export async function runInteractiveWizard(targetPath = path.resolve(process.cwd(), 'spectra.config.json')) {
  const rl = readline.createInterface({ input, output });
  console.log('\n╔══════════════════════════════════════════════════════════════════════╗');
  console.log('║ ⚙️  МАСТЕР ИНТЕРАКТИВНОЙ НАСТРОЙКИ ПРОТОТИПА SPECTRA                 ║');
  console.log('╚══════════════════════════════════════════════════════════════════════╝\n');

  const pgUrl =
    (await rl.question('1. Строка подключения PostgreSQL [postgres://spectra:spectra_pass@localhost:5432/spectra_db]: ')).trim() ||
    'postgres://spectra:spectra_pass@localhost:5432/spectra_db';

  const apiKey = (await rl.question('2. Ключ API для LLM System-2 (OPENAI_API_KEY / SPECTRA_LLM_API_KEY): ')).trim();

  const baseUrl =
    (await rl.question('3. Базовый URL LLM API [https://api.openai.com/v1]: ')).trim() ||
    'https://api.openai.com/v1';

  const frontierModel =
    (await rl.question('4. Модель для синтеза мутаций System-2 [gpt-4o-mini]: ')).trim() ||
    'gpt-4o-mini';

  const embModel =
    (await rl.question('5. Локальная модель эмбеддингов transformers.js [Xenova/multilingual-e5-small]: ')).trim() ||
    'Xenova/multilingual-e5-small';

  const hfToken = (await rl.question('6. Токен Hugging Face (Enter, если не требуется): ')).trim() || undefined;

  const logLevelInput = (await rl.question('7. Уровень детализации логов (debug/info/warn) [debug]: ')).trim() || 'debug';
  rl.close();

  /** @type {SpectraConfig} */
  const config = {
    systemName: 'spectra-autonomous-core',
    logging: {
      level: /** @type {'debug' | 'info' | 'warn' | 'error'} */ (
        ['debug', 'info', 'warn', 'error'].includes(logLevelInput) ? logLevelInput : 'debug'
      ),
      traceStateDiffs: true,
      traceGuards: true,
      jsonlFilePath: '.spectra/logs/session.jsonl'
    },
    database: {
      connectionString: pgUrl,
      embeddingDimension: 384
    },
    embeddings: {
      provider: 'transformers.js',
      modelName: embModel,
      cacheDir: './.spectra/models',
      hfToken,
      modalityWeights: [0.5, 0.3, 0.2]
    },
    system1: {
      provider: '@receptron/laya',
      cacheDir: './.spectra/laya-cache',
      hfToken,
      temperaturePlatt: 1.0,
      maxDrift: 0.65,
      minCoherence: 0.35,
      minApproveProb: 0.40
    },
    system2: {
      provider: 'openai',
      apiKey,
      baseUrl,
      frontierModel,
      lightModel: frontierModel
    },
    runtime: {
      hardStepLimit: 50,
      stateHashLoopThreshold: 2,
      pprTeleportAlpha: 0.15
    }
  };

  await fs.writeFile(targetPath, JSON.stringify(config, null, 2), 'utf-8');
  console.log(`\n✅ Конфигурация сохранена в ${targetPath}\n`);
  return config;
}
