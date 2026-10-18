// @ts-check
import fs from 'node:fs';
import fsPromises from 'node:fs/promises';
import path from 'node:path';

/**
 * @typedef {import('../../types/index.js').SpectraConfig} SpectraConfig
 * @typedef {import('../../types/index.js').CredentialsOverrides} CredentialsOverrides
 */

/**
 * Загружает переменные из файла .env в process.env, если они еще не заданы в окружении.
 * @param {string} [envFilePath]
 */
export function loadDotEnvIfPresent(envFilePath = path.resolve(process.cwd(), '.env')) {
  if (!fs.existsSync(envFilePath)) return;
  if (typeof process.loadEnvFile === 'function') {
    try {
      process.loadEnvFile(envFilePath);
      return;
    } catch {
      // Fallback для ранних минорных версий Node 20.x
    }
  }
  const content = fs.readFileSync(envFilePath, 'utf-8');
  for (const rawLine of content.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;
    const eqIdx = line.indexOf('=');
    if (eqIdx <= 0) continue;
    const key = line.slice(0, eqIdx).trim();
    const val = line.slice(eqIdx + 1).trim().replace(/^['"]|['"]$/g, '');
    if (!(key in process.env)) {
      process.env[key] = val;
    }
  }
}

/**
 * Разбирает CLI-флаги вида --api-key, --base-url, --model, --pg-url, --hf-token.
 * @param {string[]} argv
 * @returns {{ overrides: CredentialsOverrides; positional: string[]; flags: { verbose: boolean; freshGraph: boolean } }}
 */
export function parseCliArgs(argv) {
  /** @type {CredentialsOverrides} */
  const overrides = {};
  /** @type {string[]} */
  const positional = [];
  let verbose = false;
  let freshGraph = false;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--verbose' || arg === '-v') {
      verbose = true;
    } else if (arg === '--fresh-graph') {
      freshGraph = true;
    } else if (arg === '--api-key' && argv[i + 1]) {
      overrides.apiKey = argv[++i];
    } else if (arg === '--base-url' && argv[i + 1]) {
      overrides.baseUrl = argv[++i];
    } else if (arg === '--model' && argv[i + 1]) {
      overrides.frontierModel = argv[++i];
      overrides.lightModel = argv[i];
    } else if (arg === '--pg-url' && argv[i + 1]) {
      overrides.pgUrl = argv[++i];
    } else if (arg === '--hf-token' && argv[i + 1]) {
      overrides.hfToken = argv[++i];
    } else if (arg === '--laya-model-dir' && argv[i + 1]) {
      overrides.layaModelDir = argv[++i];
    } else {
      positional.push(arg);
    }
  }

  return { overrides, positional, flags: { verbose, freshGraph } };
}

/**
 * Собирает финальную конфигурацию Spectra из spectra.config.json, .env, process.env и прямых переопределений.
 * @param {CredentialsOverrides} [overrides]
 * @param {string} [configPath]
 * @returns {Promise<SpectraConfig>}
 */
export async function resolveSpectraConfig(
  overrides = {},
  configPath = path.resolve(process.cwd(), 'spectra.config.json')
) {
  loadDotEnvIfPresent();

  /** @type {Partial<SpectraConfig>} */
  let fileConfig = {};
  try {
    const raw = await fsPromises.readFile(configPath, 'utf-8');
    fileConfig = JSON.parse(raw);
  } catch {
    fileConfig = {};
  }

  const apiKey =
    overrides.apiKey ||
    process.env.SPECTRA_LLM_API_KEY ||
    process.env.OPENAI_API_KEY ||
    fileConfig.system2?.apiKey ||
    '';

  const baseUrl =
    overrides.baseUrl ||
    process.env.SPECTRA_LLM_BASE_URL ||
    process.env.OPENAI_BASE_URL ||
    fileConfig.system2?.baseUrl ||
    undefined;

  const frontierModel =
    overrides.frontierModel ||
    process.env.SPECTRA_LLM_FRONTIER_MODEL ||
    fileConfig.system2?.frontierModel ||
    'gpt-4o-mini';

  const lightModel =
    overrides.lightModel ||
    process.env.SPECTRA_LLM_LIGHT_MODEL ||
    fileConfig.system2?.lightModel ||
    frontierModel;

  const pgUrl =
    overrides.pgUrl ||
    process.env.SPECTRA_PG_URL ||
    process.env.DATABASE_URL ||
    fileConfig.database?.connectionString ||
    'postgres://spectra:spectra_pass@localhost:5432/spectra_db';

  const hfToken =
    overrides.hfToken ||
    process.env.HF_TOKEN ||
    fileConfig.system1?.hfToken ||
    fileConfig.embeddings?.hfToken ||
    undefined;

  const layaCacheDir =
    overrides.layaCacheDir ||
    process.env.LAYA_CACHE ||
    fileConfig.system1?.cacheDir ||
    './.spectra/laya-cache';

  const layaModelDir =
    overrides.layaModelDir ||
    process.env.LAYA_MODEL_DIR ||
    fileConfig.system1?.modelDir ||
    undefined;

  return {
    systemName: fileConfig.systemName || 'spectra-autonomous-core',
    logging: {
      level: fileConfig.logging?.level || 'info',
      traceStateDiffs: fileConfig.logging?.traceStateDiffs ?? true,
      traceGuards: fileConfig.logging?.traceGuards ?? true,
      jsonlFilePath: fileConfig.logging?.jsonlFilePath || '.spectra/logs/session.jsonl'
    },
    database: {
      connectionString: pgUrl,
      embeddingDimension: fileConfig.database?.embeddingDimension || 384
    },
    embeddings: {
      provider: 'transformers.js',
      modelName: fileConfig.embeddings?.modelName || 'Xenova/multilingual-e5-small',
      cacheDir: fileConfig.embeddings?.cacheDir || './.spectra/models',
      hfToken,
      modalityWeights: fileConfig.embeddings?.modalityWeights || [0.5, 0.3, 0.2]
    },
    system1: {
      provider: '@receptron/laya',
      cacheDir: layaCacheDir,
      modelDir: layaModelDir,
      hfToken,
      temperaturePlatt: fileConfig.system1?.temperaturePlatt ?? 1.0,
      maxDrift: fileConfig.system1?.maxDrift ?? 0.65,
      minCoherence: fileConfig.system1?.minCoherence ?? 0.35,
      minApproveProb: fileConfig.system1?.minApproveProb ?? 0.40
    },
    system2: {
      provider: 'openai',
      apiKey,
      baseUrl,
      frontierModel,
      lightModel
    },
    runtime: {
      hardStepLimit: fileConfig.runtime?.hardStepLimit ?? 50,
      stateHashLoopThreshold: fileConfig.runtime?.stateHashLoopThreshold ?? 2,
      pprTeleportAlpha: fileConfig.runtime?.pprTeleportAlpha ?? 0.15
    }
  };
}

/**
 * Проверяет наличие учетных данных для подключения к LLM и выбрасывает понятную ошибку при их отсутствии.
 * @param {SpectraConfig['system2']} sys2Config
 */
export function assertValidLlmCredentials(sys2Config) {
  if (!sys2Config.apiKey || sys2Config.apiKey.trim().length === 0) {
    throw new Error(
      'Не заданы учетные данные для подключения к LLM (System-2). ' +
        'Передайте их через переменную окружения SPECTRA_LLM_API_KEY (или OPENAI_API_KEY), ' +
        'файл .env, флаг CLI --api-key <key> (опционально --base-url <url> --model <name>) ' +
        'или выполните настройку через `npx spectra init`.'
    );
  }
}
