// @ts-check
import util from 'node:util';

/**
 * @typedef {'debug' | 'info' | 'warn' | 'error' | 'silent'} LogLevel
 */

/** @type {Record<LogLevel, number>} */
export const LOG_LEVELS = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
  silent: 4
};

/** @type {Record<number, LogLevel>} */
export const LEVEL_NAMES = {
  0: 'debug',
  1: 'info',
  2: 'warn',
  3: 'error',
  4: 'silent'
};

const ANSI = {
  reset: '\x1b[0m',
  dim: '\x1b[90m',
  cyan: '\x1b[36m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  red: '\x1b[31m',
  magenta: '\x1b[35m',
  bold: '\x1b[1m'
};

/**
 * @param {Date} [date]
 * @returns {string}
 */
function formatTimestamp(date = new Date()) {
  const pad = (/** @type {number} */ n, z = 2) => String(n).padStart(z, '0');
  const h = pad(date.getHours());
  const m = pad(date.getMinutes());
  const s = pad(date.getSeconds());
  const ms = pad(date.getMilliseconds(), 3);
  return `${h}:${m}:${s}.${ms}`;
}

/**
 * @param {LogLevel} level
 * @param {boolean} useColors
 * @returns {string}
 */
function getLevelBadge(level, useColors) {
  if (!useColors) {
    switch (level) {
      case 'debug': return '[DEBUG]';
      case 'info': return '[INFO] ';
      case 'warn': return '[WARN] ';
      case 'error': return '[ERROR]';
      default: return `[${level.toUpperCase()}]`;
    }
  }
  switch (level) {
    case 'debug': return `${ANSI.cyan}[DEBUG]${ANSI.reset}`;
    case 'info': return `${ANSI.green}[INFO] ${ANSI.reset}`;
    case 'warn': return `${ANSI.yellow}[WARN] ${ANSI.reset}`;
    case 'error': return `${ANSI.red}${ANSI.bold}[ERROR]${ANSI.reset}`;
    default: return `[${level.toUpperCase()}]`;
  }
}

/**
 * @param {string} tag
 * @param {boolean} useColors
 * @returns {string}
 */
function formatTag(tag, useColors) {
  if (!useColors) {
    return `[${tag}]`;
  }
  return `${ANSI.magenta}[${tag}]${ANSI.reset}`;
}

/**
 * @param {any} meta
 * @param {boolean} useColors
 * @returns {string}
 */
function formatMeta(meta, useColors) {
  if (meta === undefined) return '';
  if (meta instanceof Error) {
    return `\n${meta.stack || meta.message}`;
  }
  if (typeof meta === 'object' && meta !== null) {
    try {
      return `\n${util.inspect(meta, { colors: useColors, depth: 5, compact: false })}`;
    } catch {
      return `\n${String(meta)}`;
    }
  }
  return ` ${meta}`;
}

/**
 * Scoped child logger that automatically prefixes a component tag.
 */
export class ChildLogger {
  /**
   * @param {Logger} parent
   * @param {string} tag
   */
  constructor(parent, tag) {
    /** @type {Logger} */
    this.parent = parent;
    /** @type {string} */
    this.tag = tag;
  }

  /**
   * @param {string} msg
   * @param {any} [meta]
   */
  debug(msg, meta) {
    this.parent.debug(this.tag, msg, meta);
  }

  /**
   * @param {string} msg
   * @param {any} [meta]
   */
  info(msg, meta) {
    this.parent.info(this.tag, msg, meta);
  }

  /**
   * @param {string} msg
   * @param {any} [meta]
   */
  warn(msg, meta) {
    this.parent.warn(this.tag, msg, meta);
  }

  /**
   * @param {string} msg
   * @param {any} [meta]
   */
  error(msg, meta) {
    this.parent.error(this.tag, msg, meta);
  }

  /**
   * @param {string} subTag
   * @returns {ChildLogger}
   */
  child(subTag) {
    return new ChildLogger(this.parent, `${this.tag}:${subTag}`);
  }
}

/**
 * Structured diagnostic logger supporting levels, ANSI colors, component tags, and metadata.
 */
export class Logger {
  /**
   * @param {LogLevel | number} [initialLevel]
   */
  constructor(initialLevel) {
    /** @type {number} */
    this._level = initialLevel !== undefined ? this._normalizeLevel(initialLevel) : this._resolveEnvLevel();
  }

  /**
   * @private
   * @returns {number}
   */
  _resolveEnvLevel() {
    const envLevel = process.env.SPECTRA_LOG_LEVEL;
    if (envLevel) {
      const lower = envLevel.toLowerCase();
      if (lower in LOG_LEVELS) {
        return LOG_LEVELS[/** @type {LogLevel} */ (lower)];
      }
      const num = Number(envLevel);
      if (!isNaN(num) && num >= 0 && num <= 4) {
        return num;
      }
    }
    const debug = process.env.DEBUG;
    if (debug && (debug === '1' || debug === 'true' || debug.includes('spectra'))) {
      return LOG_LEVELS.debug;
    }
    return LOG_LEVELS.info;
  }

  /**
   * @private
   * @param {LogLevel | number} level
   * @returns {number}
   */
  _normalizeLevel(level) {
    if (typeof level === 'number') {
      if (level >= 0 && level <= 4) return level;
      return LOG_LEVELS.info;
    }
    const lower = String(level).toLowerCase();
    if (lower in LOG_LEVELS) {
      return LOG_LEVELS[/** @type {LogLevel} */ (lower)];
    }
    return LOG_LEVELS.info;
  }

  /**
   * Updates current log level.
   * @param {LogLevel | number} level
   */
  setLevel(level) {
    this._level = this._normalizeLevel(level);
  }

  /**
   * Returns current active log level name.
   * @returns {LogLevel}
   */
  getLevel() {
    return LEVEL_NAMES[this._level] || 'info';
  }

  /**
   * Core logging handler.
   * @param {LogLevel} level
   * @param {string} tag
   * @param {any} [msg]
   * @param {any} [meta]
   */
  _log(level, tag, msg, meta) {
    const numericLevel = LOG_LEVELS[level];
    if (numericLevel < this._level || this._level === LOG_LEVELS.silent) {
      return;
    }

    const useColors = Boolean(process.stdout.isTTY && !process.env.NO_COLOR);
    const ts = formatTimestamp();
    const tsStr = useColors ? `${ANSI.dim}[${ts}]${ANSI.reset}` : `[${ts}]`;
    const levelBadge = getLevelBadge(level, useColors);
    const tagBadge = formatTag(tag, useColors);
    const metaStr = formatMeta(meta, useColors);
    const messageText = typeof msg === 'string' ? msg : util.inspect(msg, { colors: useColors, depth: 3 });

    const line = `${tsStr} ${levelBadge} ${tagBadge} ${messageText}${metaStr}\n`;

    if (level === 'error' || level === 'warn') {
      process.stderr.write(line);
    } else {
      process.stdout.write(line);
    }
  }

  /**
   * @param {string} tag
   * @param {any} [msg]
   * @param {any} [meta]
   */
  debug(tag, msg, meta) {
    if (msg === undefined && meta === undefined) {
      this._log('debug', 'Spectra', tag, undefined);
    } else {
      this._log('debug', tag, msg, meta);
    }
  }

  /**
   * @param {string} tag
   * @param {any} [msg]
   * @param {any} [meta]
   */
  info(tag, msg, meta) {
    if (msg === undefined && meta === undefined) {
      this._log('info', 'Spectra', tag, undefined);
    } else {
      this._log('info', tag, msg, meta);
    }
  }

  /**
   * @param {string} tag
   * @param {any} [msg]
   * @param {any} [meta]
   */
  warn(tag, msg, meta) {
    if (msg === undefined && meta === undefined) {
      this._log('warn', 'Spectra', tag, undefined);
    } else {
      this._log('warn', tag, msg, meta);
    }
  }

  /**
   * @param {string} tag
   * @param {any} [msg]
   * @param {any} [meta]
   */
  error(tag, msg, meta) {
    if (msg === undefined && meta === undefined) {
      this._log('error', 'Spectra', tag, undefined);
    } else {
      this._log('error', tag, msg, meta);
    }
  }

  /**
   * Creates a child logger with bound tag.
   * @param {string} tag
   * @returns {ChildLogger}
   */
  child(tag) {
    return new ChildLogger(this, tag);
  }
}

/**
 * Singleton logger instance.
 */
export const logger = new Logger();
