// src/core/acceptance-gate.js
// @ts-check
// Acceptance gate: goal predicate expression + sync/async validator.
// No imports from bin/ (avoids a cycle); the expression is a self-contained
// arrow-function literal executed by DirectedGraph.compileGuardExpression
// with only `state` (and Boolean/String/Array/Object/Math) in the vm scope.

/**
 * @typedef {import('../../types/index.js').StateObject} StateObject
 * @typedef {import('../../types/index.js').Artifact} Artifact
 * @typedef {import('../../types/index.js').WorkspaceFile} WorkspaceFile
 */

/**
 * Self-contained goal predicate expression (string literal).
 * Checks: non-empty artifacts/name/content, persisted count, syntax errors,
 * local HTML src/href integrity, and destructive-overwrite guard.
 * Must stay ES5-ish: the graph vm context only exposes state/Boolean/String/Array/Object/Math.
 */
export const GOAL_PREDICATE_EXPR = `(state) => {
  const arts = Array.isArray(state.artifacts) ? state.artifacts : [];
  if (arts.length === 0) return false;
  const names = [];
  for (const a of arts) {
    const name = String((a && a.name) || '').trim();
    const content = String((a && a.content) || '');
    if (name === '' || content.trim() === '') return false;
    names.push(name.replace(/\\\\/g, '/').replace(/^\\/+/, ''));
  }
  const present = names.slice();
  if (Array.isArray(state.workspaceFiles)) {
    for (const f of state.workspaceFiles) {
      if (f && typeof f.path === 'string') {
        present.push(String(f.path).replace(/\\\\/g, '/').replace(/^\\/+/, ''));
      }
    }
  }
  if (!Array.isArray(state.persistedFiles) || state.persistedFiles.length !== arts.length) return false;
  if (!Array.isArray(state.syntaxErrors) || state.syntaxErrors.length !== 0) return false;
  for (let i = 0; i < arts.length; i++) {
    const name = names[i];
    if (!/\\.html?$/i.test(name)) continue;
    const slash = name.lastIndexOf('/');
    const dir = slash >= 0 ? name.slice(0, slash) : '';
    const content = String(arts[i].content);
    const re = /(?:src|href)\\s*=\\s*(?:"([^"]*)"|'([^']*)'|([^\\s>]+))/gi;
    let m;
    while ((m = re.exec(content)) !== null) {
      const raw = String((m[1] !== undefined ? m[1] : m[2] !== undefined ? m[2] : m[3]) || '').trim();
      if (raw === '' || raw.charAt(0) === '#' || raw.charAt(0) === '/') continue;
      if (raw.indexOf('http://') === 0 || raw.indexOf('https://') === 0 || raw.indexOf('//') === 0 ||
          raw.indexOf('data:') === 0 || raw.indexOf('mailto:') === 0) continue;
      const ref = raw.split('#')[0].split('?')[0];
      if (ref === '') continue;
      const parts = (dir ? dir + '/' + ref : ref).split('/');
      const stack = [];
      for (const p of parts) {
        if (p === '' || p === '.') continue;
        if (p === '..') stack.pop();
        else stack.push(p);
      }
      if (present.indexOf(stack.join('/')) < 0) return false;
    }
  }
  const ws = Array.isArray(state.workspaceFiles) ? state.workspaceFiles : [];
  for (let i = 0; i < arts.length; i++) {
    const name = names[i];
    const nextLen = String(arts[i].content).length;
    for (const f of ws) {
      if (!f) continue;
      const fp = String(f.path || '').replace(/\\\\/g, '/').replace(/^\\/+/, '');
      if (fp !== name) continue;
      const oldLen = Number(f.size) || (typeof f.content === 'string' ? f.content.length : 0);
      if (oldLen > 200 && nextLen < oldLen * 0.25 && state.allowDestructiveOverwrite !== true) {
        return false;
      }
    }
  }
  return true;
}`;

/**
 * @param {string} p
 * @returns {string}
 */
function normalizeRel(p) {
  return String(p || '').replace(/\\/g, '/').replace(/^\/+/, '');
}

/**
 * Resolve a relative HTML reference against the html file's directory.
 * Normalizes ./ and ../ minimally; returns POSIX path.
 * @param {string} htmlPath
 * @param {string} ref
 * @returns {string}
 */
function resolveLocalRef(htmlPath, ref) {
  const cleanRef = ref.split('#')[0].split('?')[0];
  if (!cleanRef) return '';
  const slash = htmlPath.lastIndexOf('/');
  const dir = slash >= 0 ? htmlPath.slice(0, slash) : '';
  const parts = (dir ? dir + '/' + cleanRef : cleanRef).split('/');
  /** @type {string[]} */
  const stack = [];
  for (const p of parts) {
    if (p === '' || p === '.') continue;
    if (p === '..') stack.pop();
    else stack.push(p);
  }
  return stack.join('/');
}

/**
 * Extract local src/href references that must resolve inside artifact names.
 * @param {string} content
 * @returns {string[]}
 */
function extractLocalRefs(content) {
  /** @type {string[]} */
  const refs = [];
  const re = /(?:src|href)\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/gi;
  let m;
  while ((m = re.exec(content)) !== null) {
    const raw = String((m[1] !== undefined ? m[1] : m[2] !== undefined ? m[2] : m[3]) || '').trim();
    if (!raw || raw.charAt(0) === '#' || raw.charAt(0) === '/') continue;
    if (
      raw.indexOf('http://') === 0 ||
      raw.indexOf('https://') === 0 ||
      raw.indexOf('//') === 0 ||
      raw.indexOf('data:') === 0 ||
      raw.indexOf('mailto:') === 0
    ) {
      continue;
    }
    refs.push(raw);
  }
  return refs;
}

/**
 * Synchronous + optional semantic acceptance validator.
 */
export class AcceptanceGateValidator {
  /**
   * @param {{ semanticEvaluator?: (state: StateObject) => Promise<{ approved: boolean, reason?: string }> | { approved: boolean, reason?: string } }} [options]
   */
  constructor(options = {}) {
    this.semanticEvaluator = options.semanticEvaluator;
  }

  /**
   * Local-only checks (no LLM).
   * @param {StateObject} state
   * @returns {{ passed: boolean, errors: string[], warnings: string[] }}
   */
  validateSync(state) {
    /** @type {string[]} */ const errors = [];
    /** @type {string[]} */ const warnings = [];
    const artifacts = Array.isArray(state.artifacts) ? state.artifacts : [];
    const persisted = Array.isArray(state.persistedFiles) ? state.persistedFiles : [];

    if (artifacts.length === 0) errors.push('artifacts is empty');

    /** @type {string[]} */ const names = [];
    /** @type {Set<string>} */ const seen = new Set();
    for (let i = 0; i < artifacts.length; i++) {
      const a = artifacts[i];
      const name = String((a && a.name) || '').trim();
      const content = String((a && a.content) || '');
      if (name === '') {
        errors.push(`artifact[${i}] has an empty name`);
        continue;
      }
      const rel = normalizeRel(name);
      if (content.trim() === '') errors.push(`artifact "${rel}" has empty content`);
      if (seen.has(rel)) errors.push(`duplicate artifact name "${rel}"`);
      seen.add(rel);
      names.push(rel);
    }
    const present = new Set(names);
    for (const f of Array.isArray(state.workspaceFiles) ? state.workspaceFiles : []) {
      if (f && typeof f.path === 'string') present.add(normalizeRel(f.path));
    }

    if (persisted.length !== artifacts.length) {
      errors.push(`persistedFiles count (${persisted.length}) does not match artifacts count (${artifacts.length})`);
    }
    if (!Array.isArray(state.syntaxErrors)) {
      errors.push('syntaxErrors is not an array');
    } else if (state.syntaxErrors.length > 0) {
      errors.push(`syntaxErrors is not empty: ${state.syntaxErrors.join('; ')}`);
    }

    // HTML local reference integrity.
    for (let i = 0; i < artifacts.length; i++) {
      const name = names[i];
      if (!name || !/\.html?$/i.test(name)) continue;
      const content = String((artifacts[i] && artifacts[i].content) || '');
      for (const ref of extractLocalRefs(content)) {
        const resolved = resolveLocalRef(name, ref);
        if (resolved && !present.has(resolved)) {
          errors.push(`artifact "${name}" references missing local file "${resolved}"`);
        }
      }
    }

    // Destructive overwrite guard: a same-path workspace file with existing
    // content must not be replaced by a drastically smaller artifact unless
    // the caller explicitly opts in.
    const ws = Array.isArray(state.workspaceFiles) ? state.workspaceFiles : [];
    for (let i = 0; i < artifacts.length; i++) {
      const name = names[i];
      if (!name) continue;
      const nextLen = String((artifacts[i] && artifacts[i].content) || '').length;
      for (const f of ws) {
        if (!f) continue;
        if (normalizeRel(f.path) !== name) continue;
        const oldLen = Number(f.size) || (typeof f.content === 'string' ? f.content.length : 0);
        if (oldLen > 200 && nextLen < oldLen * 0.25 && state.allowDestructiveOverwrite !== true) {
          errors.push(`destructive overwrite blocked for "${name}" (${oldLen} -> ${nextLen} bytes)`);
        }
      }
    }

    return { passed: errors.length === 0, errors, warnings };
  }

  /**
   * Sync local checks plus optional semantic evaluation (LLM).
   * A missing evaluator is a warning, never a failure.
   * @param {StateObject} state
   * @returns {Promise<{ passed: boolean, errors: string[], warnings: string[] }>}
   */
  async validate(state) {
    const base = this.validateSync(state);
    if (typeof this.semanticEvaluator === 'function') {
      try {
        const verdict = await this.semanticEvaluator(state);
        if (!verdict || verdict.approved !== true) {
          base.errors.push(`semantic evaluation rejected: ${(verdict && verdict.reason) || 'no reason provided'}`);
        }
      } catch (e) {
        base.errors.push(`semantic evaluation failed: ${e instanceof Error ? e.message : String(e)}`);
      }
    } else {
      base.warnings.push('semantic evaluator is not configured; semantic acceptance was skipped');
    }
    return { passed: base.errors.length === 0, errors: base.errors, warnings: base.warnings };
  }
}
