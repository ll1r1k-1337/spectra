#!/usr/bin/env node
// @ts-check
import { execSync, spawnSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const rootDir = path.resolve(__dirname, '..');

/**
 * Finds the working compose command ('podman compose' or 'docker compose').
 * @returns {{ cmd: string, args: string[] }}
 */
function resolveComposeCommand() {
  const tryCommands = [
    { cmd: 'docker', args: ['compose'] },
    { cmd: 'podman', args: ['compose'] },
    { cmd: 'docker-compose', args: [] },
    { cmd: 'podman-compose', args: [] }
  ];

  for (const item of tryCommands) {
    try {
      const res = spawnSync(item.cmd, [...item.args, 'version'], { stdio: 'pipe' });
      if (res.status === 0) {
        return item;
      }
    } catch {
      // try next
    }
  }

  // Fallback to podman compose
  return { cmd: 'podman', args: ['compose'] };
}

/**
 * Finds the container manager command ('podman' or 'docker').
 * @returns {string}
 */
function resolveContainerCli() {
  for (const cmd of ['podman', 'docker']) {
    try {
      const res = spawnSync(cmd, ['--version'], { stdio: 'pipe' });
      if (res.status === 0) return cmd;
    } catch {
      // try next
    }
  }
  return 'podman';
}

async function main() {
  const compose = resolveComposeCommand();
  const containerCli = resolveContainerCli();

  console.log(`🐳 Starting Spectra PostgreSQL + pgvector container using ${compose.cmd} ${compose.args.join(' ')}...`);

  const upRes = spawnSync(compose.cmd, [...compose.args, 'up', '-d'], {
    cwd: rootDir,
    stdio: 'inherit'
  });

  if (upRes.status !== 0) {
    console.error(`❌ Failed to start container with ${compose.cmd}.`);
    process.exit(upRes.status ?? 1);
  }

  const containerName = 'spectra_pgvector';
  console.log(`⏳ Waiting for PostgreSQL in "${containerName}" to be ready...`);

  let ready = false;
  for (let attempt = 1; attempt <= 30; attempt++) {
    try {
      const check = spawnSync(containerCli, ['exec', containerName, 'pg_isready', '-U', 'postgres'], {
        stdio: 'pipe'
      });
      if (check.status === 0) {
        ready = true;
        break;
      }
    } catch {
      // Wait and retry
    }
    await new Promise((r) => setTimeout(r, 1000));
  }

  if (!ready) {
    console.error('❌ PostgreSQL did not become ready within 30 seconds.');
    process.exit(1);
  }

  console.log('🔧 Ensuring role "spectra" and database "spectra_db" exist...');

  // 1. Ensure role spectra exists
  const createRoleSql = `
    DO $$
    BEGIN
      IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'spectra') THEN
        CREATE ROLE spectra WITH SUPERUSER LOGIN PASSWORD 'spectra_password';
      ELSE
        ALTER ROLE spectra WITH SUPERUSER LOGIN PASSWORD 'spectra_password';
      END IF;
    END
    $$;
  `;
  spawnSync(containerCli, ['exec', containerName, 'psql', '-U', 'postgres', '-c', createRoleSql], {
    stdio: 'inherit'
  });

  // 2. Ensure database spectra_db exists and is owned by spectra
  const checkDb = spawnSync(containerCli, [
    'exec',
    containerName,
    'psql',
    '-U',
    'postgres',
    '-tAc',
    "SELECT 1 FROM pg_database WHERE datname = 'spectra_db'"
  ], { stdio: 'pipe' });

  if (checkDb.stdout.toString().trim() !== '1') {
    spawnSync(containerCli, [
      'exec',
      containerName,
      'psql',
      '-U',
      'postgres',
      '-c',
      'CREATE DATABASE spectra_db OWNER spectra;'
    ], { stdio: 'inherit' });
  }

  // 3. Ensure DDL schema from sql/001_init_pgvector.sql is applied
  const schemaFile = path.resolve(rootDir, 'sql/001_init_pgvector.sql');
  if (fs.existsSync(schemaFile)) {
    console.log('📦 Verifying schema tables and vector extension...');
    const sqlContent = fs.readFileSync(schemaFile, 'utf-8');
    const applyRes = spawnSync(
      containerCli,
      ['exec', '-i', containerName, 'psql', '-U', 'spectra', '-d', 'spectra_db'],
      {
        input: sqlContent,
        stdio: ['pipe', 'inherit', 'inherit']
      }
    );
    if (applyRes.status !== 0) {
      console.warn('⚠️ Warning: Some DDL statements returned non-zero (tables may already exist).');
    }
  }

  console.log('\n============================================================');
  console.log('✅ PostgreSQL + pgvector is READY for Spectra & DBeaver!');
  console.log('============================================================');
  console.log('   Host:     localhost (127.0.0.1)');
  console.log('   Port:     5432');
  console.log('   Database: spectra_db');
  console.log('   Username: spectra');
  console.log('   Password: spectra_password');
  console.log('============================================================\n');
}

main().catch((err) => {
  console.error('❌ Error during db:up:', err);
  process.exit(1);
});
