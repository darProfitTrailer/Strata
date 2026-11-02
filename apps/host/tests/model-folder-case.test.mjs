import test from 'node:test';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { syncDataFolder } from '../server/model-folder.mjs';

function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'strata-folder-case-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const upper = join(root, 'Strata'), lower = join(root, 'strata');
  mkdirSync(upper);
  if (existsSync(lower)) { t.skip('requires a case-sensitive filesystem'); return null; }
  mkdirSync(lower);
  writeFileSync(join(upper, 'upper.gguf'), 'upper fixture');
  writeFileSync(join(lower, 'lower.gguf'), 'lower fixture');
  return { root, upper, lower, appFile: join(root, 'app.json'), terminalFile: join(root, 'terminal.json') };
}

function seed(file, folder, at) {
  writeFileSync(file, JSON.stringify({ data_dir: folder, data_dir_at: at, installs: ['keep'] }));
}

function assertUnmoved(f) {
  assert.deepEqual(readdirSync(f.upper), ['upper.gguf']);
  assert.deepEqual(readdirSync(f.lower), ['lower.gguf']);
  assert.equal(readFileSync(join(f.upper, 'upper.gguf'), 'utf8'), 'upper fixture');
  assert.equal(readFileSync(join(f.lower, 'lower.gguf'), 'utf8'), 'lower fixture');
}

test('a newer Terminal root differing only in case reaches the app without moving files', { skip: process.platform === 'win32' }, t => {
  const f = fixture(t); if (!f) return;
  seed(f.appFile, f.upper, 1000); seed(f.terminalFile, f.lower, 2000);
  assert.deepEqual(syncDataFolder(f), { folder: f.lower, from: 'terminal' });
  assert.deepEqual(JSON.parse(readFileSync(f.appFile, 'utf8')), { data_dir: f.lower, data_dir_at: 2000, installs: ['keep'] });
  const before = readFileSync(f.appFile);
  assert.deepEqual(syncDataFolder(f), { folder: f.lower, from: null });
  assert.deepEqual(readFileSync(f.appFile), before);
  assertUnmoved(f);
});

test('a newer app root differing only in case reaches the Terminal without moving files', { skip: process.platform === 'win32' }, t => {
  const f = fixture(t); if (!f) return;
  seed(f.appFile, f.upper, 3000); seed(f.terminalFile, f.lower, 2000);
  assert.deepEqual(syncDataFolder(f), { folder: f.upper, from: 'app' });
  assert.deepEqual(JSON.parse(readFileSync(f.terminalFile, 'utf8')), { data_dir: f.upper, data_dir_at: 3000, installs: ['keep'] });
  assert.deepEqual(syncDataFolder(f), { folder: f.upper, from: null });
  assertUnmoved(f);
});

test('a missing newer case-only Terminal root is still not adopted', { skip: process.platform === 'win32' }, t => {
  const f = fixture(t); if (!f) return;
  rmSync(f.lower, { recursive: true });
  seed(f.appFile, f.upper, 1000); seed(f.terminalFile, f.lower, 2000);
  assert.deepEqual(syncDataFolder(f), { folder: f.upper, from: 'app' });
  assert.equal(JSON.parse(readFileSync(f.terminalFile, 'utf8')).data_dir, f.upper);
  assert.equal(existsSync(f.lower), false);
  assert.equal(readFileSync(join(f.upper, 'upper.gguf'), 'utf8'), 'upper fixture');
});

test('Windows comparison still treats case-only spellings as one root (platform policy fixture)', t => {
  const root = mkdtempSync(join(tmpdir(), 'strata-folder-case-policy-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  const upper = join(root, 'Strata'), lower = join(root, 'strata');
  mkdirSync(upper);
  if (!existsSync(lower)) mkdirSync(lower);
  const f = { upper, lower, appFile: join(root, 'app.json'), terminalFile: join(root, 'terminal.json') };
  seed(f.appFile, f.upper, 1000); seed(f.terminalFile, f.lower, 2000);
  const before = [readFileSync(f.appFile), readFileSync(f.terminalFile)];
  const script = `
    import assert from 'node:assert/strict';
    import { syncDataFolder } from ${JSON.stringify(new URL('../server/model-folder.mjs', import.meta.url).href)};
    Object.defineProperty(process, 'platform', { value: 'win32' });
    assert.deepEqual(syncDataFolder(${JSON.stringify(f)}), { folder: ${JSON.stringify(f.upper)}, from: null });
  `;
  const result = spawnSync(process.execPath, ['--input-type=module', '-e', script], { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  assert.deepEqual([readFileSync(f.appFile), readFileSync(f.terminalFile)], before);
});
