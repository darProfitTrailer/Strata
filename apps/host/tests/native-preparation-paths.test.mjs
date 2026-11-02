import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, writeFileSync, mkdirSync} from 'node:fs';
import {join} from 'node:path';
import {nativeModelFolders} from '../server/model-folder.mjs';
import {nativeFixture} from './helpers/native-fixture.mjs';

test('Windows native files under Models retain their data root, including UNC and mixed separators', () => {
 for (const file of ['H:\\Strata\\Models\\IQ3_S\\model.gguf', 'H:/Strata/mOdElS/IQ3_S/model.gguf']) {
  assert.deepEqual(nativeModelFolders(file, {platform:'win32'}), {dataFolder:file.includes('/')?'H:/Strata':'H:\\Strata'});
 }
 assert.deepEqual(nativeModelFolders('\\\\server\\share\\Strata\\Models\\IQ3_S\\model.gguf', {platform:'win32'}),
  {dataFolder:'\\\\server\\share\\Strata'});
});

test('POSIX case and existing custom folders remain distinct', () => {
 assert.deepEqual(nativeModelFolders('/data/models/IQ3_S/model.gguf', {platform:'linux'}), {dataFolder:'/data'});
 assert.deepEqual(nativeModelFolders('/data/Models/IQ3_S/model.gguf', {platform:'linux',exists:()=>true}), {ownGgufDir:'/data/Models/IQ3_S'});
 assert.deepEqual(nativeModelFolders('H:\\my-ggufs\\model.gguf', {platform:'win32',exists:()=>true}), {ownGgufDir:'H:\\my-ggufs'});
 assert.deepEqual(nativeModelFolders('/missing/model.gguf', {platform:'linux',exists:()=>false}), {});
});

test('unsafe or ambiguous saved native paths never become preparation directories', () => {
 for (const [platform,files] of [['win32',['H:Models\\IQ3_S\\model.gguf','\\Models\\IQ3_S\\model.gguf','H:\\Strata\\..\\Models\\IQ3_S\\model.gguf']],
                              ['linux',['models/IQ3_S/model.gguf','/data/../models/IQ3_S/model.gguf','/data/\0/models/IQ3_S/model.gguf']]]) {
  for (const file of files) assert.deepEqual(nativeModelFolders(file,{platform,exists:()=>{throw Error('must not inspect invalid path');}}),{});
 }
 for(const file of [null,undefined,42,{}]) assert.deepEqual(nativeModelFolders(file),{});
});

test('installed native config uses recovered folder for a later preparation', async () => {
 const f=nativeFixture();try {
  const root=join(f.directory,'Strata-original'),folder=join(root,'models','IQ2_XS');mkdirSync(folder,{recursive:true});
  const file=join(folder,'model.gguf');writeFileSync(file,'synthetic');
  const path=join(f.repo,'strata-iq2_xs.json'),cfg=JSON.parse(readFileSync(path));cfg.args=['--native',file];writeFileSync(path,JSON.stringify(cfg));
  await f.control.refresh(true);await f.control.plan({family:'qwen',model:'IQ2_XS'});
  const args=f.rpcCalls.filter(c=>c.name==='strata_install').at(-1).arguments;
  assert.equal(args.data_dir,root);assert.equal(args.gguf_dir,undefined);
 } finally {await f.cleanup();}
});

test('MTP preparation is explicit, validated and forwarded unchanged', async () => {
 const f=nativeFixture();try {
  await f.control.refresh(true);
  for(const mtp of [undefined,'off','on']) {
   await f.control.plan({family:'qwen',model:'IQ2_XS',...(mtp===undefined?{}:{mtp})});
   const args=f.rpcCalls.filter(c=>c.name==='strata_install').at(-1).arguments;
   assert.equal(args.mtp,mtp);assert.equal(Object.hasOwn(args,'mtp'),mtp!==undefined);
  }
  for(const mtp of ['auto',false,null,'OFF','off --other']) {
   const before=f.rpcCalls.filter(c=>c.name==='strata_install').length;
   await assert.rejects(f.control.plan({family:'qwen',model:'IQ2_XS',mtp}),/mtp must be on or off/);
   assert.equal(f.rpcCalls.filter(c=>c.name==='strata_install').length,before);
  }
 } finally {await f.cleanup();}
});
