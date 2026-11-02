"""Offline synthetic tests; gguf writer and setup child are deliberately fakes."""
import ast
import contextlib
import errno
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class Writer:
    failure = None
    def __init__(self, *args):
        self.records = []
        Writer.last = self
    def add_string(self, *args):
        self.records.append(args)
    def add_tensor(self, name, blob, **kw):
        self.records.append((name, blob.tobytes(), kw))
    def write_header_to_file(self):
        pass
    def write_kv_data_to_file(self):
        pass
    def write_tensors_to_file(self):
        if self.failure:
            raise self.failure
    def close(self):
        pass


def module(source):
    fake = types.SimpleNamespace(GGUFWriter=Writer, GGMLQuantizationType=types.SimpleNamespace(
        Q2_0=42, Q4_0=2, Q8_0=8, BF16=30))
    ns = {'__file__': str(ROOT / 'tools/mtp_pack.py'), '__name__': 'fixture'}
    with patch.dict(sys.modules, {'gguf': fake, '_paths': types.SimpleNamespace(add_gguf_py=lambda: None)}):
        exec(compile(source, ns['__file__'], 'exec'), ns)
    return types.SimpleNamespace(**ns)


class Diagnostics(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.src = Path(self.temp.name)
        self.source = (ROOT / 'tools/mtp_pack.py').read_text()
        self.m = module(self.source)
        Writer.failure = None
        manifest = []
        for i, (name, shape) in enumerate([
            (self.m.EXPERT_TENSORS[0], [2, 1, 64]),
            (self.m.EXPERT_TENSORS[1], [2, 1, 64]),
            ('norm', [64]), ('dense', [1, 64])]):
            file = f'{i}.bin'
            x = np.linspace(-1, 1, np.prod(shape), dtype=np.float32)
            (x.view(np.uint32) >> 16).astype('<u2').tofile(self.src / file)
            manifest.append(dict(name=name, shape=shape, file=file, sha256='fixture'))
        (self.src / 'mtp-manifest.json').write_text(json.dumps(manifest))

    def run_pack(self, m=None, kind='q2_0', enabled='1'):
        stream = io.StringIO()
        self.log = stream
        with patch.dict(os.environ, STRATA_MTP_DIAGNOSTICS=enabled), patch.object(sys, 'argv', [
            'mtp_pack.py', '--src', str(self.src), '--out', str(self.src / 'out'), '--experts', kind
        ]), contextlib.redirect_stdout(stream):
            return (m or self.m).main()

    def test_bytes_and_order(self):
        baseline = module(subprocess.check_output(['git', 'show', 'v0.2.8:tools/mtp_pack.py'], cwd=ROOT, text=True))
        for kind in ('q2_0', 'q4_0', 'q8_0'):
            self.run_pack(baseline, kind)
            expected = Writer.last.records
            self.run_pack(kind=kind)
            self.assertEqual(expected, Writer.last.records)
            log = self.log.getvalue()
            labels = ['manifest-read', 'writer-init', 'expert-map', 'expert-quantize',
                      'expert-concatenate', 'expert-error-report', 'expert-add', 'write-header',
                      'write-metadata', 'write-tensors', 'writer-close', 'write-report']
            positions = [log.index('begin ' + label) for label in labels]
            self.assertEqual(positions, sorted(positions))
        self.run_pack(enabled='0')
        self.assertNotIn('[mtp-pack]', self.log.getvalue())

    def test_missing_and_truncated(self):
        for file in ('0.bin', '2.bin', '3.bin', 'mtp-manifest.json'):
            p = self.src / file
            saved = p.read_bytes()
            p.unlink()
            with self.assertRaises(FileNotFoundError):
                self.run_pack()
            self.assertIn('exception=FileNotFoundError', self.log.getvalue())
            p.write_bytes(saved[:2])
            with self.assertRaises((ValueError, json.JSONDecodeError)):
                self.run_pack()
            self.assertIn('[mtp-pack] failed', self.log.getvalue())
            p.write_bytes(saved)

    def test_memory_and_writer_errors(self):
        with patch.object(np, 'concatenate', side_effect=MemoryError('SECRET')):
            with self.assertRaises(MemoryError):
                self.run_pack()
            self.assertIn('failed expert-concatenate exception=MemoryError', self.log.getvalue())
            self.assertNotIn('SECRET', self.log.getvalue())
        for error in (PermissionError(errno.EACCES, 'SECRET'), OSError(errno.ENOSPC, 'SECRET')):
            Writer.failure = error
            with self.assertRaises(OSError):
                self.run_pack()
            self.assertIn('failed write-tensors', self.log.getvalue())
            self.assertIn(f'errno={error.errno}', self.log.getvalue())
            self.assertNotIn('SECRET', self.log.getvalue())
            self.assertNotIn('begin writer-close', self.log.getvalue())

    def test_report_error(self):
        original = Path.write_text
        def write(path, *args, **kwargs):
            if path.name.endswith('.report.json'):
                raise OSError(errno.ENOSPC, 'SECRET')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'write_text', write), self.assertRaises(OSError):
            self.run_pack()
        self.assertIn('done writer-close', self.log.getvalue())
        self.assertIn('failed write-report exception=OSError errno=28', self.log.getvalue())
        self.assertNotIn('SECRET', self.log.getvalue())

    def test_cli_failure(self):
        # Execute the actual entrypoint with fake imports: failure exits 1 without traceback text.
        ns = {'__file__': str(ROOT / 'tools/mtp_pack.py'), '__name__': '__main__'}
        fake = types.SimpleNamespace(GGUFWriter=Writer, GGMLQuantizationType=types.SimpleNamespace(
            Q2_0=42, Q4_0=2, Q8_0=8, BF16=30))
        (self.src / 'mtp-manifest.json').unlink()
        log = io.StringIO()
        with patch.dict(sys.modules, {'gguf': fake, '_paths': types.SimpleNamespace(add_gguf_py=lambda: None)}), \
             patch.dict(os.environ, STRATA_MTP_DIAGNOSTICS='1'), \
             patch.object(sys, 'argv', ['mtp_pack.py', '--src', str(self.src), '--out', 'unused']), \
             contextlib.redirect_stdout(log), self.assertRaises(SystemExit) as caught:
            exec(compile(self.source, ns['__file__'], 'exec'), ns)
        self.assertEqual(caught.exception.code, 1)
        self.assertIn('failed pack exception=FileNotFoundError', log.getvalue())
        self.assertNotIn(str(self.src), log.getvalue())

    def test_flush(self):
        with patch.dict(os.environ, STRATA_MTP_DIAGNOSTICS='1'), patch('builtins.print') as emit:
            with self.m.phase('fixture'):
                pass
        self.assertEqual(len(emit.call_args_list), 2)
        self.assertTrue(all(call.kwargs['flush'] for call in emit.call_args_list))

    def test_setup_exit_and_spawn(self):
        tree = ast.parse((ROOT / 'setup.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        ns = dict(Path=Path, ROOT=ROOT, os=os, sys=sys, subprocess=subprocess, CONTROL=False,
                  acquisition=types.SimpleNamespace(require=lambda _: None), say=lambda _: None,
                  fail=lambda _: (_ for _ in ()).throw(SystemExit(1)))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'setup.py', 'exec'), ns)
        cmd = [sys.executable, str(ROOT / 'tools/mtp_pack.py')]
        for code in (0, 7, -9, 3221225477):
            for check in (True, False):
                log = io.StringIO()
                with patch.object(subprocess, 'run', return_value=types.SimpleNamespace(returncode=code)), contextlib.redirect_stdout(log):
                    if check and code:
                        with self.assertRaises(SystemExit) as caught:
                            ns['run'](cmd, env={'STRATA_MTP_DIAGNOSTICS':'1'}, check=check)
                        self.assertEqual(caught.exception.code, 1)
                    else:
                        self.assertEqual(ns['run'](cmd, env={'STRATA_MTP_DIAGNOSTICS':'1'}, check=check).returncode, code)
                self.assertIn(f'returncode={code}', log.getvalue())
        log = io.StringIO()
        with patch.object(subprocess, 'run', side_effect=PermissionError(errno.EACCES, 'SECRET')), contextlib.redirect_stdout(log):
            with self.assertRaises(SystemExit) as caught:
                ns['run'](cmd, env={'STRATA_MTP_DIAGNOSTICS':'1'})
        self.assertEqual(caught.exception.code, 1)
        self.assertIn('spawn-failed exception=PermissionError errno=13', log.getvalue())
        self.assertNotIn('SECRET', log.getvalue())


if __name__ == '__main__':
    unittest.main()
