"""Offline accuracy regressions for the quick/legacy benchmarks and shared redaction."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import strata_bench as B
from strata_cli import bench_facts as F
from bench import bench_vs_llama as L


def event(**values):
    return b'data: ' + json.dumps(values).encode() + b'\n\n'


FINISH = event(choices=[{'delta': {}, 'finish_reason': 'stop'}])


def stream(*tail):
    return io.BytesIO(event(choices=[{'delta': {'content': 'many words in one chunk'}}]) + b''.join(tail))


class AccuracyTests(unittest.TestCase):
    def response(self, *tail):
        with mock.patch.object(B.urllib.request, 'urlopen', return_value=stream(*tail)):
            return B.chat('http://fake.invalid', '', [], 20)

    def test_truncated_stream_fails(self):
        self.assertFalse(B.digest(self.response())['ok'])

    def test_finish_without_done_still_truncated(self):
        self.assertFalse(B.digest(self.response(event(choices=[{'delta': {}, 'finish_reason': 'stop'}])))['ok'])

    def test_done_stream_succeeds_without_inventing_tokens(self):
        d = B.digest(self.response(FINISH, b'data: [DONE]\n\n'))
        self.assertTrue(d['ok'])
        self.assertIsNone(d['generated_tokens'])
        self.assertIsNone(d['decode_tps'])

    def test_usage_zero_preserved_and_engine_count_supported(self):
        for usage, timings, expected in [({'completion_tokens': 0}, {}, 0), ({}, {'predicted_n': 17}, 17)]:
            with self.subTest(expected=expected):
                r = self.response(event(usage=usage, timings=timings), FINISH, b'data: [DONE]\n\n')
                self.assertEqual(B.digest(r)['generated_tokens'], expected)

    def test_one_chunk_is_not_one_token(self):
        d = B.digest(self.response(event(usage={'completion_tokens': 19}), FINISH, b'data: [DONE]\n\n'))
        self.assertEqual(d['generated_tokens'], 19)
        # A chunk may contain multiple tokens, so first-chunk time cannot measure decode duration.
        self.assertIsNone(d['decode_tps'])

    def test_redaction_preserves_keys_and_cleans_values(self):
        for module, method, name in [(B, B.redact_obj, 'agent'), (F, F.redact_object, 'content')]:
            with self.subTest(module=module.__name__), mock.patch.object(module.getpass, 'getuser', return_value=name):
                red = module.redactor(key='secret-api-key')
                cleaned = method({name: name, 'messages': [{'content': 'secret-api-key /home/alice/file'}]}, red)
                self.assertEqual(cleaned[name], '<user>')
                self.assertNotIn('secret-api-key', json.dumps(cleaned))
                self.assertNotIn('alice', json.dumps(cleaned))

    def test_failed_quick_suite_exits_nonzero(self):
        from test_strata_bench import FakeStrata
        srv = FakeStrata(fail_after=0)
        self.addCleanup(srv.stop)
        with tempfile.TemporaryDirectory() as out, mock.patch.object(B.getpass, 'getuser', return_value='auditor'), \
                mock.patch.object(B, 'gpu_info_local', return_value={'gpus': []}):
            self.assertEqual(B.main(['--url', srv.base, '--out', out, '--runs', '1', '--quiet']), 1)

    def test_legacy_all_failed_arms_exit_nonzero(self):
        with tempfile.TemporaryDirectory() as out:
            machine = Path(out) / 'machine.json'
            machine.write_text(json.dumps({'configs': [{'name': 'fake', 'kind': 'strata'}]}))
            (Path(out) / 'prompt-4k.txt').write_text('test prompt')
            args = argparse.Namespace(machine=str(machine), out=out, configs=None, cells='4k', start_round=0, rounds=1)
            with mock.patch.object(L.Engine, 'start', side_effect=RuntimeError('controlled failure')), \
                    mock.patch.object(L.Engine, 'stop'), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(L.cmd_run(args), 1)
            self.assertIn('controlled failure', (Path(out) / 'raw/errors.log').read_text())

    def test_missing_cache_metrics_stay_unknown(self):
        raw = {'err': None, 'first': 0.1, 'total': 1, 'usage': None, 'timings': None, 'pieces': 4, 'text': 'ok'}
        with mock.patch.object(B, 'chat', return_value=raw):
            s = B.Suite('http://fake.invalid', '', 32768, 1, False, log=lambda *a: None)
            s.run_prompt(4096, '4K')
            s.run_agent()
        self.assertIsNone(s.out['prompts']['4K']['reused_tokens'])
        for field in ('hit_pct', 'reread_tokens', 'new_tokens'):
            self.assertIsNone(s.out['agent'][field])

    def test_concurrency_missing_counts_and_truncation(self):
        for tail, errors in [(FINISH + b'data: [DONE]\n\n', 0), (b'', 2)]:
            with self.subTest(errors=errors), mock.patch.object(
                    B.urllib.request, 'urlopen', side_effect=lambda *a, **k: stream(tail)):
                suite = B.Suite('http://fake.invalid', '', 32768, 1, False, log=lambda *a: None)
                suite.run_clients(2)
                self.assertIsNone(suite.out['clients']['total_tps'])
                self.assertIsNone(suite.out['clients']['rounds'][0]['tokens'])
                self.assertEqual(suite.out['clients']['errors'], errors)

    def test_legacy_arms_receive_identical_prompts(self):
        seen = {}

        class FakeEngine:
            def __init__(self, machine, cfg):
                self.name = cfg['name']
                self.sampler = argparse.Namespace(vram=0, rss=0, sysused=0)
                self.base_vram = self.base_used = self.load_s = 0
                self.cmdline = 'fake'
                seen[self.name] = []

            def start(self, out):
                pass

            def stop(self):
                pass

            def chat(self, prompt, temp, max_tokens=L.GEN):
                seen[self.name].append((prompt, temp, max_tokens))
                return dict(text='answer', prompt_tps=None, pred_tps=None, draft_acc=None,
                            draft_n=None, prompt_ms=None)

        with tempfile.TemporaryDirectory() as out:
            machine = Path(out) / 'machine.json'
            machine.write_text(json.dumps({'configs': [{'name': 'a'}, {'name': 'b'}]}))
            (Path(out) / 'prompt-4k.txt').write_text('identical corpus')
            args = argparse.Namespace(machine=str(machine), out=out, configs=None, cells='4k', start_round=0, rounds=1)
            with mock.patch.object(L, 'Engine', FakeEngine), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(L.cmd_run(args), 0)
            self.assertEqual(seen['a'], seen['b'])
            args.configs = 'missing'
            with self.assertRaises(ValueError):
                L.cmd_run(args)


if __name__ == '__main__':
    unittest.main()
