"""Standalone load-client regression tests; mocked streams only, no sockets or inference."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
import urllib.error

import serve_load as L


def event(**values):
    return b'data: ' + json.dumps(values).encode() + b'\n\n'


CONTENT = event(choices=[{'index': 0, 'delta': {'content': 'multiple tokens in a single chunk'}}])
FINISH = event(choices=[{'index': 0, 'delta': {}, 'finish_reason': 'stop'}])
DONE = b'data: [DONE]\n\n'


class LoadAccuracyTests(unittest.TestCase):
    def one(self, payload, key=''):
        result = {}
        with mock.patch.object(L.urllib.request, 'urlopen', return_value=io.BytesIO(payload)):
            L.one('http://fake.invalid', key, 'fake', 0, 100, 20, result, {}, False)
        return result[0]

    def run_main(self, payload, *args):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'summary.json'
            argv = ['serve_load.py', 'http://fake.invalid', '--clients', '2', '--rounds', '1', '--json', str(path), *args]
            with mock.patch.object(L.sys, 'argv', argv), mock.patch.object(
                    L.urllib.request, 'urlopen', side_effect=lambda *a, **k: io.BytesIO(payload)), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                code = L.main()
            text = path.read_text()
            summary = json.loads(text, parse_constant=lambda value: self.fail('nonfinite JSON: ' + value))
            return code, summary, output.getvalue()

    def test_known_usage_is_not_chunk_count_and_zero_is_preserved(self):
        for tokens in (0, 17):
            with self.subTest(tokens=tokens):
                result = self.one(CONTENT + FINISH + event(usage={'completion_tokens': tokens}) + DONE)
                self.assertIsNone(result['err'])
                self.assertEqual(result['tokens'], tokens)

    def test_missing_usage_stays_unknown_through_summary(self):
        payload = CONTENT + FINISH + DONE
        self.assertIsNone(self.one(payload)['tokens'])
        code, summary, output = self.run_main(payload)
        self.assertEqual(code, 0)
        self.assertIsNone(summary[0]['total_tps'])
        self.assertIsNone(summary[0]['per_req_tps'])
        self.assertIn('unknown', output.lower())

    def test_incomplete_malformed_and_server_error_streams_fail(self):
        cases = [CONTENT, CONTENT + DONE, CONTENT + FINISH,
                 CONTENT + b'data: {invalid}\n\n' + FINISH + DONE,
                 CONTENT + event(error={'message': 'secret-key-12345'}) + FINISH + DONE,
                 CONTENT + event(choices=[{'delta': {}, 'finish_reason': 'invented'}]) + DONE,
                 CONTENT + event(usage=[1]) + FINISH + DONE,
                 CONTENT + event(choices='bad') + FINISH + DONE,
                 CONTENT + event(choices=[{'delta': {'content': 1}}]) + FINISH + DONE]
        for payload in cases:
            with self.subTest(payload=payload):
                result = self.one(payload, key='secret-key-12345')
                self.assertTrue(result['err'])
                self.assertNotIn('secret-key-12345', result['err'])
                self.assertIsNone(result['tokens'])

    def test_invalid_usage_fails_and_does_not_poison_json(self):
        for value in ('100', True, -7, float('nan'), float('inf')):
            with self.subTest(value=value):
                payload = CONTENT + FINISH + event(usage={'completion_tokens': value}) + DONE
                code, summary, _ = self.run_main(payload)
                self.assertEqual(code, 1)
                self.assertEqual(summary[0]['errors'], 2)
                self.assertIsNone(summary[0]['total_tps'])

    def test_failed_streams_fail_exit_and_are_not_timed_as_successes(self):
        code, summary, _ = self.run_main(CONTENT + DONE)
        self.assertEqual(code, 1)
        self.assertEqual(summary[0]['errors'], 2)
        self.assertIsNone(summary[0]['first_p50'])
        self.assertIsNone(summary[0]['lat_p50'])

    def test_http_exception_does_not_leak_credentials(self):
        result = {}
        with mock.patch.object(L.urllib.request, 'urlopen', side_effect=RuntimeError('secret-key-12345')):
            L.one('http://fake.invalid', 'secret-key-12345', 'fake', 0, 100, 20, result, {}, False)
        self.assertNotIn('secret-key-12345', result[0]['err'])
        self.assertTrue(result[0]['err'])

    def test_known_complete_stream_retains_finite_throughput(self):
        code, summary, _ = self.run_main(CONTENT + FINISH + event(usage={'completion_tokens': 17}) + DONE)
        self.assertEqual(code, 0)
        self.assertEqual(summary[0]['errors'], 0)
        self.assertGreater(summary[0]['total_tps'], 0)
        self.assertGreater(summary[0]['per_req_tps'], 0)

    def test_multiline_events_keepalives_and_usage_request(self):
        payload = b': keepalive\n\n' + b'data: {"choices": [\ndata: {"delta": {"content": "hello"}}]}\n\n'
        payload += FINISH + event(usage={'completion_tokens': 5}) + DONE
        result = {}
        with mock.patch.object(L.urllib.request, 'urlopen', return_value=io.BytesIO(payload)) as opener:
            L.one('http://fake.invalid', '', 'fake', 0, 100, 20, result, {}, False)
        self.assertIsNone(result[0]['err'])
        self.assertEqual(result[0]['tokens'], 5)
        self.assertEqual(json.loads(opener.call_args.args[0].data)['stream_options'], {'include_usage': True})

    def test_usage_does_not_increment_when_more_content_arrives(self):
        payload = event(usage={'completion_tokens': 17}) + CONTENT + CONTENT + FINISH + DONE
        self.assertEqual(self.one(payload)['tokens'], 17)

    def test_content_after_finish_and_unterminated_event_fail(self):
        for payload in [CONTENT + FINISH + CONTENT + DONE, CONTENT + FINISH + b'data: [DONE]']:
            with self.subTest(payload=payload):
                self.assertTrue(self.one(payload)['err'])

    def test_request_throughput_uses_whole_request_duration(self):
        def fixed(url, key, model, i, words, max_tokens, res, extra, think, nonce):
            res[i] = dict(first=2, total=5, tokens=20, text='ok', err=None)
        with mock.patch.object(L, 'one', side_effect=fixed), mock.patch.object(L.time, 'perf_counter', side_effect=[0, 10]):
            code, summary, _ = self.run_main(b'')
        self.assertEqual(code, 0)
        self.assertEqual(summary[0]['total_tps'], 4)
        self.assertEqual(summary[0]['per_req_tps'], 4)

    def test_missing_workers_and_overflow_fail_closed(self):
        def enormous(url, key, model, i, words, max_tokens, res, extra, think, nonce):
            res[i] = dict(first=0, total=0.000001, tokens=10 ** 308, text='ok', err=None)
        for worker in [lambda *args: None, enormous]:
            with self.subTest(worker=worker), mock.patch.object(L, 'one', side_effect=worker):
                code, summary, _ = self.run_main(b'')
                self.assertEqual(code, 1)
                self.assertGreater(summary[0]['errors'], 0)
                self.assertIsNone(summary[0]['total_tps'])

    def test_save_redacts_keys_and_comparison_reports_missing_or_failed(self):
        payload = event(choices=[{'delta': {'content': 'secret-key-12345 Bearer other-secret-12345'}}])
        payload += FINISH + event(usage={'completion_tokens': 5}) + DONE
        with tempfile.TemporaryDirectory() as temp:
            code, _, _ = self.run_main(payload, '--save', temp, '--key', 'secret-key-12345')
            self.assertEqual(code, 0)
            for path in Path(temp).rglob('*.txt'):
                self.assertNotIn('secret-key-12345', path.read_text())
                self.assertNotIn('other-secret-12345', path.read_text())
            code, _, text = self.run_main(payload, '--compare', temp, '--key', 'secret-key-12345')
            self.assertEqual(code, 0)
            self.assertIn('ALL IDENTICAL', text)
            code, _, text = self.run_main(CONTENT + DONE, '--compare', temp)
            self.assertEqual(code, 1)
            self.assertIn('INCOMPLETE', text)
        with tempfile.TemporaryDirectory() as temp:
            code, _, text = self.run_main(payload, '--compare', temp)
            self.assertEqual(code, 1)
            self.assertNotIn('ALL IDENTICAL', text)

    def test_zero_clients_or_rounds_are_rejected(self):
        for option in ['--clients', '--rounds']:
            with self.subTest(option=option), mock.patch.object(L.sys, 'argv', ['serve_load.py', 'http://fake.invalid', option, '0']), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    L.main()
                self.assertEqual(error.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
