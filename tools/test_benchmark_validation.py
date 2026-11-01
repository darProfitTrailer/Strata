"""Adversarial regressions found during independent benchmark review; no live inference."""
import json
import unittest
from unittest import mock
import test_benchmark_accuracy as A
B, F = A.B, A.F

FINISH = A.event(choices=[{'delta': {}, 'finish_reason': 'stop'}])
DONE = b'data: [DONE]\n\n'


class ValidationTests(unittest.TestCase):
    def response(self, *tail):
        with mock.patch.object(B.urllib.request, 'urlopen', return_value=A.stream(*tail)):
            return B.chat('http://fake.invalid', '', [], 20)

    def test_done_requires_finish_reason(self):
        self.assertFalse(B.digest(self.response(DONE))['ok'])

    def test_supported_finish_reasons(self):
        for finish, ok in [('stop', True), ('length', True), ('tool_calls', True), ('invented', False), ([], False)]:
            with self.subTest(finish=finish):
                d = B.digest(self.response(A.event(choices=[{'delta': {}, 'finish_reason': finish}]), DONE))
                self.assertEqual(d['ok'], ok)

    def test_malformed_metrics_fail_without_raising(self):
        cases = [dict(usage=[1]), dict(usage=[]), dict(usage='bad'), dict(timings=[1]),
                 dict(usage={'prompt_tokens_details': [1]}), dict(usage={'completion_tokens': -1}),
                 dict(usage={'completion_tokens': '12'}), dict(usage={'completion_tokens': True}),
                 dict(usage={'prompt_tokens': 1.5}), dict(timings={'predicted_per_second': -5}),
                 dict(timings={'predicted_per_second': float('nan')}),
                 dict(timings={'prompt_per_second': float('inf')}),
                 dict(timings={'predicted_n': '12'}), dict(timings={'prompt_ms': -10}),
                 dict(usage={'prompt_tokens': 3, 'prompt_tokens_details': {'cached_tokens': 4}})]
        for payload in cases:
            with self.subTest(payload=payload):
                d = B.digest(self.response(A.event(**payload), FINISH, DONE))
                self.assertFalse(d['ok'])
                self.assertTrue(d['err'])
                json.dumps(d, allow_nan=False)

    def test_invalid_counts_do_not_crash_suite_aggregation(self):
        for value in ['100', float('nan'), -7, True]:
            payload = A.event(usage={'completion_tokens': value})
            with self.subTest(value=value), mock.patch.object(
                    B.urllib.request, 'urlopen', side_effect=lambda *a, **k: A.stream(payload, FINISH, DONE)):
                s = B.Suite('http://fake.invalid', '', 32768, 1, False, log=lambda *a: None)
                s.run_chat()
                s.run_agent()
                s.run_clients(2)
                self.assertTrue(s.out['failures'])
                self.assertIsNone(s.out['chat']['generated_tokens'])
                self.assertIsNone(s.out['clients']['total_tps'])
                self.assertEqual(s.out['clients']['errors'], 2)
                json.dumps(s.out, allow_nan=False)

    def test_legacy_rejects_invalid_metrics_before_writing_records(self):
        for payload in [dict(usage=[1]), dict(usage={'completion_tokens': '100'}),
                        dict(timings={'predicted_per_second': -7}),
                        dict(timings={'predicted_per_second': float('nan')})]:
            with self.subTest(payload=payload), mock.patch.object(A.L, 'http', return_value={
                    **payload, 'choices': [{'message': {'content': 'answer'}}]}):
                with self.assertRaisesRegex(ValueError, 'invalid benchmark metrics'):
                    A.L.Engine({}, {}).chat('prompt', 0)

    def test_legacy_summary_rejects_invalid_saved_rates(self):
        with A.tempfile.TemporaryDirectory() as temp:
            raw = A.Path(temp) / 'raw'
            raw.mkdir()
            for value in ['12', -7, True, float('nan')]:
                with self.subTest(value=value):
                    (raw / 'fake.jsonl').write_text(json.dumps({'pred_tps': value}) + '\n')
                    with self.assertRaisesRegex(ValueError, 'invalid saved benchmark metrics'):
                        A.L.cmd_summary(A.argparse.Namespace(out=temp))

    def test_valid_zero_and_missing_counts_remain_distinct(self):
        for value in [None, 0, 12]:
            with self.subTest(value=value):
                payload = A.event(usage={'completion_tokens': value}, timings={'predicted_per_second': 0})
                d = B.digest(self.response(payload, FINISH, DONE))
                self.assertTrue(d['ok'])
                self.assertEqual(d['generated_tokens'], value)
                self.assertEqual(d['decode_tps'], 0)

    def test_roles_are_protocol_values_but_content_is_private(self):
        for role in ['user', 'assistant', 'system', 'developer', 'tool', 'function']:
            with self.subTest(role=role), mock.patch.object(F.getpass, 'getuser', return_value=role):
                value = {'messages': [{'role': role, 'content': role + ' secret-api-key /home/alice/model'}]}
                out = F.redact_object(value, F.redactor(key='secret-api-key'))
                self.assertEqual(out['messages'][0]['role'], role)
                self.assertEqual(out['messages'][0]['content'], '<user> <key> /home/<user>/model')


if __name__ == '__main__':
    unittest.main()
