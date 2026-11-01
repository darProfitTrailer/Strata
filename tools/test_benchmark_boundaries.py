"""Overflow and incomplete speculative-counter regressions, using synthetic data only."""
import contextlib
import io
import json
import unittest
from unittest import mock
import test_benchmark_accuracy as A
from bench_metrics import validated_metrics
B, L = A.B, A.L


class BoundaryTests(unittest.TestCase):
    def summary(self, **updates):
        row = dict(config='fake', cell='4k', temp=0, prompt_tps=None, pred_tps=None,
                   prompt_ms=None, draft_n=10, vram_peak_mib=0, vram_base_mib=0, rss_peak_gib=0)
        row.update(updates)
        with A.tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as output:
            raw = A.Path(temp) / 'raw'
            raw.mkdir()
            (raw / 'fake.jsonl').write_text(json.dumps(row) + '\n' + json.dumps(row) + '\n')
            L.cmd_summary(A.argparse.Namespace(out=temp))
            return output.getvalue()

    def test_missing_and_null_accepted_count_stay_unknown(self):
        for updates in ({}, {'draft_acc': None}, {'draft_n': None, 'draft_acc': 1}, {'draft_n': 0, 'draft_acc': 0}):
            with self.subTest(updates=updates):
                self.assertIn('| - |', self.summary(**updates))

    def test_impossible_acceptance_fails(self):
        _, _, errors = validated_metrics(None, {'draft_n': 10, 'draft_n_accepted': 11})
        self.assertTrue(errors)
        with self.assertRaisesRegex(ValueError, 'accepted'):
            self.summary(draft_acc=11)

    def test_legitimate_acceptance_and_zero(self):
        self.assertIn('| 0.00 |', self.summary(draft_acc=0))
        self.assertIn('| 0.70 |', self.summary(draft_acc=7))

    def test_median_avoids_intermediate_overflow(self):
        result = B.stat([1e308, 1e308])
        self.assertEqual(result['median'], 1e308)
        json.dumps(result, allow_nan=False)
        self.assertNotIn('inf', self.summary(draft_acc=0, pred_tps=1e308))

    def test_unrepresentable_stat_is_unknown(self):
        self.assertIsNone(B.stat([10 ** 400]))

    def test_concurrency_overflow_fails_without_crashing(self):
        raw = dict(first=0.1, total=1, text='answer', usage={'completion_tokens': 10 ** 308},
                   timings=None, err=None, finish_reason='stop')
        for clients in (1, 2):
            with self.subTest(clients=clients), mock.patch.object(B, 'chat', return_value=raw), \
                    mock.patch.object(B.time, 'perf_counter', side_effect=[0, 0.01]):
                s = B.Suite('http://fake.invalid', '', 32768, 1, False, log=lambda *a: None)
                s.run_clients(clients)
                self.assertIsNone(s.out['clients']['total_tps'])
                self.assertGreater(s.out['clients']['errors'], 0)
                self.assertTrue(s.out['failures'])
                json.dumps(s.out, allow_nan=False)

    def test_agent_aggregate_overflow_stays_unknown(self):
        raw = dict(first=0.1, total=1, text='answer', usage={'completion_tokens': 10 ** 308, 'prompt_tokens': 10 ** 308,
                   'prompt_tokens_details': {'cached_tokens': 0}}, timings=None, err=None, finish_reason='stop')
        with mock.patch.object(B, 'chat', return_value=raw):
            s = B.Suite('http://fake.invalid', '', 32768, 1, False, log=lambda *a: None)
            s.run_agent()
        self.assertIsNone(s.out['agent']['reread_tokens'])
        json.dumps(s.out, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
