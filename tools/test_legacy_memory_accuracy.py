"""Memory evidence regressions using deterministic telemetry only; never probe hardware."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest import mock

from test_benchmark_accuracy import L


class MemoryTests(unittest.TestCase):
    def summary(self, rows):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()) as output:
            raw = Path(temp) / 'raw'
            raw.mkdir()
            (raw / 'fake.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            L.cmd_summary(NS(out=temp))
            return output.getvalue()

    def row(self, **changes):
        row = dict(config='fake', cell='4k', temp=0, prompt_tps=None, pred_tps=None,
                   prompt_ms=None, draft_n=None, draft_acc=None)
        row.update(changes)
        return row

    def test_vram_probe_distinguishes_unavailable_from_zero(self):
        for code, stdout, expected in [(0, '0\n', 0), (0, '12\n99\n', 12), (1, '12\n', None),
                                       (0, 'N/A\n', None), (0, '-1\n', None), (0, '', None)]:
            with self.subTest(code=code, stdout=stdout), mock.patch.object(L.subprocess, 'run', return_value=NS(returncode=code, stdout=stdout)):
                self.assertEqual(L.vram_used_mib(), expected)
        with mock.patch.object(L.subprocess, 'run', side_effect=OSError('unavailable')):
            self.assertIsNone(L.vram_used_mib())

    def test_sampler_starts_unknown(self):
        s = L.Sampler(123)
        self.assertIsNone(s.vram)
        self.assertIsNone(s.rss)
        self.assertIsNone(s.sysused)

    def test_partial_polling_preserves_observed_peak_and_coverage(self):
        with mock.patch.object(L.time, 'monotonic', side_effect=[100, 105, 110, 115, 116]):
            s = L.Sampler(123)
            with mock.patch.object(L, 'vram_used_mib', side_effect=[None, 0, 9]), \
                    mock.patch.object(L, 'rss_used_bytes', side_effect=[50, None, 20]), \
                    mock.patch.object(L, 'system_used_bytes', side_effect=[None, None, None]):
                for _ in range(3):
                    s.sample()
            snapshot = s.snapshot()
        self.assertEqual((s.vram, s.rss, s.sysused), (9, 50, None))
        self.assertEqual(snapshot['observations']['attempts'], 3)
        self.assertEqual(snapshot['observations']['valid'], {'vram': 2, 'rss': 2, 'system': 0})
        self.assertEqual(snapshot['last_sample_s'], 15)
        self.assertEqual(snapshot['snapshot_s'], 16)

    def test_incomplete_process_tree_is_not_a_complete_rss_measurement(self):
        parent, child = mock.Mock(), mock.Mock()
        parent.children.return_value = [child]
        parent.memory_info.return_value = NS(rss=100)
        child.memory_info.side_effect = L.psutil.NoSuchProcess(456)
        with mock.patch.object(L.psutil, 'Process', return_value=parent):
            self.assertIsNone(L.rss_used_bytes(123))
        child.memory_info.side_effect = None
        child.memory_info.return_value = NS(rss=0)
        parent.memory_info.return_value = NS(rss=0)
        with mock.patch.object(L.psutil, 'Process', return_value=parent):
            self.assertEqual(L.rss_used_bytes(123), 0)

    def test_baseline_failure_does_not_abort_engine_start(self):
        eng = L.Engine({}, {'name': 'fake'})
        with tempfile.TemporaryDirectory() as temp, mock.patch.object(eng, 'command', return_value=['fake']), \
                mock.patch.object(L, 'vram_used_mib', return_value=None), \
                mock.patch.object(L.psutil, 'virtual_memory', side_effect=OSError('unavailable')), \
                mock.patch.object(L.subprocess, 'Popen', return_value=NS(pid=123, poll=lambda: None)), \
                mock.patch.object(L.Sampler, 'start'), mock.patch.object(L, 'http', return_value={'status': 'ok'}):
            try:
                eng.start(temp)
                self.assertIsNone(eng.base_vram)
                self.assertIsNone(eng.base_used)
            finally:
                if eng.log:
                    eng.log.close()

    def test_memory_record_preserves_unknown_and_signed_system_delta(self):
        sampler = L.Sampler(123)
        eng = NS(sampler=sampler, base_vram=None, base_used=4 * 2**30)
        with mock.patch.object(L, 'vram_used_mib', return_value=0), \
                mock.patch.object(L, 'rss_used_bytes', return_value=None), \
                mock.patch.object(L, 'system_used_bytes', return_value=3 * 2**30):
            sampler.sample()
        record = L.memory_record(eng)
        self.assertEqual(record['vram_peak_mib'], 0)
        self.assertIsNone(record['vram_base_mib'])
        self.assertIsNone(record['rss_peak_gib'])
        self.assertEqual(record['sys_used_delta_gib'], -1)
        self.assertEqual(record['memory_scope'], 'sampler_start_to_record')
        self.assertIn('first', record['vram_scope'])
        self.assertIsNone(L.memory_record(NS(sampler=L.Sampler(123), base_vram=None, base_used=None))['sys_used_delta_gib'])
        json.dumps(record, allow_nan=False)

    def test_old_missing_null_and_sentinel_memory_values_are_unknown(self):
        for values in ({}, dict(vram_peak_mib=None, vram_base_mib=None, rss_peak_gib=None),
                       dict(vram_peak_mib=0, vram_base_mib=-1, rss_peak_gib=0)):
            with self.subTest(values=values):
                text = self.summary([self.row(**values)])
                self.assertIn('unknown', next(line for line in text.splitlines() if line.startswith('| fake |')))
                self.assertNotIn('(+1)', text)

    def test_known_zero_and_positive_legacy_data_are_distinguished(self):
        record = dict(vram_peak_mib=0, vram_base_mib=0, rss_peak_gib=0,
                      memory_scope='sampler_start_to_record',
                      memory_observations={'attempts': 1, 'valid': {'vram': 1, 'rss': 1, 'system': 1}})
        text = self.summary([self.row(**record)])
        self.assertIn('0 / +0', text)
        self.assertIn('0.0', text)
        old = self.summary([self.row(vram_peak_mib=100, vram_base_mib=20, rss_peak_gib=2)])
        self.assertIn('100 (coverage unknown)', old)
        self.assertIn('+80 (coverage unknown)', old)
        self.assertIn('coverage unknown', old)

    def test_later_record_keeps_earlier_peak_and_partial_label(self):
        s = L.Sampler(123)
        eng = NS(sampler=s, base_vram=0, base_used=0)
        with mock.patch.object(L, 'vram_used_mib', side_effect=[100, 10, None]), \
                mock.patch.object(L, 'rss_used_bytes', return_value=2**30), \
                mock.patch.object(L, 'system_used_bytes', return_value=2**30):
            s.sample()  # startup / warmup
            s.sample()  # current cell
            s.sample()  # unavailable observation
        record = L.memory_record(eng)
        self.assertEqual(record['vram_peak_mib'], 100)
        text = self.summary([self.row(**record)])
        self.assertIn('partial', text)
        self.assertIn('run-to-record', text)
        self.assertIn('earlier cells', text)

    def test_system_probe_distinguishes_zero_from_absent(self):
        for value in (0, None):
            with self.subTest(value=value), mock.patch.object(L.psutil, 'virtual_memory', return_value=NS(used=value)):
                self.assertEqual(L.system_used_bytes(), value)

    def test_run_records_sampler_scope_without_resetting_peaks_at_cells(self):
        class FakeEngine:
            def __init__(self, machine, cfg):
                self.sampler = L.Sampler(123)
                self.base_vram = self.base_used = None
                self.load_s, self.cmdline = 0, 'fake'

            def start(self, out):
                self.sampler.sample()

            def chat(self, *args):
                self.sampler.sample()
                return dict(text='answer', prompt_tps=None, pred_tps=None, draft_n=None,
                            draft_acc=None, prompt_ms=None)

            def stop(self):
                pass

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            machine = root / 'machine.json'
            machine.write_text(json.dumps({'configs': [{'name': 'fake'}]}))
            (root / 'prompt-4k.txt').write_text('corpus')
            args = NS(machine=str(machine), out=temp, configs=None, cells='4k', start_round=0, rounds=1)
            with mock.patch.object(L, 'Engine', FakeEngine), mock.patch.object(L, 'vram_used_mib', side_effect=[200] + [10] * 6), \
                    mock.patch.object(L, 'rss_used_bytes', return_value=0), mock.patch.object(L, 'system_used_bytes', return_value=None), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(L.cmd_run(args), 0)
            rows = [json.loads(line) for line in (root / 'raw/fake.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 4)
        self.assertEqual([row['memory_observations']['attempts'] for row in rows], [4, 5, 6, 7])
        for row in rows:
            self.assertEqual(row['vram_peak_mib'], 200)
            self.assertEqual(row['rss_peak_gib'], 0)
            self.assertIsNone(row['sys_used_delta_gib'])
            self.assertEqual(row['memory_scope'], 'sampler_start_to_record')
            self.assertLessEqual(row['memory_last_poll_completed_s'], row['memory_snapshot_s'])


if __name__ == '__main__':
    unittest.main()
