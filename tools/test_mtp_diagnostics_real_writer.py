"""Offline real-writer regression; requires STRATA_GGUF_PY at setup's pinned checkout."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PIN = '3cf03257f219afbe7334045ff7c6a06ac68c627d'


def main():
    gguf_path = Path(os.environ['STRATA_GGUF_PY']).resolve()
    actual = subprocess.check_output(['git', '-C', str(gguf_path), 'rev-parse', 'HEAD'], text=True).strip()
    assert actual == PIN, (actual, PIN)
    baseline = subprocess.check_output(['git', 'show', 'v0.2.8:tools/mtp_pack.py'], cwd=ROOT, text=True)
    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        src = folder / 'synthetic-private-input'
        src.mkdir()
        original = folder / 'baseline.py'
        original.write_text(baseline)
        manifest = []
        for i, (name, shape) in enumerate([
            ('mtp.layers.0.mlp.experts.gate_up_proj', [2, 1, 64]),
            ('mtp.layers.0.mlp.experts.down_proj', [2, 1, 64]),
            ('norm', [64]), ('dense', [1, 64])]):
            payload = (np.linspace(-1, 1, np.prod(shape), dtype=np.float32).view(np.uint32) >> 16).astype('<u2').tobytes()
            (src / f'{i}.bin').write_bytes(payload)
            manifest.append(dict(name=name, shape=shape, file=f'{i}.bin', sha256=hashlib.sha256(payload).hexdigest()))
        (src / 'mtp-manifest.json').write_text(json.dumps(manifest))
        def run(script, out, kind='q2_0', enabled='1'):
            env = {**os.environ, 'PYTHONPATH': str(ROOT / 'tools'), 'STRATA_MTP_DIAGNOSTICS': enabled}
            return subprocess.run([sys.executable, str(script), '--src', str(src), '--out', str(out), '--experts', kind],
                                  env=env, capture_output=True, text=True, timeout=30)
        patched = ROOT / 'tools/mtp_pack.py'
        for kind in ('q2_0', 'q4_0', 'q8_0'):
            before = folder / f'{kind}.baseline.gguf'
            after = folder / f'{kind}.patched.gguf'
            disabled = folder / f'{kind}.disabled.gguf'
            for script, out, enabled in ((original, before, '0'), (patched, after, '1'), (patched, disabled, '0')):
                result = run(script, out, kind, enabled)
                assert result.returncode == 0, result.stderr
                if enabled == '1':
                    assert '[mtp-pack] done pack' in result.stdout
                    assert str(src) not in result.stdout + result.stderr
                    assert str(out) not in result.stdout + result.stderr
            assert before.read_bytes() == after.read_bytes() == disabled.read_bytes(), kind
            reports = [json.loads(Path(str(p) + '.report.json').read_text()) for p in (before, after, disabled)]
            for report in reports:
                for tensor in report['tensors']:
                    tensor.pop('seconds', None)
            assert reports[0] == reports[1] == reports[2]
            print(f'{kind}: identical real GGUF bytes ({after.stat().st_size}), sha256={hashlib.sha256(after.read_bytes()).hexdigest()}')
        # Actual OS error, missing/truncated input and entrypoint exit propagation.
        bad = run(patched, folder / 'missing-parent' / 'out.gguf')
        assert bad.returncode == 1 and 'exception=FileNotFoundError' in bad.stdout
        assert 'Traceback' not in bad.stderr
        target = src / '0.bin'
        target.write_bytes(b'\x00\x00')
        bad = run(patched, folder / 'truncated.gguf')
        assert bad.returncode == 1 and 'failed expert-map exception=ValueError' in bad.stdout
        assert str(src) not in bad.stdout + bad.stderr
        target.unlink()
        bad = run(patched, folder / 'missing.gguf')
        assert bad.returncode == 1 and 'failed expert-map exception=FileNotFoundError' in bad.stdout
        assert 'Traceback' not in bad.stderr
        print('Real subprocess failures: missing output directory, truncated input, missing input passed; exit=1, no traceback.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
