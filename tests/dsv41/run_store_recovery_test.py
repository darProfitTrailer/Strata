"""CPU-only before/after recovery checks, no downloads. Requires Linux, Git, g++."""
from pathlib import Path
import os
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
base = 'c62d5e3a15b40ea51c96aae9e8d36928965f53ab'
with tempfile.TemporaryDirectory(prefix='strata-recovery-') as temporary:
    tmp = Path(temporary)
    original = tmp / 'original'
    original.mkdir()
    (original/'ds4_store.hpp').write_text(subprocess.check_output(['git', 'show', base+':src/dsv41/ds4_store.hpp'], cwd=root, text=True))
    flags = []
    if os.environ.get('STORE_RECOVERY_SANITIZERS'):
        flags = ['-fsanitize='+os.environ['STORE_RECOVERY_SANITIZERS'], '-fno-omit-frame-pointer', '-g']
    def build(name, source, include, sanitize=False):
        binary = tmp/name
        subprocess.run(['g++', '-std=c++17', '-O1', '-pthread', '-I'+str(include),
                        '-I'+str(root/'src/dsv41'), '-I'+str(root/'tests/dsv41'),
                        *(flags if sanitize else []), str(root/'tests/dsv41'/source), '-o', str(binary)], check=True)
        return str(binary)
    for name, include, expected in [('before', original, 1), ('after', root/'src/dsv41', 0)]:
        binary = build(name, 'store_recovery_repro.cpp', include, True)
        result = subprocess.run([binary], capture_output=True, text=True)
        print(name+': '+result.stdout, end='')
        assert result.returncode == expected, result.stderr
        if name == 'before': assert 'fails=1 no_slot=1' in result.stdout
    controlled = tmp/'controlled'
    controlled.mkdir()
    header = (root/'src/dsv41/ds4_store.hpp').read_text()
    assert header.count('#include "ds4_pool.hpp"') == 1
    (controlled/'ds4_store.hpp').write_text(header.replace('#include "ds4_pool.hpp"', '#include "store_fake_io.hpp"'))
    subprocess.run([build('paths', 'store_recovery_paths_test.cpp', controlled, True)], check=True)
    fixture = tmp/'fixture.bin'
    fixture.write_bytes(b''.join(bytes([i])*4096 for i in range(24)))
    outputs = []
    for name, include in [('original', original), ('fixed', root/'src/dsv41')]:
        binary = build('replay-'+name, 'store_success_replay_test.cpp', include)
        outputs.append(subprocess.check_output([binary, str(fixture)], text=True))
    assert outputs[0] == outputs[1], 'successful replay baseline changed'
    print(outputs[1], end='')
    fixture.write_bytes(b''.join(bytes([i%256])*4096 for i in range(384*3)))
    binary = build('store-test', 'store_test.cpp', root/'src/dsv41')
    args = [binary, str(fixture), '0', str(384*4096), str(768*4096), '4096', '0.004', '0', '20']
    subprocess.run(args, check=True)
    fixture.write_bytes(b'')
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 1, result.stdout+result.stderr
    assert 'compared 0 matrices' in result.stdout
    print('Empty-file failure correctly rejected.')
