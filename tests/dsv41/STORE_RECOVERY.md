# Free-slot recovery fix (isolated from shadow observation)

Base: c62d5e3a15b40ea51c96aae9e8d36928965f53ab.
Branch: fix/store-free-slot-recovery. No shadow instrumentation or cache admission
policy is introduced. Only take_slot changes in production source.

## Standalone real-I/O reproducer

On Linux, with an existing C++17 compiler:

```
g++ -std=c++17 -O2 -pthread -Isrc/dsv41 tests/dsv41/store_recovery_repro.cpp -o /tmp/store-recovery
/tmp/store-recovery
```

The test uses an anonymous temporary file and one cache slot. Prefetch from the
empty file fails; demand observes that failure and releases its pin. The test then
writes valid synthetic bytes to the same file and retries demand. On the original
source it returns 1: `FAIL: retry rejected; fails=1 no_slot=1`. With the fix it returns
0 and checks all three matrices against the written bytes. Return 2 indicates a
fixture/setup or unexpected result, never the expected baseline failure.

## Cause and minimal production change

finish() marks the failed slot kFree but does not rewind free_hint_. The allocator's
first scan starts at that hint, and its later eviction scan ignores already-free
slots. A prefix scan now recovers those slots before growth or eviction, preserving
the existing suffix-first free-slot order and all ready-victim scoring.

Each of the three free-slot return sites now also requires pin==0 and !visible.
Failed reads and failed puts can be kFree while demand waiters still own references.
Reusing that storage before the waiters drain could expose replacement bytes to an
old waiter. The fix does not reset pins or alter their existing release paths.
The existing mutex still guards allocation; no worker, routing, quantization,
mathematical, tensor-layout, or model-execution code changes.

This changes default failure recovery intentionally. It adds a prefix scan when the
suffix contains no eligible free slot; its cost has not been benchmarked. It does
not establish model accuracy or universal numerical equivalence.

## Tests

Run `python3 tests/dsv41/run_store_recovery_test.py`. It needs the pinned Git object
locally and never fetches it. The runner compiles the standalone reproducer against
original and fixed headers, requiring the exact expected original failure and fixed
success. It compiles deterministic allocator-path tests using a temporary header
whose sole substitution is the I/O scheduler include. The fake scheduler does no
real I/O or concurrency and must never be included in an engine build.

The focused cases cover both LRU and LFU:

- A failed expert with duplicate demand pins, allocation before any waiter drains,
  and after only one waiter drains. Reuse succeeds only after the last pin drains.
- Failed put with a demand waiter, including the suffix free-slot path.
- Cleanup through wait_slots, preload, sync, shrink, and take_slot itself.
- Pinned, visible, and loading guards; every admission origin is exercised.
- Suffix-first order and prefix recovery without unnecessary arena growth.

Focused tests verify 456 synthetic matrices. Successful real-I/O replay output is
compared to the original header for both policies (18 matrices each). The existing
store test now checks all demand return flags, reference-read failures, and expected
comparison count; Linux RAM discovery is repaired in this test only. Its small
fixture checks 90/90 matrices. An empty-file negative fixture must exit 1.

`STORE_RECOVERY_SANITIZERS=address,undefined python3 tests/dsv41/run_store_recovery_test.py`
instruments the standalone before/after real-I/O reproducers and controlled allocator
paths. The larger real-I/O fixtures run without sanitizers. These checks do not prove
race freedom; no TSan, Windows, GPU/backend build, model output, live host, or hardware
performance validation is claimed. The visible test uses an identity address token,
not GPU memory registration. Separate parent review is required before integration.

Observed validation: normal and ASan/UBSan runner invocations both exited 0.
Independent review reran the normal suite successfully and found no blocker; it did
not independently rerun sanitizers. This is local CPU validation only.
