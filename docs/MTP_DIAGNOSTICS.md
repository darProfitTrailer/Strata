# Offline MTP diagnostics patch

The original diagnostics patch targets v0.2.8, source commit
`870aedd9e1ebe1c91f94da93b59af17729f6c5c0`; `878ef09cc9092a41a37dac32fd6fe461e47fa326`
is its annotated tag object, not a source commit. This port retains the newer
`setup.py` child-output decoding behavior (`errors="replace"`) on current upstream.
The inspected user fork main was `c62d5e3a15b40ea51c96aae9e8d36928965f53ab`.
Its `tools/mtp_pack.py` matches v0.2.8; its setup.py has subsequent unrelated changes.
Do not replace a newer setup.py wholesale with this release's file.

Set `STRATA_MTP_DIAGNOSTICS=1` in an isolated source process to enable the additional
logging. This does not select a backend or change any packing option. Without the
variable, existing log text and exception behavior remain in place.

The packer flushes begin/done markers for manifest reading, writer initialization,
expert mapping, quantization, concatenation, error calculation, tensor registration,
dense processing, header/metadata/tensor writing, writer closing and report writing.
Tensor indices follow the existing sorted manifest order. Failure markers contain
only phase, exception class and numeric errno, without exception text or paths.
In diagnostic mode, the packer's own tensor/output messages use indices/fixed labels;
setup's packer command line is also hidden. The CLI exits 1 on a Python exception.
Setup records the actual subprocess return code before its existing exit-1 wrapper.
A process spawn error has no child return code: it records class/errno and exits 1.
Negative POSIX return codes and Windows status values are printed without rewriting.

These markers are not proof of a cause. A hard process termination may leave only a
begin marker and the parent's return code. Import failures happen before phase
instrumentation. Third-party writer output and other setup steps are not a general
log-redaction facility. Review complete logs before sharing them.

A log ending at down_proj alone does not prove an allocation failure: existing
messages appear after tensor completion, and later processing/writes lack markers.
No root cause has been established for the user's actual failure.

Validation: `python tools/test_mtp_diagnostics.py` uses four tiny synthetic BF16
tensors, fake GGUF writers and fake subprocess results. It covers missing/truncated
files, allocation failure, write permission/disk errors, report disk failure,
marker order/flush, CLI exception exit, and parent return-code evidence. Recorded
writer metadata and tensor payload bytes match the baseline for q2_0, q4_0 and q8_0.
Timing fields in the report are observational and may differ.

Additional validation fetched only source for the pinned llama.cpp commit
`3cf03257f219afbe7334045ff7c6a06ac68c627d` into an independent sparse checkout.
Run `STRATA_GGUF_PY=/path/to/pinned/llama.cpp/gguf-py python tools/test_mtp_diagnostics_real_writer.py`
to repeat the real writer checks. For four tiny synthetic tensors, actual serialized
GGUF files are byte-identical to v0.2.8 with diagnostics both enabled and disabled:

| Format | Bytes | SHA-256 |
| --- | ---: | --- |
| q2_0 | 1184 | 2685c4dc2b797ec98358ab431d1ba172f480b7bb7c3541cfc48edba99852ff72 |
| q4_0 | 1216 | 141bc19d2d2b9726eb91d665aa3bb98e15ecad825730d9e9423ddaf15014e83e |
| q8_0 | 1344 | ce9ceb0679b9f721dcd1bce9122014ae44f17f9d919a6b6d0b325f4bb0637720 |

Reports also match after excluding elapsed seconds. Real subprocess tests confirm
exit 1 and failure markers for a missing output directory and missing/truncated
expert input, without an exception traceback.

Windows process behavior, full-size memory use, GPU execution and live PC behavior
have not been tested. No model was downloaded; no app/configuration/model on the PC
was changed; nothing was pushed.
