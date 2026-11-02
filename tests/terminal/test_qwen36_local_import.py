"""Strata import/preflight regressions with tiny synthetic GGUFs.

Tests use this source checkout by default; STRATA_SOURCE_ROOT can select a prepared
source copy. Tests never read a production model, contact a service, start an
engine or claim installed-application integration.
Fixture catalog byte counts/hashes stand for fixture bytes, not real checkpoints.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import ntpath
import os
from pathlib import Path, PureWindowsPath
import socket
import struct
import sys
import tempfile
import unittest
from unittest import mock


Q4_NAME = "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
Q6_NAME = "Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"
Q8_NAME = "Qwen3.6-35B-A3B-Q8_0.gguf"
Q4_REAL_BYTES = 22_853_663_008
Q4_REAL_SHA256 = "55983c5a75a1ab969824077b3bb3de4146e82a9234072b48ad4e8f92ad3fe9f1"


def string(value):
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def gguf_fixture(*, architecture="qwen35moe", expert_types=(12,), tokenizer="gpt2",
                 tokens=True, merges=True, overlap=False):
    """One block per expert format, correct alignment and real directory extents."""
    # GGML geometry is declared by the pinned reader, not inferred from a name.
    from tools.gguf_reader import BLOCK_GEOMETRY, GGML_TYPES

    pairs = []
    if architecture is not None:
        pairs.append(("general.architecture", 8, string(architecture)))
    if tokenizer is not None:
        pairs.append(("tokenizer.ggml.model", 8, string(tokenizer)))
    if tokens:
        pairs.append(("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 1) + string("a")))
    if merges:
        pairs.append(("tokenizer.ggml.merges", 9, struct.pack("<IQ", 8, 1) + string("a b")))
    header = (b"GGUF" + struct.pack("<IQQ", 3, len(expert_types), len(pairs))
              + b"".join(string(key) + struct.pack("<I", kind) + value for key, kind, value in pairs))
    offset = 0
    extent = 0
    for index, kind in enumerate(expert_types):
        geometry = BLOCK_GEOMETRY.get(GGML_TYPES.get(kind))
        block, size = geometry if geometry else (32, 32)
        position = 0 if overlap else offset
        header += string(f"blk.{index}.ffn_gate_exps.weight") + struct.pack("<IQIQ", 1, block, kind, position)
        extent = max(extent, position + size)
        offset = (position + size + 31) // 32 * 32
    return header + b"\x00" * (-len(header) % 32) + b"\x00" * extent


def snapshot(directory):
    return {path.relative_to(directory).as_posix(): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in directory.rglob("*") if path.is_file()}


class Qwen36LocalImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = os.environ.get("STRATA_SOURCE_ROOT")
        cls.source = Path(raw).resolve() if raw else Path(__file__).resolve().parents[2]
        if not (cls.source / "strata_control/models.py").is_file():
            raise RuntimeError("STRATA_SOURCE_ROOT does not contain the actual prepared Strata source")
        sys.path.insert(0, str(cls.source))
        cls.addClassCleanup(lambda: sys.path.remove(str(cls.source)))
        cls.models = importlib.import_module("strata_control.models")
        cls.preflight = importlib.import_module("strata_control.gguf_preflight")
        cls.deepseek = importlib.import_module("strata_control.dsv41_sets")
        cls.ProfileError = importlib.import_module("strata_control.profiles").ProfileError
        if not Path(cls.models.__file__).resolve().is_relative_to(cls.source):
            raise RuntimeError("Another cached Strata package would invalidate source provenance")

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="strata-import-regression-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.guards = contextlib.ExitStack()
        self.addCleanup(self.guards.close)
        # Discovery is convenience state, not the parser/selection under test.
        # Avoid reading any user's remembered installs or live settings.
        self.guards.enter_context(mock.patch.object(self.models, "_already_set_up", return_value=None))
        for target in ("socket.create_connection", "urllib.request.urlopen", "subprocess.Popen"):
            self.guards.enter_context(mock.patch(target, side_effect=AssertionError("Network/acquisition/process execution is forbidden in import fixtures")))

    def put(self, name=Q4_NAME, *, folder=None, raw=None, **metadata):
        directory = self.root if folder is None else folder
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / name
        path.write_bytes(gguf_fixture(**metadata) if raw is None else raw)
        return path

    @contextlib.contextmanager
    def fixture_catalog(self, *paths, size_delta=0, digest=None):
        """Inject only fixture identity declarations, not parser/candidate logic."""
        catalog = getattr(self.models, "qwen36_catalog", None)
        if catalog is None:
            # Pristine source must fail on the real unsupported-architecture
            # result, rather than an AttributeError for a patch-only helper.
            yield
            return
        entries = {path.name: (path.stat().st_size + size_delta,
                              digest or hashlib.sha256(path.read_bytes()).hexdigest()) for path in paths}
        with mock.patch.dict(catalog.FILES, entries):
            yield

    def plan(self, path):
        return self.models.plan_import(str(path))

    def assert_valid_qwen36(self, report, model="UD-Q4_K_XL"):
        self.assertEqual(report["file_set_status"], "plausible", report["issues"])
        self.assertEqual(report["issues"], [])
        candidate = report["candidate_selection"]
        self.assertIsNotNone(candidate)
        self.assertEqual((candidate["family"], candidate["model"]), ("qwen36", model))
        self.assertEqual(report["metadata"]["architecture"], "qwen35moe")
        self.assertTrue(report["metadata"]["payload_extent_valid"])
        self.assertFalse(report["would_write"])
        self.assertEqual(report["execution_compatibility"], "unverified")

    def test_valid_q4_binary_metadata_is_recognized_without_claiming_runtime_readiness(self):
        path = self.put()
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assert_valid_qwen36(report)
        self.assertEqual(report["file_count"], 1)
        self.assertEqual(report["observed_expert_types"], ["Q4_K"])

    def test_q6_and_q8_variants_match_their_catalog_selection(self):
        for filename, kind, model in ((Q6_NAME, 14, "UD-Q6_K_XL"), (Q8_NAME, 8, "Q8_0")):
            with self.subTest(model=model):
                path = self.put(filename, expert_types=(kind,))
                with self.fixture_catalog(path):
                    self.assert_valid_qwen36(self.plan(path), model)

    def test_native_supported_mixed_expert_formats_are_not_restricted_by_filename_precision(self):
        path = self.put(expert_types=(12, 42, 8))
        with self.fixture_catalog(path):
            self.assert_valid_qwen36(self.plan(path))

    def test_unknown_architecture_cannot_pass_by_using_an_official_filename(self):
        path = self.put(architecture="llama")
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")
        self.assertIn("unsupported_or_missing_architecture", report["issues"])

    def test_missing_architecture_remains_invalid(self):
        path = self.put(architecture=None)
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assertIn("unsupported_or_missing_architecture", report["issues"])

    def test_unknown_model_architecture_reports_unsupported_instead_of_path_failure(self):
        path = self.put("unsupported.gguf", architecture="unimplemented-model")
        report = self.plan(path)
        self.assertIn("unsupported_or_missing_architecture", report["issues"])
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")

    def test_corrupt_magic_is_rejected_before_architecture_adoption(self):
        path = self.put(raw=b"NOTG\x03\x00\x00\x00")
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")
        self.assertIn("GGUF header", " ".join(report["issues"]))

    def test_truncated_header_is_rejected(self):
        path = self.put(raw=b"GGUF" + struct.pack("<IQQ", 3, 1, 1))
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")
        self.assertIn("metadata", " ".join(report["issues"]).lower())

    def test_truncated_payload_is_rejected_even_with_matching_fixture_catalog_size(self):
        path = self.put(raw=gguf_fixture()[:-1])
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")
        self.assertIn("payload is shorter", " ".join(report["issues"]))

    def test_overlapping_tensor_extents_remain_invalid(self):
        path = self.put(expert_types=(12, 12), overlap=True)
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")
        self.assertIn("overlap", " ".join(report["issues"]))

    def test_reader_unknown_quantization_encoding_is_rejected(self):
        path = self.put(expert_types=(999,))
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")
        self.assertIn("unsupported encoding", " ".join(report["issues"]))

    def test_reader_valid_but_engine_unsupported_expert_quantization_is_rejected(self):
        # Q3_K is understood structurally by the reader but absent from the
        # pinned Qwen3.6 engine's routed-expert implementations.
        path = self.put(expert_types=(11,))
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")
        self.assertIn("missing_or_unexpected_expert_quantization", report["issues"])

    def test_empty_expert_set_is_not_a_valid_model(self):
        path = self.put(expert_types=())
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")

    def test_tokenizer_model_and_required_tables_are_still_checked(self):
        for values in ({"tokenizer": "sentencepiece"}, {"tokenizer": None}, {"tokens": False}, {"merges": False}):
            with self.subTest(values=values):
                path = self.put(**values)
                with self.fixture_catalog(path):
                    report = self.plan(path)
                self.assertIn("unsupported_or_incomplete_tokenizer_metadata", report["issues"])

    def test_missing_absolute_model_path_is_diagnosed_without_download(self):
        with self.assertRaisesRegex(self.ProfileError, "does not exist"):
            self.plan(self.root / "missing.gguf")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_relative_model_path_is_rejected_without_relocation(self):
        with self.assertRaisesRegex(self.ProfileError, "absolute"):
            self.models.plan_import("relative-model.gguf")

    def test_non_gguf_file_and_empty_folder_have_specific_diagnostics(self):
        other = self.root / "model.txt"
        other.write_text("unrelated file")
        with self.assertRaisesRegex(self.ProfileError, "GGUF file or directory"):
            self.plan(other)
        folder = self.root / "empty"
        folder.mkdir()
        with self.assertRaisesRegex(self.ProfileError, "no GGUF files"):
            self.plan(folder)

    def test_qwen36_unknown_variant_is_not_silently_replaced_with_flashnext(self):
        path = self.put("Qwen3.6-unreviewed-variant.gguf")
        report = self.plan(path)
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")
        self.assertIn("unrecognized_or_unsupported_model_variant", report["issues"])
        self.assertIsNone(report["candidate_selection"])

    def test_catalog_size_mismatch_is_not_accepted_as_integrity_verified(self):
        path = self.put()
        with self.fixture_catalog(path, size_delta=1):
            report = self.plan(path)
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")
        self.assertIn("catalog_model_size_mismatch", report["issues"])

    def test_candidate_exposes_catalog_identity_without_claiming_payload_sha_validation(self):
        path = self.put()
        fixture_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assert_valid_qwen36(report)
        candidate = report["candidate_selection"]
        self.assertEqual(candidate["expected_bytes"], path.stat().st_size)
        self.assertEqual(candidate["expected_sha256"], fixture_hash)
        self.assertEqual(report["execution_compatibility"], "unverified")

    def test_preview_is_read_only_and_does_not_hash_the_full_model_or_acquire_anything(self):
        folder = self.root / "H-drive-style Models" / "Qwen local model"
        path = self.put(folder=folder)
        before = snapshot(self.root)
        with self.fixture_catalog(path), mock.patch.object(self.models.hashlib, "sha256", side_effect=AssertionError("Preview must not hash a production payload")):
            report = self.plan(path)
        self.assert_valid_qwen36(report)
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(report["source"], str(path))
        self.assertEqual(report["model_dir"], str(folder))
        self.assertEqual(report["files"][0]["path"], str(path))

    def test_qwen36_does_not_advertise_flashnext_projector_acquisition(self):
        path = self.put()
        auxiliary = path.parent / "mmproj-F16.gguf"
        auxiliary.write_bytes(gguf_fixture(architecture="clip", expert_types=(12,)))
        before = snapshot(self.root)
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assert_valid_qwen36(report)
        self.assertFalse(report["reads_pictures"])
        self.assertEqual(report["projectors"], [])
        self.assertEqual(snapshot(self.root), before)

    def test_file_and_directory_import_select_same_catalog_family_and_original_bytes(self):
        path = self.put()
        with self.fixture_catalog(path):
            selected = self.plan(path)
            directory = self.plan(path.parent)
        self.assert_valid_qwen36(selected)
        self.assert_valid_qwen36(directory)
        self.assertEqual(selected["files"], directory["files"])
        self.assertEqual(selected["candidate_selection"], directory["candidate_selection"])

    def test_control_cli_json_preview_keeps_the_desktop_host_response_contract(self):
        main = importlib.import_module("strata_cli.cli").main
        path = self.put()
        output = io.StringIO()
        before = snapshot(self.root)
        with self.fixture_catalog(path), contextlib.redirect_stdout(output):
            code = main(["models", "import", str(path), "--plan", "--output", "json"])
        response = json.loads(output.getvalue())
        self.assertEqual(code, 0, response)
        self.assertTrue(response["ok"])
        self.assert_valid_qwen36(response["plan"])
        self.assertEqual(response["plan"]["files"][0]["path"], str(path))
        self.assertEqual(snapshot(self.root), before)

    def test_multiple_qwen36_variants_require_explicit_selection(self):
        q4 = self.put()
        q6 = self.put(Q6_NAME, expert_types=(14,))
        with self.fixture_catalog(q4, q6):
            report = self.plan(self.root)
        self.assertIn("multiple_model_groups", report["issues"])
        self.assertIsNone(report["candidate_selection"])
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")

    def test_selected_file_is_preserved_when_another_variant_is_next_to_it(self):
        q4 = self.put()
        q6 = self.put(Q6_NAME, expert_types=(14,))
        with self.fixture_catalog(q4, q6):
            report = self.plan(q4)
        self.assert_valid_qwen36(report)
        self.assertEqual(report["file_count"], 1)
        self.assertEqual(report["files"][0]["path"], str(q4))

    def test_parent_folder_finds_one_nested_qwen36_model_without_moving_it(self):
        folder = self.root / "Models" / "existing Qwen folder"
        path = self.put(folder=folder)
        with self.fixture_catalog(path):
            report = self.plan(self.root)
        self.assert_valid_qwen36(report)
        self.assertEqual(report["model_dir"], str(folder))
        self.assertTrue(path.is_file())

    def test_existing_flashnext_q2_model_still_passes_its_real_name_and_quantization_checks(self):
        first = self.put("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", architecture="qwen4exp", expert_types=(42,))
        self.put("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf", architecture="qwen4exp", expert_types=(42,))
        report = self.plan(first)
        self.assertEqual(report["issues"], [])
        self.assertEqual((report["candidate_selection"]["family"], report["candidate_selection"]["model"]), ("qwen", "Q2_0"))
        self.assertEqual(report["metadata"]["architecture"], "qwen4exp")
        self.assertFalse(report["would_write"])

    def test_existing_flashnext_quantization_rejection_remains_intact(self):
        first = self.put("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", architecture="qwen4exp", expert_types=(12,))
        self.put("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf", architecture="qwen4exp", expert_types=(42,))
        report = self.plan(first)
        self.assertIn("missing_or_unexpected_expert_quantization", report["issues"])
        self.assertEqual(report["file_set_status"], "invalid_or_incomplete")

    def test_existing_deepseek_name_size_dispatch_is_not_routed_through_qwen36(self):
        one = self.put("fixture-deepseek-00001-of-00002.gguf", raw=b"D" * 48)
        two = self.put("fixture-deepseek-00002-of-00002.gguf", raw=b"E" * 64)
        table = {"sets": {"ist-das-2.00bpw": {"label": "fixture two-file set", "shards": {one.name: 48, two.name: 64}, "legacy": {}}}, "engram": {}}
        with mock.patch.object(self.deepseek, "tables", return_value=table):
            report = self.plan(one)
        self.assertEqual(report["issues"], [])
        self.assertEqual(report["candidate_selection"]["family"], "deepseek41")
        self.assertIsNone(report["metadata"])
        self.assertFalse(report["would_write"])

    def test_catalog_manifest_keeps_real_model_size_hash_and_setup_family_identity(self):
        self.assertTrue(hasattr(self.models, "qwen36_catalog"), "Pure catalog data must be shared with actual setup")
        catalog = self.models.qwen36_catalog
        self.assertEqual(catalog.FILES[Q4_NAME], (Q4_REAL_BYTES, Q4_REAL_SHA256))
        specification = importlib.util.spec_from_file_location("strata_qwen36_regression_setup", self.source / "setup.py")
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        self.addCleanup(sys.modules.pop, specification.name, None)
        specification.loader.exec_module(module)
        self.assertEqual(module.QWEN36_FILES, catalog.FILES)
        self.assertEqual(module.OWN_PATH_FAMILIES["qwen36"]["arch"], "qwen35moe")
        self.assertEqual(set(module.OWN_PATH_FAMILIES["qwen36"]["models"]), {"UD-Q4_K_XL", "UD-Q6_K_XL", "Q8_0"})

    def test_windows_drive_and_space_arguments_are_preserved_under_windows_path_semantics(self):
        # Modeled path semantics, not a claim that Windows filesystem IO ran.
        raw = r"H:\Strata\Models\Qwen local\Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
        self.assertEqual(PureWindowsPath(raw).drive, "H:")
        with mock.patch.object(self.models.os.path, "abspath", side_effect=ntpath.abspath):
            absolute = self.models.absolute(raw)
        self.assertEqual(absolute, raw)
        self.assertNotIn("OneDrive", absolute)

    @unittest.skipUnless(os.name == "nt", "Native Windows filesystem integration is unavailable in this execution environment")
    def test_native_windows_temp_model_path_recognizes_existing_fixture_in_place(self):
        path = self.put(folder=self.root / "Models with spaces")
        self.assertTrue(PureWindowsPath(str(path)).drive)
        with self.fixture_catalog(path):
            report = self.plan(path)
        self.assert_valid_qwen36(report)
        self.assertEqual(report["files"][0]["path"], str(path))


if __name__ == "__main__":
    unittest.main()
