"""Fixture export → persistent vectors → bounded and full training loaders."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import msgspec
import numpy as np
from test_candidates import store
from test_representation import training

import trustmebro.preprocessing.layout as layouts
from trustmebro.artifacts import encode_msgpack
from trustmebro.extraction import records as r
from trustmebro.learning import TrainingCfg, load_training_matrix, read_training_batches
from trustmebro.preprocessing import archives, candidates, features, records
from trustmebro.preprocessing.records import LabelPolicy


def build(root: Path) -> tuple[Path, LabelPolicy, dict[tuple[str, int], tuple[np.ndarray, int]], int]:
    db, vocab_path, policy_path = (root / name for name in ("train.db", "vocab.zst", "labels.json"))
    theorems = tuple(training(idx) for idx in range(3))
    theorems = (
        theorems[0],
        msgspec.structs.replace(
            theorems[1],
            trns=tuple(msgspec.structs.replace(trn, tactic=r.Tactic("drop", "drop")) for trn in theorems[1].trns),
        ),
        theorems[2],
    )
    store(db, theorems)
    observed = candidates.extract_cands(theorems[0], (1, 2))
    entries = tuple(records.Entry(shape.ident, shape.edges) for shape in observed.shapes)
    policy = LabelPolicy({"a": "A", "b": "B", "absent": "Absent"}, "drop")
    policy_path.write_bytes(msgspec.json.encode(policy))
    selection = records.Selection(str(db), db.stat().st_size, db.stat().st_mtime_ns, (1, 2), None, None, None)
    representation = records.Representation(
        records.AttrPolicy(hyp_slots=2, name_dims=4),
        ("f",),
        (),
        selection,
        "fixture",
        0,
        0,
        msgspec.json.encode(policy),
    )
    vocab = records.Vocab((1, 2), entries, representation=representation)
    archives.publish(vocab_path, (encode_msgpack(vocab),), sources=(db,), replace=False)
    layout = layouts.compile_vocab(vocab)
    expected: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
    for theorem in theorems:
        rows = features.encode_theorem(theorem, layout)
        for idx, tactic in enumerate(rows.tactics):
            label = policy.label(tactic)
            if label is not None:
                expected[theorem.name, idx] = (rows.matrix[idx : idx + 1].toarray(), policy.label_ids[label])
    subprocess.run(
        [
            str(Path(sys.executable).parent / "prepare-features"),
            "--db",
            str(db),
            "--vocab",
            str(vocab_path),
            "--labels",
            str(policy_path),
            "--output",
            str(root / "converted"),
            "--workers",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    # A training archive must remain usable without source data or fitting inputs.
    db.rename(root / "source.off")
    vocab_path.rename(root / "vocab.off")
    policy_path.rename(root / "labels.off")
    return root / "converted/features.zst", policy, expected, layout.width


class TrainingArchiveTests(unittest.TestCase):
    def test_export_cli_to_stream_and_preallocated_matrix_without_feature_construction(self) -> None:
        # Failures: loaders rebuild graphs, require deleted fitting inputs,
        # misalign labels/steps, lose absent class IDs or fractions, drop rows at
        # theorem/batch boundaries, exceed row/nnz limits, or mutate the archive.
        # Establishes cached public training paths and numeric equivalence, not
        # large-corpus runtime, process-RSS bounds, or classifier convergence.
        with tempfile.TemporaryDirectory() as tmp:
            path, policy, expected, width = build(Path(tmp))
            original = path.read_bytes()
            with (
                patch.object(features, "encode_theorem", side_effect=AssertionError("feature reconstruction")),
                patch.object(features, "encode_cands", side_effect=AssertionError("feature reconstruction")),
                patch.object(candidates, "extract_cands", side_effect=AssertionError("graph reconstruction")),
            ):
                for dtype in ("float64", "float32"):
                    cfg = TrainingCfg(rows=3, nnz=100000, dtype=dtype)
                    with closing(read_training_batches(path, cfg=cfg)) as stream:
                        batches = list(stream)
                    self.assertEqual([len(batch.label_ids) for batch in batches], [3, 1])
                    actual: dict[tuple[str, int], tuple[np.ndarray, int]] = {}
                    for batch in batches:
                        self.assertEqual(batch.classes, policy.labels)
                        self.assertEqual(batch.matrix.shape[1], width)
                        self.assertEqual(batch.matrix.dtype, np.dtype(dtype))
                        self.assertEqual(batch.matrix.indices.dtype, np.dtype("int32"))
                        self.assertLessEqual(batch.matrix.nnz, cfg.nnz)
                        for idx, (name, step) in enumerate(zip(batch.theorems, batch.steps, strict=True)):
                            actual[name, int(step)] = batch.matrix[idx : idx + 1].toarray(), int(batch.label_ids[idx])
                    self.assertEqual(actual.keys(), expected.keys())
                    for key in expected:
                        np.testing.assert_allclose(actual[key][0], expected[key][0], rtol=1e-6)
                        self.assertEqual(actual[key][1], expected[key][1])
                    loaded = load_training_matrix(path, cfg=cfg, max_bytes=2**20)
                    np.testing.assert_array_equal(
                        loaded.matrix.toarray(), np.vstack([b.matrix.toarray() for b in batches])
                    )
                    np.testing.assert_array_equal(loaded.label_ids, np.concatenate([b.label_ids for b in batches]))
                    self.assertEqual(loaded.theorems, tuple(name for b in batches for name in b.theorems))
                    np.testing.assert_array_equal(loaded.steps, np.concatenate([b.steps for b in batches]))
                    # The nnz bound also splits inside a theorem, independently of row limits.
                    max_row = max(np.count_nonzero(data) for data, _ in expected.values())
                    for batch in read_training_batches(path, cfg=TrainingCfg(nnz=max_row, rows=100)):
                        self.assertLessEqual(batch.matrix.nnz, max_row)
            self.assertEqual(path.read_bytes(), original)
            # Failure: the cached loader imports graph construction or solvers
            # despite no reconstruction call. A fresh interpreter checks the
            # actual dependency boundary, not the already-loaded test process.
            probe = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import sys
from pathlib import Path
from trustmebro.learning import load_training_matrix
assert len(load_training_matrix(Path(sys.argv[1])).label_ids) == 4
for name in (
    'graph_tool', 'trustmebro.preprocessing.candidates',
    'trustmebro.preprocessing.coverage', 'trustmebro.preprocessing.supervised',
    'trustmebro.preprocessing.partition', 'scipy.optimize',
):
    assert name not in sys.modules, name
""",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
            )
            self.assertEqual(probe.returncode, 0, probe.stderr)

    def test_resource_policy_and_completion_errors_fail_without_truncating_training_data(self) -> None:
        # Failures: a large frame allocates despite the guard, oversized single
        # rows silently truncate, a full load ignores its budget, changed class
        # mappings are accepted, or partial archives look complete. Establishes
        # failure behavior at the training boundary, not an OS memory cap.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, _, _, _ = build(root)
            with self.assertRaisesRegex(ValueError, "frame-size"):
                list(read_training_batches(path, cfg=TrainingCfg(frame_bytes=8)))
            with self.assertRaisesRegex(MemoryError, "one feature row"):
                list(read_training_batches(path, cfg=TrainingCfg(nnz=1)))
            with self.assertRaisesRegex(MemoryError, "numeric-buffer budget"):
                load_training_matrix(path, max_bytes=1)
            with self.assertRaisesRegex(ValueError, "frozen"):
                list(read_training_batches(path, LabelPolicy({"a": "Different"}, "drop")))
            with self.assertRaisesRegex(ValueError, "positive"):
                list(read_training_batches(path, cfg=TrainingCfg(rows=0)))
            frames = list(archives.read_frames(path))
            broken = root / "partial.zst"
            archives.publish(broken, frames[:-1], sources=(path,), replace=False)
            with self.assertRaisesRegex(ValueError, "footer"):
                load_training_matrix(broken)

    def test_legacy_policy_and_empty_selection_preserve_width_and_classes(self) -> None:
        # Failures: older cached vectors require regeneration, missing policies
        # are guessed, or an all-dropped selection loses width/classes. Checks
        # reuse without migrations and empty-data semantics, not label quality.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path, policy, _, width = build(root)
            frames = list(archives.read_frames(path))
            header = msgspec.msgpack.decode(frames[0], type=archives.FeatureHeader)
            legacy = root / "legacy.zst"
            archives.publish(
                legacy,
                (encode_msgpack(msgspec.structs.replace(header, label_policy=None)), *frames[1:]),
                sources=(path,),
                replace=False,
            )
            with self.assertRaisesRegex(ValueError, "no frozen labels"):
                list(read_training_batches(legacy))
            self.assertEqual(load_training_matrix(legacy, policy).matrix.shape, (4, width))
            empty = root / "empty.zst"
            dropped = LabelPolicy({"missing": "Absent"}, "drop")
            archives.publish(
                empty,
                (
                    encode_msgpack(msgspec.structs.replace(header, label_policy=msgspec.json.encode(dropped))),
                    *frames[1:],
                ),
                sources=(path,),
                replace=False,
            )
            self.assertEqual(list(read_training_batches(empty)), [])
            loaded = load_training_matrix(empty)
            self.assertEqual(loaded.matrix.shape, (0, width))
            self.assertEqual(loaded.classes, ("Absent",))
            self.assertEqual(loaded.theorems, ())


if __name__ == "__main__":
    unittest.main()
