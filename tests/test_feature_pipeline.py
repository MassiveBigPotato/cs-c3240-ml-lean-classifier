"""Shared candidates -> training-only vocabularies -> logical cached subsets."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from unittest.mock import patch

import msgspec
import numpy as np
from test_candidates import candidate_records, store
from test_representation import feature_rows, training

import trustmebro.preprocessing.layout as layouts
from trustmebro.artifacts import encode_msgpack
from trustmebro.extraction import records as r
from trustmebro.learning import TrainingCfg, load_training_matrix, read_training_batches
from trustmebro.preprocessing import archives, candidates, features, pipeline, records
from trustmebro.preprocessing import scan as scheduler
from trustmebro.preprocessing.partition import logical_splits
from trustmebro.preprocessing.records import LabelPolicy, MinSupport, SplitCfg, read_split


def source(root: Path) -> tuple[Path, Path, tuple[r.Theorem, ...]]:
    db, labels = root / "development.db", root / "labels.json"
    theorems = tuple(
        msgspec.structs.replace(
            training(idx),
            exprs=tuple(
                msgspec.structs.replace(expr, name=f"Only.{idx}.{expr.name}") if isinstance(expr, r.Const) else expr
                for expr in training(idx).exprs
            ),
        )
        for idx in range(6)
    )
    dropped = msgspec.structs.replace(
        training(6),
        trns=tuple(msgspec.structs.replace(trn, tactic=r.Tactic("drop", "drop")) for trn in training(6).trns),
    )
    theorems += (dropped,)
    store(db, theorems)
    labels.write_bytes(msgspec.json.encode(LabelPolicy({"a": "A", "b": "B"}, "drop")))
    return db, labels, theorems


class FeaturePipelineTests(unittest.TestCase):
    def test_build_command_uses_policy_defaults_and_keeps_explicit_overrides(self) -> None:
        # Failures: CLI shadows changed optional config fields or nested split
        # quotas; one explicit override resets unrelated defaults. This checks
        # command-to-configuration wiring, not vocabulary quality or solver speed.
        @dataclass(frozen=True, slots=True)
        class BuildCfg(pipeline.VocabBuildCfg):
            depths: tuple[int, ...] = (2, 4)
            workers: int = 3
            hyp_slots: int = 7
            name_share: float = 0.4
            min_support: int = 6
            idx_mem_mib: int = 111
            score_mem_mib: int = 222
            name_mem_mib: int = 333

        split_cfg = SplitCfg(0.3, 0, MinSupport(4, 2), MinSupport(5, 3), 17)
        base = ["--db", "unused.db", "--labels", "unused.json", "--output", "unused", "--dims", "1234"]
        for flags, expected in (([], BuildCfg(1234)), (["--hyp-slots", "9"], BuildCfg(1234, hyp_slots=9))):
            with (
                self.subTest(flags=flags),
                patch.object(pipeline, "VocabBuildCfg", BuildCfg),
                patch.object(pipeline, "DEFAULT_VALIDATION_CFG", split_cfg),
                patch.object(pipeline, "prepare_features", side_effect=RuntimeError("stop before fitting")) as prepare,
                self.assertRaisesRegex(RuntimeError, "stop before fitting"),
            ):
                pipeline.main(base + flags)
            cfg = prepare.call_args.args[3]
            self.assertEqual(cfg.vocab, expected)
            # The preparation workflow intentionally uses seed 1, not the
            # outer corpus partition's default seed 0.
            self.assertEqual(cfg.split, replace(split_cfg, seed=1))

    def test_holdout_cli_frozen_values_and_logical_loaders(self) -> None:
        # Failures: extraction happens per split; validation shapes/names/labels
        # influence fitting; slots/attributes escape the total budget; cached
        # subsets lose labels, sharing weights, fractions or transition indices;
        # a manifest from another instance is accepted. Establishes fixture
        # semantics and public integration, not corpus-scale speed or RSS bounds.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, labels, theorems = source(root)
            output = root / "learning"
            before = db.read_bytes()
            run = subprocess.run(
                [
                    str(Path(sys.executable).parent / "prepare-features"),
                    "--db",
                    str(db),
                    "--labels",
                    str(labels),
                    "--output",
                    str(output),
                    "--dims",
                    "5000",
                    "--depths",
                    "1",
                    "2",
                    "3",
                    "--workers",
                    "2",
                    "--hyp-slots",
                    "2",
                    "--min-support",
                    "1",
                    "--validation-frac",
                    "0.3",
                ],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            report = json.loads(run.stdout)
            self.assertEqual(report["instances"], 1)
            instance = output / "instance-01"
            split = read_split(instance / "split.json")
            vocab = archives.read_vocab(instance / "vocab.zst")
            layout = layouts.compile_vocab(vocab)
            self.assertLessEqual(layout.width, 5000)
            invalid = msgspec.structs.replace(vocab, budget=records.DimBudget(layout.width - 1, 0, 0.2))
            with self.assertRaisesRegex(ValueError, "exceeds total budget"):
                layouts.compile_vocab(invalid)
            self.assertEqual(set(vocab.selection.theorems), set(split.train))
            shape_ids: set[bytes] = set()
            train_consts: set[str] = set()
            for theorem in theorems:
                if theorem.name in split.train:
                    shape_ids.update(shape.ident for shape in candidates.extract_cands(theorem, (1, 2, 3)).shapes)
                    train_consts.update(expr.name for expr in theorem.exprs if isinstance(expr, r.Const))
            self.assertTrue({entry.ident for entry in vocab.entries} <= shape_ids)
            self.assertTrue(set(vocab.representation.heads) <= train_consts)
            self.assertTrue({name for _, _, name in vocab.representation.names} <= train_consts)
            if vocab.supervision is not None:
                expected = tuple(split.train_stats.labels[label].trns for label in split.policy.labels)
                self.assertEqual(vocab.supervision.counts, expected)
            archive = instance / "features.zst"
            actual: dict[tuple[str, int], np.ndarray] = {}
            for subset, wanted in (("train", set(split.train)), ("validation", set(split.validation))):
                batches = list(read_training_batches(archive, split=split, subset=subset, cfg=TrainingCfg(rows=3)))
                self.assertTrue({name for batch in batches for name in batch.theorems} <= wanted)
                for batch in batches:
                    for idx, (name, step) in enumerate(zip(batch.theorems, batch.steps, strict=True)):
                        self.assertNotIn((name, int(step)), actual)
                        actual[name, int(step)] = batch.matrix[idx : idx + 1].toarray()
                loaded = load_training_matrix(archive, split=split, subset=subset, max_bytes=2**20)
                np.testing.assert_array_equal(
                    loaded.matrix.toarray(), np.vstack([batch.matrix.toarray() for batch in batches])
                )
            for theorem in theorems:
                if theorem.name == "Train.6":
                    continue
                expected_rows = features.encode_theorem(theorem, layout)
                for idx, step in enumerate(expected_rows.steps):
                    np.testing.assert_array_equal(
                        actual[theorem.name, int(step)], expected_rows.matrix[idx : idx + 1].toarray()
                    )
            self.assertEqual(len(actual), 12)
            wrong = msgspec.structs.replace(split, cfg=SplitCfg(test_frac=0.3, seed=99))
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                list(read_training_batches(archive, split=wrong, subset="train"))
            self.assertEqual(db.read_bytes(), before)

    def test_grouped_folds_share_candidates_resume_and_reject_changed_inputs(self) -> None:
        # Failures: repeated discovery, overlapping/missing validation groups,
        # support computed using held-out theorems, unnecessary rerun of completed
        # stages, changed artifacts/settings/selector contracts silently reused or overwritten.
        # Establishes per-fold ownership, completion reuse and failure behavior;
        # does not establish optimal stratification or fault tolerance to all crashes.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, labels, _ = source(root)
            src = root / "candidates.zst"
            scheduler.scan_cands(db, src, depths=(1, 2))
            cfg = pipeline.PrepCfg(
                pipeline.VocabBuildCfg(5000, depths=(1, 2), hyp_slots=2, min_support=1), SplitCfg(seed=2), folds=3
            )
            output = root / "learning"
            with patch.object(pipeline, "scan_cands", side_effect=AssertionError("repeat discovery")):
                pipeline.prepare_features(db, labels, output, cfg, cands=src)
            held_out: set[str] = set()
            for idx in range(3):
                instance = output / f"instance-{idx + 1:02d}"
                split = read_split(instance / "split.json")
                self.assertEqual(split.cfg.test_frac, 1 / 3)
                self.assertFalse(held_out & set(split.validation))
                held_out.update(split.validation)
                self.assertTrue(all(val.trns >= 1 for val in split.validation_stats.labels.values()))
                self.assertEqual(set(archives.read_vocab(instance / "vocab.zst").selection.theorems), set(split.train))
                self.assertEqual(len(list(feature_rows(instance / "features.zst"))), 7)
            self.assertEqual(len(held_out), 7)
            stamps = {path: path.stat().st_mtime_ns for path in output.glob("instance-*/*.zst")}
            with (
                patch.object(pipeline, "fit_vocab", side_effect=AssertionError("refit")),
                patch.object(pipeline, "convert_corpus", side_effect=AssertionError("reconvert")),
            ):
                resumed = pipeline.prepare_features(db, labels, output, cfg, cands=src)
            self.assertTrue(resumed["stages"]["instance-01"]["fitting"]["reused"])
            self.assertEqual(stamps, {path: path.stat().st_mtime_ns for path in stamps})
            other = read_split(output / "instance-02/split.json")
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                list(read_training_batches(output / "instance-01/features.zst", split=other, subset="train"))
            run = output / "run.json"
            request = json.loads(run.read_text())
            outdated = {key: val for key, val in request.items() if key != "shape_selection"}
            run.write_text(json.dumps(outdated))
            with self.assertRaisesRegex(ValueError, "inputs/settings changed"):
                pipeline.prepare_features(db, labels, output, cfg, cands=src)
            run.write_text(json.dumps(request))
            labels.write_bytes(msgspec.json.encode(LabelPolicy({"a": "changed", "b": "B"}, "drop")))
            with self.assertRaisesRegex(ValueError, "inputs/settings changed"):
                pipeline.prepare_features(db, labels, output, cfg, cands=src)

    def test_preflight_and_incomplete_candidate_population_do_not_publish_features(self) -> None:
        # Failures: impossible width launches discovery; source artifacts are
        # overwritten; missing held-out candidates publish an apparently complete
        # development archive, or unsupported folds proceed to fitting. Establishes preflight and final population checks,
        # not recovery of corrupted source data or a minimum feasible width proof.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, labels, _ = source(root)
            cfg = pipeline.PrepCfg(pipeline.VocabBuildCfg(1))
            with (
                patch.object(pipeline, "scan_cands", side_effect=AssertionError("discovery")),
                self.assertRaisesRegex(ValueError, "statistics/slot"),
            ):
                pipeline.prepare_features(db, labels, root / "tiny", cfg)
            self.assertFalse((root / "tiny").exists())
            policy = LabelPolicy({"a": "A", "b": "B"}, "drop")
            split = logical_splits(db, policy, SplitCfg(test_frac=0.3, seed=1))[0]
            src, broken = root / "candidates.zst", root / "broken.zst"
            scheduler.scan_cands(db, src, depths=(1, 2))
            impossible = pipeline.PrepCfg(
                pipeline.VocabBuildCfg(5000, depths=(1, 2)), SplitCfg(test_min=MinSupport(trns=1000)), folds=3
            )
            with (
                patch.object(pipeline, "fit_vocab", side_effect=AssertionError("fitting")),
                self.assertRaisesRegex(ValueError, "fails label-support minima"),
            ):
                pipeline.prepare_features(db, labels, root / "unsupported-folds", impossible, cands=src)
            self.assertFalse((root / "unsupported-folds/instance-01/split.json").exists())
            selection = archives.read_selection(src)
            archives.publish(
                broken,
                (
                    encode_msgpack(item)
                    for item in (
                        selection,
                        *[theorem for theorem in candidate_records(src) if theorem.name != split.validation[0]],
                    )
                ),
                sources=(src,),
                replace=False,
            )
            cfg = pipeline.PrepCfg(
                pipeline.VocabBuildCfg(5000, depths=(1, 2), hyp_slots=2, min_support=1), SplitCfg(test_frac=0.3, seed=1)
            )
            with self.assertRaisesRegex(ValueError, "missing development"):
                pipeline.prepare_features(db, labels, root / "broken-run", cfg, cands=broken)
            self.assertFalse((root / "broken-run/instance-01/features.zst").exists())
