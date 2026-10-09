"""Small frozen feature archive → real SGD fitting → reloadable evaluated checkpoints.

Failure cases: skipped epoch batches or misaligned shuffling; tactic-kind counts
mistaken for class IDs; absent classes omitted; validation leakage in scaling;
live rather than best-epoch snapshots; inconsistent metrics; nondeterministic
seeds; overwritten artifacts or silently accepted policy/vocabulary mismatches.
These checks establish the CPU archive boundary, not full-corpus speed or convergence.
"""

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import msgspec
import numpy as np
from scipy.sparse import csr_array
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_recall_fscore_support

import trustmebro.preprocessing.layout as layouts
from trustmebro.artifacts import encode_msgpack
from trustmebro.extraction.records import Tactic
from trustmebro.logistic import train as logistic
from trustmebro.preprocessing import archives, records
from trustmebro.preprocessing.records import LabelPolicy


def archive_fixture(root: Path) -> tuple[csr_array, np.ndarray]:
    policy = LabelPolicy({**{f"kind-{idx}": f"class-{idx}" for idx in range(4)}, "alternate": "class-0"}, "error")
    train_names, val_names = ("train-0", "train-1"), ("val-0", "val-1")
    selection = records.Selection("absent-source.db", 1, 2, (1,), None, train_names + val_names, 0)
    support = {
        label: records.LabelSupport(6 if idx < 3 else 0, 2 if idx < 3 else 0) for idx, label in enumerate(policy.labels)
    }
    split = records.SplitManifest(
        selection,
        policy,
        records.SplitCfg(),
        1,
        1,
        train_names,
        val_names,
        records.SubsetStats(2, 18, 18, 0, support),
        records.SubsetStats(2, 9, 9, 0, support),
        "fixture",
        None,
    )
    fitted = msgspec.structs.replace(selection, theorems=train_names)
    edges = ((),)
    representation = records.Representation(
        records.AttrPolicy(hyp_slots=1, name_dims=0), (), (), fitted, "fixture", 0, 0, msgspec.json.encode(policy)
    )
    vocab = records.Vocab(
        (1,),
        (records.Entry(sha256(msgspec.msgpack.encode(edges)).digest(), edges),),
        selection=fitted,
        representation=representation,
    )
    width = layouts.compile_vocab(vocab).width
    rows: list[records.FeatureRows] = []
    validation: tuple[csr_array, np.ndarray] | None = None
    for names, count in ((train_names, 18), (val_names, 9)):
        labels = np.tile(np.arange(3, dtype=np.int64), count // 3)
        labels = labels[np.random.default_rng(7).permutation(count)]
        matrix = np.zeros((count, width), dtype=np.float32)
        matrix[np.arange(count), labels] = 2
        matrix[:, 5] = np.arange(1, count + 1)  # Row IDs make epoch coverage observable.
        matrix[:, 6] = 2 if names == train_names else 100  # Exposes validation-fitted scaling.
        if names == val_names:
            validation = csr_array(matrix), labels
        for name, ids in zip(names, np.array_split(np.arange(count), 2)):
            tactics = tuple(Tactic("alternate" if ident == 0 else f"kind-{ident}", "fixture") for ident in labels[ids])
            rows.append(records.FeatureRows(name, np.arange(len(ids)), tactics, csr_array(matrix[ids])))
    header = archives.FeatureHeader(
        vocab, selection, "absent-source.db", 1, 2, msgspec.json.encode(policy), records.split_id(split)
    )
    archives.publish(
        root / "features.zst",
        (encode_msgpack(header), *(archives.pack_rows(row) for row in rows), encode_msgpack(archives.Footer(4, 27))),
        sources=(),
        replace=False,
    )
    archives.publish(root / "vocab.zst", (encode_msgpack(vocab),), sources=(), replace=False)
    (root / "split.json").write_bytes(msgspec.json.encode(split))
    assert validation is not None
    return validation


def run(root: Path, *extra: str) -> tuple[dict, str]:
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        result = logistic.main(
            [
                "--instance",
                str(root),
                "--iters",
                "3",
                "--batch-rows",
                "4",
                "--alpha",
                "0.02",
                "--workers",
                "1",
                "--seed",
                "3",
                *extra,
            ]
        )
    if result != 0:
        raise AssertionError(result)
    return json.loads(stdout.getvalue()), stderr.getvalue()


class LogisticTests(unittest.TestCase):
    def test_archive_epochs_reports_and_reload(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix, labels = archive_fixture(root)
            calls: dict[str | None, list[np.ndarray]] = {None: [], "l2": []}
            partial_fit = logistic.SGDClassifier.partial_fit

            def observe(model, x, y, **kwargs):
                calls[model.penalty].append(np.rint(x[:, [5]].toarray().ravel() * 18).astype(int))
                return partial_fit(model, x, y, **kwargs)

            with patch.object(logistic.SGDClassifier, "partial_fit", observe):
                report, progress = run(root)
            self.assertEqual(report["classes"], ["class-0", "class-1", "class-2", "class-3"])
            self.assertEqual(report["train"]["rows"], 18)
            self.assertEqual(report["validation"]["rows"], 9)
            self.assertIn("top-3=", progress)
            self.assertIn("macro-F1=", progress)
            expected_classes = np.arange(4)
            for name, result in report["models"].items():
                penalty = logistic.VARIANTS[name]
                self.assertEqual(len(calls[penalty]), 15)
                for epoch in range(3):
                    np.testing.assert_array_equal(
                        np.sort(np.concatenate(calls[penalty][epoch * 5 : (epoch + 1) * 5])), np.arange(1, 19)
                    )
                checkpoint = joblib.load(result["checkpoint"])
                predictor = checkpoint["predictor"]
                self.assertEqual(predictor["scaling"].scale_[6], 2)
                np.testing.assert_array_equal(predictor.classes_, expected_classes)
                self.assertEqual(predictor["model"].n_jobs, 1)
                probs = predictor.predict_proba(matrix)
                ranked = np.argsort(-probs, axis=1, kind="stable")
                predicted = ranked[:, 0]
                self.assertAlmostEqual(
                    result["val_cross_entropy"], log_loss(labels, probs, labels=expected_classes), places=6
                )
                self.assertAlmostEqual(result["val_accuracy"], accuracy_score(labels, predicted))
                self.assertAlmostEqual(
                    result["val_macro_f1"],
                    f1_score(labels, predicted, labels=expected_classes, average="macro", zero_division=0),
                )
                self.assertAlmostEqual(
                    result["val_top3_accuracy"], np.any(ranked[:, :3] == labels[:, None], axis=1).mean()
                )
                self.assertEqual(result["val_top5_accuracy"], 1)
                np.testing.assert_array_equal(
                    result["confusion"], confusion_matrix(labels, predicted, labels=expected_classes)
                )
                per_class = precision_recall_fscore_support(labels, predicted, labels=expected_classes, zero_division=0)
                for field, values in zip(("precision", "recall", "f1", "support"), per_class):
                    np.testing.assert_allclose(result[field], values)
                history = checkpoint["history"]
                best = min(history, key=lambda row: row["val_cross_entropy"])
                self.assertEqual(result["best_epoch"], best["epoch"])
                self.assertEqual(result["val_cross_entropy"], best["val_cross_entropy"])
                self.assertEqual(checkpoint["vocab_id"], report["vocab_id"])
                self.assertEqual(checkpoint["split_id"], report["split_id"])
                self.assertGreater(report["duration_s"], 0)
            # The same seed must reproduce both selected predictors, independent of publication paths.
            repeated, _ = run(root, "--output", str(root / "repeat"))
            for name in report["models"]:
                first = joblib.load(report["models"][name]["checkpoint"])["predictor"]
                second = joblib.load(repeated["models"][name]["checkpoint"])["predictor"]
                np.testing.assert_array_equal(first.predict_proba(matrix), second.predict_proba(matrix))

    def test_early_stop_and_scaling_disabled(self):
        # Failure cases: ignoring patience/min_delta, or silently scaling the none policy.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            matrix, _ = archive_fixture(root)
            report, _ = run(
                root, "--iters", "8", "--patience", "2", "--min-delta", "100", "--scaling", "none", "--workers", "2"
            )
            for result in report["models"].values():
                self.assertEqual(result["epochs"], 3)
                checkpoint = joblib.load(result["checkpoint"])
                predictor = checkpoint["predictor"]
                self.assertEqual(predictor["scaling"], "passthrough")
                self.assertEqual(predictor["model"].n_jobs, 2)
                self.assertTrue(np.isfinite(predictor.predict_proba(matrix)).all())

    def test_best_checkpoint_owns_selected_epoch_even_when_training_continues(self):
        # Failure cases: aliased best weights or saving the last epoch. A real validation
        # curve need not worsen on this fixture, so control only its selection score;
        # compare the saved real estimator with a separately trained one-epoch reference.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_fixture(root)
            reference, _ = run(root, "--iters", "1", "--output", str(root / "one-epoch"))
            measured = logistic.evaluate
            counts: dict[str | None, int] = {None: 0, "l2": 0}

            def worsening(model, data, batch_rows):
                counts[model.penalty] += 1
                return replace(measured(model, data, batch_rows), ce=float(counts[model.penalty]))

            with patch.object(logistic, "evaluate", worsening):
                report, _ = run(root)
            for name, result in report["models"].items():
                self.assertEqual((result["epochs"], result["best_epoch"]), (3, 1))
                selected = joblib.load(result["checkpoint"])["predictor"]["model"]
                original = joblib.load(reference["models"][name]["checkpoint"])["predictor"]["model"]
                np.testing.assert_array_equal(selected.coef_, original.coef_)
                np.testing.assert_array_equal(selected.intercept_, original.intercept_)

    def test_rejects_invalid_inputs_without_publishing(self):
        # Failure cases: invalid counts accepted, artifacts clobbered, and external schemas trusted over the archive.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_fixture(root)
            for flag, val in (("--iters", "0"), ("--iters", "-1"), ("--workers", "0"), ("--alpha", "nan")):
                with self.subTest(flag=flag, val=val), self.assertRaises(ValueError):
                    run(root, flag, val)
            bad_policy = root / "labels.json"
            bad_policy.write_bytes(msgspec.json.encode(LabelPolicy({"alternate": "wrong"}, "error")))
            with self.assertRaisesRegex(ValueError, "policy differs"):
                run(root, "--labels", str(bad_policy))
            self.assertEqual(list(root.glob("logit-*.pkl")), [])
            run(root)
            path = root / "logit-no-regularization.pkl"
            unchanged = path.read_bytes()
            with self.assertRaises(FileExistsError):
                run(root)
            self.assertEqual(unchanged, path.read_bytes())
            run(root, "--replace")
            vocab = archives.read_vocab(root / "vocab.zst")
            archives.publish(
                root / "vocab.zst",
                (encode_msgpack(msgspec.structs.replace(vocab, depths=(1, 2))),),
                sources=(),
                replace=True,
            )
            with self.assertRaisesRegex(ValueError, "vocabulary frozen"):
                run(root, "--output", str(root / "bad-vocab"))
            self.assertFalse((root / "bad-vocab").exists())


if __name__ == "__main__":
    unittest.main()
