"""Saved experiment reports → validated result figures and manifest.

Failure cases: incompatible inputs combined; metrics from different epochs;
missing histories silently invented; existing files clobbered; figures absent
or empty; confusion transposed/column-normalized; unsupported classes dropped.
Presentation checks also catch smoothing runs appearing in figures or disappearing
from CSV metrics, score-dependent row order, indistinguishable solid curves,
and reversed heatmap darkness or gridlines drawn through cell centres.
The subprocess check establishes the reporting workflow, not training accuracy.
"""

import copy
import csv
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import matplotlib.pyplot as plt
import numpy as np

from trustmebro import learning
from trustmebro.visualization import results


def fixture(root: Path) -> tuple[Path, Path]:
    mlp, logistic = root / "mlp", root / "logistic"
    mlp.mkdir()
    logistic.mkdir()
    matrix = np.array([[8, 2, 0, 0], [1, 4, 0, 0], [1, 1, 3, 0], [0, 0, 0, 0]])
    support = matrix.sum(axis=1)
    denom = support + matrix.sum(axis=0)
    f1 = np.divide(2 * np.diag(matrix), denom, out=np.zeros(4), where=denom > 0)
    metrics = {
        "val_cross_entropy": 1.0,
        "val_accuracy": 0.75,
        "val_top3_accuracy": 1.0,
        "val_top5_accuracy": 1.0,
        "val_macro_f1": float(f1.mean()),
    }
    report = {
        "status": "complete",
        "criterion": "unweighted",
        "classes": ["induction", "rewrite", "simplify", "subst"],
        "vocab_id": "fixture",
        "width": 10,
        "train": {"rows": 40, "theorems": 4, "theorem_ids_sha256": "train"},
        "validation": {"rows": 20, "theorems": 2, "theorem_ids_sha256": "validation"},
        "epochs": 2,
        "best_epoch": 1,
        **metrics,
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion": matrix.tolist(),
        "history": [{"epoch": 1, **metrics}, {"epoch": 2, **metrics, "val_cross_entropy": 1.2}],
    }
    for name in ("baseline", "unweighted-dropout", "inverse-sqrt-dropout", "unweighted-smoothing"):
        path = mlp / name / "results.json"
        path.parent.mkdir()
        path.write_text(json.dumps(report))
    for name in ("default-regularization", "no-regularization"):
        joblib.dump(report, logistic / f"logit-{name}.pkl")
    return mlp, logistic


class ResultPlotTests(unittest.TestCase):
    def test_checkpoint_reports_avoid_predictor_loading_and_fail_closed(self):
        # Failures: new plots unpickle predictors despite a report, accept a
        # stale association, or publish a partial report after an encoding error.
        # Missing reports must keep the existing checkpoint-only read path;
        # neither path fits or evaluates. This is publication/reading evidence,
        # not a concurrent-writer or full-model performance benchmark.
        with TemporaryDirectory() as tmp:
            mlp, logistic = fixture(Path(tmp))
            expected = results.load_runs(mlp, logistic)
            paths = sorted(logistic.glob("*.pkl"))
            metadata = [joblib.load(path) for path in paths]
            for path, data in zip(paths, metadata, strict=True):
                learning.publish_model_report(path, data)
            with patch.object(results.joblib, "load", side_effect=AssertionError("predictor loading")):
                actual = results.load_runs(mlp, logistic)
                self.assertEqual(
                    [(run.ident, run.metrics) for run in actual], [(run.ident, run.metrics) for run in expected]
                )
                report = paths[0].with_suffix(".pkl.report.json")
                data = json.loads(report.read_text())
                data["checkpoint_sha256"] = "wrong"
                report.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, "does not describe"):
                    results.load_runs(mlp, logistic)
            report.unlink()
            original = paths[0].read_bytes()
            with (
                patch.object(learning.msgspec.json, "encode", side_effect=RuntimeError("interrupted encoding")),
                self.assertRaisesRegex(RuntimeError, "interrupted"),
            ):
                learning.publish_model_report(paths[0], metadata[0])
            self.assertFalse(report.exists())
            self.assertEqual(paths[0].read_bytes(), original)
            self.assertFalse(list(logistic.glob(".report-*")))
            with patch.object(results.joblib, "load", wraps=joblib.load) as loading:
                results.load_runs(mlp, logistic)
                self.assertEqual(loading.call_count, 1)

    def test_saved_reports_to_all_figures_and_overwrite_protection(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mlp, logistic = fixture(root)
            output = root / "figures"
            cmd = [
                str(Path(sys.executable).parent / "plot-results"),
                "--mlp",
                str(mlp),
                "--logistic",
                str(logistic),
                "--output",
                str(output),
                "--dpi",
                "40",
            ]
            env = os.environ | {"MPLCONFIGDIR": str(root / "mpl-cache")}
            completed = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60, check=False)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            manifest = json.loads((output / "figures.json").read_text())
            self.assertEqual(manifest["population"], "validation")
            self.assertEqual(manifest["compared"], list(results.DEFAULT_COMPARE))
            self.assertEqual(manifest["classes"], ["induction", "rewrite", "simplify", "subst"])
            self.assertEqual(manifest["chosen_configuration"], "mlp:inverse-sqrt-dropout")
            self.assertEqual(manifest["omitted_from_figures"], ["mlp:unweighted-smoothing"])
            self.assertEqual(len(manifest["plotted_runs"]), 5)
            # All three MLP histories share one plot per metric, not split panels.
            self.assertIn("learning-cross-entropy-mlp-01.pdf", manifest["files"])
            self.assertFalse(any("-02." in name for name in manifest["files"]))
            for filename in manifest["files"]:
                path = output / filename
                self.assertGreater(path.stat().st_size, 100)
                if path.suffix == ".pdf":
                    self.assertTrue(path.read_bytes().startswith(b"%PDF-"))
                elif path.suffix == ".png":
                    self.assertTrue(path.read_bytes().startswith(b"\x89PNG"))
            with (output / "metrics.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 6)
            self.assertIn("unweighted-smoothing", [row["name"] for row in rows])
            self.assertTrue(all(row["best_epoch"] == "1" for row in rows))
            with (output / "family-improvements.csv").open() as stream:
                changes = list(csv.DictReader(stream))
            self.assertEqual(len(changes), 8)
            self.assertTrue(all(row["relative_change_percent"] == "" for row in changes if row["family"] == "subst"))
            self.assertTrue(
                all(float(row["relative_change_percent"]) == 0 for row in changes if row["family"] != "subst")
            )
            saved = (output / "overall-performance.pdf").read_bytes()
            repeated = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60, check=False)
            self.assertNotEqual(repeated.returncode, 0)
            self.assertEqual((output / "overall-performance.pdf").read_bytes(), saved)
            repeated = subprocess.run(
                cmd + ["--replace"], capture_output=True, text=True, env=env, timeout=60, check=False
            )
            self.assertEqual(repeated.returncode, 0, repeated.stderr)
            self.assertFalse(any(path.name.startswith(".results-") for path in output.iterdir()))

    def test_confusion_values_orientation_and_zero_support_survive_rendering(self):
        # Pixel/file checks alone cannot diagnose transposed or incorrectly normalized
        # numerical images, so inspect the public figure's actual raster and axes.
        with TemporaryDirectory() as tmp:
            mlp, logistic = fixture(Path(tmp))
            runs = results.load_runs(mlp, logistic)
            fig = results.confusion(runs[0])
            try:
                image = fig.axes[0].images[0]
                expected = np.array([[0.8, 0.2, 0, 0], [0.2, 0.8, 0, 0], [0.2, 0.2, 0.6, 0], [0, 0, 0, 0]])
                np.testing.assert_allclose(image.get_array().filled(0), expected)
                self.assertTrue(image.get_array().mask[-1].all())
                self.assertEqual(image.get_clim(), (0, 1))
                self.assertEqual(fig.axes[0].get_xlabel(), "Predicted tactic family")
                self.assertEqual(fig.axes[0].get_ylabel(), "True tactic family")
                self.assertEqual(len(fig.axes[0].get_yticklabels()), 4)
                colours = np.array([image.cmap(val)[:3] for val in (0.0, 0.5, 1.0)])
                luminance = colours @ np.array([0.2126, 0.7152, 0.0722])
                self.assertTrue(np.all(np.diff(luminance) < 0))
                np.testing.assert_allclose(colours[0], [1, 1, 1])
                np.testing.assert_allclose(fig.axes[0].get_xticks(minor=True), np.arange(5) - 0.5)
                self.assertTrue(all(tick.gridline.get_visible() for tick in fig.axes[0].xaxis.get_minor_ticks()))
            finally:
                plt.close(fig)
            compared = tuple(next(run for run in runs if run.ident == ident) for ident in results.DEFAULT_COMPARE)
            fig = results.family_changes(compared)
            try:
                for ax in fig.axes[:2]:
                    offsets = ax.collections[0].get_offsets()
                    self.assertEqual(len(offsets), 3)
                    self.assertFalse(np.ma.getmaskarray(offsets).any())
                    self.assertTrue(any("undefined" in text.get_text() for text in ax.texts))
                    self.assertEqual(len(ax.get_yticks()), 4)
            finally:
                plt.close(fig)

    def test_relative_changes_bars_and_merged_curves_retain_values(self):
        # Failures: percentage points substituted for relative changes; epsilon
        # invented for a zero baseline; extreme improvements clipped; bars given
        # nonzero origins; histories dropped during merging. These numerical
        # boundaries cannot be diagnosed by successful CLI output/file signatures.
        src = np.array([0.2, 0.4, 0, 0, 0.0001])
        dst = np.array([0.3, 0.2, 0.1, 0, 0.4])
        np.testing.assert_allclose(results.relative_f1(src, dst), [50, -50, np.nan, np.nan, 399900], equal_nan=True)
        with TemporaryDirectory() as tmp:
            mlp, logistic = fixture(Path(tmp))
            runs = results.load_runs(mlp, logistic)
            fig = results.overall(runs)
            try:
                ids = (
                    "logistic:default-regularization",
                    "logistic:no-regularization",
                    "mlp:baseline",
                    "mlp:unweighted-dropout",
                    "mlp:inverse-sqrt-dropout",
                )
                ordered = [next(run for run in runs if run.ident == ident) for ident in ids]
                self.assertEqual(
                    [label.get_text().removesuffix(" ★") for label in fig.axes[0].get_yticklabels()],
                    [run.label for run in ordered],
                )
                for ax, key in ((fig.axes[0], "val_cross_entropy"), (fig.axes[2], "val_macro_f1")):
                    self.assertEqual(ax.get_xlim()[0], 0)
                    np.testing.assert_allclose(
                        [bar.get_width() for bar in ax.patches], [r.metrics[key] for r in ordered]
                    )
                    self.assertTrue(all(bar.get_x() == 0 for bar in ax.patches))
            finally:
                plt.close(fig)
            group = [run for run in ordered if run.model == "mlp"]
            fig = results.learning(group, "val_macro_f1")
            try:
                self.assertEqual(len(fig.axes), 1)
                self.assertEqual(len(fig.axes[0].lines), len(group))
                self.assertTrue(all(line.get_linestyle() == "-" for line in fig.axes[0].lines))
                self.assertEqual(len({line.get_color() for line in fig.axes[0].lines}), len(group))
                for line, run in zip(fig.axes[0].lines, group):
                    np.testing.assert_allclose(line.get_ydata(), [row["val_macro_f1"] for row in run.history])
                lo, hi = fig.axes[0].get_ylim()
                self.assertEqual(lo, 0)
                self.assertGreaterEqual(hi, max(row["val_macro_f1"] for run in group for row in run.history))
            finally:
                plt.close(fig)

    def test_inconsistent_reports_and_missing_histories_fail_before_publication(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            mlp, logistic = fixture(root)
            path = mlp / "baseline/results.json"
            original = json.loads(path.read_text())
            mutations = (
                ("vocab_id", "other"),
                ("f1", [1, 1, 1, 0]),
                ("val_accuracy", 0.5),
                ("best_epoch", 2),
                ("history", [{**original["history"][0], "epoch": 2}]),
            )
            for field, val in mutations:
                data = copy.deepcopy(original)
                data[field] = val
                path.write_text(json.dumps(data))
                with self.subTest(field=field), self.assertRaises(ValueError):
                    results.load_runs(mlp, logistic)
            data = copy.deepcopy(original)
            del data["history"]
            path.write_text(json.dumps(data))
            argv = ["--mlp", str(mlp), "--logistic", str(logistic), "--output", str(root / "missing")]
            with self.assertRaises(SystemExit):
                results.main(argv)
            self.assertFalse((root / "missing").exists())
            # Historical reports without curves still support selected-checkpoint figures.
            self.assertEqual(results.main(argv + ["--plots", "overall", "--formats", "png", "--dpi", "40"]), 0)
            self.assertTrue((root / "missing/overall-performance.png").exists())


if __name__ == "__main__":
    unittest.main()
