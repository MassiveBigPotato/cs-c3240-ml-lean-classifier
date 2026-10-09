"""Retained installed workflows, frozen conversion, and reusable current statistics."""

import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.metadata import distribution
from pathlib import Path
from unittest.mock import patch

import msgspec
from test_candidates import store
from test_feature_pipeline import source
from test_representation import feature_rows

import trustmebro.preprocessing.layout as layouts
from trustmebro.preprocessing import archives, pipeline
from trustmebro.visualization import cli, scan, vocab_render
from trustmebro.visualization.products import AnalysisPaths


class CommandTests(unittest.TestCase):
    def test_installed_surface_and_removed_options(self):
        # Failures: stale registrations, unusable installed imports, old CLI flags accepted.
        # Establishes packaging and help, not every command's computational correctness.
        expected = {
            "extract-dataset",
            "find-files",
            "proof-states",
            "partition-corpus",
            "graphs",
            "prepare-features",
            "train-logistic",
            "train-mlp",
            "plot-results",
        }
        self.assertEqual({entry.name for entry in distribution("trustmebro").entry_points}, expected)
        for name in sorted(expected):
            result = subprocess.run(
                [str(Path(sys.executable).parent / name), "--help"],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("--list-graphs", result.stdout)
            self.assertNotIn("--node-types", result.stdout)
            if name == "proof-states":
                self.assertNotIn("--dicts", result.stdout)
            if name == "prepare-features":
                self.assertNotIn("{build,convert}", result.stdout)

    def test_preparation_and_frozen_conversion_preserve_columns_and_resume(self):
        # Failures: accidental fitting/splitting on test input; mismatched policy accepted;
        # repeated completed conversion; fitting-only flags ignored; vocabulary mutation.
        # Tiny real fit/convert checks identities and rows, not predictive quality.
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            db, labels, theorems = source(root)
            learning = root / "learning"
            pipeline.main(
                [
                    "--db",
                    str(db),
                    "--labels",
                    str(labels),
                    "--output",
                    str(learning),
                    "--dims",
                    "5000",
                    "--min-support",
                    "1",
                    "--depths",
                    "1",
                    "2",
                ]
            )
            vocab = learning / "instance-01/vocab.zst"
            original = vocab.read_bytes()
            heldout = root / "test.db"
            store(heldout, tuple(msgspec.structs.replace(t, name="test." + t.name) for t in theorems))
            output = root / "test-features"
            args = ["--db", str(heldout), "--vocab", str(vocab), "--output", str(output)]
            with patch.object(pipeline, "fit_vocab", side_effect=AssertionError("test fitting")):
                pipeline.main(args)
            header = archives.prepare_feature_source(output / "features.zst").header
            self.assertEqual(header.vocab, archives.read_vocab(vocab))
            self.assertEqual(
                layouts.compile_vocab(header.vocab).width, layouts.compile_vocab(archives.read_vocab(vocab)).width
            )
            self.assertEqual(
                {row.name for row in feature_rows(output / "features.zst")}, {"test." + t.name for t in theorems}
            )
            self.assertEqual(msgspec.json.decode(header.label_policy), json.loads(labels.read_text()))
            before = (output / "features.zst").read_bytes()
            with patch.object(pipeline, "convert_corpus", side_effect=AssertionError("reconversion")):
                pipeline.main(args)
            self.assertEqual((output / "features.zst").read_bytes(), before)
            self.assertEqual(vocab.read_bytes(), original)
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                pipeline.main(args + ["--dims", "5000"])
            wrong = root / "wrong.json"
            wrong.write_text('{"kinds":{"a":"Different"},"unmapped":"other","other_label":"OTHER"}')
            with self.assertRaisesRegex(ValueError, "frozen vocabulary"):
                pipeline.main(
                    [
                        "--db",
                        str(heldout),
                        "--vocab",
                        str(vocab),
                        "--output",
                        str(root / "wrong-output"),
                        "--labels",
                        str(wrong),
                    ]
                )
            # Current frozen vocabulary, not an old frequency inventory, reaches the renderer.
            with patch.object(vocab_render, "compile_vocab", wraps=layouts.compile_vocab) as compile_:
                vocab_render.render_inventory(vocab, root / "vocab-plots", min_nodes=2)
            compile_.assert_called_once()
            self.assertTrue((root / "vocab-plots/vocabulary-dimensions.png").is_file())

    def test_graph_discovery_missing_complete_damaged_and_obsolete(self):
        # Failures: repeated analysis; missing DB accepted for new data; complete saved
        # products require DB queries; corruption accepted; --replace cannot recover.
        # Uses real fixture analysis with a stub renderer to isolate cache orchestration.
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            db, _, _ = source(root)
            args = [
                "--db",
                str(db),
                "--stats",
                str(root / "stats"),
                "--output",
                str(root / "plots"),
                "--graphs",
                "heads",
                "--workers",
                "1",
            ]
            calls = []

            def stages(cmd, stages, cfg):
                calls.append((cmd, tuple(stages)))
                if cmd == "analyze":
                    scan.collect_analysis(cfg.db, cfg.stats, analyses=cfg.analyses, workers=1)
                return {}

            with patch.object(cli, "_run_stages", side_effect=stages):
                cli.main(args)
                self.assertEqual(calls, [("analyze", ("shared",)), ("graphs", ("heads",))])
                calls.clear()
                original_topo = AnalysisPaths(root / "stats").manifest["topology"]
                cli.main(args + ["--graphs", "local-patterns"])
                self.assertEqual(calls, [("analyze", ("shared",)), ("graphs", ("local-patterns",))])
                self.assertEqual(AnalysisPaths(root / "stats").manifest["topology"], original_topo)
                self.assertFalse(AnalysisPaths(root / "stats").analysis("metrics").exists())
                calls.clear()
                db.rename(root / "unavailable.db")
                cli.main(args)
                self.assertEqual(calls, [("graphs", ("heads",))])
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main(args + ["--graphs", "complexity"])
                paths = AnalysisPaths(root / "stats")
                paths.heads.write_bytes(b"damaged")
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main(args)
                (root / "unavailable.db").rename(db)
                cli.main(args + ["--replace"])
                from trustmebro.visualization.products import MANIFEST

                (root / "stats" / MANIFEST).write_bytes(b"obsolete")
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    cli.main(args)
                cli.main(args + ["--replace"])
