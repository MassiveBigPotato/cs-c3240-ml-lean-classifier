"""End-to-end checks with known sharing and an exponentially large tree."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from dataclasses import fields
from pathlib import Path

from analysis_fixture import analyze

from trustmebro.extraction import records as r
from trustmebro.extraction.storage import encode_exprs, encode_trns, open_extraction_db
from trustmebro.visualization import state_render
from trustmebro.visualization.archives import read_stats, row_vals
from trustmebro.visualization.metrics import DESCR_GROUP_NAMES, DESCR_KINDS
from trustmebro.visualization.products import AnalysisPaths


def snapshot(path):
    data = {"states": [], "expressions": [], "examples": [], "mdata": None}
    data["global_reuse"] = {}
    for record in read_stats(path):
        if record[0] == "theorem":
            data["states"].extend(map(row_vals, record[2]))
            data["expressions"].extend(map(row_vals, record[3]))
        elif record[0] == "shape":
            data["examples"].append(record[1:])
        elif record[0] == "mdata":
            mdata = record[1]
            data["mdata"] = {
                field.name: getattr(mdata, field.name).tolist()
                if field.name in ("conc", "rotated_conc")
                else getattr(mdata, field.name)
                for field in fields(mdata)
            }
        elif record[0] == "global_reuse":
            data["global_reuse"][record[1]] = [[list(row_vals(row)) for row in rows] for rows in record[2:]]
    for key in ("states", "expressions", "examples"):
        data[key] = sorted(
            tuple(None if isinstance(val, float) and math.isnan(val) else val for val in row) for row in data[key]
        )
    return data


class VisualizationTests(unittest.TestCase):
    def test_database_to_metrics_and_figures_then_render_only(self):
        with tempfile.TemporaryDirectory() as dir:
            root = Path(dir)
            database = root / "source.sqlite"
            base = (r.Const("A", ()), r.Const("B", ()), r.App(0, (1,)), r.App(2, (2,)))
            # Expanded size exceeds floating-point range; the DAG stays small.
            deep = list(base)
            for _ in range(1100):
                deep.append(r.App(len(deep) - 1, (len(deep) - 1,)))
            locals_ = (
                r.LocalConst(0, 2, r.LocalDeclKind.DEFAULT, False, r.BinderInfo.DEFAULT),
                r.LocalConst(1, 0, r.LocalDeclKind.DEFAULT, False, r.BinderInfo.DEFAULT),
            )
            with closing(open_extraction_db(database)) as connection:
                for name, exprs, locals_used in (("small", base, locals_), ("deep", tuple(deep), ())):
                    state = r.ProofState(len(exprs) - 1, locals_used, (), ())
                    trn = r.Trn(r.Tactic("test", "test"), (0, 1), 1, state)
                    connection.execute(
                        """INSERT INTO theorems
                      (name,module,expr_count,trn_count,exprs,trns)
                      VALUES (?, 'Test', ?, 1, ?, ?)""",
                        (name, len(exprs), encode_exprs(exprs), encode_trns((trn,))),
                    )
            original = database.read_bytes()
            # A one-theorem corpus has no cross-theorem patterns: parallel
            # workers must handle an empty shared index without a special file.
            analyze(database, root / "empty-index", analyses=("metrics",), limit=1, workers=2)
            empty_idx_metrics = AnalysisPaths(root / "empty-index").analysis("metrics")
            empty_reuse = snapshot(empty_idx_metrics)["global_reuse"]["small"]
            self.assertEqual(empty_reuse[0][0][4:], [4, 4])
            binder_database = root / "binder.sqlite"
            binder_exprs = (r.Const("A", ()), r.Bvar(0), r.Lambda(("x",), 0, 1, r.BinderInfo.DEFAULT))
            with closing(open_extraction_db(binder_database)) as connection:
                state = r.ProofState(2, (), (), ())
                trn = r.Trn(r.Tactic("test", "test"), (0, 1), 1, state)
                connection.execute(
                    """INSERT INTO theorems
                    (name,module,expr_count,trn_count,exprs,trns)
                    VALUES ('binder', 'Test', 3, 1, ?, ?)""",
                    (encode_exprs(binder_exprs), encode_trns((trn,))),
                )
            analyze(binder_database, root / "binder", analyses=("metrics",), workers=2)
            binder_metrics = AnalysisPaths(root / "binder").analysis("metrics")
            binder_shapes = next(record[1] for record in read_stats(binder_metrics) if record[0] == "stral")
            descriptor = binder_shapes[0].topo
            self.assertAlmostEqual(descriptor.groups[DESCR_GROUP_NAMES.index("binder_nesting_0")], 2 / 3)
            self.assertAlmostEqual(descriptor.groups[DESCR_GROUP_NAMES.index("binder_nesting_1")], 1 / 3)
            self.assertEqual(descriptor.groups[DESCR_GROUP_NAMES.index("binder_ref_0")], 1)
            output = root / "plots"
            output.mkdir()
            retired = output / "constrs-goal.png"
            retired.write_bytes(b"obsolete generated plot")
            note = output / "notes.txt"
            note.write_text("keep this user file")
            command = [
                str(Path(sys.executable).parent / "graphs"),
                "--graphs",
                "complexity",
                "constructors",
                "context",
                "reuse",
                "frequencies",
                "embedding",
                "expr-atlas",
                "--db",
                str(database),
                "--output",
                str(output),
                "--stats",
                str(output),
                "--atlas-min-nodes",
                "3",
            ]
            env = dict(os.environ, MPLCONFIGDIR=str(root / "matplotlib"))
            subprocess.run(command, check=True, capture_output=True, env=env, timeout=120)
            metrics = AnalysisPaths(output).analysis("metrics")
            data = snapshot(metrics)
            self.assertTrue(data["examples"])
            reuse = {record[1]: record[2:] for record in read_stats(metrics) if record[0] == "global_reuse"}
            # Both theorems share the entire small goal: largest-first must not
            # also charge its nested patterns. The context still keeps its roots.
            small_state = reuse["small"][0][0]
            for actual, expected in zip(
                (small_state.expanded, small_state.local_dag, small_state.reduced_tree, small_state.reduced_dag),
                (11, 4, 3, 3),
                strict=True,
            ):
                self.assertEqual(actual, expected)
            self.assertEqual((small_state.novel, small_state.nodes), (0, 4))
            deep_state = reuse["deep"][0][0]
            self.assertEqual((deep_state.novel, deep_state.nodes), (1100, 1104))
            small_expr = reuse["small"][1][0]
            self.assertEqual((small_expr.reduced_tree, small_expr.reduced_dag), (1, 1))
            row = next(row for row in data["states"] if row[0] == "small")
            self.assertEqual(row[3], 7)
            self.assertEqual(tuple(row[idx] for idx in (4, 5, 11, 10, 12)), (4, 3, 0.75, 4, 8))
            self.assertGreater(next(row[3] for row in data["states"] if row[0] == "deep"), 10**308)
            grid = data["mdata"]["conc"]
            self.assertEqual(len(grid), 1000)
            self.assertEqual(len(grid[0]), 1000)
            self.assertEqual(sum(map(sum, grid)), 1000)
            self.assertEqual(grid[500][750], 1)
            rotated = data["mdata"]["rotated_conc"]
            self.assertEqual(sum(map(sum, rotated)), 1000)
            self.assertEqual(data["mdata"]["empty_ctxts"], 1)
            self.assertFalse(list(output.glob("*.sqlite")))
            self.assertFalse(list(output.glob("*.part")))
            for name in (
                "complexity-state.png",
                "context-composition.png",
                "context-largest-to-average.png",
                "global-reuse-state-expanded.png",
                "global-reuse-state-local-dag.png",
                "global-reuse-expression-expanded.png",
                "global-reuse-expression-local-dag.png",
                "global-novelty-state.png",
                "global-novelty-expression.png",
                "constructors-state.png",
                "expr-dag-atlas.png",
                "expr-adj-atlas.png",
                "expr-layer-profiles.png",
                "expr-layer-depth.png",
                "context-concentration.png",
                "context-concentration-excess.png",
                "reuse-nesting.png",
                "sharing-nesting.png",
                "structural-umap-shared.png",
                "cross-root-sharing.png",
                "vocabulary-coverage.png",
                "subexpression-repetition.png",
            ):
                self.assertTrue((output / name).read_bytes().startswith(b"\x89PNG"))
            shapes = json.loads((output / "expr-shapes.json").read_text())
            self.assertEqual(retired.read_bytes(), b"obsolete generated plot")
            self.assertEqual(note.read_text(), "keep this user file")
            self.assertEqual(len(shapes), len({(shape["theorem"], shape["root"]) for shape in shapes}))
            for shape in shapes:
                self.assertGreaterEqual(shape["nodes"], 3)
                if shape["theorem"] == "deep":
                    self.assertEqual(shape["layer_width"], [1] * 1102 + [2])
                    self.assertEqual(shape["shared_frac"], [0] + [1] * 1101 + [0])
                elif shape["root"] == 3:
                    self.assertEqual(shape["layer_width"], [1, 1, 2])
                    self.assertEqual(shape["shared_frac"], [0, 1, 0])
                elif shape["root"] == 2:
                    self.assertEqual(shape["layer_width"], [1, 2])
            self.assertEqual(row[17], {"App": 2, "Const": 2})
            stral = next(record[1] for record in read_stats(metrics) if record[0] == "stral")
            shared_root = next(row for row in stral if (row.theorem, row.root) == ("small", 3))
            self.assertEqual(shared_root.shared_frac, 0.25)
            self.assertAlmostEqual(sum(shared_root.topo.edge_weights), 1)
            self.assertAlmostEqual(sum(shared_root.topo.groups[:64]), 1)
            self.assertAlmostEqual(
                dict(zip(shared_root.topo.edges, shared_root.topo.edge_weights, strict=True))[
                    (DESCR_KINDS.index("App"), 0, DESCR_KINDS.index("App"))
                ],
                0.25,
            )
            import numpy as np

            with (
                np.load(output / "structural-observations.npz") as observations,
                np.load(output / "structural-embedding.npz") as embedding,
            ):
                self.assertTrue(np.isfinite(embedding["coordinates"]).all())
                self.assertTrue(any(name.startswith("motif_") for name in observations["feature_names"]))
                self.assertIn("edge_App_0_App", observations["feature_names"])
                self.assertNotIn("log10_reuse", embedding["input_names"])
                self.assertNotIn("shared_frac", embedding["input_names"])
                self.assertEqual(observations["features"].shape[0], len(stral))
                self.assertTrue(np.isfinite(embedding["inputs"]).all())
            with np.load(output / "structural-embedding-shape-only.npz") as embedding:
                self.assertTrue(np.isfinite(embedding["coordinates"]).all())
                self.assertNotIn("log10_nodes", embedding["input_names"])
                self.assertNotIn("log10_depth", embedding["input_names"])
            self.assertTrue((output / "structural-umap-shape-only-nodes_log.png").is_file())
            original_metrics = metrics.read_bytes()
            original_plot = (output / "complexity-state.png").read_bytes()
            subprocess.run(
                [
                    str(Path(sys.executable).parent / "graphs"),
                    "--stats",
                    str(output),
                    "--output",
                    str(output),
                    "--graphs",
                    "complexity",
                    "--dot-diameter",
                    "5",
                ],
                check=True,
                capture_output=True,
                env=env,
                timeout=120,
            )
            self.assertEqual(metrics.read_bytes(), original_metrics)
            self.assertNotEqual((output / "complexity-state.png").read_bytes(), original_plot)
            invalid = subprocess.run(
                [str(Path(sys.executable).parent / "graphs"), "--dot-diameter", "0"],
                check=False,
                capture_output=True,
                env=env,
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertIn(b"must be positive", invalid.stderr)
            self.assertEqual(database.read_bytes(), original)
            # A worker failure must not publish a partial archive or leave a
            # temporary cache behind. The source deliberately has a bad edge.
            invalid_src = root / "invalid.db"
            with closing(open_extraction_db(invalid_src)) as connection:
                connection.execute(
                    """INSERT INTO theorems
                    (name,module,expr_count,trn_count,exprs,trns)
                    VALUES ('bad','Test',1,1,?,?)""",
                    (
                        encode_exprs((r.App(9, (9,)),)),
                        encode_trns((r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(0, (), (), ())),)),
                    ),
                )
            invalid_output = root / "failed"
            failure = subprocess.run(
                [
                    str(Path(sys.executable).parent / "graphs"),
                    "--graphs",
                    "complexity",
                    "constructors",
                    "context",
                    "reuse",
                    "frequencies",
                    "embedding",
                    "expr-atlas",
                    "--db",
                    str(invalid_src),
                    "--output",
                    str(invalid_output),
                    "--stats",
                    str(invalid_output),
                ],
                check=False,
                capture_output=True,
                env=env,
                timeout=120,
            )
            self.assertNotEqual(failure.returncode, 0)
            self.assertFalse(list(invalid_output.iterdir()))
            # Failures: shared descriptors are duplicated in mode exports,
            # copying re-encodes bytes, provenance order differs, or mismatched
            # shared observations are accepted. Uses the existing fixture's
            # products and a no-op drawer, not another analysis or plot run.
            paths = AnalysisPaths(output)
            for path in (paths.embedding_shared, paths.embedding("size-aware"), paths.embedding("shape-only")):
                self.assertEqual(path.read_bytes(), (output / path.name).read_bytes())
            with np.load(paths.embedding_shared) as observations:
                self.assertEqual(list(observations["theorems"]), [row.theorem for row in stral])
                np.testing.assert_array_equal(observations["roots"], [row.root for row in stral])
            mode_path = paths.embedding("size-aware")
            original_mode = mode_path.read_bytes()
            with np.load(mode_path) as encoded:
                mode = {name: encoded[name] for name in encoded.files}
            bad_dir = root / "bad-alignment"
            bad_dir.mkdir()
            try:
                for field, bad in (("shared_sha256", np.asarray("wrong")), ("rows", np.asarray(-1))):
                    np.savez_compressed(mode_path, **(mode | {field: bad}))
                    with self.subTest(field=field), self.assertRaisesRegex(ValueError, "observations disagree"):
                        state_render._plot_stral(metrics, len(stral), bad_dir, output, lambda *args, **kwargs: None)
            finally:
                mode_path.write_bytes(original_mode)
