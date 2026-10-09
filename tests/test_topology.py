"""End-to-end unlabeled topology counts, plots, and parallel determinism."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from analysis_fixture import analyze, read_pairs, shape_counts

from trustmebro.artifacts import encode_msgpack
from trustmebro.extraction import records as r
from trustmebro.extraction.storage import encode_exprs, encode_trns, open_extraction_db
from trustmebro.graph import build_graph
from trustmebro.visualization import archives
from trustmebro.visualization.archives import write_pairs
from trustmebro.visualization.measurements import VIEW_MODES as MODES
from trustmebro.visualization.measurements import GraphSize
from trustmebro.visualization.products import AnalysisPaths
from trustmebro.visualization.stral_render import _plot_comparisons, render_shapes
from trustmebro.visualization.views import prepare_views


class TopologyTests(unittest.TestCase):
    def test_database_to_figures_keeps_independent_complexity_ranges(self):
        # Inspect the actual figures at the pipeline's save boundary, not just
        # PNG existence: inflated x ranges, clipped equality lines, unequal
        # before/after scales, and minor-tick label spam all produce valid PNGs.
        # Also check occurrence mass at the image boundary: streamed native
        # buffers must not lose or duplicate weights between archive batches.
        import datashader as ds
        import numpy as np
        from matplotlib.figure import Figure
        from matplotlib.ticker import NullFormatter

        from trustmebro.visualization.measurements import VIEW_MODES

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            database = out / "source.db"
            exprs: list[r.Expr] = [r.Const("A", ())]
            for _ in range(40):
                exprs.append(r.App(0, (len(exprs) - 1, len(exprs) - 1)))
            trn = r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(40, (), (), ()))
            with closing(open_extraction_db(database)) as db:
                db.execute(
                    "INSERT INTO theorems (name,module,expr_count,trn_count,exprs,trns) VALUES ('deep', 'Test', ?, 1, ?, ?)",
                    (len(exprs), encode_exprs(tuple(exprs)), encode_trns((trn,))),
                )
                db.commit()
            analyze(database, out, analyses=("topology",), workers=1)
            pairs = read_pairs(AnalysisPaths(out).comparisons)
            archive = out / "pairs.zst"
            write_pairs(archive, pairs)
            pairs = read_pairs(archive)
            savefig = Figure.savefig
            inspected: set[str] = set()

            def inspect(fig, filename, *args, **kwargs):
                name = Path(filename).stem
                if name.endswith("-complexity"):
                    for ax in fig.axes[:-1]:
                        self.assertLess(ax.get_xlim()[1], 2)  # fewer than 100 DAG nodes
                        self.assertGreater(ax.get_ylim()[1], 12)  # >10^12 expanded nodes
                        for line in ax.lines:
                            self.assertLessEqual(max(line.get_xdata()), ax.get_xlim()[1])
                        for axis in (ax.xaxis, ax.yaxis):
                            self.assertEqual(axis.get_major_formatter()(2, 0), "100")
                    inspected.add(name)
                elif name.endswith("-nodes"):
                    lvl = "expression" if "-expression-" in name else "state"
                    for mode, ax in zip(VIEW_MODES[1:], fig.axes[:-1], strict=True):
                        self.assertEqual(ax.get_xlim(), ax.get_ylim())
                        grid = np.asarray(ax.images[0].get_array())
                        self.assertAlmostEqual(
                            np.power(10, grid[np.isfinite(grid)]).sum(), sum(pairs[mode][lvl].values())
                        )
                    inspected.add(name)
                elif name == "plumbing-topology-coverage":
                    for ax in fig.axes:
                        self.assertIsInstance(ax.xaxis.get_minor_formatter(), NullFormatter)
                    inspected.add(name)
                elif name.startswith(("topology-frequency-", "plumbing-topology-frequency-")):
                    for ax in fig.axes[:-1]:
                        for axis in (ax.xaxis, ax.yaxis):
                            self.assertEqual(axis.get_major_formatter()(2, 0), "100")
                            self.assertEqual(axis.get_major_formatter()(6, 0), "1,000,000")
                            self.assertEqual(axis.get_major_formatter()(7, 0), "1e7")
                    inspected.add(name)
                return savefig(fig, filename, *args, **kwargs)

            with patch.object(Figure, "savefig", inspect):
                # Failure cases: bounds and raster replay the same archive,
                # budget fallback drops weight, or the baseline topology plot
                # rereads coordinates already prepared for the mode panels.
                # This diagnoses source passes, not wall-time performance.
                from trustmebro.visualization import stral_render

                with (
                    patch.object(archives, "_pair_batches", wraps=archives._pair_batches) as reads,
                    patch.object(ds.Canvas, "points", side_effect=AssertionError("per-batch raster construction")),
                ):
                    _plot_comparisons(archive, out, 1)
                    self.assertEqual(reads.call_count, 2)
                with (
                    patch.object(archives, "_pair_batches", wraps=archives._pair_batches) as reads,
                    patch.object(stral_render, "POINT_MEMORY_BUDGET", 0),
                ):
                    _plot_comparisons(archive, out, 1)
                    # One bounds pass per level, then one replay per raster group.
                    self.assertEqual(reads.call_count, 8)
                with patch.object(stral_render, "shape_cols", wraps=stral_render.shape_cols) as reads:
                    render_shapes(out, out, 1)
                    self.assertEqual(reads.call_count, 2 * len(VIEW_MODES))
            self.assertEqual(
                inspected,
                {
                    "plumbing-expression-complexity",
                    "plumbing-state-complexity",
                    "plumbing-expression-nodes",
                    "plumbing-state-nodes",
                    "plumbing-topology-coverage",
                    "topology-frequency-top",
                    "topology-frequency-all",
                    "plumbing-topology-frequency-top",
                    "plumbing-topology-frequency-all",
                },
            )

    def test_exported_baseline_preserves_nested_operands_and_grouped_binders(self):
        # Failure cases: reflattening changes nested operands or binder domains;
        # grouped multiplicity/repeated roots are lost; removed modes survive;
        # rendering reads the original DB. Checks archive/figure integration,
        # not throughput or Lean elaboration itself.
        with tempfile.TemporaryDirectory() as tmp:
            dir = Path(tmp)
            database = dir / "source.db"
            shared = (
                r.Const("f", ()),
                r.Const("a", ()),
                r.Const("b", ()),
                r.Const("g", ()),
                r.Const("x", ()),
                r.Const("y", ()),
                r.Const("z", ()),
                r.Const("w", ()),
                r.App(3, (4, 5, 6, 7)),
                r.App(0, (1, 8, 2, 1)),
            )
            grouped = (
                r.Const("Nat", ()),
                r.Const("Bool", ()),
                r.Bvar(0),
                r.Forall(("z",), 1, 2, r.BinderInfo.IMPLICIT),
                r.Forall(("x", "y"), 0, 3, r.BinderInfo.DEFAULT),
                r.Lambda(("u", "v"), 0, 4, r.BinderInfo.DEFAULT),
                r.Metadata((), 5),
            )
            with closing(open_extraction_db(database)) as db:
                for name, exprs, dst, hyps in (("shared", shared, 9, (8, 8)), ("grouped", grouped, 6, (4,))):
                    locals_ = tuple(
                        r.LocalConst(i, root, r.LocalDeclKind.DEFAULT, False, r.BinderInfo.DEFAULT)
                        for i, root in enumerate(hyps)
                    )
                    trn = r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(dst, locals_, (), ()))
                    db.execute(
                        "INSERT INTO theorems (name,module,expr_count,trn_count,exprs,trns) VALUES (?, 'Test', ?, 1, ?, ?)",
                        (name, len(exprs), encode_exprs(exprs), encode_trns((trn,))),
                    )
            original = database.read_bytes()
            views = prepare_views(build_graph(grouped), MODES, matches={})
            for view in views.values():
                self.assertEqual(view.graph.edges[4], (0, 3))
                self.assertEqual(view.binders[4], (4, 4))
                self.assertEqual(view.binders[5], (5, 5))
                self.assertIs(view.graph.exprs, grouped)
            for view in prepare_views(build_graph(shared), MODES, matches={}).values():
                self.assertEqual(view.graph.edges[9], (0, 1, 8, 2, 1))
                self.assertEqual(view.graph.edges[8], (3, 4, 5, 6, 7))
            out = dir / "plots"
            analyze(database, out, analyses=("topology",), workers=2, atlas_min_nodes=2)
            paths = AnalysisPaths(out)
            pairs = read_pairs(paths.comparisons)
            self.assertEqual(set(pairs), set(MODES[1:]))
            for levels in pairs.values():
                self.assertEqual(set(levels), {"expression", "state"})
                self.assertIn((GraphSize(10, 10, 0, 3, 23, 5),) * 2, levels["state"])
                self.assertIn((GraphSize(7, 7, 5, 5, 13, 2),) * 2, levels["state"])
                self.assertTrue(all(src == dst for hist in levels.values() for src, dst in hist))
            saved = paths.comparisons.read_bytes()
            database.rename(dir / "unavailable.db")
            subprocess.run(
                [
                    str(Path(sys.executable).parent / "graphs"),
                    "--stats",
                    str(out),
                    "--output",
                    str(out),
                    "--graphs",
                    "topology",
                    "comparisons",
                    "heads",
                    "topology-atlas",
                    "--atlas-min-nodes",
                    "2",
                ],
                check=True,
                capture_output=True,
                env=dict(os.environ, MPLCONFIGDIR=str(dir / "mpl")),
                timeout=180,
            )
            coverage = json.loads((out / "plumbing-topology-coverage.json").read_text())
            for curves in coverage.values():
                self.assertEqual(set(curves), set(MODES))
                self.assertTrue(all(curve == curves["original"] for curve in curves.values()))
            for name in ("plumbing-state-nodes", "plumbing-topology-atlas", "plumbing-common-heads"):
                self.assertTrue((out / f"{name}.png").read_bytes().startswith(b"\x89PNG"))
            self.assertFalse(list(out.glob("flatten-*")))
            self.assertEqual(paths.comparisons.read_bytes(), saved)
            self.assertEqual((dir / "unavailable.db").read_bytes(), original)

    def test_ordered_topo_identities_preserve_full_edges_after_bfs_numbering(self):
        # Failure cases: discovery-tree edges replace the full ordered DAG;
        # repeated operands disappear; shared/equal leaves become interchangeable;
        # child slots reorder; theorem IDs/unreachable nodes affect the identity.
        # DB -> analysis -> archive checks independent canonical descriptors and
        # occurrence weights, not just equality between two implementations.
        exprs = (r.Const("A", ()), r.Const("B", ()), r.App(0, (0,)), r.App(0, (1,)), r.App(2, (3,)), r.App(3, (2,)))
        # Unreachable nodes and unrelated table IDs do not affect identity.
        # Original exports require child-before-parent order, unlike the local BFS.
        renumbered = (
            r.Const("unused", ()),
            r.Const("Y", ()),
            r.Const("X", ()),
            r.App(1, (1,)),
            r.App(1, (2,)),
            r.App(3, (4,)),
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.db"
            with closing(open_extraction_db(src)) as db:
                for name, data, dst, weight in (
                    ("shared_first", exprs, 4, 2),
                    ("shared_second", exprs, 5, 1),
                    ("renumbered", renumbered, 5, 3),
                ):
                    trn = r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(dst, (), (), ()))
                    db.execute(
                        "INSERT INTO theorems(name,module,expr_count,trn_count,exprs,trns) VALUES (?, 'Test', ?, ?, ?, ?)",
                        (name, len(data), weight, encode_exprs(data), encode_trns((trn,) * weight)),
                    )
                db.commit()
            out = root / "stats"
            analyze(src, out, analyses=("topology",), workers=1)
            counts = shape_counts(AnalysisPaths(out).topo())
            expected = (
                ([[1, 2], [3, 3], [3, 4], [], []], (5, 5, 5)),
                ([[1, 2], [3, 4], [3, 3], [], []], (5, 1, 1)),
                ([[1, 1], []], (2, 0, 6)),
                ([[1, 2], [], []], (3, 0, 6)),
                ([[]], (1, 0, 12)),
            )
            self.assertEqual(
                {sig: (row.nodes, row.top, row.all) for sig, row in counts.items()},
                {hashlib.sha256(encode_msgpack(descr)).digest(): weights for descr, weights in expected},
            )

    def test_topo_scan_counts_and_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            dir = Path(tmp)
            db_path = dir / "source.db"

            def trn(root: int) -> r.Trn:
                return r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(root, (), (), ()))

            rows = (
                ("app", (r.Const("A", ()), r.Const("B", ()), r.App(0, (1,))), (trn(2),)),
                (
                    "forall",
                    (r.Const("C", ()), r.Const("D", ()), r.Forall(("x",), 0, 1, r.BinderInfo.DEFAULT)),
                    (trn(2),),
                ),
                ("shared", (r.Const("E", ()), r.App(0, (0,))), (trn(1),)),
            )
            with closing(open_extraction_db(db_path)) as db:
                db.executemany(
                    "INSERT INTO theorems (name,module,expr_count,trn_count,exprs,trns) VALUES (?, 'Test', ?, ?, ?, ?)",
                    [
                        (name, len(exprs), len(trns), encode_exprs(exprs), encode_trns(trns))
                        for name, exprs, trns in rows
                    ],
                )
            original = db_path.read_bytes()
            env = dict(os.environ, MPLCONFIGDIR=str(dir / "matplotlib"))
            # Parallel numerical parity is covered by the shared collector fixtures.
            for workers in (2,):
                out = dir / str(workers)
                command = [
                    str(Path(sys.executable).parent / "graphs"),
                    "--graphs",
                    "topology",
                    "comparisons",
                    "heads",
                    "topology-atlas",
                    "--stats",
                    str(out),
                    "--db",
                    str(db_path),
                    "--output",
                    str(out),
                    "--workers",
                    str(workers),
                ]
                subprocess.run(command, check=True, capture_output=True, env=env, timeout=120)
                counts = shape_counts(AnalysisPaths(out).topo())
                rows = sorted((row.nodes, row.top, row.all) for row in counts.values())
                self.assertEqual(rows, [(1, 0, 5), (2, 1, 1), (3, 2, 2)])
                coverage = json.loads((out / "topology-coverage.json").read_text())
                self.assertAlmostEqual(coverage["top"]["All sizes"]["coverage"][0], 2 / 3)
                self.assertAlmostEqual(coverage["all"]["All sizes"]["coverage"][0], 5 / 8)
                for name in (
                    "topology-frequency-top.png",
                    "topology-frequency-all.png",
                    "topology-coverage.png",
                    "topology-atlas.png",
                ):
                    self.assertTrue((out / name).read_bytes().startswith(b"\x89PNG"))
            self.assertEqual(db_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
