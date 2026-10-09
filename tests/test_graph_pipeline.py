"""Exercise separate analysis/render commands, default coverage, and selective rerendering."""

import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import closing
from functools import partial
from pathlib import Path
from unittest.mock import create_autospec, patch

import msgspec
import numpy as np
from analysis_fixture import (
    analyze,
    exact_weights,
    pack_patterns,
    pattern_batches,
    pattern_key,
    read_pairs,
    read_pattern_cols,
)

import trustmebro.graph as graph_ops
from trustmebro import runtime as diagnostics
from trustmebro.artifacts import encode_msgpack
from trustmebro.extraction import records as r
from trustmebro.extraction.storage import encode_exprs, encode_trns, open_extraction_db
from trustmebro.visualization import archives, cli, identities, metrics, patterns, products, scan, state_render, views
from trustmebro.visualization import patterns as pattern_metrics
from trustmebro.visualization import views as processing
from trustmebro.visualization.archives import json_counts, read_stats, stats_writer, write_pairs
from trustmebro.visualization.measurements import (
    PATTERN_MODES,
    ComparisonLvl,
    Descriptors,
    FreqRow,
    GraphSize,
    Pair,
    ReuseRow,
    StateRow,
    StralRow,
    ViewMode,
)
from trustmebro.visualization.metrics import FreqCounts, FreqTotals, add_freq_totals, add_freqs, freq_stats
from trustmebro.visualization.products import MANIFEST, AnalysisPaths


class GraphPipelineTests(unittest.TestCase):
    def test_reuse_counts_survive_cache_eviction_sharing_and_huge_expansion(self) -> None:
        # Failure cases: expanded arithmetic narrowed to uint64; replacement cuts
        # discard independently reachable descendants; duplicate roots recounted
        # in DAG unions; raw/cut cache keys collide; cache eviction changes results;
        # basic reuse activates sampled descriptors. This isolated boundary permits
        # testing zero/tiny budgets without adding a public benchmark option.
        import numpy as np

        exprs: list[r.Expr] = [r.Const("base", ())]
        exprs.extend(r.App(node - 1, (node - 1,)) for node in range(1, 81))
        graph = graph_ops.build_graph(tuple(exprs))
        locals_ = tuple(r.LocalConst(idx, 0, r.LocalDeclKind.DEFAULT, False, r.BinderInfo.DEFAULT) for idx in (0, 1))
        trns = (r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(80, locals_, (), ())),)
        digest = identities.Idents(graph).sig(40)

        class Index:
            def find(self, digests):
                return np.fromiter((item == digest for item in digests), dtype=bool)

        expected = ReuseRow((1 << 81) + 1, 81, (1 << 41) + 1, 42, 80, 81)
        for budget in (0, 32, 128, 64 * 1024 * 1024):
            with patch.object(metrics, "expr_arrays", side_effect=AssertionError("unrequested descriptors")):
                result = metrics.measure_reuse("test", graph, trns, Index(), (), reach_budget=budget)
            self.assertEqual(result.states, [expected])
            self.assertEqual(result.exprs[1:], [ReuseRow(1, 1, 1, 1, 1, 1)] * 2)
            self.assertEqual(result.stral, {})

    def test_parallel_results_refill_before_consumption_and_stop_on_failure(self) -> None:
        # Failure cases: yield delays replacement until aggregation finishes;
        # refill loses/duplicates jobs or exceeds the outstanding-future bound;
        # a failed result schedules replacement before propagating its error.
        # Immediate futures isolate the ordering boundary without scheduling
        # races that an end-to-end corpus test cannot reliably distinguish.
        submitted: list[int] = []
        pool = create_autospec(ProcessPoolExecutor, instance=True)

        def submit(measure, row: int) -> Future[int]:
            submitted.append(row)
            future: Future[int] = Future()
            future.set_result(measure(row))
            return future

        def complete(futures, *, return_when):
            self.assertLessEqual(len(futures), 2)
            return set(futures), set()

        pool.submit.side_effect = submit
        with patch.object(diagnostics, "wait", side_effect=complete):
            results = diagnostics.ready_results(pool, range(5), 1, lambda row: row)
            received: list[int] = []
            for idx, result in enumerate(results):
                received.append(result)
                # All slots freed by the completed batch are refilled before
                # handing results to aggregation; no arrival-order buffer.
                self.assertEqual(submitted, list(range(4 if idx < 2 else 5)))
            self.assertEqual(sorted(received), list(range(5)))

        failed: Future[int] = Future()
        failed.set_exception(ValueError("worker failed"))
        pool.submit.reset_mock(side_effect=True)
        pool.submit.return_value = failed
        with self.assertRaisesRegex(ValueError, "worker failed"):
            next(diagnostics.ready_results(pool, range(5), 1, lambda row: row))
        self.assertEqual(pool.submit.call_count, 2)

    def test_batch_timing_excludes_consumers_and_records_failure_and_thread_cpu(self) -> None:
        # Failure cases: a suspended generator charges consumer work to production;
        # process CPU is mistaken for thread CPU; exceptions drop/mark spans as OK.
        # Numerical end-to-end results cannot expose these diagnostic errors.
        # Scripted clocks establish timer boundaries, not actual clock accuracy.
        tick = 100

        def batches():
            nonlocal tick
            for val in (1, 2):
                tick += 8
                yield val

        with tempfile.TemporaryDirectory() as tmp, closing(diagnostics.TimingLog(Path(tmp), "coordinator")) as log:
            with (
                patch.object(diagnostics.time, "perf_counter_ns", side_effect=lambda: tick),
                patch.object(diagnostics.time, "process_time_ns", side_effect=lambda: tick // 2),
                patch.object(diagnostics.time, "thread_time_ns", side_effect=lambda: tick // 4),
            ):
                received = []
                for val in diagnostics.timed_batches(log, "test", "produce", batches()):
                    received.append(val)
                    tick += 8000
                with self.assertRaisesRegex(ValueError, "deliberate"), diagnostics.checkpoint(log, "test", "fail"):
                    tick += 8
                    raise ValueError("deliberate")
            log.flush()
            with Path(log.stream.name).open(newline="") as stream:
                rows = list(csv.DictReader(stream))
        self.assertEqual(received, [1, 2])
        self.assertEqual([int(row["wall_ns"]) for row in rows], [8, 8, 0, 8])
        self.assertEqual([int(row["cpu_ns"]) for row in rows], [4, 4, 0, 4])
        self.assertEqual([int(row["thread_ns"]) for row in rows], [2, 2, 0, 2])
        self.assertEqual([row["ok"] for row in rows], ["1", "1", "1", "0"])

    def test_published_measurements_preserve_pre_cleanup_identities_and_descriptors(self) -> None:
        # Failure cases: a consistent-but-changed signature algorithm passes identity
        # comparisons; histogram splitting changes normalization; packing changes
        # archive adapters lose fields or exact counts.
        # Golden hashes were captured from the pre-cleanup implementation on this
        # fixture; DB -> analysis -> published archives is checked, not corpus speed.
        # View-dependent pattern/comparison products intentionally changed with
        # the exported baseline. Their weights, identities, and independent
        # computation are checked by the dedicated pipeline tests below.
        exprs = (
            r.Const("A", ()),
            r.Fvar(10),
            r.Fvar(20),
            r.ExprMvar(10),
            r.Const("f", (r.LvlMvar(1), r.LvlMvar(1))),
            r.App(4, (1,)),
            r.App(5, (1,)),
            r.App(5, (2,)),
            r.NatLiteral(1 << 15000),
            r.Const("Int.ofNat", ()),
            r.App(9, (8,)),
            r.Metadata((("tag", "keep"),), 10),
            r.Bvar(0),
            r.Forall(("x",), 0, 12, r.BinderInfo.DEFAULT),
            r.Forall(("y",), 0, 13, r.BinderInfo.IMPLICIT),
            r.Lambda(("z",), 0, 14, r.BinderInfo.DEFAULT),
            r.Let("v", 0, 6, 15, False),
            r.Proj("P", 0, 16),
        )
        locals_ = (r.LocalConst(0, 7, r.LocalDeclKind.DEFAULT, False, r.BinderInfo.DEFAULT),)
        roots = (6, 7, 11, 17, 3)
        trns = tuple(
            r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(root, locals_, (), ())) for root in (*roots, 6)
        )

        def canonical(data):
            if isinstance(data, dict):
                return [
                    (canonical(key), canonical(val))
                    for key, val in sorted(data.items(), key=lambda item: repr(item[0]))
                ]
            if isinstance(data, (tuple, list)):
                return [canonical(val) for val in data]
            return data

        def legacy_descriptor(row):
            # Reconstruct the old wire view only in this golden oracle.
            # Fixed numerical bins must preserve the old histogram meanings/order.
            topo = {
                f"edge_{metrics.DESCR_KINDS[a]}_{slot}_{metrics.DESCR_KINDS[b]}": val
                for (a, slot, b), val in zip(row.topo.edges, row.topo.edge_weights, strict=True)
            }
            topo.update(
                {
                    name: val
                    for idx, (name, val) in enumerate(zip(metrics.DESCR_GROUP_NAMES, row.topo.groups, strict=True))
                    if val or idx >= metrics.DENSE_ALWAYS
                }
            )
            record = {name: getattr(row, name) for name in row.__struct_fields__}
            record["topo"] = topo
            return msgspec.Raw(encode_msgpack(record))

        expected = {
            "states": "38acf8e5d8671ee05793b51005f47e9a41907a9053c5373e28f009bbe19e1f41",
            "exprs": "1bf72368384155c374a88352cb040e4b082857c100f2fb790eacf10bf931cc11",
            "freqs": "6230c43dbdaa16c41d72f3e5a82d71490cd10205cf17bdd4e0d28f83c5867cfd",
            "descriptors": "abfadbdcc04a050e36604a59347dd516f7065b5fe5e8e7048e1e38bef2ae25cc",
            "concentration": "e43a94f1ae157ccf4924c7b1770f5da1f26faf643602e3cabf8ca41074fd137c",
        }
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "source.db", Path(tmp) / "stats"
            with closing(open_extraction_db(src)) as db:
                db.execute(
                    "INSERT INTO theorems(name,module,expr_count,trn_count,exprs,trns) "
                    "VALUES('fixture','Test',?,?,?,?)",
                    (len(exprs), len(trns), encode_exprs(exprs), encode_trns(trns)),
                )
                db.commit()
            with patch.object(metrics, "CONCENTRATION_BATCH_CURVES", 4):
                # Six repeated context curves cross a partial-batch boundary;
                # the independent grid hashes catch repeated-cell undercounts.
                analyze(src, out, analyses=("metrics", "topology", "patterns"), depths=(1, 2), workers=1)
            paths = AnalysisPaths(out)
            records = list(read_stats(paths.analysis("metrics")))
            theorem = next(record for record in records if record[0] == "theorem")
            descriptors = {row.root: row for record in records if record[0] == "stral" for row in record[1]}
            mdata = next(record[1] for record in records if record[0] == "mdata")
            observed = {
                "states": theorem[2],
                "exprs": theorem[3],
                "freqs": sorted(
                    (row for record in records if record[0] == "freqs" for row in record[1]), key=lambda row: row.digest
                ),
                "descriptors": [legacy_descriptor(descriptors[root]) for root in roots],
                "concentration": (mdata.conc.tobytes(), mdata.rotated_conc.tobytes()),
            }
            for name, data in observed.items():
                with self.subTest(product=name):
                    self.assertEqual(hashlib.sha256(encode_msgpack(canonical(data))).hexdigest(), expected[name])

    def test_selected_analysis_prepares_only_consumed_views_and_sampled_descriptors(self) -> None:
        # Final counts/PNGs cannot reveal eager unused views, repeated preparation,
        # or descriptors computed for discarded sample candidates. Observe those
        # boundaries during real DB -> analysis -> archive -> rendering runs.
        exprs = (
            r.Const("A", ()),
            r.NatLiteral(1),
            r.Const("Int.ofNat", ()),
            r.App(2, (1,)),
            r.App(0, (3,)),
            r.Metadata((("test", "mdata"),), 3),
            r.App(0, (5,)),
            r.App(4, (6,)),
            r.App(2, (0,)),
            r.App(7, (8,)),
        )
        trns = tuple(
            r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(root, (), (), ())) for root in (3, 4, 6, 7, 9)
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.db"
            with closing(open_extraction_db(src)) as db:
                for name in ("first", "second"):
                    db.execute(
                        "INSERT INTO theorems(name,module,expr_count,trn_count,exprs,trns) VALUES(?,'Test',?,?,?,?)",
                        (name, len(exprs), len(trns), encode_exprs(exprs), encode_trns(trns)),
                    )
                db.commit()
            original = src.read_bytes()
            prepared: list[scan.TheoremInput] = []
            built_views: list[dict[ViewMode, views.TopoView]] = []
            prepare = scan._prepare
            prepare_views = scan.prepare_views

            def capture(*args):
                data = prepare(*args)
                prepared.append(data)
                return data

            def capture_views(*args, **kwargs):
                views = prepare_views(*args, **kwargs)
                built_views.append(views)
                return views

            with (
                patch.object(scan, "_prepare", side_effect=capture),
                patch.object(scan, "prepare_views", side_effect=capture_views) as prepared_views,
                patch.object(scan, "graph_stats", wraps=scan.graph_stats) as sizes,
                patch.object(scan, "find_matches", wraps=scan.find_matches) as matches,
                patch.object(scan, "observe_roots", wraps=scan.observe_roots) as observations,
                patch.object(metrics, "STRAL_SAMPLE_SIZE", 2),
                patch.object(metrics, "_stral_row", wraps=metrics._stral_row) as descriptors,
                patch.object(metrics, "expr_arrays", wraps=metrics.expr_arrays) as arrays,
                patch.object(metrics, "view_arrs", wraps=metrics.view_arrs) as topology_arrays,
                patch.object(processing, "_plumbing_view", wraps=processing._plumbing_view) as policies,
            ):
                with patch.object(scan, "PatternBatches", side_effect=AssertionError("unselected pattern resources")):
                    report = analyze(src, root / "metrics", analyses=("metrics",), workers=1, atlas_min_nodes=2)
                self.assertEqual(report["pattern_spills"], 0)
                self.assertFalse(any("pattern-counts-" in str(path) for path in (root / "metrics").rglob("*")))
                self.assertEqual(descriptors.call_count, 2)
                # Both selected roots belong to one theorem: arrays are prepared once.
                selected_theorems = {call.args[0] for call in descriptors.call_args_list}
                self.assertEqual(arrays.call_count, len(selected_theorems))
                self.assertEqual(policies.call_count, 0)
                self.assertEqual(prepared_views.call_count, 0)
                self.assertEqual(matches.call_count, 0)
                self.assertEqual(observations.call_count, 0)
                self.assertEqual(sizes.call_count, 2)
                self.assertEqual(topology_arrays.call_count, 0)
                rows = [
                    row
                    for record in read_stats(AnalysisPaths(root / "metrics").analysis("metrics"))
                    if record[0] == "stral"
                    for row in record[1]
                ]
                self.assertEqual(len(rows), 2)
                # Reservoir fill/replacement must preserve bottom-k selection,
                # not merely the number of measured descriptors.
                expected = sorted(
                    (hashlib.sha256(f"{name}:{root}".encode()).digest(), name, root)
                    for name in ("first", "second")
                    for root in (3, 4, 6, 7, 9)
                )[:2]
                self.assertEqual(
                    [(row.theorem, row.root) for row in rows], [(name, root) for _, name, root in expected]
                )
                self.assertTrue(all(len(row.topo.groups) == len(metrics.DESCR_GROUP_NAMES) for row in rows))
                self.assertTrue(all(row.topo.edges for row in rows))
                analyze(src, root / "patterns", analyses=("patterns",), workers=1, depths=(1, 2))
                self.assertEqual(descriptors.call_count, 2)
                self.assertEqual(sizes.call_count, 2)  # No size preparation for pattern-only analysis.
                self.assertEqual(topology_arrays.call_count, 0)
                self.assertEqual(policies.call_count, 6)
                self.assertEqual(matches.call_count, 2)
                self.assertEqual(observations.call_count, 2)
                for selected in built_views:
                    self.assertEqual(set(selected), set(PATTERN_MODES))
                    self.assertEqual(views.resolve_root(selected[ViewMode.ERASED], 3), 1)
                    self.assertEqual(selected[ViewMode.COMPACT].markers[3].kind, views.Kind.VAL_COE)
                built_views.clear()
                sizes.reset_mock()
                matches.reset_mock()
                observations.reset_mock()
                analyze(
                    src,
                    root / "combined",
                    analyses=("metrics", "topology", "patterns"),
                    workers=1,
                    depths=(1, 2),
                    atlas_min_nodes=2,
                )
                # One original graph's stats/observations/matches per theorem, despite
                # three consumers and two pattern radii; derived views share prerequisites.
                self.assertEqual(sizes.call_count, 2)
                self.assertEqual(matches.call_count, 2)
                self.assertEqual(observations.call_count, 2)
                self.assertEqual(policies.call_count, 12)
                self.assertEqual(topology_arrays.call_count, 8)
                self.assertTrue(all(set(selected) == set(processing.ViewMode) for selected in built_views))
            # Rendering consumes the published, fully populated sample after the
            # source goes away; reduced sampling did not change population size.
            summary = archives.metric_summary(AnalysisPaths(root / "metrics").analysis("metrics"))
            self.assertEqual(summary["states"], 10)
            src.rename(root / "unavailable.db")
            from trustmebro.visualization.state_render import render

            render(AnalysisPaths(root / "metrics").analysis("metrics"), root / "plots", families={"complexity"})
            self.assertTrue((root / "plots/complexity-state.png").exists())
            self.assertEqual((root / "unavailable.db").read_bytes(), original)

    def test_shared_worker_traversals_preserve_radii_alias_weights_and_repeated_roots(self) -> None:
        # Failure cases: alias caching loses raw occurrence/affected/head weights;
        # smaller BFS prefixes keep deeper edges or mislabel true leaves; sorting
        # measurement keys drops repeated roots; caches cross view boundaries;
        # vocabulary and baseline patterns repeat the same fragment traversal.
        # Published counts check behavior. Spies additionally establish reuse and
        # the leaf fast path, which unchanged archives alone cannot demonstrate.
        exprs = (
            r.Const("f", ()),
            r.Const("A", ()),
            r.NatLiteral(1),
            r.Const("Int.ofNat", ()),
            r.App(3, (2,)),
            r.Metadata((("tag", "keep"),), 4),
            r.App(0, (1,)),
            r.App(0, (4,)),
            r.App(6, (6,)),
            r.App(8, (7,)),
            r.Bvar(0),
            r.Forall(("x",), 1, 10, r.BinderInfo.DEFAULT),
        )
        locals_ = tuple(r.LocalConst(idx, 6, r.LocalDeclKind.DEFAULT, False, r.BinderInfo.DEFAULT) for idx in (0, 1))
        roots = (4, 5, 6, 7, 8, 9, 11, 6)
        trns = tuple(r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(root, locals_, (), ())) for root in roots)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.db"
            with closing(open_extraction_db(src)) as db:
                db.execute(
                    "INSERT INTO theorems(name,module,expr_count,trn_count,exprs,trns) VALUES('fixture','Test',?,?,?,?)",
                    (len(exprs), len(trns), encode_exprs(exprs), encode_trns(trns)),
                )
                db.commit()
            with (
                patch.object(metrics, "topo_sig", wraps=metrics.topo_sig) as sigs,
                patch.object(metrics, "measure_view", wraps=metrics.measure_view) as sizes,
                patch.object(pattern_metrics, "extract_patterns", wraps=pattern_metrics.extract_patterns) as patterns,
                patch.object(
                    pattern_metrics, "pattern_observations", wraps=pattern_metrics.pattern_observations
                ) as observations,
                patch.object(identities, "_root_order", wraps=identities._root_order) as traversals,
            ):
                analyze(src, root / "combined", analyses=("topology", "patterns"), depths=(3, 1, 5))
                self.assertEqual(observations.call_count, 1)
                for spy, key in (
                    (sigs, lambda call: (id(call.args[0]), call.args[1])),
                    (sizes, lambda call: (id(call.args[0]), tuple(call.args[-1]))),
                    (patterns, lambda call: (id(call.args[0]), call.args[1])),
                ):
                    keys = [key(call) for call in spy.call_args_list]
                    self.assertEqual(len(keys), len(set(keys)))
                self.assertTrue(all(call.args[2] == (3, 1, 5) for call in patterns.call_args_list))
                self.assertTrue(all(call.args[0].edges[call.args[1]] for call in traversals.call_args_list))
                reused_pattern_calls = patterns.call_count
                reused_topo_calls = sigs.call_count
            # Changed ordered edges/labels, aliases, and binder provenance must
            # still match independently computed views. Spies establish actual
            # skips, while published products establish their correctness.
            with (
                patch.object(scan, "sig_reuse", return_value={}),
                patch.object(
                    pattern_metrics, "extract_patterns", wraps=pattern_metrics.extract_patterns
                ) as independent_patterns,
                patch.object(metrics, "topo_sig", wraps=metrics.topo_sig) as independent_topo,
            ):
                analyze(src, root / "independent", analyses=("topology", "patterns"), depths=(3, 1, 5))
                self.assertLess(reused_pattern_calls, independent_patterns.call_count)
                self.assertLess(reused_topo_calls, independent_topo.call_count)
            paths = AnalysisPaths(root / "combined")
            independent = AnalysisPaths(root / "independent")
            self.assertEqual(read_pairs(paths.comparisons), read_pairs(independent.comparisons))
            pairs = read_pairs(paths.comparisons)
            self.assertTrue(
                any(pair[0].expanded > pair[0].nodes for pair in pairs[ViewMode.INSTS]["state"]),
                "repeated hypotheses must survive measurement-cache key construction",
            )
            expected_top = len(trns) * (1 + len(locals_))
            for mode in PATTERN_MODES:
                for rad in (3, 1, 5):
                    rows = [row for batch in pattern_batches(paths.patterns(mode, rad)) for row in batch]
                    self.assertEqual(
                        sum(weights[0] for _, flavour, head, weights in rows if flavour == 0 and not head), expected_top
                    )
                    independent_rows = [
                        row for batch in pattern_batches(independent.patterns(mode, rad)) for row in batch
                    ]
                    self.assertEqual(sorted(rows), sorted(independent_rows))
            for rad in (1, 5):
                analyze(src, root / f"radius-{rad}", analyses=("patterns",), depths=(rad,))
                single = AnalysisPaths(root / f"radius-{rad}")
                for mode in PATTERN_MODES:
                    observed = sorted(row for batch in pattern_batches(single.patterns(mode, rad)) for row in batch)
                    expected = sorted(row for batch in pattern_batches(paths.patterns(mode, rad)) for row in batch)
                    self.assertEqual(observed, expected)

    def test_archived_batches_match_pop_rasters_and_selective_figures(self) -> None:
        # Figure existence cannot catch wrong bin totals, chunk-mean averaging,
        # duplicate/outlier loss, max initialization, or approximate medians.
        # Exercise archive -> selected columns -> raster -> figure; also poison
        # unrelated payloads to prove a context-only render never decodes them.
        # Count buffer creation/finalization too: numerical equivalence alone
        # cannot expose full-image allocation/work on every small theorem batch.
        # Also catch duplicate decoding/projection, a frame per tiny theorem,
        # and budget fallback losing observations or altering the final image.
        import datashader as ds
        import numpy as np
        import pandas as pd
        from matplotlib.figure import Figure

        from trustmebro.visualization import drawing
        from trustmebro.visualization.drawing import (
            Axis,
            DensityCfg,
            DensityPlot,
            density,
            density_figure,
            density_raster,
            prepare_points,
        )
        from trustmebro.visualization.measurements import StateRow
        from trustmebro.visualization.state_render import _metric_src, render

        huge = 1 << 15000
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "metrics.msgpack.zst"
            rows = [
                StateRow("dense", i, 2, 4, 3, 2, 8, 4, 2, 12, 5, (i % 5) / 4, 7, 0, 0, {}, {}, {}) for i in range(37)
            ]
            rows.append(
                StateRow("outlier", 0, 1, huge, 15001, 15001, 1, 1, 1, huge + 1, 15002, 1, 15002, 0, 0, {}, {}, {})
            )
            with stats_writer(path) as write:
                for row in rows:
                    write(("theorem", row.theorem, [row], []))
            src = _metric_src(
                path,
                "state",
                ("state_distinct", "state_expanded", "largest_frac"),
                {"state_distinct", "state_expanded"},
            )
            reference = pd.concat([pd.DataFrame(cols) for cols in src()], ignore_index=True)
            self.assertEqual([len(next(iter(cols.values()))) for cols in src()], [len(rows)])
            with patch.object(state_render, "col_batches", partial(drawing.col_batches, size=8)):
                chunks = list(src())
                self.assertEqual([len(next(iter(cols.values()))) for cols in chunks], [8, 8, 8, 8, 6])
                pd.testing.assert_frame_equal(
                    pd.concat([pd.DataFrame(cols) for cols in chunks], ignore_index=True), reference
                )
            plot = DensityPlot(
                "raster.png",
                "All observations",
                Axis("state_distinct_log", "Nodes", "log10"),
                Axis("state_expanded_log", "Expanded", "log10"),
                "",
            )
            bounds = prepare_points(src, plot, DensityCfg()).bounds
            self.assertIsNotNone(bounds)
            self.assertGreater(bounds[1][1], 4500)  # enormous outlier still included
            # Array handoff must neither concatenate a complete chunk again nor
            # copy already-owned coordinates. DataFrame/sliced input must detach
            # its backing storage so unused columns do not evade the budget.
            columns = next(src())
            with patch.object(np, "concatenate", side_effect=AssertionError("unnecessary column concatenation")):
                prepared = drawing.prepare_points(lambda: iter((columns,)), plot, DensityCfg())
            self.assertIs(prepared.chunks[0][plot.x.field], columns[plot.x.field])
            framed = drawing.prepare_points(reference, plot, DensityCfg())
            self.assertTrue(all(col.base is None for cols in framed.chunks for col in cols.values()))
            canvas = ds.Canvas(plot_width=31, plot_height=23, x_range=bounds[0], y_range=bounds[1])
            counts = canvas.points(reference, plot.x.field, plot.y.field, agg=ds.count()).values
            self.assertEqual(counts.sum(), len(rows))
            for cfg in (
                DensityCfg(),
                DensityCfg(weight="largest_frac"),
                DensityCfg(colour="largest_frac", reduce_color="max", colour_proj="linear", colour_range=(0, 1)),
                DensityCfg(colour="largest_frac", reduce_color="mean", colour_proj="linear", colour_range=(0, 1)),
                DensityCfg(median="largest_frac", colour_proj="linear", colour_range=(0, 1)),
            ):
                operations: list[str] = []
                compile_components = drawing.compile_components

                def compile_spy(*args, compile_components=compile_components, operations=operations, **kwargs):
                    components = list(compile_components(*args, **kwargs))
                    create, finalize = components[0], components[4]

                    def created(*args, **kwargs):
                        operations.append("create")
                        return create(*args, **kwargs)

                    def finalized(*args, **kwargs):
                        operations.append("finalize")
                        return finalize(*args, **kwargs)

                    components[0], components[4] = created, finalized
                    return tuple(components)

                with (
                    patch.object(drawing, "compile_components", side_effect=compile_spy),
                    patch.object(ds.Canvas, "points", side_effect=AssertionError("per-batch raster construction")),
                ):
                    actual = density_raster(src, plot, cfg, canvas).vals
                self.assertEqual(operations, ["create", "finalize"])
                if cfg.median:
                    # Both dense batches occupy one pixel; median includes all
                    # 37 observations, while the isolated outlier is masked (<3).
                    expected = np.full_like(actual, np.nan)
                    expected[counts > 3] = np.median(reference.largest_frac.iloc[:-1])
                elif cfg.colour:
                    reducer = ds.mean(cfg.colour) if cfg.reduce_color == "mean" else ds.max(cfg.colour)
                    expected = canvas.points(reference, plot.x.field, plot.y.field, agg=reducer).values
                else:
                    vals = canvas.points(
                        reference, plot.x.field, plot.y.field, agg=ds.sum(cfg.weight) if cfg.weight else ds.count()
                    ).values
                    expected = np.where(vals > 0, np.log10(np.maximum(vals, np.finfo(float).tiny)), np.nan)
                np.testing.assert_allclose(actual, expected, equal_nan=True)

            saved: list[str] = []

            def inspect(fig, filename, **kwargs):
                ax = fig.axes[0]
                for axis in (ax.xaxis, ax.yaxis):
                    formatter = axis.get_major_formatter()
                    self.assertEqual(formatter(2, 0), "100")
                    self.assertEqual(formatter(6, 0), "1,000,000")
                    self.assertEqual(formatter(7, 0), "1e7")
                    self.assertEqual(formatter(4500, 0), "1e4500")
                saved.append(Path(filename).name)

            with (
                patch.object(Figure, "savefig", inspect),
                patch.object(state_render, "metric_cols", wraps=state_render.metric_cols) as reads,
                patch.object(state_render, "log_counts", wraps=state_render.log_counts) as projections,
            ):
                density(root, src, plot)
                self.assertEqual(reads.call_count, 1)
                self.assertEqual(projections.call_count, 2)
            self.assertEqual(saved, ["raster.png"])
            images: list[np.ndarray] = []

            def capture(fig, filename, **kwargs):
                images.append(np.asarray(fig.axes[0].images[0].get_array()).copy())

            with patch.object(Figure, "savefig", capture):
                density(root, src, plot)
                with (
                    patch.object(drawing, "POINT_MEMORY_BUDGET", 150),
                    patch.object(drawing, "col_batches", partial(drawing.col_batches, size=8)),
                    patch.object(state_render, "metric_cols", wraps=state_render.metric_cols) as reads,
                ):
                    density(root, src, plot)
                    self.assertEqual(reads.call_count, 2)
            np.testing.assert_allclose(images[0], images[1], equal_nan=True)
            # Prepared-raster builders and the runner share the actual numerical
            # image path, not a separate appendix-only plotting implementation.
            import matplotlib as mpl
            import matplotlib.pyplot as plt

            with mpl.rc_context({"axes.facecolor": "pink", "image.cmap": "viridis"}):
                before = mpl.rcParams.copy()
                prepared = density_raster(src, plot, DensityCfg(), canvas)
                fig = density_figure(prepared, plot, DensityCfg())
                try:
                    np.testing.assert_allclose(fig.axes[0].images[0].get_array(), prepared.vals, equal_nan=True)
                    self.assertEqual(fig.axes[0].get_xlabel(), plot.x.label)
                    self.assertEqual(fig.axes[0].get_facecolor(), (0, 0, 0, 1))
                    self.assertEqual(fig.axes[0].images[0].get_cmap().name, "turbo")
                    self.assertEqual(dict(mpl.rcParams), dict(before))
                finally:
                    plt.close(fig)

            # A linear colour is not necessarily a fraction. Its data range
            # must survive archive decoding, raster aggregation, and plotting.
            def inspect_colour(fig, filename, **kwargs):
                norm = fig.axes[0].images[0].norm
                self.assertEqual((norm.vmin, norm.vmax), (5, 15002))
                self.assertEqual(fig.axes[-1].get_ylabel(), "Distinct nodes")

            colour_src = _metric_src(
                path,
                "state",
                ("state_distinct", "state_expanded"),
                {"state_expanded"},
                lambda cols: {**cols, "state_distinct_log": np.log10(cols["state_distinct"])},
            )
            with patch.object(Figure, "savefig", inspect_colour):
                density(
                    root,
                    colour_src,
                    plot,
                    DensityCfg(colour="state_distinct", colour_proj="linear", label="Distinct nodes"),
                    dot_diam=3,
                )

            # Publication creates valid envelopes; normal end-to-end runs would
            # miss a missing completion marker or trailing records. Rendering
            # corrupted archives must reject both before saving a population plot.
            for suffix in ("incomplete", "trailing"):
                broken = root / f"{suffix}.msgpack.zst"
                with archives.writer(broken) as write:
                    for encoded in archives.records(path):
                        record = archives.decode_record(encoded)
                        if record[0] != "complete":
                            write(record)
                    if suffix == "trailing":
                        write(("complete",))
                        write(("theorem", "trailing", [], []))
                with self.assertRaises(ValueError):
                    render(broken, root / suffix, families={"complexity"})
                self.assertFalse(list((root / suffix).glob("*.png")))

            # Context consumer requires two grids, not theorem/frequency/atlas
            # payloads, nor the unused excess grid. All are deliberately invalid.
            grid = np.zeros((10, 10), dtype=np.int64)
            grid[5, 7] = 3
            selective = root / "selective.msgpack.zst"
            with archives.writer(selective) as write:
                write(("theorem", "broken", "do not decode", "do not decode"))
                stral = archives.col_blocks(
                    [StralRow("unused", 0, 15001, 15001, huge, 0.25, {}, Descriptors((), (), ()))], StralRow
                )
                for name in ("constrs", "topo"):
                    stral[0]["cols"][name] = archives.PackedCol(
                        "msgpack", 1, encode_msgpack(msgspec.msgpack.Ext(127, b"unrequested"))
                    )
                write(("stral", stral))
                write(("shape", "do not decode"))
                encoded = {
                    "conc": archives.array_data(grid),
                    "rotated_conc": archives.array_data(grid),
                    "empty_ctxts": 0,
                    "stral_roots": 38,
                    "excess_conc": "do not decode",
                }
                write(("mdata", encoded))
                write(
                    (
                        "stats",
                        {"states": 38, "theorems": 2, "expr_idents": 0, "pcentiles_expanded": {}},
                        "do not decode",
                    )
                )
                write(("complete",))
            with patch.object(archives, "block_rows", side_effect=AssertionError("unrelated rows decoded")):
                render(selective, root / "context", families={"context"})
                render(selective, root / "embedding", families={"embedding"})
            self.assertTrue((root / "context/context-concentration.png").exists())
            self.assertTrue((root / "embedding/sharing-nesting.png").exists())

    def test_failed_replacement_preserves_published_inputs_and_selective_rendering(self) -> None:
        # Pipeline failures: fresh/partial publication, replacement losing the
        # previous generation, embedding interruption, missing required files,
        # and head-only rendering incorrectly requiring unrelated products.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, stats = root / "source.db", root / "stats"
            exprs = (r.Const("f", ()), r.Const("A", ()), r.App(0, (1,)))
            trns = (r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(2, (), (), ())),)
            with closing(open_extraction_db(src)) as db:
                db.execute(
                    "INSERT INTO theorems(name,module,expr_count,trn_count,exprs,trns) VALUES('test','Test',3,1,?,?)",
                    (encode_exprs(exprs), encode_trns(trns)),
                )
                db.commit()
            original_db = src.read_bytes()
            analyze(src, stats, depths=(1,))
            manifest = (stats / MANIFEST).read_bytes()
            paths = AnalysisPaths(stats)
            original_metric = paths.analysis("metrics").read_bytes()
            generations = set((stats / ".generations").iterdir())
            save_shapes = scan.write_shapes
            calls = 0

            def fail_after_first_view(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected topology failure")
                return save_shapes(*args, **kwargs)

            for output in (stats, root / "fresh"):
                calls = 0
                with (
                    patch.object(scan, "write_shapes", fail_after_first_view),
                    self.assertRaisesRegex(RuntimeError, "injected topology failure"),
                ):
                    analyze(src, output, analyses=("topology",), replace=True)
                self.assertFalse(list(output.rglob(".pending-*")))
            self.assertFalse((root / "fresh" / MANIFEST).exists())
            self.assertEqual((stats / MANIFEST).read_bytes(), manifest)
            self.assertEqual(set((stats / ".generations").iterdir()), generations)

            # Exercise the public CLI's transaction across both producer stages
            # without adding an expensive UMAP computation to failure coverage.
            def interrupted_stages(command, stages, args):
                self.assertEqual(command, "analyze")
                self.assertIn("embedding", stages)
                cli._analyze("shared", args)
                AnalysisPaths(args.stats).embedding("size-aware").write_bytes(b"unfinished embedding")
                raise KeyboardInterrupt("injected embedding interruption")

            with patch.object(cli, "_run_stages", interrupted_stages), self.assertRaises(KeyboardInterrupt):
                cli.main(["--db", str(src), "--stats", str(stats), "--replace", "--graphs", "embedding"])
            self.assertEqual((stats / MANIFEST).read_bytes(), manifest)
            self.assertEqual(AnalysisPaths(stats).analysis("metrics").read_bytes(), original_metric)
            self.assertEqual(set((stats / ".generations").iterdir()), generations)

            # Interruption just after the manifest swap is a completed commit,
            # not permission to delete the newly referenced generation.
            committed_stats = root / "committed"
            save_summary = archives.write_summary

            def interrupt_after_manifest(path, val):
                save_summary(path, val)
                if path.name == MANIFEST:
                    raise KeyboardInterrupt("interrupted after publication")

            with (
                patch.object(products, "write_summary", interrupt_after_manifest),
                self.assertRaises(KeyboardInterrupt),
            ):
                analyze(src, committed_stats, analyses=("metrics",))
            committed_paths = AnalysisPaths(committed_stats)
            committed_paths.require(("metrics",), {"metrics": ("metrics.msgpack.zst",)})
            self.assertTrue(list(read_stats(committed_paths.analysis("metrics"))))

            src.rename(root / "unavailable.db")
            command = ["--stats", str(stats), "--output", str(root / "plots")]
            self.assertEqual(cli.main([*command, "--graphs", "heads"]), 0)
            # A selected consumer must fail preflight, but unrelated consumers
            # can still use their intact artifacts in the same generation.
            paths.topo().unlink()
            self.assertEqual(cli.main([*command, "--graphs", "heads"]), 0)
            with self.assertRaises(SystemExit):
                cli.main([*command, "--graphs", "topology"])
            paths.heads.unlink()
            with self.assertRaises(SystemExit):
                cli.main([*command, "--graphs", "heads"])
            self.assertEqual((root / "unavailable.db").read_bytes(), original_db)

    def test_huge_counts_cross_archive_and_plot_boundaries_without_decimal_conversion(self) -> None:
        # Boundary integration is needed beyond the normal DB-to-figure fixture:
        # a 15k-deep doubling DAG would make reachability/signature work costly.
        # Failures covered: decimal limits, integer truncation, extension rejection,
        # lost aggregation, float overflow during projection, lossy JSON output,
        # and changed array layouts/huge-int conversion after switching records.
        from trustmebro.visualization.state_render import frequency_frame

        huge = 1 << 15000
        conversion_limit = sys.get_int_max_str_digits()
        counts: FreqCounts = {}
        local = [FreqRow(b"a" * 32, 15001, huge, 1, 1, 1), FreqRow(b"b" * 32, 1, 1, 0, 1, huge)]
        add_freqs(counts, local)
        add_freqs(counts, local)
        expected = list(counts.values())
        summary = FreqTotals()
        add_freq_totals(summary, expected)
        stats = freq_stats(summary)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "counts.msgpack.zst"
            reuse = ReuseRow(huge, 15001, 1, 1, 0, 15001)
            with stats_writer(path) as write:
                write(("freqs", expected))
                write(("global_reuse", "huge", [reuse], [reuse]))
            record, reused = read_stats(path)
            self.assertEqual(record[1], expected)
            self.assertEqual(reused, ("global_reuse", "huge", [reuse], [reuse]))
            size = GraphSize(15001, 30000, 0, 15001, huge, 2)
            pairs: dict[ViewMode, dict[ComparisonLvl, Counter[Pair]]] = {
                ViewMode.INSTS: {"expression": Counter({(size, size): huge})}
            }
            pairs_path = Path(tmp) / "pairs.msgpack.zst"
            write_pairs(pairs_path, pairs)
            self.assertEqual(read_pairs(pairs_path), pairs)
            decoded = FreqTotals()
            add_freq_totals(decoded, record[1])
            self.assertEqual(freq_stats(decoded), stats)
            self.assertNotIn("reuse_bins", stats)
            self.assertEqual(counts[b"a" * 32].expanded, huge)
            self.assertEqual(counts[b"b" * 32].expanded_tree, 2 * huge)
            for pop in range(3):
                frame = frequency_frame(((a, b, weights) for (a, b), weights in decoded.pairs.items()), pop, "total")
                self.assertTrue(frame.expanded_log.notna().all())
                self.assertTrue((frame.expanded_log < float("inf")).all())
            displayed = json.loads(json.dumps(json_counts(stats)))
            self.assertEqual(int(displayed["pairs"][0][1], 16), huge)
        self.assertEqual(sys.get_int_max_str_digits(), conversion_limit)

    def test_selected_column_batches_validate_without_decoding_unrelated_fields(self) -> None:
        # Failure cases: aligned observation keys or exact uint64/huge values are
        # lost across blocks; invalid lengths/schema/exceptions are accepted;
        # unrelated packed payloads are decoded; completion/truncation is missed.
        # This archive boundary test inspects native bytes independently of its
        # reader, and establishes exactness/selection, not throughput or RSS.
        from copy import deepcopy

        huge = 1 << 15000
        maximum = (1 << 64) - 1
        expected = [0, maximum, maximum + 1, huge] * 5
        rows = [
            StateRow("selected", idx, 1, 1, 1, 1, 1, 1, 1, val, 1, 0.5, 1, 0, 0, {}, {}, {})
            for idx, val in enumerate(expected)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "selected.zst"

            def archive(blocks):
                with archives.writer(path) as write:
                    write(("theorem", "selected", blocks, msgspec.msgpack.Ext(127, b"unrequested expressions")))
                    write(("complete",))

            with patch.object(archives, "COL_BLOCK_ROWS", 7):
                blocks = archives.col_blocks(rows, StateRow)
            self.assertEqual([block["rows"] for block in blocks], [7, 7, 6])
            base = blocks[0]["cols"]["state_expanded"]
            np.testing.assert_array_equal(np.frombuffer(base.data, dtype="<u8"), [0, maximum, 0, 0, 0, maximum, 0])
            self.assertEqual(base.indices, (2, 3, 6))
            self.assertEqual(base.exceptions, (maximum + 1, huge, maximum + 1))
            for block in blocks:
                block["cols"]["goal_constrs"] = archives.PackedCol(
                    "msgpack", block["rows"], encode_msgpack(msgspec.msgpack.Ext(127, b"unrequested"))
                )
            archive(blocks)
            batches = list(archives.metric_cols(path, "theorem", "state", ("state_expanded", "largest_frac"), size=3))
            self.assertEqual([int(val) for cols in batches for val in cols["state_expanded"]], expected)
            np.testing.assert_array_equal(
                np.concatenate([cols["largest_frac"] for cols in batches]), np.full(len(rows), 0.5)
            )
            for defect in ("length", "keys", "schema", "exception", "duplicates", "packed"):
                bad = deepcopy(blocks)
                block = bad[0]
                if defect == "length":
                    block["cols"]["state_expanded"] = archives.CountCol("<u8", b"", (), ())
                elif defect == "keys":
                    block["ordinal"] = np.arange(1, 8, dtype="<u8").tobytes()
                elif defect == "schema":
                    block["schema"] = tuple(reversed(block["schema"]))
                elif defect == "exception":
                    block["cols"]["state_expanded"] = archives.CountCol(
                        "<u8", base.data, base.indices, (1, huge, maximum + 1)
                    )
                elif defect == "duplicates":
                    block["cols"]["state_expanded"] = archives.CountCol("<u8", base.data, (2, 2, 6), base.exceptions)
                else:
                    block["cols"]["goal_constrs"] = archives.PackedCol("msgpack", 1, b"")
                archive(bad)
                with self.subTest(defect=defect), self.assertRaises((ValueError, msgspec.ValidationError)):
                    list(archives.metric_cols(path, "theorem", "state", ("state_expanded",)))
            archive([])
            self.assertEqual(list(archives.metric_cols(path, "theorem", "state", ("state_expanded",))), [])

    def test_raw_frequency_archive_preserves_reuse_plot_statistics(self) -> None:
        # Archive-to-raster failures: display bins persisted instead of raw rows;
        # bins computed independently per batch; arithmetic replacing geometric
        # mean; occurrence weights lost; huge counts cast to float before logs.
        # This checks numerical equivalence, not full-corpus runtime/memory.
        import datashader as ds
        import numpy as np

        from trustmebro.visualization.drawing import DensityCfg, density_raster, prepare_points
        from trustmebro.visualization.state_render import _reuse_breadth_plot, reuse_breadth

        rows = [
            FreqRow(b"a" * 32, 4, 8, 10, 20, 80, 3),
            FreqRow(b"b" * 32, 9, 18, 0, 5, 20, 3),
            FreqRow(b"c" * 32, 1, 1 << 15000, 1, 1, 1 << 15000),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "freqs.zst"
            with stats_writer(path) as write:
                for row in rows:
                    write(("freqs", [row]))
            columns = ("theorems", "expanded_tree", "root_dag", "distinct")
            frame = reuse_breadth(archives.metric_cols(path, "freqs", None, columns, size=1))
            self.assertEqual(frame.identities.tolist(), [2, 1])
            self.assertEqual(frame.breadth.tolist(), [0.48, 0.0])
            self.assertEqual(frame.repetition.iloc[0], 0.60)
            self.assertGreater(frame.repetition.iloc[1], 4500)
            np.testing.assert_allclose(frame["size"], [np.log10(6), 0])
            plot = _reuse_breadth_plot("reuse.png")
            cfg = DensityCfg(weight="identities")
            bounds = prepare_points(frame, plot, cfg).bounds
            canvas = ds.Canvas(plot_width=31, plot_height=23, x_range=bounds[0], y_range=bounds[1])
            raster = density_raster(frame, plot, cfg, canvas).vals
            self.assertAlmostEqual(np.power(10, raster[np.isfinite(raster)]).sum(), 3)
            self.assertEqual([row for _, batch in read_stats(path) for row in batch], rows)
            # Bin sorting, regrouped floating sums, and huge-int NumPy logs can
            # subtly change this plot. Compare varied chunking against the exact
            # scalar reference, including first-occurrence order and empty input.
            import math

            import pandas as pd

            rows += [FreqRow(bytes([i]), i % 9 + 1, 8, 1, i % 5 + 1, 80 + i, i % 7 + 1) for i in range(80)]
            bins: dict[tuple[float, float], tuple[int, float]] = {}
            for row in rows:
                key = (
                    round(math.log10(row.theorems), 2),
                    round(math.log10(row.expanded_tree) - math.log10(row.root_dag), 2),
                )
                count, size = bins.get(key, (0, 0.0))
                bins[key] = count + 1, size + math.log10(row.distinct)
            expected = pd.DataFrame(
                [(x, y, size / count, count) for (x, y), (count, size) in bins.items()],
                columns=["breadth", "repetition", "size", "identities"],
            )
            path = Path(tmp) / "batching.zst"
            with stats_writer(path) as write:
                write(("freqs", rows))
            for size in (1, 7, 100):
                actual = reuse_breadth(archives.metric_cols(path, "freqs", None, columns, size=size))
                pd.testing.assert_frame_equal(actual, expected, check_exact=True)
            self.assertTrue(reuse_breadth(()).empty)

    def test_streamed_points_preserve_edge_bins_and_counters_above_uint32(self) -> None:
        # Failure cases: replacing Canvas.points changes upper-edge inclusion,
        # neighbouring boundary bins, clipping, invalid-row filtering, signed
        # maxima, sums/means across unequal chunks, or median/count alignment;
        # persistent uint32 counts
        # wrap after many chunks. Real proof fixtures cannot reliably target
        # floating pixel boundaries or produce four billion rows, so compare
        # the adapter to public Datashader and seed only the large-count case.
        import datashader as ds
        import numpy as np
        import pandas as pd

        from trustmebro.visualization import drawing

        frame = pd.DataFrame(
            {
                "x": [
                    -1,
                    np.nextafter(-1.0, -2),
                    -0.5,
                    0,
                    0,
                    0,
                    np.nextafter(1.0, 0),
                    1,
                    3,
                    np.nextafter(3.0, 4),
                    np.nan,
                    0,
                ],
                "y": [-2, 0, -1, 1, 1, 1, 2, 2, 5, 2, 1, np.inf],
                "val": [-3, 2, -1, 0, 7, 2, 1, 5, 4, 2, 1, 1],
            },
            dtype=float,
        )
        boundaries = -1 + np.arange(8) * (4 / 7)
        nearby = np.repeat(
            np.concatenate((boundaries, np.nextafter(boundaries, -np.inf), np.nextafter(boundaries, np.inf))), 3
        )
        frame = pd.concat(
            (frame, pd.DataFrame({"x": nearby, "y": 1.0, "val": np.arange(len(nearby), dtype=float)})),
            ignore_index=True,
        )
        canvas = ds.Canvas(plot_width=7, plot_height=5, x_range=(-1, 3), y_range=(-2, 5))
        plot = drawing.DensityPlot("", "", drawing.Axis("x", ""), drawing.Axis("y", ""), "")

        def src():
            yield frame.iloc[:0]
            for start, stop in ((0, 2), (2, 7), (7, 8), (8, len(frame))):
                yield frame.iloc[start:stop]

        valid = frame[np.isfinite(frame.to_numpy()).all(axis=1)]
        for cfg, agg in (
            (drawing.DensityCfg(), ds.count()),
            (drawing.DensityCfg(weight="val"), ds.sum("val")),
            (drawing.DensityCfg(colour="val", reduce_color="max"), ds.max("val")),
            (drawing.DensityCfg(colour="val", reduce_color="mean"), ds.mean("val")),
        ):
            expected = canvas.points(valid, "x", "y", agg=agg).values
            if not cfg.colour:
                expected = np.where(expected > 0, np.log10(np.maximum(expected, np.finfo(float).tiny)), np.nan)
            actual = drawing.density_raster(src, plot, cfg, canvas).vals
            np.testing.assert_allclose(actual, expected, equal_nan=True)
            empty = drawing.density_raster(lambda: iter(()), plot, cfg, canvas).vals
            self.assertTrue(np.isnan(empty).all())

        # Use the public single-point raster as an independent pixel-ID oracle;
        # duplicating the renderer's scale/translate formula would miss mistakes.
        cells: dict[int, list[float]] = {}
        for idx in range(len(valid)):
            point_counts = canvas.points(valid.iloc[[idx]], "x", "y", agg=ds.count()).values
            for cell in np.flatnonzero(point_counts):
                cells.setdefault(int(cell), []).append(float(valid.iloc[idx]["val"]))
        expected_medians = np.full((5, 7), np.nan)
        for cell, vals in cells.items():
            if len(vals) >= 3:
                expected_medians.ravel()[cell] = np.median(vals)
        actual_medians = drawing.density_raster(src, plot, drawing.DensityCfg(median="val"), canvas).vals
        np.testing.assert_allclose(actual_medians, expected_medians, equal_nan=True)

        original = drawing.compile_components

        def seeded(*args, **kwargs):
            components = list(original(*args, **kwargs))
            create = components[0]

            def near_overflow(shape):
                bases = create(shape)
                for base in bases:
                    if base.dtype == np.uint32:
                        base.fill(np.iinfo(np.uint32).max - 1)
                return bases

            components[0] = near_overflow
            return tuple(components)

        point = pd.DataFrame({"x": [0.1], "y": [0.1]})
        small = ds.Canvas(plot_width=2, plot_height=2, x_range=(0, 1), y_range=(0, 1))
        with patch.object(drawing, "compile_components", side_effect=seeded):
            reduction = drawing.prepare_point_raster(point, small, "x", "y", ds.count())
        for _ in range(3):
            reduction.append(point)
        self.assertEqual(int(reduction.finish().values[0, 0]), 2**32 + 1)

    def test_constructor_archive_is_read_once_and_keeps_exact_percentile_totals(self) -> None:
        # Failure cases: compact columns overflow individual counts or group
        # sums, percentile ties move bands, batching changes constructor keys,
        # or a second pass is still needed despite a small retained population.
        # The synthetic huge counts exercise boundaries real Lean fixtures do
        # not reach; archive -> renderer -> figure is the checked public path.
        import numpy as np
        from matplotlib.figure import Figure

        from trustmebro.visualization import drawing
        from trustmebro.visualization.measurements import StateRow

        huge = 1 << 15000
        rows = [
            StateRow("T", idx, 1, 1, 1, 1, 1, 1, 1, size, 1, 1.0, 1, 0, 0, {}, {}, constrs)
            for idx, (size, constrs) in enumerate(
                [(2, {"const": 2**63 - 1, "app": 1})] * 7
                + [(3, {"const": 3, "app": 8}), (huge, {"const": huge, "mdata": huge})]
            )
        ]
        sizes = sorted(row.state_expanded for row in rows)
        bands = [Counter() for _ in range(6)]
        boundaries = [0.5, 0.9, 0.95, 0.99, 0.999, 1]
        for row in rows:
            left = sum(size < row.state_expanded for size in sizes)
            right = sum(size <= row.state_expanded for size in sizes)
            rank = (left + right + 1) / (2 * len(sizes))
            idx = min(sum(boundary < rank for boundary in boundaries), 5)
            bands[idx].update(row.state_constrs)
        kinds = sorted({name for row in rows for name in row.state_constrs})
        expected = np.array([[band[name] / sum(band.values()) if band else np.nan for name in kinds] for band in bands])
        images: list[np.ndarray] = []

        def inspect(fig, filename, **kwargs):
            images.append(np.asarray(fig.axes[0].images[0].get_array()).copy())

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "metrics.msgpack.zst"
            with stats_writer(path) as write:
                for row in rows:
                    write(("theorem", row.theorem, [row], []))
                write(("stats", {"states": len(rows), "theorems": 1, "expr_idents": 0, "pcentiles_expanded": {}}))
            with patch.object(Figure, "savefig", inspect):
                for budget, passes in ((1 << 30, 1), (0, 2)):
                    with (
                        patch.object(state_render, "POINT_MEMORY_BUDGET", budget),
                        patch.object(state_render, "metric_cols", wraps=state_render.metric_cols) as reads,
                        patch.object(state_render, "col_batches", partial(drawing.col_batches, size=3)),
                    ):
                        state_render.render(path, root, families={"constructors"})
                        self.assertEqual(reads.call_count, passes)
        for image in images:
            np.testing.assert_allclose(image, expected, equal_nan=True)

    def test_concentration_batches_preserve_boundary_cells_and_repeated_weights(self) -> None:
        # Failure cases: fastmath or reassociation moves boundary pixels; fused
        # writes lose repetitions; clipping/truncation changes the rotated grid;
        # partial batches are dropped. Proof fixtures cannot reliably produce
        # nextafter neighbours of every pixel boundary, so check this kernel's
        # boundary against independent NumPy raster construction, not runtime.
        import numpy as np

        positions = np.arange(1000)
        x = (positions + 0.5) / 1000
        boundaries = positions / 1000
        patterns = np.array((boundaries, np.nextafter(boundaries, 0), np.nextafter(boundaries, 1), x))
        curves = np.concatenate((np.tile(patterns, (17, 1)), patterns[:1]))
        expected = np.zeros((1000, 1000), np.int64), np.zeros((1000, 1000), np.int64)
        actual = np.zeros((1000, 1000), np.int64), np.zeros((1000, 1000), np.int64)
        for vals in curves:
            bins = np.clip((vals * 1000).astype(int), 0, 999)
            along = np.clip(((x + vals) / 2 * 1000).astype(int), 0, 999)
            above = np.clip(((vals - x) * 1000).astype(int), 0, 999)
            np.add.at(expected[0], (positions, bins), 1)
            np.add.at(expected[1], (along, above), 1)
        metrics.add_concentration(*actual, iter(curves))
        for raster, reference in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(raster, reference)
            self.assertEqual(raster.sum(), len(curves) * 1000)

    def test_frequency_batches_preserve_exact_group_sums_maxima_and_histograms(self) -> None:
        # Failure cases: native grouping ignores one size field, counts identities
        # as occurrences, wraps sums, includes zeros in frequency histograms, or
        # lets one huge row corrupt its normal neighbours. Real tactic fixtures
        # cannot practically produce uint64-boundary invocation counts. Exercise
        # aggregation -> archive -> aggregation against Python-integer totals.
        rows = [
            FreqRow(bytes((idx % 256,)) * 32, idx % 5 + 1, idx % 7 + 1, idx % 3, idx % 4, idx % 11)
            for idx in range(700)
        ]
        rows.extend((FreqRow(b"u" * 32, 2, 5, 2**64 - 1, 3, 4), FreqRow(b"v" * 32, 2, 5, 9, 1, 0)))
        huge = 1 << 15000
        rows.append(FreqRow(b"h" * 32, 1, huge, 1, huge, huge))
        expected = FreqTotals()
        for row in rows:
            key = row.distinct, row.expanded
            values = expected.pairs.setdefault(key, [0] * 6)
            for col, val in enumerate((row.top, row.root_dag, row.expanded_tree)):
                if val:
                    values[2 * col] += val
                    values[2 * col + 1] = max(values[2 * col + 1], val)
                    expected.hgrams[col][val] += 1
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "freqs.zst"
            with stats_writer(path) as write:
                write(("freqs", rows))
            decoded = next(read_stats(path))[1]
            for batches in ((decoded,), (decoded[:300], decoded[300:700], decoded[700:])):
                counts = FreqTotals()
                for batch in batches:
                    add_freq_totals(counts, batch)
                self.assertEqual(counts, expected)

    def test_prepared_figures_are_independent_and_style_scoped(self) -> None:
        # Fresh-process public builder checks catch hidden computational imports,
        # caller rc mutation, missing labels, leaked figures, and empty-constructor
        # failures. Ordinary in-process pipeline tests preload graph machinery
        # and cannot establish the import boundary; no image-byte comparison.
        program = """
import sys
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from trustmebro.visualization import state_render
from trustmebro.visualization.archives import stats_writer
from trustmebro.visualization.drawing import save_single
matplotlib.use("Agg")

for name in ("metrics", "processing", "scan", "atlas"):
    assert "trustmebro.visualization." + name not in sys.modules, name
assert "graph_tool" not in sys.modules
out = Path(sys.argv[1])
with matplotlib.rc_context({"axes.facecolor": "pink", "image.cmap": "viridis"}):
    before = matplotlib.rcParams.copy()
    fig = state_render.concentration_figure(np.array([[1, 2], [3, 4]]), 7)
    assert fig.axes[0].get_xlabel() == "Fraction of hypotheses (largest first)"
    assert fig.axes[0].get_title().endswith("7 empty contexts excluded")
    assert fig.axes[0].get_facecolor() == (0, 0, 0, 1)
    assert fig.axes[0].images[0].get_cmap().name == "turbo"
    save_single(fig, out / "prepared.png")
    assert dict(matplotlib.rcParams) == dict(before)
    assert not plt.get_fignums()
with stats_writer(out / "empty.zst") as write:
    write(("stats", {"states": 0, "theorems": 0, "expr_idents": 0, "pcentiles_expanded": {}},
           {"pairs": [], "coverage": {}}))
state_render.render(out / "empty.zst", out, families={"constructors"})
assert not (out / "constructors-state.png").exists()
assert not plt.get_fignums()
"""
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run([sys.executable, "-c", program, tmp], check=True, capture_output=True, timeout=60)
            self.assertTrue((Path(tmp) / "prepared.png").exists())

    def test_shared_parallel_pass_and_spills_preserve_exact_counts(self) -> None:
        # End-to-end failures covered: skipped views/radii, head-ID remapping,
        # lost spill batches, duplicate counts, extra source passes, and leftovers.
        # Timing must reach subprocesses, keep separate writers, preserve exact
        # results, and log coordinator spill costs independently of worker costs.
        # Coordinator checkpoints must cover indexing/native updates/publication,
        # keep process and thread CPU distinct, and never alter archive counts.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "source.db"
            exprs = (r.Const("f", ()), r.Const("A", ()), r.App(0, (1,)), r.App(2, (2,)))
            trns = tuple(r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(i, (), (), ())) for i in (2, 3))
            with closing(open_extraction_db(src)) as db:
                for name in ("first", "second", "third"):
                    named_exprs = (r.Const(name, ()), *exprs[1:])
                    db.execute(
                        "INSERT INTO theorems (name,module,expr_count,trn_count,exprs,trns) VALUES (?, 'Test', 4, 2, ?, ?)",
                        (name, encode_exprs(named_exprs), encode_trns(trns)),
                    )
                db.commit()
            original = src.read_bytes()
            env = dict(os.environ, NUMBA_CACHE_DIR=str(root / "numba"))
            for mode, workers, budget, analyses in (
                ("separate", 1, 512, ("patterns",)),
                ("shared", 2, 1, ("metrics", "topology", "patterns")),
            ):
                command = [
                    str(Path(sys.executable).parent / "graphs"),
                    "--skip-embedding",
                    "--db",
                    str(src),
                    "--stats",
                    str(root / mode),
                    "--workers",
                    str(workers),
                    "--aggregation-memory-mib",
                    str(budget),
                    "--skip-embedding",
                    "--pattern-depths",
                    "1",
                    "2",
                    "--graphs",
                    *(("local-patterns",) if mode == "separate" else ("local-patterns", "complexity", "heads")),
                    "--output",
                    str(root / (mode + "-plots")),
                ]
                if mode == "shared":
                    command.extend(("--timing-dir", str(root / "timings")))
                result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=120, check=False)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(result.stdout)["stages"]["analysis"]["shared"]
                self.assertEqual(report["source_passes"], 2 if mode == "shared" else 1)
                self.assertEqual(report["theorems"], 3)
                self.assertEqual(bool(report["pattern_spills"]), mode == "shared")
                self.assertFalse(list((root / mode).glob("pattern-counts-*")))
                self.assertFalse(list((root / mode).glob("*.part")))

            def timing_rows(path: Path) -> list[dict[str, str]]:
                with path.open(newline="") as stream:
                    return list(csv.DictReader(stream))

            (coordinator,) = (root / "timings").glob("coordinator-*.csv")
            parent_rows = timing_rows(coordinator)
            worker_paths = list((root / "timings").glob("worker-*.csv"))
            self.assertTrue(worker_paths)
            # Three jobs and two workers require at least one persistent worker
            # to consume more than one theorem, not a fresh process per job.
            self.assertGreaterEqual(
                max(sum(row["phase"] == "worker_total" for row in timing_rows(path)) for path in worker_paths), 2
            )
            worker_rows = [row for path in worker_paths for row in timing_rows(path)]
            self.assertEqual(
                {row["label"] for row in worker_rows if row["phase"] == "worker_total"}, {"first", "second", "third"}
            )
            self.assertEqual(sum(row["phase"] == "worker_total" for row in worker_rows), 3)
            phases = Counter(row["phase"] for row in worker_rows)
            for mode in PATTERN_MODES:
                self.assertEqual(phases[f"patterns.{mode}"], 3)
                self.assertEqual(phases[f"patterns.{mode}.signatures"], 3)
                # All radii/flavours/cohorts now share a single native count-and-
                # pack operation per theorem/view, not separate timed loops.
                self.assertEqual(phases[f"patterns.{mode}.counts_pack"], 3)
            for phase in (
                "decode_exprs",
                "decode_trns",
                "build_graph",
                "prepare_views",
                "signature_reuse",
                "measure_theorem",
                "measure_topo",
                "pattern_observations",
                "topo.prepare_measurements",
                "topo.shape_counts",
                "topo.sizes",
                "topo.heads",
            ):
                self.assertIn(phase, {row["phase"] for row in worker_rows})
            for phase in ("wait_results", "add_metrics", "add_topo", "write_metrics", "reuse_pass"):
                self.assertIn(phase, {row["phase"] for row in parent_rows})
            self.assertTrue(any(row["phase"].startswith("spill_patterns.") for row in parent_rows))
            for phase in (
                "pattern.index",
                "pattern.buffer_update",
                "pattern.sort",
                "pattern.gather",
                "pattern.produce_batch",
                "pattern.remap",
                "pattern.cols",
                "pattern.concat_cols",
                "pattern.statistics",
                "archive.encode",
                "archive.compress_write",
                "archive.read",
                "archive.decode_validate",
            ):
                self.assertIn(phase, {row["phase"] for row in parent_rows})
            for row in (*parent_rows, *worker_rows):
                self.assertGreater(int(row["start_ns"]), 0)
                self.assertGreaterEqual(int(row["wall_ns"]), 0)
                self.assertGreaterEqual(int(row["cpu_ns"]), 0)
                self.assertGreaterEqual(int(row["thread_ns"]), 0)
                self.assertEqual(row["ok"], "1")

            def counts(path: Path) -> dict[tuple[bytes, int, str], tuple[int, ...]]:
                return {
                    (digest, flavour, head): weights
                    for batch in pattern_batches(path)
                    for digest, flavour, head, weights in batch
                }

            for mode in PATTERN_MODES:
                for depth in (1, 2):
                    expected = counts(AnalysisPaths(root / "separate").patterns(mode, depth))
                    self.assertTrue(expected)
                    self.assertEqual(counts(AnalysisPaths(root / "shared").patterns(mode, depth)), expected)
            # Repeated in-process runs must not depend on another run's worker
            # configuration, selected metrics, or sampling radii.
            analyze(src, root / "serial-topology", analyses=("topology",), workers=1)
            with patch.object(scan, "PATTERN_MERGE_FAN_IN", 2):
                analyze(src, root / "serial-patterns", analyses=("patterns",), depths=(1, 2), workers=1, aggr_mem=1)
            for mode in PATTERN_MODES:
                for depth in (1, 2):
                    self.assertEqual(
                        counts(AnalysisPaths(root / "serial-patterns").patterns(mode, depth)),
                        counts(AnalysisPaths(root / "separate").patterns(mode, depth)),
                    )
            # Reverse insertion order changes interned head IDs and merge order,
            # but must not change canonical patterns or observation weights.
            reversed_src = root / "reversed.db"
            with closing(open_extraction_db(reversed_src)) as db:
                for name in ("third", "second", "first"):
                    named_exprs = (r.Const(name, ()), *exprs[1:])
                    db.execute(
                        "INSERT INTO theorems(name,module,expr_count,trn_count,exprs,trns) VALUES(?,'Test',4,2,?,?)",
                        (name, encode_exprs(named_exprs), encode_trns(trns)),
                    )
                db.commit()
            analyze(reversed_src, root / "reversed", analyses=("patterns",), depths=(1, 2), workers=1, aggr_mem=1)
            for mode in PATTERN_MODES:
                for depth in (1, 2):
                    self.assertEqual(
                        counts(AnalysisPaths(root / "reversed").patterns(mode, depth)),
                        counts(AnalysisPaths(root / "separate").patterns(mode, depth)),
                    )
            self.assertEqual(src.read_bytes(), original)

    def test_packed_pattern_spills_preserve_keys_weights_and_statistics(self) -> None:
        # Boundary failures: buffer growth loses/reorders weights; fixed-width
        # digests drop trailing zero bytes; little-endian head IDs above 255 sort
        # incorrectly; spill-local IDs collide; unique keys are double-counted or
        # repeated keys across runs are lost; large counts pass through floats.
        # Pending-batch failures: count/row-triggered drains or the final tail
        # lose data; oversized unique inputs are double-reduced; native grouped
        # cohorts change positive populations, totals, or entropy normalization.
        # Tiny buffers/batches and forced compaction exercise aggregation -> archive
        # -> statistics against independent Python integer totals. Graph identity
        # construction itself is covered by the database/golden fixture above.
        names = ["", *(f"head{idx:03}" for idx in range(270))]
        inputs = [
            [
                (
                    ((idx + turn) % 3).to_bytes(32, "little"),
                    flavour,
                    head,
                    (2**54 + idx, idx + 1, turn + flavour, idx % 5),
                )
                for idx, head in (reversed(list(enumerate(names))) if turn % 2 == 0 else enumerate(names))
                for flavour in (0, 1)
            ]
            for turn in range(4)
        ]
        expected: dict[tuple[bytes, int, str], list[int]] = {}
        for batch in inputs:
            for digest, flavour, head, weights in batch:
                totals = expected.setdefault((digest, flavour, head), [0, 0, 0, 0])
                for idx, weight in enumerate(weights):
                    totals[idx] += weight
        expected_rows = {key: tuple(weights) for key, weights in expected.items()}
        key = ViewMode.ORIGINAL, 1
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.object(pattern_metrics, "PATTERN_CAP_ROWS", 7),
            patch.object(pattern_metrics, "PATTERN_BATCH_ROWS", 17),
            patch.object(scan, "PATTERN_MERGE_FAN_IN", 2),
        ):
            baseline = None
            for scenario, budget, batch_rows, batch_count in (
                ("memory", 512, 10_000, 64),
                ("batch-bounded", 512, 10_000, 2),
                ("row-bounded", 512, 1_600, 64),
                ("oversized", 512, 500, 64),
                ("immediate", 512, 10_000, 1),
                ("spill", 0, 10_000, 2),
                ("mixed", 0, 10_000, 2),
                ("empty", 512, 10_000, 2),
            ):
                with (
                    self.subTest(scenario=scenario),
                    patch.object(pattern_metrics, "PATTERN_BUF_ROWS", batch_rows),
                    patch.object(pattern_metrics, "PATTERN_BUF_BATCHES", batch_count),
                ):
                    dir = Path(tmp) / scenario
                    dir.mkdir()
                    counts = scan.PatternBatches(dir, budget)
                    if scenario != "empty":
                        for idx, batch in enumerate(inputs):
                            if scenario == "mixed" and idx == len(inputs) - 1:
                                counts.budget = 512 * 1024**2
                            counts.add({key: pack_patterns(batch)})
                    path = dir / "final.msgpack.zst"
                    for _, _, heads, rows in counts.finish((key,)):
                        archives.write_pattern_batches(path, heads, rows, {}, with_cols=True)
                    # Multiple compaction rounds must not multiply decoded frame
                    # sizes by fan-in. This checks the real persisted run frames.
                    self.assertTrue(all(len(batch) <= 17 for _, batch in archives.packed_pattern_batches(path)))
                    actual = {
                        (digest, flavour, head): weights
                        for batch in pattern_batches(path)
                        for digest, flavour, head, weights in batch
                    }
                    self.assertEqual(actual, {} if scenario == "empty" else expected_rows)
                    heads, cols = read_pattern_cols(path)
                    stats = pattern_metrics.pattern_stats(heads, cols)
                    if scenario == "memory":
                        baseline = stats
                    elif scenario != "empty":
                        self.assertEqual(stats, baseline)
                    if scenario != "empty":
                        for label, scope, cohort, head, patterns, occs, entropy in stats.head_rows:
                            flavour = 0 if label == "topology" else 1
                            col = {
                                ("top", "all"): 0,
                                ("all", "all"): 1,
                                ("top", "affected"): 2,
                                ("all", "affected"): 3,
                            }[scope, cohort]
                            vals = [
                                weights[col]
                                for (_, kind, name), weights in expected_rows.items()
                                if kind == flavour and name == head and weights[col] > 0
                            ]
                            total = sum(vals)
                            reference = -sum((val / total) * math.log2(val / total) for val in vals) if total else 0
                            self.assertEqual((patterns, occs), (len(vals), total))
                            self.assertAlmostEqual(entropy, reference, places=12)
                    self.assertEqual(list(dir.iterdir()), [path])

    def test_pattern_buffers_preserve_exceptional_counts_through_spills_and_statistics(self) -> None:
        # Failure cases: uint64 addition wraps in memory or during native run
        # reduction; object weights lose precision in transport/archive frames;
        # statistics sum/cumulative counts overflow or huge probabilities become
        # inf/NaN; head-only summaries lose exact totals or mistake occurrence
        # weights for the count of positive patterns after sharing preparation.
        # A real graph fixture cannot practically invoke 2**64 tactics,
        # so exercise the aggregate -> spill -> archive -> statistics boundary.
        key = ViewMode.ORIGINAL, 1
        digest = b"\0" * 32
        maximum = 2**64 - 1
        huge = 1 << 15000
        inputs = [
            [(digest, 0, "", (maximum, maximum, 0, maximum))],
            [(digest, 0, "", (2, 2, 0, 2))],
            [(b"x" * 32, 0, "", (huge, huge, 0, huge))],
            [(b"h" * 32, 0, "head", (huge, huge, 0, huge)), (b"l" * 32, 0, "head", (1, 1, 0, 1))],
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for budget in (0, 512):
                with self.subTest(budget=budget):
                    root = Path(tmp) / str(budget)
                    root.mkdir()
                    counts = scan.PatternBatches(root, budget)
                    for rows in inputs:
                        counts.add({key: pack_patterns(rows)})
                    path = root / "counts.msgpack.zst"
                    for _, _, heads, batches in counts.finish((key,)):
                        archives.write_pattern_batches(path, heads, batches, {})
                    observed = {row[0]: row[3] for batch in pattern_batches(path) for row in batch}
                    self.assertEqual(observed[digest], (maximum + 2, maximum + 2, 0, maximum + 2))
                    self.assertEqual(observed[b"x" * 32], (huge, huge, 0, huge))
                    heads, cols = read_pattern_cols(path)
                    stats = patterns.pattern_stats(heads, cols)
                    stat = stats.descrs["topology:all:all"]
                    self.assertEqual(stat["occs"], huge + maximum + 2)
                    self.assertEqual(stat["coverage"][-1], 1.0)
                    self.assertTrue(math.isfinite(stat["entropy_bits"]))
                    head_rows = [row for row in stats.head_rows if row[:4] == ("topology", "all", "affected", "head")]
                    self.assertEqual(len(head_rows), 1)
                    self.assertEqual(head_rows[0][4:6], (2, huge + 1))
                    self.assertTrue(math.isfinite(head_rows[0][6]))
                    self.assertEqual(stats.points[0], [(huge + 1, 2)])
            # Distinct keys can each fit uint64 while their head's grouped total
            # exceeds it. This must exercise the grouped-sum fallback even when
            # the persisted input matrix itself contains only native counts.
            root = Path(tmp) / "native-head"
            root.mkdir()
            counts = scan.PatternBatches(root, 512)
            weight = 2**63 + 1
            counts.add(
                {
                    key: pack_patterns(
                        [(digest, 0, "head", (weight, weight, 0, weight)) for digest in (b"a" * 32, b"b" * 32)]
                    )
                }
            )
            path = root / "counts.msgpack.zst"
            for _, _, heads, batches in counts.finish((key,)):
                archives.write_pattern_batches(path, heads, batches, {})
            heads, cols = read_pattern_cols(path)
            self.assertEqual(cols.dtype.name, "uint64")
            stats = patterns.pattern_stats(heads, cols)
            row = next(row for row in stats.head_rows if row[:4] == ("topology", "all", "affected", "head"))
            self.assertEqual(row[4:], (2, 2 * weight, 1.0))
            self.assertEqual(stats.points[0], [(2 * weight, 2)])
            # Repeated keys in separate native pending batches overflow only
            # when coalesced: no huge/object input may mask that guard.
            root = Path(tmp) / "native-repeat"
            root.mkdir()
            counts = scan.PatternBatches(root, 512)
            for _ in range(2):
                counts.add({key: pack_patterns([(digest, 0, "head", (weight, weight, 0, weight))])})
            path = root / "counts.msgpack.zst"
            for _, _, heads, batches in counts.finish((key,)):
                archives.write_pattern_batches(path, heads, batches, {})
            observed = [row for batch in pattern_batches(path) for row in batch]
            self.assertEqual(observed, [(digest, 0, "head", (2 * weight, 2 * weight, 0, 2 * weight))])
            heads, cols = read_pattern_cols(path)
            stats = patterns.pattern_stats(heads, cols)
            row = next(row for row in stats.head_rows if row[:4] == ("topology", "all", "affected", "head"))
            self.assertEqual(row[4:], (1, 2 * weight, -0.0))
            self.assertEqual(stats.points[0], [(2 * weight, 1)])

    def test_worker_pattern_batches_preserve_head_ids_aliases_and_exact_counts(self) -> None:
        # Failure cases: combined radius/flavour keys collide; trailing digest
        # zeros or head IDs above 255 are truncated; resolved aliases drop raw
        # weights; native repeated sums wrap; object counts lose precision;
        # empty observations enter grouping. These counts cannot practically
        # arise in a theorem fixture, so check worker -> aggregate -> archive.
        import numpy as np

        from trustmebro.visualization.measurements import PatternObs

        exprs = (r.Const("f", ()), r.Const("X", ()), r.App(0, (1,)), r.Metadata((), 2))
        graph = graph_ops.build_graph(exprs, edges=[(), (), (0, 1), ()])
        view = views.TopoView(graph, ((),) * 4, (0, 1, 2, 2))
        heads = ("", *(f"head-{idx}" for idx in range(1, 301)))
        roots = np.array((2, 3, 2), dtype=np.int64)
        head_ids = np.array((300, 300, 0), dtype=np.uint32)
        # Fixed full-width digests exercise zero suffixes and identical keys
        # across radii; actual identity construction has its own golden fixture.
        sigs = dict.fromkeys((3, 1), (b"\0" * 32, b"l" + b"\0" * 31))
        with tempfile.TemporaryDirectory() as tmp:
            for count in (2**63 + 1, 1 << 15000):
                weights = exact_weights([(count, count, 0, count)] * 3)
                obs = PatternObs(roots, weights, heads, head_ids)
                with patch.object(pattern_metrics, "extract_patterns", return_value=sigs):
                    batches, _ = patterns.measure_patterns(view, (3, 1), obs, mode=ViewMode.ERASED)
                for rad, batch in batches.items():
                    expected: Counter[tuple[bytes, int, str]] = Counter()
                    for flavour, digest in enumerate(sigs[rad]):
                        expected[digest, flavour, ""] = 3 * count
                        expected[digest, flavour, heads[300]] = 2 * count
                    path = Path(tmp) / f"counts-{rad}.msgpack.zst"
                    counts = patterns.PatternCounts()
                    counts.add(batch)
                    archives.write_pattern_batches(path, tuple(counts.heads), counts.batches(), {})
                    observed = [row for rows in pattern_batches(path) for row in rows]
                    self.assertEqual(
                        sorted(observed),
                        sorted(
                            (digest, flavour, head, (val, val, 0, val))
                            for (digest, flavour, head), val in expected.items()
                        ),
                    )
            empty = PatternObs(roots[:0], np.empty((0, 4), np.uint64), heads, head_ids[:0])
            batches, sigs = patterns.measure_patterns(view, (3, 1), empty, mode=ViewMode.ERASED)
            self.assertEqual(sigs, {})
            self.assertTrue(all(len(batch) == 0 for batch in batches.values()))

    def test_pattern_archive_rejects_invalid_buffers(self) -> None:
        # Failure cases: truncated buffer lengths silently reshape; unknown head
        # IDs/flavours reach renderers; negative exceptional counts pass through.
        # Valid-source end-to-end tests cannot manufacture these corrupt archives.
        good = pack_patterns([(b"\0" * 32, 0, "", (1, 2, 0, 0))])
        keys = good.keys.tobytes()
        weights = good.weights.astype("<u8", copy=False).tobytes()
        invalid = (
            (keys[:-1], weights),
            (keys, weights[:-1]),
            (pattern_key(b"\0" * 32, 2, 0), weights),
            (pattern_key(b"\0" * 32, 0, 1), weights),
            (keys, [(-1, 2, 0, 0)]),
            (keys, []),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "corrupt.msgpack.zst"
            for data in invalid:
                with self.subTest(data=data), archives.writer(path) as write:
                    write({"heads": [""]})
                    write(data)
                with self.assertRaises(ValueError):
                    list(archives.packed_pattern_batches(path))

    def test_analysis_and_rendering_have_separate_inputs_and_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, stats, plots = root / "source.db", root / "stats", root / "plots"
            exprs = (r.Const("A", ()), r.Const("B", ()), r.App(0, (1,)), r.App(2, (2,)))
            trns = tuple(r.Trn(r.Tactic("test", "test"), (0, 1), 1, r.ProofState(dst, (), (), ())) for dst in (2, 3))
            with closing(open_extraction_db(src)) as db:
                db.execute(
                    "INSERT INTO theorems (name,module,expr_count,trn_count,exprs,trns) VALUES (?, ?, ?, ?, ?, ?)",
                    ("fixture", "Test", len(exprs), len(trns), encode_exprs(exprs), encode_trns(trns)),
                )
                db.commit()
            original = src.read_bytes()
            env = dict(os.environ, MPLCONFIGDIR=str(root / "mpl"), NUMBA_CACHE_DIR=str(root / "numba"))

            def run(command: str, *args: str, success: bool = True) -> subprocess.CompletedProcess[str]:
                result = subprocess.run(
                    [str(Path(sys.executable).parent / "graphs"), "--stats", str(stats), *args],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=180,
                    check=False,
                )
                if success:
                    self.assertEqual(result.returncode, 0, result.stderr)
                else:
                    self.assertNotEqual(result.returncode, 0)
                return result

            run(
                "graphs",
                "--db",
                str(src),
                "--workers",
                "2",
                "--pattern-depths",
                "1",
                "--atlas-min-nodes",
                "2",
                "--skip-embedding",
                "--graphs",
                "heads",
                "--output",
                str(plots),
            )
            self.assertTrue(plots.exists())
            self.assertFalse(list(stats.glob("*.png")))
            self.assertFalse(list(stats.glob("*.db")))
            self.assertFalse(list(stats.glob("*.part")))
            archives = {
                str(p.relative_to(stats)): hashlib.sha256(p.read_bytes()).digest()
                for p in stats.rglob("*")
                if p.is_file()
            }
            # Every renderer, including the topology atlas, is archive-only.
            src.rename(root / "unavailable.db")
            result = run("graphs", "--output", str(plots), "--atlas-min-nodes", "2", "--graphs", "heads")
            self.assertEqual(set(json.loads(result.stdout)["stages"]["rendering"]), {"heads"})
            for filename in ("plumbing-common-heads.png",):
                self.assertTrue((plots / filename).read_bytes().startswith(b"\x89PNG"))
            self.assertEqual(
                archives,
                {
                    str(p.relative_to(stats)): hashlib.sha256(p.read_bytes()).digest()
                    for p in stats.rglob("*")
                    if p.is_file()
                },
            )
            self.assertEqual((root / "unavailable.db").read_bytes(), original)

            # Non-atlas redraws must work without access to the SQLite corpus.
            subset = root / "subset"
            run("graphs", "--output", str(subset), "--graphs", "heads")
            self.assertEqual(
                {p.name for p in subset.glob("*.png")}, {"plumbing-common-heads.png", "plumbing-head-vocabulary.png"}
            )
            run("graphs", "--db", str(root / "absent.db"), "--graphs", "heads", "--replace", success=False)
            self.assertEqual(
                archives,
                {
                    str(p.relative_to(stats)): hashlib.sha256(p.read_bytes()).digest()
                    for p in stats.rglob("*")
                    if p.is_file()
                },
            )


if __name__ == "__main__":
    unittest.main()
