"""Candidate archive -> supervised cover -> unchanged positional feature counts."""

from __future__ import annotations

import json
import tempfile
import unittest
from compression import zstd
from contextlib import redirect_stderr
from dataclasses import replace
from io import StringIO
from itertools import batched
from pathlib import Path
from unittest.mock import patch

import msgspec
import numpy as np
from scipy.sparse import csr_array, vstack
from test_candidates import candidate_records
from test_representation import feature_rows

import trustmebro.preprocessing.layout as layouts
from trustmebro.artifacts import encode_msgpack
from trustmebro.extraction import records as r
from trustmebro.preprocessing import archives, coverage, features, pipeline, records, supervised
from trustmebro.preprocessing import candidates as c
from trustmebro.preprocessing import scan as scheduler
from trustmebro.preprocessing.records import LabelPolicy

DEPTHS = (1, 2, 3)
LABELS = LabelPolicy({"a": "A", "b": "B"}, "drop")


def fixture(idx: int) -> records.Cands:
    # Distinct internal arity, identical names, repeated hyp roots and depth
    # matches, a leaf-only hypothesis and an oversized literal retained exactly.
    exprs = (
        r.Const("f", ()),
        r.Fvar(0),
        r.App(0, (1,)),
        r.App(0, (1, 1)),
        r.App(0, (2,)),
        r.App(0, (3,)),
        r.NatLiteral(1 << 100),
    )
    kind = "a" if idx % 2 == 0 else "b"
    goal, hyp = (4, 3) if kind == "a" else (5, 2)
    hyps = tuple(
        r.LocalConst(local, ref, r.LocalDeclKind.DEFAULT, False, r.BinderInfo.DEFAULT)
        for local, ref in enumerate((hyp, hyp, 6))
    )
    state = r.ProofState(goal, hyps, (), ())
    trns = tuple(
        r.Trn(r.Tactic(tactic, "same source spelling"), (step, step + 1), 1, state)
        for step, tactic in enumerate((kind, kind, "unmapped"))
    )
    return c.extract_cands(r.Theorem(f"Fixture.{idx}", "Fixture", None, exprs, trns), DEPTHS)


def write_candidates(path: Path, theorems: tuple[records.Cands, ...]) -> records.Selection:
    selection = records.Selection("exploratory-full.db", 42, 123, DEPTHS, None, None)
    with zstd.open(path, "wb") as stream:
        for record in (selection, *theorems):
            archives.write_frame(stream, encode_msgpack(record))
    return selection


def serial_batches(source):
    """Test-side batch owner; algorithms receive only the one execution boundary."""

    def run(fn, phase):
        rows = iter(source())
        try:
            yield from (fn(batch) for batch in batched(rows, 16))
        finally:
            if close := getattr(rows, "close", None):
                close()

    return run


class SupervisedTests(unittest.TestCase):
    def test_parallel_batched_fit_matches_serial_vocabulary_and_features(self) -> None:
        # Failures: batch seams or completion order change support/weights,
        # roles collapse, worker decoding loses oversized naturals, held-out
        # names enter screening, or the CLI worker setting misses a selection
        # pass. Establishes public archive -> fit -> conversion equivalence
        # with real spawned workers, not corpus-scale speed or memory bounds.
        theorems = tuple(fixture(idx) for idx in range(18))
        theorems = theorems[:16] + tuple(
            msgspec.structs.replace(
                theorem,
                nodes=tuple(
                    msgspec.structs.replace(node, expr=r.Const("heldout.only", ()))
                    if isinstance(node.expr, r.Const)
                    else node
                    for node in theorem.nodes
                ),
            )
            for theorem in theorems[16:]
        )
        selected = tuple(theorem.name for theorem in theorems[:16])
        cfg = pipeline.VocabBuildCfg(5000, depths=DEPTHS, hyp_slots=2, name_share=0.25, min_support=2)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, _, labels = (root / name for name in ("candidates.zst", "coverage.zst", "labels.json"))
            write_candidates(src, theorems)
            labels.write_bytes(msgspec.json.encode(LABELS))
            before = src.read_bytes()
            results: list[records.Vocab] = []
            progress = StringIO()
            with patch.object(scheduler, "SELECTION_BATCH_THEOREMS", 2), redirect_stderr(progress):
                for workers in (1, 2):
                    dst = root / f"vocab-{workers}.zst"
                    pipeline.fit_vocab(src, labels, dst, replace(cfg, workers=workers), theorems=selected, replace=True)
                    results.append(archives.read_vocab(dst))
            self.assertEqual(msgspec.to_builtins(results[0]), msgspec.to_builtins(results[1]))
            info = results[1].supervision
            assert info is not None
            self.assertEqual((info.counts, info.labeled_theorems), ((16, 16), 16))
            assert results[1].representation is not None
            self.assertTrue(results[1].representation.names)
            self.assertNotIn("heldout.only", {name for _, _, name in results[1].representation.names})
            self.assertNotIn("heldout.only", results[1].representation.heads)
            self.assertEqual(results[1].representation.selection.theorems, selected)
            for phase in (
                "Collect count-label associations",
                "Score count-label channels",
                "Collect count redundancy",
                "Select shapes within complete-column budget",
                "Screen constant names",
                "Collect selected name-position associations",
            ):
                self.assertIn(f"{phase} completed", progress.getvalue())
            self.assertIn("16/16 theorems", progress.getvalue())
            self.assertIn("2 workers", progress.getvalue())
            compiled = tuple(layouts.compile_vocab(vocab) for vocab in results)
            for theorem in theorems:
                matrices = tuple(features.encode_cands(theorem, layout).matrix for layout in compiled)
                np.testing.assert_array_equal(matrices[0].toarray(), matrices[1].toarray())
            self.assertEqual(src.read_bytes(), before)

    def test_reordered_batches_preserve_profiles_and_reject_scope_corruption(self) -> None:
        # Failures: the coordinator assumes submission order, counts repeated
        # transitions as theorem support, or misses duplicate/omitted theorems
        # and altered root sets after receiving reduced results. This narrow
        # boundary deliberately corrupts worker output; an ordinary successful
        # E2E run cannot induce these IPC/population failures reliably.
        theorems = tuple(fixture(idx) for idx in range(6))
        index = coverage.build_coverage_idx(theorems, DEPTHS)
        cfg = records.SupervisedPolicy(min_support=2)
        cols = coverage.eligible_shapes(index, min_nodes=2, min_support=2)
        space = supervised._count_space(index, cols, DEPTHS)
        oracle = supervised.gather_count_profiles(serial_batches(lambda: iter(theorems)), index, LABELS, cfg, space)

        def reordered(fn, phase):
            yield from (fn(theorems[idx : idx + 2]) for idx in (4, 2, 0))

        actual = supervised.gather_count_profiles(reordered, index, LABELS, cfg, space)
        np.testing.assert_array_equal(actual.totals, oracle.totals)
        np.testing.assert_array_equal(actual.support, oracle.support)
        np.testing.assert_array_equal(actual.channels, oracle.channels)
        np.testing.assert_allclose(actual.scores, oracle.scores, atol=1e-12)
        batches = [supervised.count_batch(theorems[idx : idx + 2], space, LABELS) for idx in (4, 2, 0)]
        for broken in (
            batches[:-1],
            batches + batches[:1],
            [replace(batches[0], scope=((theorems[4].name, (-1,)),)), *batches[1:]],
        ):
            with self.assertRaisesRegex(ValueError, "candidate and coverage"):
                list(supervised._checked_batches(broken, index))

    def test_sparse_moment_reduction_preserves_batches_and_enforces_guard(self) -> None:
        # Failures: pairwise merges drop a level, remapped/duplicate columns lose
        # sums, a final short batch is omitted, or retained growth bypasses the
        # guard. Tiny E2E fixtures do not naturally reach merge/memory boundaries;
        # this narrow arithmetic test forces them without a corpus-size workload.
        # It establishes native reduction results, not total process memory.
        cfg = records.SupervisedPolicy(memory_mib=1)
        moments = supervised._MomentSum(4, 20, cfg, 0)
        moments.batch_bytes = 100
        expected = np.zeros((4, 20))
        for idx in range(19):
            values = np.asarray([[1, idx + 1, 2], [3, 0, 1], [0, 1, idx], [1, 1, 1]], dtype=float)
            cols = np.asarray([2, idx % 20, 2])
            moments.add(csr_array(values), cols)
            for col, dst in enumerate(cols):
                expected[:, dst] += values[:, col]
        np.testing.assert_array_equal(moments.finish().toarray(), expected)
        guarded = supervised._MomentSum(4, 20, cfg, 2**20 - 1)
        guarded.batch_bytes = 1
        with self.assertRaisesRegex(MemoryError, "score/pair buffers"):
            guarded.add(csr_array([[1.0], [0], [0], [0]]), np.asarray([0]))

    def test_archive_selection_balances_decorated_signals_not_shape_presence(self) -> None:
        # Failures: shape-presence shortlisting discards universally present but
        # informative kinds; final selection buys redundant A-only entries rather
        # than serving B; negative associations with C are ignored; depth matches
        # duplicate channels; or an absent label forces meaningless allocations.
        # Establishes archive -> count profiles -> budgeted balanced selection.
        # The controlled empty backbone isolates enrichment; the next test covers
        # actual full-cover preservation and conversion. No predictive claim.
        policy = LabelPolicy({"a": "A", "b": "B", "c": "C", "never": "absent"}, "drop")
        theorems: list[records.Cands] = []
        for idx in range(18):
            kind = ("a", "b", "c")[idx % 3]
            exprs = (
                r.Const("f", ()),
                r.Const("x", ()),
                r.Fvar(0),
                r.App(2 if kind == "a" else 0, (1,)),
                r.App(2 if kind == "a" else 0, (1, 1)),
                r.App(2 if kind == "b" else 0, (1, 1, 1)),
                r.App(0, (3, 4, 5)),
            )
            trn = r.Trn(r.Tactic(kind, kind), (0, 1), 0, r.ProofState(6, (), (), ()))
            theorems.append(c.extract_cands(r.Theorem(f"Balanced.{idx}", "Balanced", None, exprs, (trn,)), (1,)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.zst"
            write_candidates(path, tuple(theorems))
            index = coverage.build_coverage_idx(theorems, (1,))
            cfg = records.SupervisedPolicy(dims=2 * 2 * 3 * len(records.NODE_NAMES), min_support=3, redundancy=0)
            result = supervised.select_supervised(
                serial_batches(lambda: candidate_records(path)),
                index,
                np.empty(0, dtype=np.int64),
                policy,
                cfg,
                depths=(1,),
            )
            selected = [coverage.coverage_shape(index.shapes, int(col)) for col in result.cols]
            arities = {len(shape.edges[0]) - 1 for shape in selected if shape.edges[0] is not None}
            self.assertEqual(len(selected), 2)
            self.assertIn(3, arities)  # the B-specific entry, not both A duplicates
            self.assertTrue(arities.intersection((1, 2)))
            self.assertEqual(result.dims, cfg.dims)
            np.testing.assert_allclose(result.selected_assoc, (1, 1, 0.25, 0), atol=1e-12)
            self.assertEqual(result.label_strength[-1], 0)
            np.testing.assert_allclose(result.label_strength[:3], (1.25, 1.25, 2), atol=1e-12)

    def test_count_magnitude_and_theorem_support_affect_selection(self) -> None:
        # Failures: one vs two copies is binarized, duplicate source-depth matches
        # inflate counts, or many transitions in one theorem satisfy the channel
        # support minimum. Establishes public selection with real archived counts,
        # not the ability to learn a nonlinear count/label relationship.
        exprs = (r.Const("f", ()), r.Const("x", ()), r.App(0, (1,)))
        theorems: list[records.Cands] = []
        for idx in range(8):
            kind = "a" if idx % 2 == 0 else "b"
            hyps = tuple(
                r.LocalConst(local, 2, r.LocalDeclKind.DEFAULT, False, r.BinderInfo.DEFAULT)
                for local in range(1 if kind == "a" else 2)
            )
            state = r.ProofState(2, hyps, (), ())
            trns = tuple(r.Trn(r.Tactic(kind, kind), (step, step + 1), 0, state) for step in range(4))
            theorems.append(c.extract_cands(r.Theorem(f"Counts.{idx}", "Counts", None, exprs, trns), DEPTHS))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.zst"
            write_candidates(path, tuple(theorems))
            index = coverage.build_coverage_idx(theorems, DEPTHS)
            cfg = records.SupervisedPolicy(dims=1000, min_support=3, redundancy=0)
            result = supervised.select_supervised(
                serial_batches(lambda: candidate_records(path)),
                index,
                np.empty(0, dtype=np.int64),
                LABELS,
                cfg,
                depths=DEPTHS,
            )
            self.assertGreater(len(result.cols), 0)
            np.testing.assert_allclose(result.selected_assoc, (1, 1), atol=1e-12)
            strict = msgspec.structs.replace(cfg, min_support=9)
            empty = supervised.select_supervised(
                serial_batches(lambda: candidate_records(path)),
                index,
                np.empty(0, dtype=np.int64),
                LABELS,
                strict,
                depths=DEPTHS,
            )
            self.assertEqual(len(empty.cols), 0)

    def test_public_selection_preserves_cover_budget_policy_and_feature_values(self) -> None:
        # Failures: supervision removes coverage, exceeds the complete-dimension budget,
        # drops unknown rows during feature conversion, changes node-type counts,
        # re-mines graphs, loses label-policy/corpus provenance or alters sources.
        # Establishes the unified fit/archive/feature contract, not predictive usefulness,
        # leakage-safe experiment scope or corpus-scale memory/performance.
        theorems = tuple(fixture(idx) for idx in range(8))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, _, labels, dst = (
                root / name for name in ("candidates.zst", "coverage.zst", "labels.json", "vocab.zst")
            )
            selection = write_candidates(src, theorems)
            labels.write_bytes(msgspec.json.encode(LABELS))
            index = pipeline.prepare_coverage(src)
            baseline = coverage.select_coverage(index)
            immutable = {path: path.read_bytes() for path in (src, labels)}
            entries = tuple(
                records.Entry(shape.ident, shape.edges)
                for col in baseline.cols
                for shape in (coverage.coverage_shape(index.shapes, int(col)),)
            )
            width = layouts.fixed_dimensions(entries, 2)
            with patch.object(c, "prepare_adjacency", side_effect=AssertionError("graph rediscovery")):
                report = pipeline.fit_vocab(
                    src,
                    labels,
                    dst,
                    pipeline.VocabBuildCfg(width + 500, depths=DEPTHS, hyp_slots=2, name_share=0, min_support=1),
                    replace=True,
                )
            vocab = archives.read_vocab(dst)
            self.assertEqual(
                vocab.selection,
                msgspec.structs.replace(selection, theorems=tuple(theorem.name for theorem in theorems)),
            )
            info = vocab.supervision
            self.assertIsNotNone(info)
            assert info is not None
            self.assertEqual(json.loads(info.label_policy_json), msgspec.to_builtins(LABELS))
            self.assertEqual((info.labels, info.counts, info.labeled_theorems), (("A", "B"), (8, 8), 8))
            self.assertEqual(info.screening, records.SUPERVISED_SCREENING)
            np.testing.assert_allclose(info.available_assoc, info.selected_assoc)
            self.assertEqual(set(report["shape_associations"]), {"A", "B"})
            self.assertGreater(info.added_shapes, 0)
            self.assertLessEqual(info.added_dims, 500)
            self.assertEqual(info.candidates, str(src.resolve()))
            base_ids = {index.shapes.idents[col].tobytes() for col in baseline.cols}
            selected_ids = {entry.ident for entry in vocab.entries}
            self.assertTrue(base_ids.issubset(selected_ids))
            cols = [col for col, ident in enumerate(index.shapes.idents) if ident.tobytes() in selected_ids]
            coverage.require_coverage(index, cols)
            self.assertLessEqual(report["dimensions"], width + 500)
            self.assertEqual(report["baseline_dimensions"], width)
            out, stats = root / "features.zst", root / "stats.zst"
            with patch.object(features, "extract_cands", side_effect=AssertionError("graph rediscovery")):
                scheduler.convert_corpus(src, dst, out, stats, cands=True)
            actual = {rows.name: rows for rows in feature_rows(out)}
            layout = layouts.compile_vocab(vocab)
            for theorem in theorems:
                np.testing.assert_array_equal(
                    actual[theorem.name].matrix.toarray(), features.encode_cands(theorem, layout).matrix.toarray()
                )
                self.assertEqual(len(actual[theorem.name].steps), 3)
            self.assertEqual({path: path.read_bytes() for path in immutable}, immutable)
            # Zero enrichment budget must retain exactly the original cover.
            zero = root / "zero.zst"
            pipeline.fit_vocab(
                src, labels, zero, pipeline.VocabBuildCfg(width, depths=DEPTHS, hyp_slots=2, min_support=1)
            )
            self.assertEqual({entry.ident for entry in archives.read_vocab(zero).entries}, base_ids)

    def test_streamed_count_profiles_and_redundancy_match_real_feature_oracle(self) -> None:
        # Failures: count evidence becomes binary, repeats/depths change weights,
        # roles merge, absent/constant channels produce NaN, negative associations
        # disappear, support counts transitions instead of theorems, or redundancy
        # uses undecorated shape presence rather than representative count channels.
        # This narrow statistical boundary independently uses emitted features;
        # final selection alone cannot establish correct correlation/pair values.
        # It does not establish predictive improvement or corpus-scale resources.
        theorems = tuple(fixture(idx) for idx in range(8))
        index = coverage.build_coverage_idx(theorems, DEPTHS)
        cfg = records.SupervisedPolicy(min_support=3)
        cols = coverage.eligible_shapes(index, min_nodes=2, min_support=cfg.min_support)
        space = supervised._count_space(index, cols, DEPTHS)
        entries = tuple(
            records.Entry(shape.ident, shape.edges)
            for col in cols
            for shape in (coverage.coverage_shape(index.shapes, int(col)),)
        )
        layout = layouts.compile_vocab(records.Vocab(DEPTHS, entries))
        for policy in (
            LabelPolicy({"a": "A", "b": "B", "never": "absent"}, "drop"),
            LabelPolicy({"a": "A", "b": "B"}, "other", "C"),
        ):
            evidence = supervised.gather_count_profiles(
                serial_batches(lambda: iter(theorems)), index, policy, cfg, space
            )
            rows_per_theorem = 2 if policy.unmapped == "drop" else 3
            matrix = (
                vstack(
                    [features.encode_cands(theorem, layout).matrix[:rows_per_theorem] for theorem in theorems],
                    format="csr",
                )
                .toarray()
                .astype(float)
            )
            self.assertGreater(matrix.max(), 1)
            y = np.asarray([label for idx in range(8) for label in (idx % 2, idx % 2, 2)[:rows_per_theorem]])
            np.testing.assert_array_equal(evidence.totals, np.bincount(y, minlength=len(policy.labels)))
            support = np.zeros(matrix.shape[1], dtype=np.int64)
            for start in range(0, len(y), rows_per_theorem):
                support += np.any(matrix[start : start + rows_per_theorem] > 0, axis=0)
            centered = matrix - matrix.mean(axis=0)
            expected = np.zeros_like(evidence.scores)
            representatives = np.full(expected.shape, -1, dtype=np.int64)
            for label in range(len(policy.labels)):
                indicator = (y == label).astype(float)
                indicator -= indicator.mean()
                covariance = centered.T @ indicator
                variance = np.square(centered).sum(axis=0) * np.square(indicator).sum()
                corr = np.divide(covariance**2, variance, out=np.zeros_like(covariance), where=variance > 0)
                corr[support < cfg.min_support] = 0
                for entry in range(len(entries)):
                    channels = np.r_[
                        np.arange(layout.offsets[entry], layout.offsets[entry + 1]),
                        np.arange(layout.offsets[entry], layout.offsets[entry + 1]) + layout.block_width,
                    ]
                    best = channels[np.argmax(corr[channels])]
                    expected[entry, label] = corr[best]
                    if corr[best] > 0:
                        representatives[entry, label] = best
            np.testing.assert_allclose(evidence.scores, expected, atol=1e-12)
            np.testing.assert_array_equal(evidence.channels, representatives)
            self.assertTrue(np.isfinite(evidence.scores).all())
            projected = np.zeros((len(y) * len(policy.labels), 2 * len(entries)))
            for entry, channels in enumerate(evidence.channels):
                for label, channel in enumerate(channels):
                    if channel >= 0:
                        role = channel // layout.block_width
                        projected[label * len(y) : (label + 1) * len(y), role * len(entries) + entry] = matrix[
                            :, channel
                        ]
            for additions in (1, len(entries)):
                pairs, norms = supervised._cooccurrence(
                    serial_batches(lambda: iter(theorems)),
                    index,
                    policy,
                    space,
                    evidence,
                    np.arange(len(entries)),
                    additions,
                    cfg,
                )
                expected_pairs = sum(
                    projected[:, role * len(entries) : role * len(entries) + additions].T
                    @ projected[:, role * len(entries) : (role + 1) * len(entries)]
                    for role in (0, 1)
                )
                np.testing.assert_array_equal(pairs, expected_pairs)
                np.testing.assert_array_equal(
                    norms,
                    np.square(projected[:, : len(entries)]).sum(axis=0)
                    + np.square(projected[:, len(entries) :]).sum(axis=0),
                )

    def test_failed_selection_does_not_publish_or_replace_sources(self) -> None:
        # Failures: unknown tactics silently
        # bypass strict policy, one-class evidence is accepted, invalid settings
        # or changed sources publish a partial vocabulary, or source overwritten.
        # Establishes fail-closed publication using fixture archives, not OS OOM.
        theorems = tuple(fixture(idx) for idx in range(4))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src, _, labels, dst = (
                root / name for name in ("candidates.zst", "coverage.zst", "labels.json", "vocab.zst")
            )
            write_candidates(src, theorems)
            labels.write_bytes(msgspec.json.encode(LABELS))
            cfg = pipeline.VocabBuildCfg(5000, depths=DEPTHS, hyp_slots=2, min_support=1)
            pipeline.fit_vocab(src, labels, dst, cfg)
            before = dst.read_bytes()
            for policy, message in (
                (LabelPolicy({"a": "A", "b": "B"}, "error"), "unmapped"),
                (LabelPolicy({"a": "same", "b": "same"}, "drop"), "two observed"),
            ):
                labels.write_bytes(msgspec.json.encode(policy))
                with self.assertRaisesRegex(ValueError, message):
                    pipeline.fit_vocab(src, labels, dst, cfg, replace=True)
                self.assertEqual(dst.read_bytes(), before)
            labels.write_bytes(msgspec.json.encode(LABELS))
            with self.assertRaisesRegex(ValueError, "budget"):
                pipeline.fit_vocab(src, labels, dst, pipeline.VocabBuildCfg(-1), replace=True)
            with self.assertRaisesRegex(ValueError, "cannot replace"):
                pipeline.fit_vocab(src, labels, labels, cfg, replace=True)
            # A large declared label set exercises the retained-buffer guard
            # with tiny graphs; allocations never approach a server OOM.
            large = LabelPolicy(LABELS.kinds | {f"kind.{idx}": f"label.{idx}" for idx in range(20_000)}, "drop")
            labels.write_bytes(msgspec.json.encode(large))
            with self.assertRaisesRegex(MemoryError, "score/pair buffers"):
                pipeline.fit_vocab(
                    src,
                    labels,
                    dst,
                    pipeline.VocabBuildCfg(5000, depths=DEPTHS, min_support=1, score_mem_mib=1),
                    replace=True,
                )
            labels.write_bytes(msgspec.json.encode(LABELS))
            original_select = pipeline.select_supervised

            def changed(*args, **kwargs):
                result = original_select(*args, **kwargs)
                labels.write_bytes(labels.read_bytes() + b" ")
                return result

            with (
                patch.object(pipeline, "select_supervised", side_effect=changed),
                self.assertRaisesRegex(ValueError, "label policy changed"),
            ):
                pipeline.fit_vocab(src, labels, dst, cfg, replace=True)
            self.assertEqual(dst.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
