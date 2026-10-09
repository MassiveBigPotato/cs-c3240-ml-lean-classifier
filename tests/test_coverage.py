"""Tiny candidate-archive -> coverage -> feature-equivalence checks; no tracing/JIT."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from compression import zstd
from pathlib import Path
from unittest.mock import patch

import msgspec
import numpy as np
from scipy.sparse import csr_array
from test_candidates import candidate_records

import trustmebro.preprocessing.layout as layouts
from trustmebro.artifacts import encode_msgpack
from trustmebro.extraction import records as r
from trustmebro.preprocessing import archives, coverage, features, pipeline, records


def fixture(name: str) -> records.Cands:
    edges = (((),), ((1, 2), (), ()), ((1, 2, 2), (), ()))
    shapes = tuple(records.Shape(hashlib.sha256(msgspec.msgpack.encode(edge)).digest(), edge) for edge in edges)
    nodes = tuple(
        records.Node(idx, expr, 0, 0)
        for idx, expr in enumerate((r.Const("f", ()), r.Const("g", ()), r.Fvar(0), r.App(0, (2,)), r.App(1, (2, 2))))
    )
    occurrences = tuple(records.Occurrence(0, depth, (ref,)) for ref in (0, 1, 2) for depth in (1, 2))
    occurrences += (
        records.Occurrence(1, 1, (3, 0, 2)),
        records.Occurrence(1, 2, (3, 0, 2)),
        records.Occurrence(2, 1, (4, 1, 2)),
    )
    roots = (records.Root(3, (0, 2, 3)), records.Root(4, (1, 2, 4)), records.Root(0, (0,)))
    tactic = r.Tactic("Fixture.tactic", "fixture")
    return records.Cands(
        name,
        nodes,
        shapes,
        occurrences,
        roots,
        (records.State(0, tactic, 3, (4, 0, 4)), records.State(1, tactic, 4, (3, 0))),
    )


def pack_shapes(shapes):
    """Build synthetic solver inputs without adding a production packing API."""
    shapes = tuple(shapes)
    data = tuple(msgspec.msgpack.encode(shape.edges) for shape in shapes)
    return coverage.PackedShapes(
        np.frombuffer(b"".join(shape.ident for shape in shapes), dtype="V32"),
        memoryview(b"".join(data)),
        np.r_[0, np.cumsum(tuple(map(len, data)), dtype=np.int64)],
        np.fromiter((len(shape.edges) for shape in shapes), dtype=np.int64),
    )


class CoverageTests(unittest.TestCase):
    def test_projected_filtered_archive_preserves_memberships_and_full_reader(self) -> None:
        # Failures: omitted expression data is still decoded; held-out payloads
        # reach the full decoder; request order changes the positional join;
        # unsorted/repeated closures alter binary membership or occurrence counts;
        # subset duplicates/missing names go unnoticed; large naturals regress in
        # the full reader. Establishes archive -> coverage parity and reader scope,
        # not a timing claim or validation of omitted feature-only fields.
        first = fixture("first")
        first = msgspec.structs.replace(
            first,
            nodes=(
                first.nodes[0],
                msgspec.structs.replace(first.nodes[1], expr=r.NatLiteral(1 << 100)),
                *first.nodes[2:],
            ),
            roots=(records.Root(3, (3, 2, 0, 2)), records.Root(4, (4, 2, 1, 2)), first.roots[2]),
        )
        second = msgspec.structs.replace(first, name="second")
        expected = coverage.build_coverage_idx((first, second), (1, 2))
        held_out = ["held-out", "not decoded as candidate fields"]
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "candidates.zst"
            selection = records.Selection("unused.db", 0, 0, (1, 2), None, None)
            archives.publish(
                src,
                (encode_msgpack(selection), encode_msgpack(first), encode_msgpack(held_out), encode_msgpack(second)),
                sources=(),
                replace=False,
            )
            before = src.read_bytes()
            with (
                patch.object(archives, "decode_candidates", side_effect=AssertionError("full candidate decoding")),
                patch.object(archives, "decode_nat_ext", side_effect=AssertionError("omitted expression decoding")),
            ):
                actual = pipeline.prepare_coverage(src, theorems=("second", "first"))
            self.assertEqual(actual.names, expected.names)
            for index in (actual,):
                np.testing.assert_array_equal(index.matches.toarray(), expected.matches.toarray())
                np.testing.assert_array_equal(
                    index.matches.toarray(), np.tile(((1, 1, 0), (1, 0, 1), (1, 0, 0)), (2, 1))
                )
                for col in ("goals", "hyps", "refs", "offsets", "support"):
                    np.testing.assert_array_equal(getattr(index, col), getattr(expected, col))
                np.testing.assert_array_equal(index.shapes.idents, expected.shapes.idents)
                self.assertEqual(index.shapes.data, expected.shapes.data)
                self.assertTrue(index.matches.has_canonical_format)
            decoder = archives.decode_candidates
            with patch.object(archives, "decode_candidates", wraps=decoder) as decoded:
                full = tuple(candidate_records(src, names=("second", "first")))
            self.assertEqual(decoded.call_count, 2)
            self.assertEqual(full, (first, second))
            for names, message in (
                (("missing",), "lacks requested"),
                (("first", "first"), "distinct"),
                ((), "nonempty"),
            ):
                with self.subTest(names=names), self.assertRaisesRegex(ValueError, message):
                    tuple(archives.read_coverage_candidates(src, names=names))
            with self.assertRaises(msgspec.ValidationError):
                tuple(candidate_records(src))
            self.assertEqual(src.read_bytes(), before)

    def test_projected_shape_validation_and_failed_publication(self) -> None:
        # Failures: raw adjacency bypasses conflict/identity checks; an equivalent
        # noncanonical integer encoding is rejected; malformed relevant fields
        # or duplicate selected names replace an existing artifact. Establishes
        # projected-reader validation/publication, not full feature validation.
        first, second = fixture("first"), fixture("second")
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "candidates.zst"
            selection = records.Selection("unused.db", 0, 0, (1, 2), None, None)

            def source(record: object) -> None:
                archives.publish(
                    src,
                    (encode_msgpack(selection), encode_msgpack(first), msgspec.msgpack.encode(record)),
                    sources=(),
                    replace=True,
                )

            record = msgspec.msgpack.decode(encode_msgpack(second))
            # Same edges ((1, 2), (), ()), with 1 encoded as uint8, not fixint.
            record[2][1][1] = msgspec.Raw(b"\x93\x92\xcc\x01\x02\x90\x90")
            source(record)
            actual = pipeline.prepare_coverage(src)
            expected = coverage.build_coverage_idx((first, second), (1, 2))
            self.assertEqual(actual.shapes.data, expected.shapes.data)
            np.testing.assert_array_equal(actual.matches.toarray(), expected.matches.toarray())
            record[2][1][1] = [[2, 1], [], []]
            source(record)
            with self.assertRaisesRegex(ValueError, "conflicting"):
                pipeline.prepare_coverage(src)
            for field in ("identity", "reference", "anchor", "adjacency", "duplicate"):
                record = msgspec.msgpack.decode(encode_msgpack(second))
                match field:
                    case "identity":
                        record[2][1][0] = b"invalid"
                    case "reference":
                        record[1][0][0] = "invalid"
                    case "anchor":
                        record[3][0][2] = []
                    case "adjacency":
                        record[2][1][1] = "invalid"
                    case "duplicate":
                        record[0] = "first"
                source(record)
                with self.subTest(field=field), self.assertRaises(ValueError):
                    pipeline.prepare_coverage(src, theorems=("first", "second"))

    def test_cost_objectives_fallback_and_native_pruning(self) -> None:
        # A narrow incidence fixture is needed to distinguish entry costs from
        # dimension costs unambiguously, independent of candidate discovery.
        # Failures: costs/rows/columns transposed, fallback trivializes selection,
        # support thresholds discard population, search worsens cost, or local
        # column numbering changes the result. Does not establish optimality.
        base = coverage.build_coverage_idx((fixture("first"),), (1, 2))
        chain = tuple((idx + 1,) if idx < 6 else () for idx in range(7))
        wide = records.Shape(hashlib.sha256(msgspec.msgpack.encode(chain)).digest(), chain)
        index = coverage.CoverageIdx(
            pack_shapes((*fixture("first").shapes, wide)),
            csr_array(np.asarray(((1, 1, 0, 1), (1, 0, 1, 1), (1, 0, 0, 0)), dtype=bool)),
            base.goals,
            base.hyps,
            base.names,
            base.offsets,
            base.refs,
            np.ones(4, dtype=np.int64),
        )
        for steps in (0, 10):
            entries = coverage.select_coverage(index, policy=records.CoverPolicy("entries", steps))
            dims = coverage.select_coverage(index, policy=records.CoverPolicy("dims", steps))
            self.assertEqual(set(entries.cols), {0, 3})
            self.assertEqual(set(dims.cols), {0, 1, 2})
            for result in (entries, dims):
                self.assertEqual(result.summary.total.covered_roots, 3)
                self.assertEqual(result.summary.rich.covered_roots, 2)
                self.assertLessEqual(result.summary.selected_cost, result.summary.greedy_cost)
                rich = result.cols[1:]
                for col in rich:
                    others = rich[rich != col]
                    self.assertTrue(np.any(coverage.coverage_counts(index, others)[:2] == 0))
        fallback = coverage.select_coverage(index, min_support=2)
        self.assertEqual(tuple(fallback.cols), (0,))
        self.assertEqual(fallback.summary.fallback_roots, 3)
        self.assertEqual(fallback.summary.rich.covered_roots, 0)
        for policy in (records.CoverPolicy("bad"), records.CoverPolicy("dims", -1)):
            with self.subTest(policy=policy), self.assertRaises(ValueError):
                coverage.select_coverage(index, policy=policy)
        with self.assertRaisesRegex(ValueError, "min_nodes"):
            coverage.select_coverage(index, min_nodes=1)
        # The real leaf is not interchangeable with a frontier wildcard.
        missing = coverage.CoverageIdx(
            pack_shapes((*fixture("first").shapes[1:], wide)),
            index.matches[:, 1:],
            index.goals,
            index.hyps,
            index.names,
            index.offsets,
            index.refs,
            index.support[1:],
        )
        with self.assertRaisesRegex(ValueError, "real-leaf"):
            coverage.select_coverage(missing)
        unary_edges = ((1,), ())
        unary = records.Shape(hashlib.sha256(msgspec.msgpack.encode(unary_edges)).digest(), unary_edges)
        pruning = coverage.CoverageIdx(
            pack_shapes((*fixture("first").shapes, unary)),
            csr_array(np.asarray(((1, 1, 0, 1), (1, 0, 1, 1), (1, 1, 0, 0), (1, 0, 1, 0)), dtype=bool)),
            np.ones(4, dtype=np.int64),
            np.zeros(4, dtype=np.int64),
            ("pruning",),
            np.asarray((0, 4), dtype=np.int64),
            np.arange(4, dtype=np.int64),
            np.ones(4, dtype=np.int64),
        )
        # Greedy first selects the cheaper unary pattern; later necessary
        # entries cover its roots too, so pruning must remove it.
        result = coverage.select_coverage(pruning)
        self.assertEqual(set(result.cols), {0, 1, 2})
        self.assertEqual((result.summary.greedy_cost, result.summary.selected_cost), (208, 156))

    def test_archive_to_coverage_preserves_individual_hypotheses_and_converter_membership(self) -> None:
        # Failures: a covered hypothesis hides another; duplicate depths/anchors
        # inflate coverage; repeats lose occurrence weight; theorem-local IDs
        # merge across proofs; global column remapping changes shapes; archive
        # codecs alter raw buffers or shape identities; fallback is misreported
        # as rich coverage. Establishes membership parity with feature conversion,
        # not solver behavior, full-corpus resource use or predictive quality.
        first = fixture("first")
        second = msgspec.structs.replace(
            fixture("second"),
            shapes=tuple(reversed(first.shapes)),
            occs=tuple(msgspec.structs.replace(occ, shape=2 - occ.shape) for occ in first.occs),
        )
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "candidates.zst"
            selection = records.Selection("unread-source.db", 0, 0, (1, 2), None, None)
            with zstd.open(src, "wb") as stream:
                for record in (selection, first, second):
                    archives.write_frame(stream, encode_msgpack(record))
            before = src.read_bytes()
            index = pipeline.prepare_coverage(src)
            loaded = index
            self.assertEqual(src.read_bytes(), before)
            np.testing.assert_array_equal(index.shapes.idents, loaded.shapes.idents)
            np.testing.assert_array_equal(index.shapes.offsets, loaded.shapes.offsets)
            np.testing.assert_array_equal(index.shapes.nodes, loaded.shapes.nodes)
            self.assertEqual(index.shapes.data, loaded.shapes.data)
            np.testing.assert_array_equal(loaded.matches.toarray(), np.tile(((1, 1, 0), (1, 0, 1), (1, 0, 0)), (2, 1)))
            np.testing.assert_array_equal(loaded.goals, (1, 1, 0, 1, 1, 0))
            np.testing.assert_array_equal(loaded.hyps, (1, 2, 2, 1, 2, 2))
            np.testing.assert_array_equal(loaded.offsets, (0, 3, 6))
            np.testing.assert_array_equal(loaded.support, (2, 2, 2))
            rich = coverage.eligible_shapes(loaded)
            report = coverage.coverage_report(loaded, rich)
            self.assertEqual((report.goals, report.covered_goals, report.hyps, report.covered_hyps), (4, 4, 10, 6))
            with self.assertRaisesRegex(ValueError, "4 individual hypothesis"):
                coverage.require_coverage(loaded, rich)
            coverage.require_coverage(loaded, (0,))
            self.assertEqual(coverage.coverage_report(loaded, ()).covered_roots, 0)
            np.testing.assert_array_equal(coverage.shape_costs(loaded, "dims"), (26, 78, 78))
            self.assertEqual(coverage.coverage_report(loaded, (0,)).hyps, 10)
            for col, shape in enumerate(first.shapes):
                layout = layouts.compile_vocab(records.Vocab((1, 2), (records.Entry(shape.ident, shape.edges),)))
                rows = features.encode_cands(first, layout)
                goal_match = rows.matrix[:, : layout.block_width].sum(axis=1) > 0
                hyp_match = rows.matrix[:, layout.block_width :].sum(axis=1) > 0
                matches = index.matches[:3, col].toarray().ravel().astype(bool)
                root_rows = {root.ref: idx for idx, root in enumerate(first.roots)}
                np.testing.assert_array_equal(goal_match, [matches[root_rows[state.goal]] for state in first.states])
                np.testing.assert_array_equal(
                    hyp_match, [any(matches[root_rows[ref]] for ref in state.hyps) for state in first.states]
                )

    def test_invalid_input_and_failed_publication_preserve_existing_artifacts(self) -> None:
        # Successful public-path fixtures cannot exercise corrupt references,
        # identity conflicts, index guard exhaustion or incomplete archives.
        # This narrow failure check establishes rejection/publication behavior,
        # not a total-memory cap or comprehensive adversarial archive validation.
        first = fixture("first")
        with self.assertRaisesRegex(ValueError, "unknown"):
            coverage.root_coverage(
                msgspec.structs.replace(first, states=(msgspec.structs.replace(first.states[0], goal=999),)), (1, 2)
            )
        with self.assertRaisesRegex(ValueError, "duplicate coverage theorem"):
            coverage.build_coverage_idx((first, first), (1, 2))
        invalid = msgspec.structs.replace(
            first, shapes=(msgspec.structs.replace(first.shapes[0], ident=b"bad"), *first.shapes[1:])
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            coverage.build_coverage_idx((invalid,), (1, 2))
        # Inject accounting pressure: testing actual exhaustion would require
        # an unnecessarily large workload during the active corpus extraction.
        with patch.object(coverage.sys, "getsizeof", return_value=2**20), self.assertRaises(MemoryError):
            coverage.build_coverage_idx((first,), (1, 2), idx_mem_mib=1)
        index = coverage.build_coverage_idx((first,), (1, 2))
        for columns in ((1, 1), (-1,), (99,), (0.5,)):
            with self.subTest(columns=columns), self.assertRaises(ValueError):
                coverage.coverage_counts(index, columns)
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "candidates.zst"
            selection = records.Selection("unused.db", 0, 0, (1, 2), None, None)
            with zstd.open(src, "wb") as stream:
                for record in (selection, first):
                    archives.write_frame(stream, encode_msgpack(record))
            pipeline.prepare_coverage(src)
            with self.assertRaisesRegex(ValueError, "depths"):
                pipeline.prepare_coverage(src, depths=(5,))
            with zstd.open(src, "ab") as stream:
                archives.write_frame(stream, b"\xc1")
            with self.assertRaises(msgspec.DecodeError):
                pipeline.prepare_coverage(src)
            self.assertFalse(list(Path(tmp).glob(".archive-*")))
