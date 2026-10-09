"""Frozen representation: training selection -> DB/candidates -> sparse rows.

Fixture-scale checks establish representation semantics, not predictive quality
or corpus-scale runtime/memory. No user datasets or artifacts are modified.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import msgspec
import numpy as np
from test_candidates import fixture, reachable, reference_fragment, store

import trustmebro.graph as graph_ops
import trustmebro.preprocessing.layout as layouts
from trustmebro.extraction import records as r
from trustmebro.preprocessing import archives, candidates, features, pipeline, records
from trustmebro.preprocessing import scan as scheduler
from trustmebro.preprocessing.layout import ROOT_FIELDS, SLOT_FLAGS
from trustmebro.preprocessing.records import LabelPolicy


def training(idx: int) -> r.Theorem:
    theorem = fixture()
    locals = theorem.trns[0].state.locals
    locals = (
        r.LocalLet(0, locals[0].type, r.LocalDeclKind.DEFAULT, False, 7, False),
        msgspec.structs.replace(locals[1], is_instance=True),
        *locals[2:],
    )
    trns = tuple(
        msgspec.structs.replace(
            trn,
            tactic=r.Tactic("a" if step == 0 else "b", "irrelevant syntax"),
            state=msgspec.structs.replace(trn.state, locals=locals),
        )
        for step, trn in enumerate(theorem.trns)
    )
    return msgspec.structs.replace(theorem, name=f"Train.{idx}", trns=trns)


def stat_fields(node_names: tuple[str, ...], hyp_slots: int) -> tuple[str, ...]:
    """Column names in state_stats order; operands includes an App's function.

    Shared fractions are root-local indegree>1 / distinct nodes. Context/overflow
    summaries average fractions but sum all counts; maxima remain separate.
    Local slots follow export order, including non-propositional declarations.
    """
    root = (*ROOT_FIELDS, *(f"root_{name}" for name in node_names), *(f"nodes_{name}" for name in node_names))
    flags = SLOT_FLAGS

    def summary(prefix: str) -> tuple[str, ...]:
        return tuple(f"{prefix}_{'mean' if name == 'shared_frac' else 'sum'}_{name}" for name in root)

    fields = ["locals", "lets", "instances", *(f"goal_{name}" for name in root)]
    fields.extend((*summary("ctxt"), *(f"ctxt_max_{name}" for name in root)))
    for slot in range(hyp_slots):
        fields.extend(f"hyp_{slot}_{name}" for name in (*flags, *root))
    fields.extend(
        (*(f"overflow_{name}" for name in flags), *summary("overflow"), *(f"overflow_max_{name}" for name in root))
    )
    return tuple(fields)


def feature_rows(path: Path, *, max_frame_bytes: int | None = None) -> Iterator[records.FeatureRows]:
    """One theorem matrix at a time; validate the final completion counters."""
    with archives.open_features(path, max_frame_bytes=max_frame_bytes) as source:
        yield from source.rows


class RepresentationTests(unittest.TestCase):
    def test_full_representation_matches_public_sources_and_frozen_columns(self) -> None:
        # Failures: mixed-count serialization rounds integers; shared children or
        # repeated hypotheses change weights; declarations lose let/instance or
        # order/overflow; enrichment grows on held-out names; full columns pollute
        # structural coverage; DB/reuse/API/worker paths disagree or re-traverse
        # already prepared graphs. Establishes exact fixture channels/provenance
        # and shared preparation, not held-out generalization or model quality.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            db, src, _, labels, full = (
                root / name for name in ("train.db", "candidates.zst", "coverage.zst", "labels.json", "full.zst")
            )
            theorems = tuple(training(idx) for idx in range(4))
            policy = LabelPolicy({"a": "A", "b": "B"}, "drop")
            store(db, theorems)
            labels.write_bytes(msgspec.json.encode(policy))
            scheduler.scan_cands(db, src, depths=(1, 2, 3))
            before = {path: path.read_bytes() for path in (db, src, labels)}
            pipeline.fit_vocab(
                src, labels, full, pipeline.VocabBuildCfg(5000, depths=(1, 2, 3), hyp_slots=2, min_support=1)
            )
            layout = layouts.compile_vocab(archives.read_vocab(full))
            cfg = layout.vocab.representation
            assert cfg is not None
            self.assertEqual(json.loads(cfg.label_policy_json), msgspec.to_builtins(policy))
            self.assertEqual(cfg.selection.db, str(db.resolve()))
            self.assertEqual(set(cfg.selection.theorems), {theorem.name for theorem in theorems})
            self.assertEqual(layout.head_index.keys(), {"f", "g"})
            self.assertGreater(len(cfg.names), 0)
            blocks = layouts.feature_blocks(layout)
            self.assertEqual(max(hi for _, hi in blocks.values()), layout.width)
            with patch.object(candidates, "prepare_adjacency", wraps=graph_ops.prepare_adjacency) as prepare:
                expected = features.encode_theorem(theorems[0], layout)
                self.assertEqual(prepare.call_count, 1)
            stat_names = stat_fields(records.NODE_NAMES, 2)
            lo, hi = blocks["statistics"]
            self.assertEqual(len(stat_names), hi - lo)
            vals = dict(zip(stat_names, expected.matrix[0:1, lo:hi].toarray()[0], strict=True))
            for name, val in {
                "locals": 5,
                "lets": 1,
                "instances": 1,
                "goal_distinct": 6,
                "goal_operand_refs": 7,
                "goal_depth": 3,
                "goal_shared_nodes": 1,
                "goal_shared_frac": 1 / 6,
                "goal_operands": 4,
                "goal_nodes_App": 2,
                "hyp_0_present": 1,
                "hyp_0_is_let": 1,
                "hyp_1_is_instance": 1,
                "overflow_present": 3,
                "overflow_sum_distinct": 7,
                "ctxt_sum_distinct": 15,
                "ctxt_sum_operand_refs": 10,
                "ctxt_max_distinct": 4,
                "ctxt_max_binders": 2,
            }.items():
                self.assertAlmostEqual(vals[name], val, msg=name)
            # Reuse the exact selected positions to inspect meaningful attribute
            # columns; the independent values here depend only on fixture syntax.
            attr_ids = {
                attr: layout.attr_cols[:, layouts.ATTR_NAMES.index(attr)]
                for attr in ("nat_other", "bvar_0", "forall_binders", "forall_implicit")
            }
            alo, _ = blocks["hyp_attributes"]
            for attr, count in (("nat_other", 2), ("bvar_0", 1), ("forall_binders", 2), ("forall_implicit", 1)):
                active = attr_ids[attr][attr_ids[attr] >= 0] + alo
                self.assertGreaterEqual(expected.matrix[0:1, active].sum(), count)
            named = np.zeros((2, 2 * len(cfg.names)), dtype=np.int64)
            name_cols = {key: idx for idx, key in enumerate(cfg.names)}
            for step, trn in enumerate(theorems[0].trns):
                for role, roots in enumerate(((trn.state.target,), tuple(local.type for local in trn.state.locals))):
                    for ref in roots:
                        seen: set[tuple[int, bytes]] = set()
                        for anchor in reachable(theorems[0].exprs, ref):
                            for depth in layout.vocab.depths:
                                edges, nodes = reference_fragment(theorems[0].exprs, anchor, depth)
                                ident = hashlib.sha256(msgspec.msgpack.encode(edges)).digest()
                                if (anchor, ident) in seen:
                                    continue
                                seen.add((anchor, ident))
                                for pos, node in enumerate(nodes):
                                    expr = theorems[0].exprs[node]
                                    col = name_cols.get((ident, pos, expr.name)) if isinstance(expr, r.Const) else None
                                    if col is not None:
                                        named[step, role * len(cfg.names) + col] += 1
            nlo, _ = blocks["goal_names"]
            _, nhi = blocks["hyp_names"]
            np.testing.assert_array_equal(expected.matrix[:, nlo:nhi].toarray(), named)
            structural_layout = layouts.compile_vocab(records.Vocab(layout.vocab.depths, layout.vocab.entries))
            structural = features.encode_theorem(theorems[0], structural_layout)
            np.testing.assert_array_equal(
                expected.matrix[:, : 2 * layout.block_width].toarray(), structural.matrix.toarray()
            )
            np.testing.assert_array_equal(
                features.pattern_presence(expected.matrix, layout).toarray(),
                features.pattern_presence(structural.matrix, structural_layout).toarray(),
            )
            for reuse, workers in ((False, 1), (True, 2)):
                out, stats = root / f"rows-{reuse}.zst", root / f"stats-{reuse}.zst"
                scheduler.convert_corpus(src if reuse else db, full, out, stats, cands=reuse, workers=workers)
                rows = list(feature_rows(out))
                self.assertEqual(len(rows), 4)
                for row in rows:
                    np.testing.assert_array_equal(row.matrix.toarray(), expected.matrix.toarray())
                    self.assertEqual(row.matrix.dtype, np.float64)
            # Failure: rendering interprets extra attribute columns as shapes.
            # Establishes the existing public diagnostics-to-image boundary.
            run = subprocess.run(
                [
                    str(Path(sys.executable).parent / "graphs"),
                    "--graphs",
                    "features",
                    "--feature-stats",
                    str(root / "stats-True.zst"),
                    "--output",
                    str(root / "plots"),
                    "--stats",
                    str(root / "no-analysis"),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(run.returncode, 0, run.stderr)
            self.assertEqual(len(tuple((root / "plots").glob("*.png"))), 3)
            heldout = r.Theorem(
                "Heldout",
                "H",
                None,
                (r.Const("False", ()),),
                (r.Trn(r.Tactic("a", "x"), (0, 1), 1, r.ProofState(0, (), (), ())),),
            )
            row = features.encode_theorem(heldout, layout).matrix
            self.assertEqual(row.shape[1], layout.width)
            self.assertEqual(row[0, lo + stat_names.index("goal_is_false")], 1)
            for name in ("goal_names", "hyp_names", "goal_heads", "hyp_heads", "hyp_slot_heads"):
                start, end = blocks[name]
                self.assertEqual(row[:, start:end].nnz, 0)
            self.assertEqual({path: path.read_bytes() for path in before}, before)

    def test_metadata_validation_and_fractional_roundtrip(self) -> None:
        # Failures: legacy candidate declarations silently become ordinary locals;
        # fractional values are misread as integer bytes. Establishes rejection at public
        # boundaries and exact fractional round-trip, not corpus memory limits.
        theorem = candidates.extract_cands(training(0), (1, 2))
        entries = tuple(records.Entry(shape.ident, shape.edges) for shape in theorem.shapes)
        selection = records.Selection("train.db", 1, 1, (1, 2), None, None)
        cfg = records.Representation(records.AttrPolicy(name_dims=0), (), (), selection, "x", 1, 1, b"{}")
        layout = layouts.compile_vocab(records.Vocab((1, 2), entries, representation=cfg))
        rows = features.encode_cands(theorem, layout)
        restored = archives.unpack_rows(archives.pack_rows(rows), layout.width, real=True)
        np.testing.assert_array_equal(restored.matrix.toarray(), rows.matrix.toarray())
        old = msgspec.structs.replace(
            theorem, states=tuple(msgspec.structs.replace(state, locals=None) for state in theorem.states)
        )
        with self.assertRaisesRegex(ValueError, "regenerate candidates"):
            features.encode_cands(old, layout)
        structural = layouts.compile_vocab(records.Vocab((1, 2), entries))
        self.assertEqual(features.encode_cands(old, structural).matrix.dtype, np.int64)
        # Failure: a theorem without retained transitions breaks empty native
        # incidence/slot assembly. Establishes the zero-row public entry case.
        empty = r.Theorem("Empty", "Empty", None, (), ())
        rows = features.encode_theorem(empty, layout)
        self.assertEqual(rows.matrix.shape, (0, layout.width))

    def test_categorical_literals_binders_and_unselected_names(self) -> None:
        # Failures: oversized naturals become floats/decimal strings; binder
        # attributes conflate kind, group length or BinderInfo; zero literals
        # disappear; Sort0 is conflated with other sorts; unknown names suppress
        # generic Const/leaf/slot channels. Establishes export-entry conversion
        # and archive values, not semantic simplification or tactic applicability.
        exprs = (
            r.Const("unknown", ()),
            r.Sort(r.LvlZero()),
            r.NatLiteral(0),
            r.NatLiteral(1),
            r.Bvar(1),
            r.Bvar(2),
            r.Bvar(1 << 15000),
            r.Lambda(("x", "y", "z"), 1, 4, r.BinderInfo.INST_IMPLICIT),
            r.Forall(("a", "b"), 1, 7, r.BinderInfo.STRICT_IMPLICIT),
            r.App(0, (1, 2, 3, 4, 5, 6, 8)),
        )
        trn = r.Trn(r.Tactic("a", "x"), (0, 1), 1, r.ProofState(9, (), (), ()))
        theorem = r.Theorem("Attrs", "Attrs", None, exprs, (trn,))
        observed = candidates.extract_cands(theorem, (1,))
        selection = records.Selection("train", 1, 1, (1,), None, None)
        cfg = records.Representation(records.AttrPolicy(name_dims=0), (), (), selection, "x", 1, 1, b"{}")
        vocab = records.Vocab(
            (1,), tuple(records.Entry(shape.ident, shape.edges) for shape in observed.shapes), representation=cfg
        )
        layout = layouts.compile_vocab(vocab)
        row = features.encode_theorem(theorem, layout)
        restored = archives.unpack_rows(archives.pack_rows(row), layout.width, real=True)
        np.testing.assert_array_equal(row.matrix.toarray(), restored.matrix.toarray())
        alo, _ = layouts.feature_blocks(layout)["goal_attributes"]
        for attr, min_count in (
            ("nat_0", 1),
            ("nat_1", 1),
            ("bvar_1", 1),
            ("bvar_2", 1),
            ("bvar_3_plus", 1),
            ("sort_prop", 1),
            ("lambda_binders", 3),
            ("lambda_instImplicit", 1),
            ("forall_binders", 2),
            ("forall_strictImplicit", 1),
        ):
            cols = layout.attr_cols[:, layouts.ATTR_NAMES.index(attr)]
            self.assertGreaterEqual(row.matrix[:, alo + cols[cols >= 0]].sum(), min_count, attr)
