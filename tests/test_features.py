"""Fixture-scale export → fixed sparse features → archives/diagnostic rendering.

No corpus benchmark, label quality, fitted preprocessing or predictive claims.
"""

from __future__ import annotations

import hashlib
import unittest

import msgspec
import numpy as np
from test_candidates import fixture, reachable, reference_fragment

import trustmebro.preprocessing.layout as layouts
from trustmebro.extraction import records as r
from trustmebro.preprocessing import candidates, features, records


def oracle(theorem: r.Theorem, layout: layouts.Layout) -> np.ndarray:
    """Independent small-fixture traversal/counting, not production sparse products."""
    rows = np.zeros((len(theorem.trns), 2 * layout.block_width), dtype=np.int64)
    for step, trn in enumerate(theorem.trns):
        roles = ((trn.state.target,), tuple(local.type for local in trn.state.locals))
        for role, roots in enumerate(roles):
            for root in roots:
                seen: set[tuple[int, bytes]] = set()
                for anchor in reachable(theorem.exprs, root):
                    for depth in layout.vocab.depths:
                        edges, nodes = reference_fragment(theorem.exprs, anchor, depth)
                        ident = hashlib.sha256(msgspec.msgpack.encode(edges)).digest()
                        entry = layout.index.get(ident)
                        if entry is None or (anchor, ident) in seen:
                            continue
                        seen.add((anchor, ident))
                        for pos, node in enumerate(nodes):
                            kind = layout.vocab.node_kinds.index(type(theorem.exprs[node]).__name__)
                            col = (
                                role * layout.block_width
                                + layout.offsets[entry]
                                + pos * len(layout.vocab.node_kinds)
                                + kind
                            )
                            rows[step, col] += 1
    return rows


def leaf_theorem() -> r.Theorem:
    return r.Theorem(
        "Heldout.leaf",
        "Heldout",
        None,
        (r.Const("unknown", ()),),
        (r.Trn(r.Tactic("fresh", "fresh"), (0, 1), 1, r.ProofState(0, (), (), ())),),
    )


class FeatureTests(unittest.TestCase):
    def test_tactic_names_values_and_out_of_scope_state_fields_do_not_change_features(self) -> None:
        # Failures: labels leak into X; names/literal values unexpectedly become
        # channels; local-let values/instance flags or other goals are included;
        # removing duplicate hypotheses doesn't alter context counts correctly.
        # Compact column IDs escape into output when early/middle vocabulary
        # entries are absent. Establishes the agreed feature boundary and global
        # column mapping, not future named channels or throughput.
        theorem = fixture()
        observed = candidates.extract_cands(theorem, (1, 2, 3))
        entries = tuple(records.Entry(shape.ident, shape.edges) for shape in observed.shapes)
        unused_edges: records.Edges = ((1, 2, 3, 4, 5, 6), (), (), (), (), (), ())
        unused = records.Entry(hashlib.sha256(msgspec.msgpack.encode(unused_edges)).digest(), unused_edges)
        vocab = records.Vocab((1, 2, 3), (unused, *entries))
        layout = layouts.compile_vocab(vocab)
        original = features.encode_theorem(theorem, layout).matrix.toarray()
        np.testing.assert_array_equal(original, oracle(theorem, layout))
        exprs = list(theorem.exprs)
        exprs[0] = r.Const("completely.different", ())
        exprs[7] = r.NatLiteral(19)
        trns = tuple(
            msgspec.structs.replace(
                trn,
                tactic=r.Tactic("different", "different arguments"),
                open_goal_count=5,
                state=msgspec.structs.replace(
                    trn.state,
                    locals=tuple(
                        r.LocalLet(local.id, local.type, r.LocalDeclKind.AUXILIARY, True, 7, True)
                        for local in trn.state.locals
                    ),
                ),
            )
            for trn in theorem.trns
        )
        changed = msgspec.structs.replace(theorem, exprs=tuple(exprs), trns=trns)
        np.testing.assert_array_equal(features.encode_theorem(changed, layout).matrix.toarray(), original)
        single_hyp = msgspec.structs.replace(
            theorem,
            trns=tuple(
                msgspec.structs.replace(trn, state=msgspec.structs.replace(trn.state, locals=trn.state.locals[1:]))
                for trn in theorem.trns
            ),
        )
        actual = features.encode_theorem(single_hyp, layout).matrix.toarray()
        np.testing.assert_array_equal(actual, oracle(single_hyp, layout))
        np.testing.assert_array_equal(actual[:, : layout.block_width], original[:, : layout.block_width])
        self.assertTrue(np.any(actual[:, layout.block_width :] < original[:, layout.block_width :]))


if __name__ == "__main__":
    unittest.main()
