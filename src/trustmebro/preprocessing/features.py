"""Fixed-column, label-neutral structural features from observed DAG fragments.

Only exact canonical identities match. Frontier/leaf distinctions, operand
order and sharing remain those of candidate discovery. A column counts a node
kind at one canonical position across anchored occurrences, never query depths.
Goal and summed hypothesis-type blocks are separate. A frozen representation can
add structural statistics, local-declaration slots and selected node attributes.
No local identifiers, let values, proof search or fitted scaling enter the values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from numba import njit
from scipy.sparse import csr_array, hstack

from trustmebro.extraction import records as r
from trustmebro.preprocessing.candidates import extract_cands
from trustmebro.preprocessing.layout import ATTR_NAMES, LEAF_ATTRS, SHARED_FRAC, Layout
from trustmebro.preprocessing.records import KIND_COLS, MAX_COUNT, NODE_NAMES, Cands, FeatureRows


@dataclass(frozen=True, slots=True)
class Matches:
    anchors: np.ndarray  # one entry per matched canonical position
    entries: np.ndarray
    positions: np.ndarray
    nodes: np.ndarray  # compact theorem-node rows, not fragment-local IDs


def matched_positions(theorem: Cands, layout: Layout, proj: Projection) -> Matches:
    entries = np.asarray([layout.index.get(shape.ident, -1) for shape in theorem.shapes], dtype=np.int64)
    for shape, entry in zip(theorem.shapes, entries, strict=True):
        if entry >= 0 and shape.edges != layout.vocab.entries[int(entry)].edges:
            raise ValueError("matching shape identities have different adjacency")
    return match_positions(theorem, proj, layout.vocab.depths, entries, layout.entry_sizes)


def match_positions(
    theorem: Cands, proj: Projection, depths: tuple[int, ...], entries: np.ndarray, sizes: np.ndarray
) -> Matches:
    """Match stored occurrences once per anchor/entry, shared by selection and conversion.

    entries maps theorem-local shapes to a caller's column order; -1 skips a shape.
    All returned references are compact theorem-node rows, not vocabulary positions.
    """
    anchors: list[int] = []
    matched_entries: list[int] = []
    positions: list[tuple[int, ...]] = []
    matched: set[tuple[int, int]] = set()
    allowed_depths = set(depths)
    for occ in theorem.occs:
        if occ.depth not in allowed_depths:
            continue
        entry = int(entries[occ.shape])
        if entry < 0:
            continue
        anchor = proj.node_rows[occ.nodes[0]]
        if (anchor, entry) in matched:
            continue
        if len(occ.nodes) != sizes[entry]:
            raise ValueError("occurrence positions disagree with its shape")
        matched.add((anchor, entry))
        anchors.append(anchor)
        matched_entries.append(entry)
        positions.append(occ.nodes)
    lengths = np.fromiter(map(len, positions), dtype=np.int64)
    ends = np.r_[0, np.cumsum(lengths)]
    refs = np.fromiter((proj.node_rows[ref] for nodes in positions for ref in nodes), dtype=np.intp)
    return Matches(
        np.repeat(np.asarray(anchors, dtype=np.int64), lengths),
        np.repeat(np.asarray(matched_entries, dtype=np.int64), lengths),
        np.arange(len(refs)) - np.repeat(ends[:-1], lengths),
        refs,
    )


def _root_cols(
    proj: Projection, anchors: np.ndarray, cols: np.ndarray, vals: np.ndarray | None = None
) -> tuple[csr_array, np.ndarray]:
    # Products need only this theorem's active columns, not the full vocabulary
    # width. Restore global IDs after the goal/context products are complete.
    global_cols, local_cols = np.unique(cols, return_inverse=True)
    vals = np.ones(len(cols), dtype=np.int64) if vals is None else vals
    anchored = csr_array(
        (vals, (anchors, local_cols)), shape=(cast(tuple[int, int], proj.closures.shape)[1], len(global_cols))
    )
    return cast(csr_array, proj.closures @ anchored), global_cols


def pos_counts(
    proj: Projection, matches: Matches, offsets: np.ndarray, kinds: np.ndarray
) -> tuple[csr_array, np.ndarray]:
    """Count positional node kinds per root in compact active columns."""
    cols = offsets[matches.entries] + matches.positions * len(NODE_NAMES) + kinds[matches.nodes]
    return _root_cols(proj, matches.anchors, cols)


def _roles(proj: Projection, roots: csr_array, cols: np.ndarray, width: int) -> csr_array:
    local = cast(csr_array, hstack((proj.goals @ roots, proj.hyps @ roots), format="csr"))
    idxs = local.indices
    if len(cols):
        idxs = cols[idxs % len(cols)] + (idxs // len(cols)) * width
    return csr_array((local.data, idxs, local.indptr), shape=(cast(tuple[int, int], proj.goals.shape)[0], 2 * width))


def _node_attrs(theorem: Cands, proj: Projection, matches: Matches, layout: Layout) -> csr_array:
    values = [
        (idx, ATTR_NAMES.index(attr), val)
        for idx, node in enumerate(theorem.nodes)
        for attr, val in expr_attrs(node.expr)
    ]
    attrs = csr_array(
        ([val for _, _, val in values], ([idx for idx, _, _ in values], [col for _, col, _ in values])),
        shape=(len(theorem.nodes), len(ATTR_NAMES)),
        dtype=np.int64,
    )[matches.nodes].tocoo()
    positions = layout.offsets[matches.entries] // len(NODE_NAMES) + matches.positions
    cols = layout.attr_cols[positions[attrs.row], attrs.col]
    if np.any(cols < 0):
        raise ValueError("node attributes conflict with matched shape arity")
    roots, active_cols = _root_cols(proj, matches.anchors[attrs.row], cols, attrs.data)
    return _roles(proj, roots, active_cols, layout.attr_width)


def _named_positions(theorem: Cands, proj: Projection, matches: Matches, layout: Layout) -> csr_array:
    cfg = layout.vocab.representation
    if cfg is None:
        raise ValueError("named positions require a frozen representation")
    node_names = np.fromiter(
        (layout.name_index.get(node.expr.name, -1) if isinstance(node.expr, r.Const) else -1 for node in theorem.nodes),
        dtype=np.int64,
    )
    rows = np.flatnonzero(node_names[matches.nodes] >= 0)
    positions = layout.offsets[matches.entries[rows]] // len(NODE_NAMES) + matches.positions[rows]
    cols = (
        np.asarray(layout.name_cols[positions, node_names[matches.nodes[rows]]]).ravel() - 1
        if len(rows)
        else np.empty(0, dtype=np.int64)
    )
    keep = cols >= 0
    roots, active_cols = _root_cols(proj, matches.anchors[rows[keep]], cols[keep])
    return _roles(proj, roots, active_cols, len(cfg.names))


def root_heads(theorem: Cands, proj: Projection, names: dict[str, int]) -> csr_array:
    exprs = {node.ref: node.expr for node in theorem.nodes}
    entries = [
        (idx, names[name])
        for idx, root in enumerate(theorem.roots)
        if (name := head_name(root.ref, exprs)) is not None and name in names
    ]
    return incidence([idx for idx, _ in entries], [col for _, col in entries], (len(theorem.roots), len(names)))


def _extra_features(theorem: Cands, proj: Projection, matches: Matches, layout: Layout, kinds: np.ndarray) -> csr_array:
    cfg = layout.vocab.representation
    if cfg is None:
        raise ValueError("extra features require a frozen representation")
    heads = root_heads(theorem, proj, layout.head_index)
    blocks = [
        _node_attrs(theorem, proj, matches, layout),
        _named_positions(theorem, proj, matches, layout),
        _roles(proj, heads, np.arange(len(layout.head_index)), len(layout.head_index)),
        state_stats(theorem, proj, root_stats(theorem, proj, kinds, len(NODE_NAMES)), cfg.policy.hyp_slots),
    ]
    for slot in range(cfg.policy.hyp_slots):
        keep = proj.hyp_slots == slot
        local = heads[proj.hyp_roots[keep]].tocoo()
        blocks.append(
            csr_array(
                (local.data, (proj.hyp_rows[keep][local.row], local.col)),
                shape=(len(theorem.states), len(layout.head_index)),
            )
        )
    return cast(csr_array, hstack(blocks, format="csr"))


def encode_cands(theorem: Cands, layout: Layout) -> FeatureRows:
    """Prepare one projection and matched population for every requested block.

    Structural-only counts stay int64. Full vectors include sharing fractions
    and use float64; integer counts are checked before conversion, never silently
    rounded. Arbitrary-size source literals only enter categorical channels.
    """
    bound = (
        len(theorem.nodes)
        * len(layout.vocab.depths)
        * max((1 + len(state.hyps) for state in theorem.states), default=1)
    )
    if bound > MAX_COUNT:
        raise OverflowError("feature occurrence counts exceed exact int64 sparse arithmetic")
    if layout.vocab.representation is not None and bound > 2**53:
        raise OverflowError("full-vector count bound exceeds exact float64 integers")
    if layout.vocab.representation is not None:
        factor = max(
            (
                max(
                    len(r.expr_refs(node.expr)),
                    len(node.expr.names) if isinstance(node.expr, r.Lambda | r.Forall) else 1,
                )
                for node in theorem.nodes
            ),
            default=1,
        )
        if bound * max(factor, 1) > 2**53:
            raise OverflowError("full-vector attribute/statistic counts exceed exact float64 integers")
    proj = prepare_projection(theorem)
    matches = matched_positions(theorem, layout, proj)
    kinds = np.fromiter((KIND_COLS[type(node.expr)] for node in theorem.nodes), dtype=np.int64)
    roots, active_cols = pos_counts(proj, matches, layout.offsets, kinds)
    matrix = _roles(proj, roots, active_cols, layout.block_width)
    if layout.vocab.representation is not None:
        matrix = cast(csr_array, hstack((matrix, _extra_features(theorem, proj, matches, layout, kinds)), format="csr"))
    matrix.eliminate_zeros()
    matrix.sort_indices()
    return FeatureRows(
        theorem.name,
        np.fromiter((state.step for state in theorem.states), dtype=np.int64),
        tuple(state.tactic for state in theorem.states),
        matrix,
    )


def encode_theorem(theorem: r.Theorem, layout: Layout, *, validated: bool = False) -> FeatureRows:
    """Prepare and encode an export entry; validate unless its caller already did."""
    return encode_cands(extract_cands(theorem, layout.vocab.depths, validated=validated), layout)


def pattern_presence(matrix: csr_array, layout: Layout, *, separate_roles: bool = True) -> csr_array:
    """Derive binary pattern presence from feature columns without graph searches."""
    matrix = matrix[:, : 2 * layout.block_width]
    entries = np.searchsorted(layout.offsets, matrix.indices % layout.block_width, side="right") - 1
    if separate_roles:
        entries += (matrix.indices // layout.block_width) * len(layout.vocab.entries)
    width = len(layout.vocab.entries) * (2 if separate_roles else 1)
    presence = csr_array(
        (np.ones(matrix.nnz, dtype=np.int64), entries, matrix.indptr.copy()),
        shape=(cast(tuple[int, int], matrix.shape)[0], width),
    )
    presence.sum_duplicates()
    presence.data.fill(1)
    return presence


@dataclass
class FeatureStats:
    """Own coordinated corpus totals and a bounded co-occurrence diagnostic."""

    layout: Layout
    pair_limit: int
    theorems: int
    covered_theorems: int
    states: int
    covered_states: int
    goal_covered: int
    hyp_covered: int
    active_dims: dict[int, int]
    active_patterns: dict[int, int]
    support: np.ndarray
    pairs: np.ndarray


def prepare_stats(layout: Layout, pair_limit: int = 128) -> FeatureStats:
    if not 0 <= pair_limit <= 2048:
        raise ValueError("co-occurrence prefix must be between 0 and 2048 shapes")
    count = min(pair_limit, len(layout.vocab.entries))
    support = np.zeros(2 * len(layout.vocab.entries), dtype=np.int64)
    pairs = np.zeros((count, count), dtype=np.int64)
    return FeatureStats(layout, count, 0, 0, 0, 0, 0, 0, {}, {}, support, pairs)


def _add_hgram(dst: dict[int, int], vals: np.ndarray) -> None:
    keys, counts = np.unique(vals, return_counts=True)
    for key, count in zip(keys, counts, strict=True):
        dst[int(key)] = dst.get(int(key), 0) + int(count)


def add_stats(stats: FeatureStats, rows: FeatureRows) -> None:
    layout, matrix = stats.layout, rows.matrix
    count = cast(tuple[int, int], matrix.shape)[0]
    if stats.states + count > MAX_COUNT:
        raise OverflowError("diagnostic counts exceed exact native arithmetic")
    presence = pattern_presence(matrix, layout)
    vocab_size = len(layout.vocab.entries)
    state_rows = np.repeat(np.arange(count), np.diff(presence.indptr))
    nontrivial = layout.nontrivial[presence.indices % vocab_size]
    goal = np.bincount(state_rows[nontrivial & (presence.indices < vocab_size)], minlength=count) > 0
    hyp = np.bincount(state_rows[nontrivial & (presence.indices >= vocab_size)], minlength=count) > 0
    stats.theorems += 1
    stats.covered_theorems += int(np.any(goal | hyp))
    stats.states += count
    stats.covered_states += int(np.count_nonzero(goal | hyp))
    stats.goal_covered += int(np.count_nonzero(goal))
    stats.hyp_covered += int(np.count_nonzero(hyp))
    _add_hgram(stats.active_dims, np.diff(matrix.indptr))
    _add_hgram(stats.active_patterns, np.diff(presence.indptr))
    cols, support = np.unique(presence.indices, return_counts=True)
    stats.support[cols] += support
    if stats.pair_limit:
        # Union roles, then binarize again: goal+context counts as one state.
        selected = presence[:, : stats.pair_limit] + presence[:, vocab_size : vocab_size + stats.pair_limit]
        selected.data.fill(1)
        stats.pairs += (selected.T @ selected).toarray()


@dataclass(frozen=True, slots=True)
class Projection:
    """Theorem-local coordinate maps and native root/state incidence matrices."""

    node_rows: dict[int, int]
    root_rows: dict[int, int]
    closures: csr_array
    goals: csr_array
    hyps: csr_array
    hyp_rows: np.ndarray
    hyp_roots: np.ndarray
    hyp_slots: np.ndarray


def incidence(rows: list[int] | np.ndarray, cols: list[int] | np.ndarray, shape: tuple[int, int]) -> csr_array:
    return csr_array((np.ones(len(rows), dtype=np.int64), (rows, cols)), shape=shape)


def prepare_projection(theorem: Cands) -> Projection:
    node_rows = {node.ref: idx for idx, node in enumerate(theorem.nodes)}
    root_rows = {root.ref: idx for idx, root in enumerate(theorem.roots)}
    lengths = np.fromiter((len(root.anchors) for root in theorem.roots), dtype=np.int64)
    closures = incidence(
        np.repeat(np.arange(len(theorem.roots)), lengths),
        np.fromiter((node_rows[ref] for root in theorem.roots for ref in root.anchors), dtype=np.int64),
        (len(theorem.roots), len(theorem.nodes)),
    )
    counts = np.fromiter((len(state.hyps) for state in theorem.states), dtype=np.int64)
    hyp_rows = np.repeat(np.arange(len(theorem.states)), counts)
    hyp_roots = np.fromiter((root_rows[ref] for state in theorem.states for ref in state.hyps), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    hyp_slots = np.arange(len(hyp_roots)) - np.repeat(offsets[:-1], counts)
    shape = len(theorem.states), len(theorem.roots)
    goals = incidence(np.arange(len(theorem.states)), [root_rows[state.goal] for state in theorem.states], shape)
    return Projection(
        node_rows, root_rows, closures, goals, incidence(hyp_rows, hyp_roots, shape), hyp_rows, hyp_roots, hyp_slots
    )


def expr_attrs(expr: r.Expr) -> tuple[tuple[str, int], ...]:
    match expr:
        case r.Bvar(idx=idx):
            return ((LEAF_ATTRS[min(idx, 3)], 1),)
        case r.NatLiteral(val=val):
            return (("nat_0" if val == 0 else "nat_1" if val == 1 else "nat_other", 1),)
        case r.Sort(lvl=r.LvlZero()):
            return (("sort_prop", 1),)
        case r.Lambda(names=names, binder_info=info) | r.Forall(names=names, binder_info=info):
            kind = "lambda" if isinstance(expr, r.Lambda) else "forall"
            return ((f"{kind}_binders", len(names)), (f"{kind}_{info}", 1))
        case _:
            return ()


def head_name(ref: int, exprs: dict[int, r.Expr]) -> str | None:
    expr = exprs[ref]
    if isinstance(expr, r.App):
        expr = exprs[expr.fn]
    return expr.name if isinstance(expr, r.Const) else None


@njit(cache=True)
def _depths(offsets: np.ndarray, refs: np.ndarray) -> np.ndarray:
    depths = np.ones(len(offsets) - 1, np.int64)
    for node in range(len(depths)):
        for idx in range(offsets[node], offsets[node + 1]):
            depths[node] = max(depths[node], depths[refs[idx]] + 1)
    return depths


def root_stats(theorem: Cands, proj: Projection, kinds: np.ndarray, node_kind_count: int) -> np.ndarray:
    """One native adjacency product computes root-local incoming multiplicity.

    Repeated operands count as repeated edges. External parents do not affect
    shared-node counts; the compact nodes retain validated post-order numbering.
    Leaf depth is one. Binder counts refer to flattened binder groups only.
    """
    operands = [tuple(proj.node_rows[ref] for ref in r.expr_refs(node.expr)) for node in theorem.nodes]
    counts = np.fromiter(map(len, operands), dtype=np.int64)
    offsets = np.r_[0, np.cumsum(counts)]
    refs = np.fromiter((ref for items in operands for ref in items), dtype=np.int64)
    parents = np.repeat(np.arange(len(counts)), counts)
    if np.any(refs >= parents):
        raise ValueError("candidate nodes are not in expression post-order")
    graph = incidence(parents, refs, (len(counts), len(counts)))
    incoming = cast(csr_array, proj.closures @ graph)
    root_nodes = np.fromiter((proj.node_rows[root.ref] for root in theorem.roots), dtype=np.int64)
    distinct = np.diff(proj.closures.indptr)
    shared = np.bincount(
        np.repeat(np.arange(len(theorem.roots)), np.diff(incoming.indptr))[incoming.data > 1],
        minlength=len(theorem.roots),
    )
    roots = [theorem.nodes[idx].expr for idx in root_nodes]
    scalars = np.column_stack(
        (
            distinct,
            np.asarray(incoming.sum(axis=1)).ravel(),
            _depths(offsets, refs)[root_nodes],
            shared,
            shared / np.maximum(distinct, 1),
            counts[root_nodes],
            [len(expr.names) if isinstance(expr, r.Lambda | r.Forall) else 0 for expr in roots],
            [isinstance(expr, r.Const) and expr.name == "False" for expr in roots],
        )
    )
    node_kinds = incidence(np.arange(len(kinds)), kinds, (len(kinds), node_kind_count))
    root_kinds = np.zeros((len(roots), node_kind_count), dtype=np.int64)
    root_kinds[np.arange(len(roots)), kinds[root_nodes]] = 1
    return np.column_stack((scalars, root_kinds, cast(csr_array, proj.closures @ node_kinds).toarray()))


def _summary(proj: Projection, stats: np.ndarray, keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows, roots = proj.hyp_rows[keep], proj.hyp_roots[keep]
    counts = np.bincount(rows, minlength=cast(tuple[int, int], proj.goals.shape)[0])
    totals = incidence(rows, roots, cast(tuple[int, int], proj.hyps.shape)) @ stats
    totals[:, SHARED_FRAC] /= np.maximum(counts, 1)
    maxima = np.zeros_like(totals)
    np.maximum.at(maxima, rows, stats[roots])
    return totals, maxima


def state_stats(theorem: Cands, proj: Projection, stats: np.ndarray, hyp_slots: int) -> csr_array:
    """Sparse slot assembly avoids a states-by-all-slots dense intermediate.

    Context/overflow summaries sum counts, average shared fractions, and retain
    maxima. Overflow never removes collective patterns or context summaries.
    """
    if any(state.locals is None or len(state.locals) != len(state.hyps) for state in theorem.states):
        raise ValueError("full features require local metadata; regenerate candidates from the database")
    info = [local for state in theorem.states for local in state.locals or ()]
    flags = np.asarray([(1, int(local.is_let), int(local.is_instance)) for local in info], dtype=np.int64).reshape(
        -1, 3
    )
    totals = np.zeros((len(theorem.states), 3), dtype=np.int64)
    np.add.at(totals, proj.hyp_rows, flags)
    all_hyps = np.ones(len(info), dtype=bool)
    ctxt_sum, ctxt_max = _summary(proj, stats, all_hyps)
    blocks = [csr_array(totals), csr_array(proj.goals @ stats), csr_array(ctxt_sum), csr_array(ctxt_max)]
    for slot in range(hyp_slots):
        keep = proj.hyp_slots == slot
        local = csr_array(np.column_stack((flags[keep], stats[proj.hyp_roots[keep]]))).tocoo()
        blocks.append(
            csr_array(
                (local.data, (proj.hyp_rows[keep][local.row], local.col)),
                shape=(len(theorem.states), 3 + stats.shape[1]),
            )
        )
    overflow = proj.hyp_slots >= hyp_slots
    over_flags = np.zeros_like(totals)
    np.add.at(over_flags, proj.hyp_rows[overflow], flags[overflow])
    over_sum, over_max = _summary(proj, stats, overflow)
    blocks.extend((csr_array(over_flags), csr_array(over_sum), csr_array(over_max)))
    return cast(csr_array, hstack(blocks, format="csr"))
