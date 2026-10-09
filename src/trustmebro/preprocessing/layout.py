"""Frozen feature-column compilation and categorical allocation, without fitting."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_array

from trustmebro.artifacts import data_digest
from trustmebro.extraction import records as r

from .records import NODE_NAMES, AttrPolicy, Edges, Entry, Vocab

LEAF_ATTRS = ("bvar_0", "bvar_1", "bvar_2", "bvar_3_plus", "nat_0", "nat_1", "nat_other", "sort_prop")
BINDER_ATTRS = tuple(
    attr for kind in ("lambda", "forall") for attr in (f"{kind}_binders", *(f"{kind}_{info}" for info in r.BinderInfo))
)
ATTR_NAMES = LEAF_ATTRS + BINDER_ATTRS
ROOT_FIELDS = ("distinct", "operand_refs", "depth", "shared_nodes", "shared_frac", "operands", "binders", "is_false")
ROOT_SCALARS = len(ROOT_FIELDS)
SHARED_FRAC = ROOT_FIELDS.index("shared_frac")
SLOT_FLAGS = ("present", "is_let", "is_instance")


@dataclass(frozen=True)
class Layout:
    """Explicitly compiled once, then reused across theorem conversions."""

    vocab: Vocab
    index: dict[bytes, int]
    offsets: np.ndarray  # entry boundaries within one role block
    entry_sizes: np.ndarray  # prepared once; selection/conversion validate canonical position counts
    block_width: int
    nontrivial: np.ndarray
    attr_cols: np.ndarray
    attr_width: int
    name_cols: csr_array
    name_index: dict[str, int]
    head_index: dict[str, int]
    width: int  # complete vector; block_width remains structural-only


def position_attrs(refs: tuple[int, ...] | None) -> tuple[str, ...]:
    """Allocate only attributes compatible with a complete node's operand arity."""
    return (LEAF_ATTRS if refs is None or not refs else ()) + (BINDER_ATTRS if refs is None or len(refs) == 2 else ())


def compile_vocab(vocab: Vocab) -> Layout:
    if not vocab.entries or not vocab.depths or any(depth < 1 for depth in vocab.depths):
        raise ValueError("vocabulary requires entries and positive discovery depths")
    if vocab.node_kinds != NODE_NAMES:
        raise ValueError("vocabulary node-kind order differs from this converter")
    index: dict[bytes, int] = {}
    boundaries = [0]
    for idx, entry in enumerate(vocab.entries):
        if entry.ident in index:
            raise ValueError("vocabulary contains a duplicate shape identity")
        if not entry.edges or data_digest(entry.edges) != entry.ident:
            raise ValueError("vocabulary shape identity does not match canonical adjacency")
        if any(ref < 0 or ref >= len(entry.edges) for refs in entry.edges if refs is not None for ref in refs):
            raise ValueError("vocabulary contains an invalid fragment-local reference")
        index[entry.ident] = idx
        boundaries.append(boundaries[-1] + len(entry.edges) * len(NODE_NAMES))
    if 2 * boundaries[-1] > np.iinfo(np.intp).max:
        raise OverflowError("feature dimension count exceeds native sparse indexing")
    position_count = boundaries[-1] // len(NODE_NAMES)
    attr_cols = np.empty((0, len(ATTR_NAMES)), dtype=np.int64)
    attrs = 0
    name_rows: list[int] = []
    name_cols: list[int] = []
    name_index: dict[str, int] = {}
    head_index: dict[str, int] = {}
    cfg = vocab.representation
    extra_width = 0
    if cfg is not None:
        attr_cols = np.full((position_count, len(ATTR_NAMES)), -1, dtype=np.int64)
        check_attr_policy(cfg.policy)
        if len(set(cfg.heads)) != len(cfg.heads) or len(set(cfg.names)) != len(cfg.names):
            raise ValueError("representation contains duplicate name channels")
        head_index = {name: idx for idx, name in enumerate(cfg.heads)}
        name_index = {name: idx for idx, name in enumerate(sorted({name for _, _, name in cfg.names}))}
        for entry, offset in zip(vocab.entries, boundaries[:-1], strict=True):
            for pos, refs in enumerate(entry.edges):
                for attr in position_attrs(refs):
                    attr_cols[offset // len(NODE_NAMES) + pos, ATTR_NAMES.index(attr)] = attrs
                    attrs += 1
        for ident, pos, name in cfg.names:
            entry = index.get(ident)
            if entry is None or not 0 <= pos < len(vocab.entries[entry].edges) or not name:
                raise ValueError("invalid representation name position")
            refs = vocab.entries[entry].edges[pos]
            if refs is not None and refs:
                raise ValueError("constant name channel occupies a non-leaf position")
            name_rows.append(boundaries[entry] // len(NODE_NAMES) + pos)
            name_cols.append(name_index[name])
        named_cost = 2 * len(cfg.names) + (2 + cfg.policy.hyp_slots) * len(cfg.heads)
        if named_cost > cfg.policy.name_dims or any(not name for name in cfg.heads):
            raise ValueError("representation exceeds its name budget or contains empty names")
        extra_width = (
            2 * (attrs + len(cfg.names) + len(cfg.heads))
            + stat_width(len(NODE_NAMES), cfg.policy.hyp_slots)
            + cfg.policy.hyp_slots * len(cfg.heads)
        )
    named_lookup = csr_array(
        (np.arange(1, len(name_rows) + 1, dtype=np.int64), (name_rows, name_cols)),
        shape=(position_count, len(name_index)),
    )
    width = 2 * boundaries[-1] + extra_width
    if vocab.budget is not None:
        budget = vocab.budget
        if budget.baseline < 0 or budget.total < budget.baseline or not 0 <= budget.name_share <= 1:
            raise ValueError("invalid complete dimension budget")
        if width > budget.total:
            raise ValueError(f"complete feature width {width:,} exceeds total budget {budget.total:,}")
    if width > np.iinfo(np.intp).max:
        raise OverflowError("complete feature dimension count exceeds native sparse indexing")
    return Layout(
        vocab,
        index,
        np.asarray(boundaries, dtype=np.int64),
        np.diff(boundaries) // len(NODE_NAMES),
        boundaries[-1],
        np.asarray([len(entry.edges) > 1 for entry in vocab.entries]),
        attr_cols,
        attrs,
        named_lookup,
        name_index,
        head_index,
        width,
    )


def entry_dims(edges: Edges, *, attributes: bool = False) -> int:
    """Both roles, including every fixed attribute allocated to these positions."""
    attrs = sum(len(position_attrs(refs)) for refs in edges) if attributes else 0
    return 2 * (len(edges) * len(NODE_NAMES) + attrs)


def fixed_dimensions(entries: tuple[Entry, ...], hyp_slots: int) -> int:
    return sum(entry_dims(entry.edges, attributes=True) for entry in entries) + stat_width(len(NODE_NAMES), hyp_slots)


def check_attr_policy(cfg: AttrPolicy) -> None:
    if min(cfg.hyp_slots, cfg.name_dims, cfg.max_names, cfg.max_heads) < 0 or min(cfg.min_support, cfg.memory_mib) < 1:
        raise ValueError("attribute budgets/slots must be nonnegative and support/memory positive")


def feature_blocks(layout: Layout) -> dict[str, tuple[int, int]]:
    """Half-open column spans; structural coverage consumes only the first two."""
    sizes = [("goal_patterns", layout.block_width), ("hyp_patterns", layout.block_width)]
    cfg = layout.vocab.representation
    if cfg is not None:
        sizes.extend(
            (
                ("goal_attributes", layout.attr_width),
                ("hyp_attributes", layout.attr_width),
                ("goal_names", len(cfg.names)),
                ("hyp_names", len(cfg.names)),
                ("goal_heads", len(cfg.heads)),
                ("hyp_heads", len(cfg.heads)),
                ("statistics", stat_width(len(NODE_NAMES), cfg.policy.hyp_slots)),
                ("hyp_slot_heads", cfg.policy.hyp_slots * len(cfg.heads)),
            )
        )
    blocks: dict[str, tuple[int, int]] = {}
    offset = 0
    for name, width in sizes:
        blocks[name] = offset, offset + width
        offset += width
    return blocks


def count_columns(layout: Layout) -> np.ndarray:
    """Columns eligible for count compression, not data-dependent presence guesses.

    Pattern/attribute occurrences and summed context indicators remain counts.
    Goal/local indicators, fractions, and per-slot head flags remain untransformed.
    """
    counts = np.ones(layout.width, dtype=bool)
    cfg = layout.vocab.representation
    if cfg is not None:
        blocks = feature_blocks(layout)
        lo, hi = blocks["statistics"]
        counts[lo:hi] = stat_count_cols(len(NODE_NAMES), cfg.policy.hyp_slots)
        for block in ("goal_heads", "hyp_slot_heads"):
            lo, hi = blocks[block]
            counts[lo:hi] = False
    return counts


def stat_width(node_kinds: int, hyp_slots: int) -> int:
    root_width = ROOT_SCALARS + 2 * node_kinds
    return 3 + 3 * root_width + hyp_slots * (3 + root_width) + 3 + 2 * root_width


def stat_count_cols(node_kinds: int, hyp_slots: int) -> np.ndarray:
    """Count mask in state_stats order; sums of flags are counts, maxima are flags."""
    root = np.ones(ROOT_SCALARS + 2 * node_kinds, dtype=bool)
    root[SHARED_FRAC] = False
    root[ROOT_FIELDS.index("is_false")] = False
    root[ROOT_SCALARS : ROOT_SCALARS + node_kinds] = False
    summed = np.ones_like(root)
    summed[SHARED_FRAC] = False
    slot = np.r_[np.zeros(len(SLOT_FLAGS), dtype=bool), root]
    totals = np.ones(len(SLOT_FLAGS), dtype=bool)
    return np.concatenate((totals, root, summed, root, np.tile(slot, hyp_slots), totals, summed, root))
