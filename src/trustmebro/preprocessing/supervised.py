"""Label-balanced enrichment of an immutable individual-root coverage vocabulary.

The first pass streams actual positional node-kind counts into sparse sufficient
statistics, never a corpus transition-feature matrix. Squared count/label
correlation screens both positive and negative associations, with theorem support
per channel. Each entry takes its strongest supported channel per label, not a
sum biased toward large entries. A second pass measures count-based redundancy
only for shortlisted entries' representative channels and the preserved cover.
Names retain their separate presence-based screening. No graphs are re-mined.
"""

from __future__ import annotations

from collections.abc import Callable, Generator, Iterable, Iterator
from contextlib import AbstractContextManager, closing, nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_array, csc_array, csr_array, hstack, vstack

from trustmebro.extraction import records as r
from trustmebro.preprocessing.coverage import CoverageIdx, IntArray, eligible_shapes, shape_costs
from trustmebro.preprocessing.features import (
    Projection,
    head_name,
    incidence,
    match_positions,
    matched_positions,
    pos_counts,
    prepare_projection,
)
from trustmebro.preprocessing.layout import Layout, check_attr_policy
from trustmebro.preprocessing.records import KIND_COLS, NODE_NAMES, AttrPolicy, Cands, LabelPolicy, SupervisedPolicy

type NameKey = tuple[bytes, int, str]  # empty identity denotes a root/head channel during selection
type BatchScope = tuple[tuple[str, tuple[int, ...]], ...]
type Stage = Callable[[str], AbstractContextManager[object]]


@dataclass(frozen=True, slots=True)
class CountBatch:
    scope: BatchScope
    cols: IntArray
    moments: csr_array
    totals: IntArray
    shape_cols: IntArray
    shape_support: IntArray
    labeled_theorems: int


@dataclass(frozen=True, slots=True)
class PairBatch:
    scope: BatchScope
    pairs: csr_array
    norms: np.ndarray


@dataclass(frozen=True, slots=True)
class PairSpace:
    counts: CountSpace
    channels: IntArray
    dst: IntArray
    label_cols: IntArray
    labels: int
    entries: int
    additions: int


@dataclass(frozen=True, slots=True)
class NameBatch:
    scope: BatchScope
    keys: tuple[NameKey, ...]
    counts: csr_array  # role/label rows x batch-local name columns
    totals: IntArray
    support: IntArray
    labeled_theorems: int


@dataclass(frozen=True, slots=True)
class NameEvidence:
    cols: IntArray
    counts: IntArray  # role x name channel x label, each transition counts at most once
    totals: IntArray  # labeled transitions per label
    support: IntArray  # distinct labeled theorem support per name channel
    labeled_theorems: int


@dataclass(frozen=True, slots=True)
class Enrichment:
    cols: IntArray  # additions only; the cover is never altered
    evidence: CountProfs
    shortlist_shapes: int
    dims: int
    label_strength: tuple[float, ...]
    selected_assoc: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CountProfs:
    cols: IntArray  # coverage shape IDs in profile order, including the backbone
    scores: np.ndarray  # shape x label, best supported squared correlation
    channels: IntArray  # shape x label, corresponding role/position/kind column; -1 when uninformative
    totals: IntArray  # labeled transitions per family
    support: IntArray  # labeled theorem support per shape
    labeled_theorems: int


@dataclass(frozen=True, slots=True)
class CountSpace:
    cols: IntArray
    sizes: IntArray
    offsets: IntArray  # positional node-kind column boundaries, one role
    sorted_ids: np.ndarray
    sorted_rows: IntArray
    depths: tuple[int, ...]

    @property
    def width(self) -> int:
        return int(self.offsets[-1])


type SelectionBatch = CountBatch | PairBatch | NameBatch


class BatchMapper(Protocol):
    """Orchestration supplies completion-order, persistent-worker batch results."""

    def __call__[Result: SelectionBatch](
        self, fn: Callable[[Iterable[Cands]], Result], phase: str
    ) -> Generator[Result]: ...


def _scope(theorems: tuple[Cands, ...]) -> BatchScope:
    return tuple((theorem.name, tuple(root.ref for root in theorem.roots)) for theorem in theorems)


def _checked_batches[Result: CountBatch | PairBatch](results: Iterable[Result], idx: CoverageIdx) -> Iterator[Result]:
    """Validate the complete population independently of completion order."""
    rows = {name: idx for idx, name in enumerate(idx.names)}
    seen = np.zeros(len(idx.names), dtype=bool)
    for result in results:
        for name, refs in result.scope:
            i = rows.get(name)
            if i is None or seen[i]:
                raise ValueError("candidate and coverage theorem populations differ or contain duplicates")
            lo, hi = int(idx.offsets[i]), int(idx.offsets[i + 1])
            if set(map(int, idx.refs[lo:hi])) != set(refs):
                raise ValueError("candidate and coverage root references differ")
            seen[i] = True
        yield result
    if not seen.all():
        raise ValueError("candidate and coverage theorem populations differ")


class _MomentSum:
    """Own bounded triplet batches and pairwise sparse reductions.

    Binary merge levels avoid copying the entire growing corpus index at every
    theorem/batch. The guard covers retained arrays, not native operation scratch
    or decoded theorem/projection memory. Moments are derived float64 statistics;
    emitted feature counts and exact population totals are never replaced.
    """

    def __init__(self, rows: int, cols: int, cfg: SupervisedPolicy, base_bytes: int) -> None:
        self.shape = rows, cols
        self.cfg = cfg
        self.base_bytes = base_bytes
        self.pending: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.levels: list[csr_array | None] = []
        self.pending_bytes = self.retained_bytes = 0
        self.batch_bytes = min(8 * 2**20, cfg.memory_mib * 2**20 // 16)

    def add(self, stats: csr_array, cols: IntArray) -> None:
        if not stats.nnz:
            return
        batch = stats.tocoo()
        data = batch.data, batch.row, cols[batch.col]
        self.pending.append(data)
        self.pending_bytes += sum(array.nbytes for array in data)
        if self.pending_bytes >= self.batch_bytes:
            self.flush()
        _guard(self.base_bytes + self.pending_bytes + self.retained_bytes, self.cfg)

    def flush(self) -> None:
        if not self.pending:
            return
        data, rows, cols = (np.concatenate(parts) for parts in zip(*self.pending, strict=True))
        merged = coo_array((data, (rows, cols)), shape=self.shape).tocsr()
        self.pending.clear()
        self.pending_bytes = 0
        level = 0
        while level < len(self.levels) and (prev := self.levels[level]) is not None:
            self.retained_bytes -= _sparse_bytes(prev)
            self.levels[level] = None
            merged = merged + prev
            level += 1
        if level == len(self.levels):
            self.levels.append(None)
        self.levels[level] = merged
        self.retained_bytes += _sparse_bytes(merged)
        _guard(self.base_bytes + self.retained_bytes, self.cfg)

    def finish(self) -> csr_array:
        self.flush()
        result = csr_array(self.shape, dtype=np.float64)
        for idx, matrix in enumerate(self.levels):
            if matrix is not None:
                result = result + matrix
                self.levels[idx] = None
        return result


def _sparse_bytes(matrix: csr_array | csc_array) -> int:
    return matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes


def _check_policy(cfg: SupervisedPolicy) -> None:
    if min(cfg.dims, cfg.shortlist, cfg.per_label, cfg.common) < 0:
        raise ValueError("supervised budgets and shortlist sizes must be nonnegative")
    if cfg.min_support < 1 or cfg.memory_mib < 1 or not np.isfinite(cfg.redundancy) or cfg.redundancy < 0:
        raise ValueError("supervised support/memory must be positive and redundancy finite/nonnegative")


def _guard(byte_count: int, cfg: SupervisedPolicy) -> None:
    if byte_count > cfg.memory_mib * 2**20:
        raise MemoryError(
            f"supervised score/pair buffers require {byte_count / 2**20:.1f} MiB; "
            "increase --score-memory-mib or reduce eligible candidates/shortlists; "
            "this guard excludes the coverage index and per-theorem scratch, and is not a peak-RSS cap"
        )


def _count_space(index: CoverageIdx, cols: IntArray, depths: tuple[int, ...]) -> CountSpace:
    sizes = index.shapes.nodes[cols]
    width = sum(map(int, sizes)) * len(NODE_NAMES)
    if 2 * width > np.iinfo(np.int64).max:
        raise OverflowError("positional selection columns exceed native sparse indexing")
    offsets = np.r_[0, np.cumsum(sizes * len(NODE_NAMES))]
    order = np.argsort(index.shapes.idents[cols])
    return CountSpace(cols, sizes, offsets, index.shapes.idents[cols][order], order, depths)


def _count_rows(
    theorem: Cands, space: CountSpace, policy: LabelPolicy, labels: dict[str, int]
) -> tuple[IntArray, IntArray, csr_array]:
    """One shared projection and native sparse product for actual feature counts.

    No Layout, adjacency decoding or graph discovery is needed. Global columns
    are arithmetic position/kind/role IDs; the working matrix is theorem-local.
    Repeated hypotheses count repeatedly, while repeated depths at one anchor do not.
    """
    labeled = [
        (idx, labels[label])
        for idx, state in enumerate(theorem.states)
        if (label := policy.label(state.tactic)) is not None
    ]
    y = np.fromiter((label for _, label in labeled), dtype=np.int64)
    if not len(y) or not len(space.cols):
        return y, np.empty(0, dtype=np.int64), csr_array((len(y), 0), dtype=np.int64)
    # Selection moments are floating point, but their source counts must still
    # fit the encoder's exact int64 arithmetic and float64 conversion boundary.
    bound = len(theorem.nodes) * len(space.depths) * max((1 + len(s.hyps) for s in theorem.states), default=1)
    if bound > 2**53:
        raise OverflowError("selection feature counts exceed exact float64 integers")
    ids = np.frombuffer(b"".join(shape.ident for shape in theorem.shapes), dtype="V32")
    pos = np.searchsorted(space.sorted_ids, ids)
    found = pos < len(space.cols)
    found[found] &= space.sorted_ids[pos[found]] == ids[found]
    entries = np.full(len(ids), -1, dtype=np.int64)
    entries[found] = space.sorted_rows[pos[found]]
    proj = prepare_projection(theorem)
    matches = match_positions(theorem, proj, space.depths, entries, space.sizes)
    kinds = np.fromiter((KIND_COLS[type(node.expr)] for node in theorem.nodes), dtype=np.int64)
    roots, cols = pos_counts(proj, matches, space.offsets, kinds)
    keep = np.fromiter((idx for idx, _ in labeled), dtype=np.int64)
    vals = cast(csr_array, hstack((proj.goals[keep] @ roots, proj.hyps[keep] @ roots), format="csr"))
    return y, np.r_[cols, cols + space.width], vals


def _stack_counts(rows: list[tuple[IntArray, IntArray, csr_array]]) -> tuple[IntArray, IntArray, csr_array, IntArray]:
    """Align theorem-local columns once; never allocate the global feature width.

    The extra row IDs retain theorem boundaries for support, while the joined
    transition rows permit one native class/square reduction per worker batch.
    """
    cols = np.unique(np.concatenate([cols for _, cols, _ in rows]))
    lengths = np.asarray([len(y) for y, _, _ in rows], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    parts = [counts.tocoo() for _, _, counts in rows]
    vals = np.concatenate([part.data for part in parts])
    row_ids = np.concatenate([part.row + offset for part, offset in zip(parts, offsets[:-1], strict=True)])
    col_ids = np.concatenate(
        [np.searchsorted(cols, local[part.col]) for (_, local, _), part in zip(rows, parts, strict=True)]
    )
    counts = csr_array((vals, (row_ids, col_ids)), shape=(int(offsets[-1]), len(cols)))
    return np.concatenate([y for y, _, _ in rows]), cols, counts, np.repeat(np.arange(len(rows)), lengths)


def count_batch(theorems: Iterable[Cands], space: CountSpace, policy: LabelPolicy) -> CountBatch:
    theorems = tuple(theorems)
    labels = policy.label_ids
    rows = [_count_rows(theorem, space, policy, labels) for theorem in theorems]
    y, cols, counts, theorem_rows = _stack_counts(rows)
    totals = np.bincount(y, minlength=len(labels))
    vals = counts.astype(np.float64)
    sums = incidence(y, np.arange(len(y)), (len(labels), len(y))) @ vals
    squares = csr_array(np.asarray(vals.power(2).sum(axis=0)))
    present = counts.copy()
    present.data.fill(1)
    support = incidence(theorem_rows, np.arange(len(y)), (len(rows), len(y))) @ present
    support.data.fill(1)  # multiple transitions in a theorem are still one support vote
    channel_support = csr_array(np.asarray(support.sum(axis=0)))
    shape_rows = [
        np.unique(np.searchsorted(space.offsets, local[np.unique(matrix.indices)] % space.width, side="right") - 1)
        for _, local, matrix in rows
        if matrix.nnz
    ]
    shape_cols, shape_support = np.unique(
        np.concatenate(shape_rows) if shape_rows else np.empty(0, dtype=np.int64), return_counts=True
    )
    return CountBatch(
        _scope(theorems),
        cols,
        cast(csr_array, vstack((sums, squares, channel_support), format="csr")),
        totals,
        shape_cols,
        shape_support,
        sum(bool(len(y)) for y, _, _ in rows),
    )


def gather_count_profiles(
    batches: BatchMapper,
    idx: CoverageIdx,
    policy: LabelPolicy,
    cfg: SupervisedPolicy,
    space: CountSpace,
    *,
    stage: Stage = nullcontext,
) -> CountProfs:
    """Stream class sums, squared sums and per-channel theorem support.

    Float64 moments are only derived association estimates; totals/support stay
    exact. Zero-valued transitions are implicit, and neither count thresholds nor
    binning discard magnitude information. Native sparse reductions batch across
    theorems before scoring observed channels in bounded dense blocks.
    """
    labels = len(policy.labels)
    base_bytes = len(space.cols) * (16 * labels + 72) + (labels + 1) * 8
    _guard(base_bytes, cfg)
    moments = _MomentSum(labels + 2, 2 * space.width, cfg, base_bytes)
    totals = np.zeros(labels, dtype=np.int64)
    support = np.zeros(len(space.cols), dtype=np.int64)
    labeled_theorems = 0
    results = batches(partial(count_batch, space=space, policy=policy), "Collect count-label associations")
    with closing(results):
        for batch in _checked_batches(results, idx):
            labeled_theorems += batch.labeled_theorems
            totals += batch.totals
            moments.add(batch.moments, batch.cols)
            support[batch.shape_cols] += batch.shape_support
    if np.count_nonzero(totals) < 2:
        raise ValueError("supervised selection requires at least two observed tactic labels")
    with stage("Reduce count-label moments"):
        stats = moments.finish()
        used = np.unique(stats.indices)
        # Compact observed columns before CSC conversion, avoiding a pointer
        # array for every possible channel in the global candidate inventory.
        compact = csr_array(
            (stats.data, np.searchsorted(used, stats.indices), stats.indptr), shape=(labels + 2, len(used))
        ).tocsc()
        _guard(base_bytes + _sparse_bytes(stats) + _sparse_bytes(compact) + used.nbytes, cfg)
        del stats, moments
    with stage("Score count-label channels"):
        scores, channels = _channel_profiles(compact, used, totals, space, cfg)
    return CountProfs(space.cols, scores, channels, totals, support, labeled_theorems)


def _channel_profiles(
    compact: csc_array, used: NDArray[np.int32 | np.int64], totals: IntArray, space: CountSpace, cfg: SupervisedPolicy
) -> tuple[np.ndarray, IntArray]:
    labels = len(totals)
    scores = np.zeros((len(space.cols), labels))
    channels = np.full(scores.shape, np.iinfo(np.int64).max, dtype=np.int64)
    total = float(totals.sum())
    probs = totals / total
    # Bound dense conversion by both channel count and declared label count.
    batch_size = max(1, min(8192, cfg.memory_mib * 2**20 // max(128 * labels, 1)))
    for lo in range(0, len(used), batch_size):
        hi = min(lo + batch_size, len(used))
        data = compact[:, lo:hi].toarray()
        sums = data[:labels].sum(axis=0)
        variance = np.maximum(data[labels] - sums * sums / total, 0)
        covariance = data[:labels] - probs[:, None] * sums
        denom = variance * (total * probs * (1 - probs))[:, None]
        corr = np.divide(covariance * covariance, denom, out=np.zeros_like(covariance), where=denom > 0)
        np.clip(corr, 0, 1, out=corr)
        corr[:, data[labels + 1] < cfg.min_support] = 0
        shapes = np.searchsorted(space.offsets, used[lo:hi] % space.width, side="right") - 1
        active = np.unique(shapes)
        for label in range(labels):
            prev = scores[active, label].copy()
            np.maximum.at(scores[:, label], shapes, corr[label])
            channels[active[scores[active, label] > prev], label] = np.iinfo(np.int64).max
            best = (corr[label] > 0) & (corr[label] == scores[shapes, label])
            np.minimum.at(channels[:, label], shapes[best], used[lo:hi][best])
    channels[channels == np.iinfo(np.int64).max] = -1
    return scores, channels


def _name_scores(evidence: NameEvidence) -> tuple[np.ndarray, np.ndarray]:
    """Pearson presence-count association, as in sklearn.feature_selection.chi2.

    Compute in bounded shape batches from sufficient statistics rather than
    retaining the transition-feature matrix. P-values are not used as claims of
    independent observations: transitions from the same theorem are correlated.
    Per-label scores use each label versus the rest, normalized for shortlisting.
    """
    total = evidence.totals.sum()
    probs = evidence.totals / total
    scores = np.zeros(len(evidence.cols))
    per_label = np.zeros((len(evidence.cols), len(probs)))
    for lo in range(0, len(scores), 8192):
        hi = min(lo + 8192, len(scores))
        observed = evidence.counts[:, lo:hi].astype(np.float64)
        present = observed.sum(axis=2, keepdims=True)
        expected = present * probs
        residual = (observed - expected) ** 2
        stats = np.divide(residual, expected, out=np.zeros_like(residual), where=expected > 0)
        scores[lo:hi] = stats.sum(axis=2).max(axis=0)
        other_expected = present * (1 - probs)
        other = np.divide(residual, other_expected, out=np.zeros_like(residual), where=other_expected > 0)
        per_label[lo:hi] = (stats + other).max(axis=0)
    return scores, per_label


def _name_roots(
    theorem: Cands, proj: Projection, layout: Layout | None, allowed: set[str] | None
) -> tuple[list[NameKey], csr_array]:
    """One sparse projection: either screen names or count selected positions.

    Name screening counts constants anywhere reachable. The second pass counts
    a constant only at a selected shape position, plus root/application heads.
    No state graph is rebuilt, and repeated depths share the matching rule used
    by conversion. Final evidence binarizes these counts per transition/role.
    """
    keys: dict[NameKey, int] = {}
    rows: list[int] = []
    cols: list[int] = []

    def add(row: int, key: NameKey) -> None:
        rows.append(row)
        cols.append(keys.setdefault(key, len(keys)))

    if allowed is None:
        for idx, node in enumerate(theorem.nodes):
            if isinstance(node.expr, r.Const):
                add(idx, (b"", 0, node.expr.name))
        return list(keys), proj.closures @ incidence(rows, cols, (len(theorem.nodes), len(keys)))
    if layout is None:
        raise ValueError("positional name selection requires a compiled vocabulary")
    matches = matched_positions(theorem, layout, proj)
    for anchor, entry, pos, node in zip(
        matches.anchors, matches.entries, matches.positions, matches.nodes, strict=True
    ):
        expr = theorem.nodes[int(node)].expr
        if isinstance(expr, r.Const) and expr.name in allowed:
            add(int(anchor), (layout.vocab.entries[int(entry)].ident, int(pos), expr.name))
    # Head columns join the same theorem-local key index, but count roots
    # directly rather than all nested occurrences of that function name.
    exprs = {node.ref: node.expr for node in theorem.nodes}
    heads = [
        (idx, name)
        for idx, root in enumerate(theorem.roots)
        if (name := head_name(root.ref, exprs)) is not None and name in allowed
    ]
    head_rows: list[int] = []
    head_cols: list[int] = []
    for row, name in heads:
        head_rows.append(row)
        head_cols.append(keys.setdefault((b"", 0, name), len(keys)))
    roots = proj.closures @ incidence(rows, cols, (len(theorem.nodes), len(keys)))
    return list(keys), roots + incidence(head_rows, head_cols, (len(theorem.roots), len(keys)))


def name_batch(
    theorems: Iterable[Cands], layout: Layout | None, policy: LabelPolicy, allowed: set[str] | None
) -> NameBatch:
    """Reduce binary transition presence directly into role/label triplets.

    Native COO coalescing replaces per-theorem class matrices/products. Support
    still counts each theorem once, across both roles and all its transitions.
    """
    theorems = tuple(theorems)
    labels = policy.label_ids
    keys: dict[NameKey, int] = {}
    row_parts: list[IntArray] = []
    col_parts: list[IntArray] = []
    support_parts: list[IntArray] = []
    totals = np.zeros(len(labels), dtype=np.int64)
    labeled_theorems = 0
    for theorem in theorems:
        if any(state.locals is None for state in theorem.states):
            raise ValueError("full features require local metadata; regenerate candidates from the database")
        state_labels = [
            (idx, labels[label])
            for idx, state in enumerate(theorem.states)
            if (label := policy.label(state.tactic)) is not None
        ]
        if not state_labels:
            continue
        proj = prepare_projection(theorem)
        local_keys, roots = _name_roots(theorem, proj, layout, allowed)
        global_cols = np.fromiter((keys.setdefault(key, len(keys)) for key in local_keys), dtype=np.int64)
        y = np.fromiter((label for _, label in state_labels), dtype=np.int64)
        keep = np.fromiter((idx for idx, _ in state_labels), dtype=np.int64)
        totals += np.bincount(y, minlength=len(labels))
        labeled_theorems += 1
        active = np.zeros(len(local_keys), dtype=bool)
        for role, state_roots in enumerate((proj.goals, proj.hyps)):
            presence = (state_roots[keep] @ roots).tocoo()
            active[presence.col] = True
            row_parts.append(y[presence.row] + role * len(labels))
            col_parts.append(global_cols[presence.col])
        support_parts.append(global_cols[active])
    rows = np.concatenate(row_parts) if row_parts else np.empty(0, dtype=np.int64)
    cols = np.concatenate(col_parts) if col_parts else np.empty(0, dtype=np.int64)
    counts = incidence(rows, cols, (2 * len(labels), len(keys)))
    support = np.bincount(
        np.concatenate(support_parts) if support_parts else np.empty(0, dtype=np.int64), minlength=len(keys)
    )
    return NameBatch(_scope(theorems), tuple(keys), counts, totals, support, labeled_theorems)


def _name_evidence(
    batches: BatchMapper, layout: Layout, policy: LabelPolicy, cfg: AttrPolicy, allowed: set[str] | None
) -> tuple[list[NameKey], NameEvidence]:
    """Own global name keys and exact counts; workers return reduced batches.

    The guard estimates retained buffers and key/index storage, excluding
    decoded batches, worker configurations and sparse scratch, not peak RSS.
    """
    keys: dict[NameKey, int] = {}
    counts = np.zeros((2, 0, len(policy.labels)), dtype=np.int64)
    support = np.zeros(0, dtype=np.int64)
    totals = np.zeros(len(policy.labels), dtype=np.int64)
    labels = policy.label_ids
    key_bytes = 0
    labeled_theorems = 0
    phase = "Screen constant names" if allowed is None else "Collect selected name-position associations"
    work = partial(name_batch, layout=layout if allowed is not None else None, policy=policy, allowed=allowed)
    results = batches(work, phase)
    with closing(results):
        for batch in results:
            local_cols: list[int] = []
            for key in batch.keys:
                if key not in keys:
                    keys[key] = len(keys)
                    key_bytes += 256 + len(key[0]) + 4 * len(key[2])
                local_cols.append(keys[key])
            capacity = max(len(keys), max(64, len(support) * 2)) if len(keys) > len(support) else len(support)
            if key_bytes + (2 * capacity * len(labels) + capacity) * 8 > cfg.memory_mib * 2**20:
                raise MemoryError("retained name evidence exceeds --name-memory-mib; no representation published")
            if capacity > len(support):
                counts = np.pad(counts, ((0, 0), (0, capacity - len(support)), (0, 0)))
                support = np.pad(support, (0, capacity - len(support)))
            totals += batch.totals
            labeled_theorems += batch.labeled_theorems
            global_cols = np.asarray(local_cols, dtype=np.int64)
            values = batch.counts.tocoo()
            counts[values.row // len(labels), global_cols[values.col], values.row % len(labels)] += values.data
            support[global_cols] += batch.support
    if np.count_nonzero(totals) < 2:
        raise ValueError("name selection requires at least two observed tactic labels")
    size = len(keys)
    evidence = NameEvidence(np.arange(size), counts[:, :size], totals, support[:size], labeled_theorems)
    return list(keys), evidence


def _rank_names(keys: list[NameKey], evidence: NameEvidence, min_support: int) -> list[int]:
    _, per_label = _name_scores(evidence)
    allowed = np.flatnonzero(evidence.support >= min_support)
    maxima = per_label[allowed].max(axis=0, initial=0)
    normalized = np.divide(per_label, maxima, out=np.zeros_like(per_label), where=maxima > 0)
    scores = normalized.max(axis=1, initial=0)
    common = evidence.counts.sum(axis=(0, 2))
    return sorted(map(int, allowed), key=lambda idx: (-scores[idx], -int(common[idx]), keys[idx]))


def select_attributes(
    batches: BatchMapper, layout: Layout, policy: LabelPolicy, cfg: AttrPolicy, *, stage: Stage = nullcontext
) -> tuple[tuple[str, ...], tuple[NameKey, ...]]:
    """Two training-only passes, then freeze a bounded set of name channels.

    Screen a configurable name pool with 3/4 label association and 1/4 common
    support. Among those names, reserve up to max_heads root/head channels, then
    spend the remaining budget on shape-position names. Generic attributes and
    statistics are not charged to this optional name-dimension budget.
    """
    check_attr_policy(cfg)
    if not cfg.name_dims or not cfg.max_names:
        return (), ()
    keys, evidence = _name_evidence(batches, layout, policy, cfg, None)
    with stage("Select constant-name pool"):
        ranked = _rank_names(keys, evidence, cfg.min_support)
        associated = ranked[: cfg.max_names - cfg.max_names // 4]
        common = sorted(ranked, key=lambda idx: (-int(evidence.support[idx]), keys[idx]))
        selected = dict.fromkeys(associated + common)
        pool = {keys[idx][2] for idx in list(selected)[: cfg.max_names]}
        del keys, evidence
    if not pool:
        return (), ()
    keys, evidence = _name_evidence(batches, layout, policy, cfg, pool)
    with stage("Freeze name-position channels"):
        ranked = _rank_names(keys, evidence, cfg.min_support)
        heads = [keys[idx][2] for idx in ranked if not keys[idx][0]][
            : min(cfg.max_heads, cfg.name_dims // (2 + cfg.hyp_slots))
        ]
        remaining = (cfg.name_dims - len(heads) * (2 + cfg.hyp_slots)) // 2
        names = [keys[idx] for idx in ranked if keys[idx][0]][:remaining]
        # Freeze canonical column order independently of score ties or iteration.
        return tuple(sorted(heads)), tuple(sorted(names))


def _top(scores: np.ndarray, allowed: IntArray, count: int, idents: np.ndarray) -> IntArray:
    """Bound the expensive tie sort to the top score threshold, not all shapes."""
    if not count or not len(allowed):
        return np.empty(0, dtype=np.int64)
    count = min(count, len(allowed))
    vals = scores[allowed]
    cutoff = np.partition(vals, len(vals) - count)[len(vals) - count]
    finalists = allowed[vals >= cutoff]
    return finalists[np.lexsort((idents[finalists], -scores[finalists]))[:count]]


def _shortlist(
    idx: CoverageIdx, evidence: CountProfs, baseline: IntArray, cfg: SupervisedPolicy
) -> tuple[IntArray, np.ndarray]:
    per_label = evidence.scores
    scores = per_label.max(axis=1, initial=0)
    excluded = np.isin(evidence.cols, baseline)
    allowed = np.flatnonzero((evidence.support >= cfg.min_support) & ~excluded)
    idents = idx.shapes.idents[evidence.cols]
    positive = allowed[scores[allowed] > 0]
    candidates = [_top(scores, positive, cfg.shortlist, idents)]
    for col in range(len(evidence.totals)):
        informative = allowed[per_label[allowed, col] > 0]
        candidates.append(_top(per_label[:, col], informative, cfg.per_label, idents))
    # Support is a shortlist hedge, not a substitute for association. Entries
    # with no supported count/label signal never consume the enrichment budget.
    common = evidence.support
    candidates.append(_top(common, allowed, cfg.common, idents))
    selected = np.unique(np.concatenate(candidates))
    maxima = per_label.max(axis=0, initial=0)
    normalized = np.divide(per_label[selected], maxima, out=np.zeros_like(per_label[selected]), where=maxima > 0)
    return evidence.cols[selected], normalized


def _cooccurrence(
    batches: BatchMapper,
    idx: CoverageIdx,
    policy: LabelPolicy,
    space: CountSpace,
    evidence: CountProfs,
    rows: IntArray,
    additions: int,
    cfg: SupervisedPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    """Cosine sufficient statistics for representative decorated count signals.

    Each entry uses its strongest supported channel for each family. Place each
    family's observations in separate row blocks so unlike channels never sum
    into a constant (e.g. Const + Fvar), and keep goal/context separate too.
    This is a bounded redundancy proxy, not a claim of conditional independence.
    Only addition x (addition + cover) pairs are allocated; cover-cover is unused.
    """
    subset = _count_space(idx, evidence.cols[rows], space.depths)
    channels = evidence.channels[rows]
    entry_rows, label_cols = np.nonzero(channels >= 0)
    channels = channels[entry_rows, label_cols]
    role = channels // space.width
    positions = channels % space.width - space.offsets[rows[entry_rows]]
    channels = role * subset.width + subset.offsets[entry_rows] + positions
    dst = entry_rows + role * len(rows)
    _guard(
        len(evidence.cols) * (16 * len(evidence.totals) + 72)
        + (len(evidence.totals) + 1) * 8
        + 64 * len(rows)  # second-pass subset coordinates
        + 8 * (additions * len(rows) + len(rows))
        + channels.nbytes
        + dst.nbytes
        + label_cols.nbytes,
        cfg,
    )
    pairs = np.zeros((additions, len(rows)))
    norms = np.zeros(len(rows))
    work = PairSpace(subset, channels, dst, label_cols, len(evidence.totals), len(rows), additions)
    results = batches(partial(pair_batch, space=work, policy=policy), "Collect count redundancy")
    with closing(results):
        for result in _checked_batches(results, idx):
            norms += result.norms
            batch = result.pairs.tocoo()
            pairs[batch.row, batch.col] += batch.data
    return pairs, norms


def pair_batch(theorems: Iterable[Cands], space: PairSpace, policy: LabelPolicy) -> PairBatch:
    theorems = tuple(theorems)
    labels = policy.label_ids
    _, cols, counts, _ = _stack_counts([_count_rows(theorem, space.counts, policy, labels) for theorem in theorems])
    pos = np.searchsorted(cols, space.channels)
    keep = pos < len(cols)
    keep[keep] &= cols[pos[keep]] == space.channels[keep]
    local = counts[:, pos[keep]].tocoo()
    row_count = cast(tuple[int, int], counts.shape)[0]
    projected = csr_array(
        (
            local.data.astype(np.float64),
            (local.row + space.label_cols[keep][local.col] * row_count, space.dst[keep][local.col]),
        ),
        shape=(row_count * space.labels, 2 * space.entries),
    )
    pairs = csr_array((space.additions, space.entries), dtype=np.float64)
    norms = np.zeros(space.entries)
    for role in (0, 1):
        vals = projected[:, role * space.entries : (role + 1) * space.entries]
        norms += np.asarray(vals.power(2).sum(axis=0)).ravel()
        pairs = pairs + vals[:, : space.additions].T @ vals
    return PairBatch(_scope(theorems), pairs, norms)


def select_supervised(
    batches: BatchMapper,
    idx: CoverageIdx,
    baseline: IntArray,
    policy: LabelPolicy,
    cfg: SupervisedPolicy,
    *,
    depths: tuple[int, ...],
    min_nodes: int = 2,
    dim_costs: Callable[[IntArray], IntArray] | None = None,
    stage: Stage = nullcontext,
) -> Enrichment:
    """Select label-balanced, cost-aware count signals without removing coverage.

    The log-utility objective gives diminishing returns per family. Baseline
    entries already contribute strength; one addition may help several families.
    Families are normalized by their strongest supported available association,
    not their transition frequency. A 0.1 prior keeps an initially unserved family
    from taking unlimited priority. Absent/no-signal labels contribute nothing.
    Neither balance nor marginal correlation guarantees held-out usefulness.
    """
    _check_policy(cfg)
    eligible = eligible_shapes(idx, min_nodes=min_nodes, min_support=cfg.min_support)
    all_cols = np.union1d(eligible, baseline)
    _guard(len(all_cols) * (16 * len(policy.labels) + 72) + (len(policy.labels) + 1) * 8, cfg)
    space = _count_space(idx, all_cols, depths)
    evidence = gather_count_profiles(batches, idx, policy, cfg, space, stage=stage)
    maxima = evidence.scores.max(axis=0, initial=0)
    base_rows = np.searchsorted(evidence.cols, baseline)
    base_scores = evidence.scores[base_rows]
    strength = np.divide(base_scores, maxima, out=np.zeros_like(base_scores), where=maxima > 0).sum(axis=0)
    selected_assoc = evidence.scores[base_rows].max(axis=0, initial=0)
    cols, profiles = _shortlist(idx, evidence, baseline, cfg)
    if not len(cols) or not cfg.dims:
        return Enrichment(
            np.empty(0, dtype=np.int64),
            evidence,
            len(cols),
            0,
            tuple(map(float, strength)),
            tuple(map(float, selected_assoc)),
        )
    # Include the entire preserved cover when estimating redundancy. A pair
    # table is limited to the shortlist + cover, never the global shape index.
    pair_rows = np.searchsorted(evidence.cols, np.concatenate((cols, baseline)))
    if cfg.redundancy:
        pairs, squares = _cooccurrence(batches, idx, policy, space, evidence, pair_rows, len(cols), cfg)
    else:
        pairs, squares = None, None
    with stage("Select shapes within complete-column budget"):
        costs = shape_costs(idx, "dims")[cols] if dim_costs is None else dim_costs(cols)
        if costs.shape != cols.shape or costs.dtype.kind not in "iu" or np.any(costs <= 0):
            raise ValueError("supervised dimension costs must be positive integers aligned with shortlisted shapes")
        similarity = np.zeros(len(cols))
        norms = np.sqrt(squares) if squares is not None else np.empty(0)
        if pairs is not None:
            # One cover column at a time avoids another dense pair-sized float array.
            for col in range(len(cols), len(pair_rows)):
                denom = norms[: len(cols)] * norms[col]
                cosine = np.divide(pairs[: len(cols), col], denom, out=np.zeros(len(cols)), where=denom > 0)
                np.maximum(similarity, cosine, out=similarity)
        chosen: list[int] = []
        available = np.ones(len(cols), dtype=bool)
        remaining = cfg.dims
        idents = idx.shapes.idents[cols]
        while np.any(allowed := available & (costs <= remaining)):
            gains = profiles / (1 + cfg.redundancy * similarity[:, None])
            priorities = np.log1p(gains / (0.1 + strength)).sum(axis=1) / costs
            allowed &= priorities > 0
            if not np.any(allowed):
                break
            best = _top(priorities, np.flatnonzero(allowed), 1, idents)[0]
            chosen.append(int(best))
            available[best] = False
            remaining -= int(costs[best])
            strength += gains[best]
            np.maximum(selected_assoc, evidence.scores[pair_rows[best]], out=selected_assoc)
            if pairs is not None:
                denom = norms[best] * norms[: len(cols)]
                cosine = np.divide(pairs[best, : len(cols)], denom, out=np.zeros(len(cols)), where=denom > 0)
                np.maximum(similarity, cosine, out=similarity)
    return Enrichment(
        cols[chosen],
        evidence,
        len(cols),
        cfg.dims - remaining,
        tuple(map(float, strength)),
        tuple(map(float, selected_assoc)),
    )
