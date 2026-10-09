"""Measure prepared graphs and aggregate corpus statistics; no corpus/archive I/O."""

from __future__ import annotations

import heapq
import sys
from collections.abc import Generator, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import repeat
from typing import cast

import numpy as np
from numba import njit
from numpy.typing import NDArray

from trustmebro.graph import Adjacency, prepare_search
from trustmebro.runtime import TimingLog, checkpoint
from trustmebro.visualization.identities import extract_patterns
from trustmebro.visualization.metrics import _group_totals, _positive_coverage, exact_count_sum
from trustmebro.visualization.views import Observations, PatternSigs, TopoView, resolve_root

from .measurements import *

# Analysis constants and result records.


PATTERN_WEIGHT_COUNT = 4
PATTERN_CAP_ROWS = 65_536
PATTERN_BUF_ROWS = 16_384
PATTERN_BUF_BATCHES = 64
PATTERN_WORK_DTYPE = np.dtype([("key", PATTERN_KEY_DTYPE), ("rad", "<u4")])


@dataclass(slots=True)
class HeadDistrs:
    counts: IntArr
    totals: CountArray
    entropies: FloatArr


def pattern_observations(
    observations: Observations, affected: NDArray[np.bool_], heads: Mapping[int, str]
) -> PatternObs:
    """Prepare raw observation weights once, without resolving lossy-view aliases."""
    top, all_ = observations.top, observations.all
    roots = np.flatnonzero(all_)
    weights = np.column_stack((top[roots], all_[roots], top[roots] * affected[roots], all_[roots] * affected[roots]))
    names = {"": 0}
    head_ids = np.fromiter(
        (names.setdefault(heads.get(int(root), ""), len(names)) for root in roots), dtype=np.uint32, count=len(roots)
    )
    return PatternObs(roots, weights.astype(np.uint64), tuple(names), head_ids)


def measure_patterns(
    view: TopoView,
    radii: Sequence[int],
    observations: PatternObs,
    *,
    mode: ViewMode,
    timings: TimingLog | None = None,
    name: str = "",
    inherited: tuple[NDArray[np.bool_], PatternSigs] | None = None,
) -> tuple[dict[int, PatternBatch], PatternSigs]:
    """Return packed counts and scratch signatures for the immediately following view."""
    phase = f"patterns.{mode}"
    with checkpoint(timings, name, f"{phase}.signatures"):
        resolved = tuple(resolve_root(view, int(root)) for root in observations.roots)
        sigs: PatternSigs = {}
        graph = view.graph
        scratch = prepare_search(Adjacency(graph.graph, graph.children, graph.offsets))
        for root in dict.fromkeys(resolved):
            if inherited is not None and inherited[0][root] and root in inherited[1]:
                sigs[root] = inherited[1][root]
            else:
                sigs[root] = extract_patterns(view, root, radii, scratch=scratch)
    with checkpoint(timings, name, f"{phase}.counts_pack"):
        if not resolved or not radii:
            return {
                rad: PatternBatch(np.empty(0, PATTERN_KEY_DTYPE), np.empty((0, 4), np.uint64), observations.heads)
                for rad in radii
            }, sigs
        # One native grouping call covers all radii/flavours/head cohorts for
        # this view. Radius IDs exist only in scratch, not the archive keys.
        work = np.empty((len(radii), len(resolved), 2, 2), dtype=PATTERN_WORK_DTYPE)
        work["rad"] = np.arange(len(radii), dtype=np.uint32)[:, None, None, None]
        work["key"]["flavour"] = np.arange(2, dtype=np.uint8)[None, None, :, None]
        work["key"]["head"] = np.column_stack((np.zeros(len(resolved), np.uint32), observations.head_ids))[
            None, :, None, :
        ]
        for idx, rad in enumerate(radii):
            work["key"]["digest"][idx] = np.asarray([sigs[root][rad] for root in resolved], dtype="V32")[:, :, None]
        keep = np.ones(work.shape, dtype=bool)
        keep[..., 1] = observations.head_ids[None, :, None] != 0
        weights = np.broadcast_to(observations.weights[None, :, None, None, :], (*work.shape, 4))[keep]
        keys, totals = _coalesce_patterns(work[keep], weights)
        grouped = np.frombuffer(b"".join(keys), dtype=PATTERN_WORK_DTYPE)
        return {
            rad: PatternBatch(grouped["key"][grouped["rad"] == idx], totals[grouped["rad"] == idx], observations.heads)
            for idx, rad in enumerate(radii)
        }, sigs


class PatternCounts:
    """Own pending arrays/index/counts; use an append-only, possibly shared head table.

    Reconcile theorem-local head IDs immediately; coalesce pending native arrays
    before querying the global index. Neither graph references nor theorem data
    are retained. A large unique input bypasses buffering and regrouping.
    """

    def __init__(self, heads: dict[str, int] | None = None) -> None:
        self.idx: dict[bytes, int] = {}
        self.heads: dict[str, int] = {"": 0} if heads is None else heads
        self.weights: CountArray = np.empty((0, PATTERN_WEIGHT_COUNT), dtype=np.uint64)
        self.length = 0
        self.pending: list[tuple[NDArray[np.void], CountArray]] = []
        self.pending_rows = 0
        self.pending_bytes = 0

    def add(
        self,
        batch: PatternBatch,
        timings: TimingLog | None = None,
        label: str = "",
        remaps: dict[tuple[str, ...], NDArray[np.uint32]] | None = None,
    ) -> None:
        """Retain only packed pattern arrays; drain before exceeding the row limit."""
        if not len(batch):
            return
        with checkpoint(timings, label, "pattern.heads"):
            # The cache is scoped to this destination table and one theorem;
            # sibling indexes may share both without repeating reconciliation.
            remap = None if remaps is None else remaps.get(batch.heads)
            if remap is None:
                remap = np.fromiter(
                    (self.heads.setdefault(head, len(self.heads)) for head in batch.heads), dtype=np.uint32
                )
                if remaps is not None:
                    remaps[batch.heads] = remap
            identity = np.array_equal(remap, np.arange(len(remap), dtype=np.uint32))
            if identity:
                # Borrow without modifying; retained workers' arrays are never
                # mutated by aggregation, coalescing or publication.
                packed = batch.keys
            else:
                packed = batch.keys.copy()
                packed["head"] = remap[packed["head"]]
        if self.pending_rows + len(batch) > PATTERN_BUF_ROWS:
            self.flush(timings, label)
        if len(batch) >= PATTERN_BUF_ROWS:
            self._add_unique(packed.view(PATTERN_BYTES_DTYPE).tolist(), batch.weights, timings, label)
            return
        self.pending.append((packed, batch.weights))
        self.pending_rows += len(batch)
        self.pending_bytes += packed.nbytes + batch.weights.nbytes
        if batch.weights.dtype == object:
            self.pending_bytes += batch.weights.size * 32
        if len(self.pending) >= PATTERN_BUF_BATCHES:
            self.flush(timings, label)

    def flush(self, timings: TimingLog | None = None, label: str = "") -> None:
        """Consume pending arrays once; all published counts include the final tail."""
        if not self.pending:
            return
        with checkpoint(timings, label, "pattern.coalesce"):
            if len(self.pending) == 1:
                packed, weights = self.pending[0]
                keys = packed.view(PATTERN_BYTES_DTYPE).tolist()
            else:
                packed = np.concatenate([keys for keys, _ in self.pending])
                weights = np.concatenate([weights for _, weights in self.pending])
                keys, weights = _coalesce_patterns(packed, weights)
            self.pending.clear()
            self.pending_rows = self.pending_bytes = 0
        self._add_unique(keys, weights, timings, label)

    def _add_unique(self, keys: list[bytes], weights: CountArray, timings: TimingLog | None, label: str) -> None:
        """Query each coalesced key once and update its coordinated weight row."""
        with checkpoint(timings, label, "pattern.index"):
            idxs = np.fromiter(map(self.idx.get, keys, repeat(-1)), dtype=np.int64, count=len(keys))
            missing = np.flatnonzero(idxs < 0)
            new = np.arange(self.length, self.length + len(missing), dtype=np.int64)
            self.idx.update(zip(map(keys.__getitem__, missing.tolist()), new.tolist(), strict=True))
            idxs[missing] = new
            self.length += len(missing)
        with checkpoint(timings, label, "pattern.buffer_update"):
            capacity = len(self.weights)
            dtype = object if weights.dtype == object or self.weights.dtype == object else np.uint64
            if self.length > capacity or self.weights.dtype != dtype:
                size = max(self.length, capacity * 2, PATTERN_CAP_ROWS)
                grown = np.empty((size, PATTERN_WEIGHT_COUNT), dtype=dtype)
                live = self.length - len(missing)
                grown[:live] = self.weights[:live]
                self.weights = grown
            self.weights[new] = 0
            if dtype == object:
                self.weights[idxs] += weights
            else:
                overflow = _update_pattern_totals(
                    cast(NDArray[np.uint64], self.weights), idxs, cast(NDArray[np.uint64], weights)
                )
                if np.any(overflow):
                    # Overflowing cells still contain their old value; other
                    # cells were updated once. Promote only after the native pass.
                    grown = np.empty(self.weights.shape, dtype=object)
                    grown[: self.length] = self.weights[: self.length]
                    self.weights = grown
                    rows, cols = np.nonzero(overflow)
                    self.weights[idxs[rows], cols] += weights[rows, cols].astype(object)

    @property
    def memory_bytes(self) -> int:
        boxed = self.weights.size * 32 if self.weights.dtype == object else 0
        return sys.getsizeof(self.idx) + len(self.idx) * 98 + self.weights.nbytes + boxed + self.pending_bytes

    def batches(self, timings: TimingLog | None = None, label: str = "") -> Generator[PatternBatch]:
        """Sort natively, retaining packed keys and gathering counters in bounded batches."""
        self.flush(timings, label)
        with checkpoint(timings, label, "pattern.sort"):
            keys = np.frombuffer(b"".join(self.idx), dtype=PATTERN_KEY_DTYPE)
            heads = tuple(self.heads)
            lexical = np.argsort(np.argsort(heads))
            order = np.lexsort((lexical[keys["head"]], keys["flavour"], keys["digest"]))
        for start in range(0, len(order), PATTERN_BATCH_ROWS):
            with checkpoint(timings, label, "pattern.gather"):
                idxs = order[start : start + PATTERN_BATCH_ROWS]
                batch = PatternBatch(keys[idxs], self.weights[idxs], heads)
            yield batch


@njit(cache=True, nogil=True)
def _update_pattern_totals(buffer: NDArray[np.uint64], idxs: IntArr, weights: NDArray[np.uint64]) -> NDArray[np.bool_]:
    """Update unique indexed rows in place, leaving carry cells for exact repair."""
    overflow = np.zeros(weights.shape, dtype=np.bool_)
    for row, idx in enumerate(idxs):
        for col in range(weights.shape[1]):
            old = buffer[idx, col]
            val = old + weights[row, col]
            if val < old:
                overflow[row, col] = True
            else:
                buffer[idx, col] = val
    return overflow


@njit(cache=True, nogil=True)
def _packed_groups(raw: NDArray[np.uint8], words: NDArray[np.uint64]) -> tuple[IntArr, IntArr]:
    """Factor complete 37/41-byte keys through Numba's native typed hash table."""
    zero = np.uint64(0)
    idx = {(zero, zero, zero, zero, zero, zero): 0}
    idx.clear()
    codes = np.empty(len(raw), dtype=np.int64)
    first = np.empty(len(raw), dtype=np.int64)
    for row in range(len(raw)):
        tail = zero
        for col in range(32, min(40, raw.shape[1])):
            tail |= np.uint64(raw[row, col]) << np.uint64(8 * (col - 32))
        last = np.uint64(raw[row, 40]) if raw.shape[1] > 40 else zero
        key = (words[row, 0], words[row, 1], words[row, 2], words[row, 3], tail, last)
        if key in idx:
            code = idx[key]
        else:
            code = len(idx)
            idx[key] = code
            first[code] = row
        codes[row] = code
    return codes, first[: len(idx)]


@njit(cache=True, nogil=True)
def _idxd_totals(
    codes: IntArr, weights: NDArray[np.uint64], count: int
) -> tuple[NDArray[np.uint64], NDArray[np.bool_]]:
    """Accumulate all cohorts in one native pass and mark actual carry events."""
    totals = np.zeros((count, weights.shape[1]), dtype=np.uint64)
    overflow = np.zeros(totals.shape, dtype=np.bool_)
    for row, code in enumerate(codes):
        for col in range(weights.shape[1]):
            old = totals[code, col]
            val = old + weights[row, col]
            totals[code, col] = val
            overflow[code, col] |= val < old
    return totals, overflow


def _coalesce_patterns(packed: NDArray[np.void], weights: CountArray) -> tuple[list[bytes], CountArray]:
    """Box only distinct keys; ordinary grouping and accumulation stay native."""
    if not len(packed):
        return [], weights
    packed = np.ascontiguousarray(packed)
    width = packed.dtype.itemsize
    if width not in (PATTERN_KEY_DTYPE.itemsize, PATTERN_WORK_DTYPE.itemsize):
        raise ValueError("unsupported packed pattern key layout")
    # Four unaligned little-endian word views cover the digest without copying.
    # The tail includes flavour, head, and (for worker scratch) radius in full.
    words = np.ndarray((len(packed), 4), dtype="<u8", buffer=packed, strides=(width, 8))
    codes, first = _packed_groups(packed.view(np.uint8).reshape(-1, width), words)
    if weights.dtype == object:
        totals = np.zeros((len(first), weights.shape[1]), dtype=object)
        np.add.at(totals, codes, weights)
    else:
        totals, overflow = _idxd_totals(codes, cast(NDArray[np.uint64], weights), len(first))
        if np.any(overflow):
            totals = totals.astype(object)
            for group, col in zip(*np.nonzero(overflow), strict=True):
                totals[group, col] = sum(map(int, weights[codes == group, col]))
    return packed[first].view(np.dtype((np.void, width))).tolist(), totals


def remap_pattern_heads(
    src_heads: Sequence[str],
    dst_heads: Sequence[str],
    batches: Iterable[PatternBatch],
    timings: TimingLog | None = None,
    label: str = "",
) -> Generator[PatternBatch]:
    """Reconcile run-local head IDs in bulk without unpacking digests or weights."""
    with checkpoint(timings, label, "pattern.headmap"):
        dst_ids = {head: idx for idx, head in enumerate(dst_heads)}
        remap = np.fromiter((dst_ids[head] for head in src_heads), dtype=np.uint32, count=len(src_heads))
        heads = tuple(dst_heads)
    if np.array_equal(remap, np.arange(len(src_heads), dtype=np.uint32)):
        for batch in batches:
            yield PatternBatch(batch.keys, batch.weights, heads)
        return
    for batch in batches:
        with checkpoint(timings, label, "pattern.remap"):
            keys = batch.keys.copy()
            keys["head"] = remap[keys["head"]]
        yield PatternBatch(keys, batch.weights, heads)


@njit(cache=True, inline="always")
def _pattern_heap_key(
    words: NDArray[np.uint64], flavours: NDArray[np.uint8], heads: NDArray[np.uint32], row: int, run: int
) -> tuple[np.uint64, ...]:
    # Numeric big-endian digest words preserve byte ordering. Head IDs are
    # numeric, not ordered by their little-endian packed representation.
    return (
        words[row, 0],
        words[row, 1],
        words[row, 2],
        words[row, 3],
        np.uint64(flavours[row]),
        np.uint64(heads[row]),
        np.uint64(run),
    )


@njit(cache=True, nogil=True)
def _merge_pattern_order(
    words: NDArray[np.uint64], flavours: NDArray[np.uint8], heads: NDArray[np.uint32], offsets: IntArr
) -> IntArr:
    """Compile standard-library heap operations over already ordered prefixes."""
    pos = offsets[:-1].copy()
    heap = [_pattern_heap_key(words, flavours, heads, row, run) for run, row in enumerate(pos)]
    heapq.heapify(heap)
    order = np.empty(len(words), dtype=np.int64)
    for idx in range(len(words)):
        run = int(heap[0][-1])
        order[idx] = pos[run]
        pos[run] += 1
        if pos[run] == offsets[run + 1]:
            heapq.heappop(heap)
        else:
            heapq.heapreplace(heap, _pattern_heap_key(words, flavours, heads, pos[run], run))
    return order


def merge_patterns(
    streams: Iterable[Iterable[PatternBatch]], timings: TimingLog | None = None, label: str = ""
) -> Generator[PatternBatch]:
    """Merge sorted buffers in bounded prefixes, reducing equal keys natively.

    The smallest buffered last key bounds a prefix for which no unseen input
    can contribute. At most one chunk per run is retained, plus that prefix.
    Head IDs must already refer to a shared lexically ordered head table.
    """
    streams = [iter(stream) for stream in streams]
    if len(streams) == 1:
        yield from streams[0]
        return

    pending: list[PatternBatch | None] = [None] * len(streams)
    ends: list[tuple[bytes, int, int]] = [(b"", 0, 0)] * len(streams)
    while True:
        for idx, stream in enumerate(streams):
            batch = pending[idx]
            if batch is None or not len(batch):
                pending[idx] = next((batch for batch in stream if len(batch)), None)
                if (batch := pending[idx]) is not None:
                    last = batch.keys[-1]
                    ends[idx] = bytes(last["digest"]), int(last["flavour"]), int(last["head"])
        active = [batch for batch in pending if batch is not None]
        if not active:
            return
        if len(active) == 1:
            # The remaining run is already unique, sorted and rechunked.
            yield active[0]
            idx = next(idx for idx, batch in enumerate(pending) if batch is not None)
            yield from (batch for batch in streams[idx] if len(batch))
            return
        last_idx = min((idx for idx, batch in enumerate(pending) if batch is not None), key=ends.__getitem__)
        last_batch = pending[last_idx]
        assert last_batch is not None
        last = last_batch.keys[-1]
        key_parts: list[NDArray[np.void]] = []
        weight_parts: list[CountArray] = []
        for idx, batch in enumerate(pending):
            if batch is None:
                continue
            stop = int(np.searchsorted(batch.keys, last, side="right"))
            if not stop:
                continue
            key_parts.append(batch.keys[:stop])
            weight_parts.append(batch.weights[:stop])
            pending[idx] = PatternBatch(batch.keys[stop:], batch.weights[stop:], batch.heads)
        if len(key_parts) == 1:
            unique, totals = key_parts[0], weight_parts[0]
        else:
            keys = np.concatenate(key_parts)
            weights = np.concatenate(weight_parts)
            offsets = np.r_[0, np.cumsum([len(part) for part in key_parts])]
            with checkpoint(timings, label, "pattern.merge.order"):
                words = np.ndarray((len(keys), 4), dtype=">u8", buffer=keys, strides=(keys.dtype.itemsize, 8))
                order = _merge_pattern_order(words.astype(np.uint64), keys["flavour"], keys["head"], offsets)
                keys, weights = keys[order], weights[order]
            with checkpoint(timings, label, "pattern.merge.reduce"):
                starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
                totals = _group_totals(weights, starts)
                unique = keys[starts]
        # Rechunk every merge level. Otherwise fan-in multiplies frame sizes
        # again at each compaction round and defeats the decoded-buffer bound.
        for start in range(0, len(unique), PATTERN_BATCH_ROWS):
            stop = start + PATTERN_BATCH_ROWS
            yield PatternBatch(unique[start:stop], totals[start:stop], active[0].heads)


# Coverage, comparisons, and corpus summaries.


def _pattern_distr(weights: CountArray) -> tuple[CountArray, int, float]:
    """Return the positive population, exact total, and entropy for both consumers."""
    vals = weights[weights > 0]
    if not len(vals):
        return vals, 0, -0.0
    total = exact_count_sum(vals)
    if total > sys.float_info.max:
        probs = np.fromiter((int(val) / total for val in vals), dtype=float, count=len(vals))
    else:
        probs = vals.astype(float) / float(total)
    positive = probs[probs > 0]
    entropy = float(-np.sum(positive * np.log2(positive)))
    return vals, total, entropy


def _head_distrs(weights: CountArray, starts: IntArr) -> HeadDistrs:
    """Measure all sorted heads/cohorts together, not one small array at a time.

    Exceptional totals retain the scalar arbitrary-natural division path. Normal
    groups use native segmented reductions; entropy may vary in its last bits
    from the scalar sum order, while all population counts remain exact.
    """
    totals = _group_totals(weights, starts)
    counts = np.add.reduceat((weights > 0).astype(np.int64), starts, axis=0)
    entropies = np.empty(totals.shape, dtype=np.float64)
    lengths = np.diff(np.r_[starts, len(weights)])
    if totals.dtype == object:
        for idx, (start, length) in enumerate(zip(starts, lengths, strict=True)):
            for col in range(PATTERN_WEIGHT_COUNT):
                _, _, entropies[idx, col] = _pattern_distr(weights[start : start + length, col])
        return HeadDistrs(counts, totals, entropies)
    for col in range(PATTERN_WEIGHT_COUNT):
        probs = np.zeros(len(weights), dtype=np.float64)
        denominators = np.repeat(totals[:, col], lengths)
        np.divide(weights[:, col], denominators, out=probs, where=denominators > 0)
        terms = np.zeros_like(probs)
        np.log2(probs, out=terms, where=probs > 0)
        terms *= probs
        entropies[:, col] = -np.add.reduceat(terms, starts)
    return HeadDistrs(counts, totals, entropies)


def pattern_descr(weights: CountArray) -> PatternDescr:
    vals, total, entropy = _pattern_distr(weights)
    curve = _positive_coverage(vals, total)
    return {
        "patterns": len(vals),
        "occs": total,
        "entropy_bits": entropy,
        "effective_patterns": 2**entropy if total else 0,
        "singletons": int((vals == 1).sum()),
        "rank": curve["rank"],
        "coverage": curve["coverage"],
    }


def pattern_stats(heads: list[str], arr: CountArray) -> PatternStats:
    descrs: dict[str, PatternDescr] = {}
    rows: list[HeadPatternRow] = []
    points: dict[int, list[tuple[int, int]]] = {}
    cohorts = (
        (PatternCol.TOP, "top", "all"),
        (PatternCol.ALL, "all", "all"),
        (PatternCol.AFFECTED_TOP, "top", "affected"),
        (PatternCol.AFFECTED_ALL, "all", "affected"),
    )
    head_ids = arr[:, PatternCol.HEAD]
    unnamed, named = head_ids == 0, head_ids > 0
    for flavour, label in enumerate(("topology", "constructors")):
        included = arr[:, PatternCol.FLAVOUR] == flavour
        global_rows = included & unnamed
        for col, scope, cohort in cohorts:
            descrs[f"{label}:{scope}:{cohort}"] = pattern_descr(arr[:, col][global_rows])
        grouped = arr[included & named]
        grouped = grouped[np.argsort(grouped[:, PatternCol.HEAD], kind="stable")]
        points[flavour] = []
        if not len(grouped):
            continue
        head_ids = grouped[:, PatternCol.HEAD]
        starts = np.r_[0, np.flatnonzero(head_ids[1:] != head_ids[:-1]) + 1]
        distrs = _head_distrs(grouped[:, PatternCol.TOP :], starts)
        for head_id, counts, totals, entropies in zip(
            head_ids[starts], distrs.counts.tolist(), distrs.totals.tolist(), distrs.entropies.tolist(), strict=True
        ):
            head = heads[int(head_id)]
            for (_, scope, cohort), count, total, entropy in zip(cohorts, counts, totals, entropies, strict=True):
                rows.append((label, scope, cohort, head, count, total, entropy))
            if totals[-1]:
                points[flavour].append((totals[-1], counts[-1]))
    return PatternStats(descrs, rows, points)
