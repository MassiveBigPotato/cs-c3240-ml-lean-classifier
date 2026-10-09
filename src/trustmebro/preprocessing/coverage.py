"""Individual-expression memberships and coverage-based vocabulary selection.

Sparse Boolean products combine the already observed anchor matches and root
closures. They do not rediscover graphs or construct feature vectors. Query
depths, repeated anchors and repeated hypotheses cannot inflate binary coverage.
Occurrence weights remain separate from the solver's coverage requirements.
The native heuristic adapter is isolated from preparation and verification.
"""

from __future__ import annotations

import hashlib
import sys
from array import array
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

import msgspec
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array

from trustmebro.preprocessing.records import (
    DEFAULT_COVER_POLICY,
    NODE_NAMES,
    Cands,
    CoverageCands,
    CoverPolicy,
    Edges,
    Occurrence,
    Shape,
    _CoverageOcc,
    _CoverageShape,
)

type IntArray = NDArray[np.int64]
DEFAULT_IDX_MEM_MIB = 1024


# Positional projections of the candidate archive, not a second wire format.
# msgspec skips trailing fields natively, including expression/extension data.
@dataclass(frozen=True, slots=True)
class RootCoverage:
    refs: IntArray
    matches: csr_array  # root x theorem-local shape, Boolean
    goals: IntArray  # goal occurrence count for each root
    hyps: IntArray  # individual hypothesis occurrence count for each root
    shapes: IntArray  # compact column -> archived theorem-local shape ID


@dataclass(frozen=True, slots=True)
class PackedShapes:
    idents: NDArray[np.void]  # fixed-width SHA-256 digests, not Python bytes per row
    data: memoryview  # concatenated ordinary MessagePack adjacency
    offsets: IntArray
    nodes: IntArray


@dataclass(frozen=True, slots=True)
class CoverageIdx:
    shapes: PackedShapes
    matches: csr_array  # theorem-local root x global shape, Boolean
    goals: IntArray
    hyps: IntArray
    names: tuple[str, ...]
    offsets: IntArray  # root row boundaries for each theorem
    refs: IntArray  # original theorem-local expression IDs
    support: IntArray  # distinct theorem support, not occurrence counts


class CoverageReport(msgspec.Struct, frozen=True):
    shapes: int
    dims: int
    roots: int
    covered_roots: int
    goals: int
    covered_goals: int
    hyps: int
    covered_hyps: int


class CoverSummary(msgspec.Struct, frozen=True):
    policy: CoverPolicy
    min_nodes: int
    min_support: int
    eligible_shapes: int
    eligible_roots: int
    fallback_roots: int
    greedy_cost: int  # richer entries only; the fixed fallback is not optimized
    selected_cost: int
    rich: CoverageReport
    total: CoverageReport


@dataclass(frozen=True, slots=True)
class CoverSelection:
    cols: IntArray  # fixed leaf first, followed by selected richer shapes
    summary: CoverSummary


def _append_shape(
    ident: bytes, edges: Edges | msgspec.Raw, idents: bytearray, data: bytearray, offsets: array, nodes: array
) -> None:
    # Decode/validate raw adjacency once per new identity, not per theorem.
    if isinstance(edges, msgspec.Raw):
        edges = msgspec.msgpack.decode(edges, type=Edges)
    packed = msgspec.msgpack.encode(edges)
    if not edges or hashlib.sha256(packed).digest() != ident:
        raise ValueError("invalid coverage shape identity")
    idents.extend(ident)
    data.extend(packed)
    offsets.append(len(data))
    nodes.append(len(edges))


def coverage_shape(shapes: PackedShapes, col: int) -> Shape:
    """Decode only an explicitly requested shape, checking stored cost metadata."""
    lo, hi = shapes.offsets[col : col + 2]
    data = shapes.data[int(lo) : int(hi)]
    ident = shapes.idents[col].tobytes()
    edges = msgspec.msgpack.decode(data, type=Edges)
    if len(edges) != shapes.nodes[col] or hashlib.sha256(data).digest() != ident:
        raise ValueError("invalid packed coverage shape identity or node count")
    return Shape(ident, edges)


def _refs(refs: Iterable[int], positions: dict[int, int]) -> IntArray:
    result = np.fromiter((positions.get(ref, -1) for ref in refs), dtype=np.int64)
    if np.any(result < 0):
        raise ValueError("coverage references an unknown node or root")
    return result


def root_coverage(theorem: Cands | CoverageCands, depths: tuple[int, ...]) -> RootCoverage:
    """Prepare each root once, preserving every goal and hypothesis occurrence."""
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("coverage depths must be positive and nonempty")
    if not theorem.states:
        raise ValueError("coverage theorem has no observed states")
    nodes = {node.ref: idx for idx, node in enumerate(theorem.nodes)}
    roots = {root.ref: idx for idx, root in enumerate(theorem.roots)}
    if len(nodes) != len(theorem.nodes) or len(roots) != len(theorem.roots):
        raise ValueError("coverage node/root references must be distinct")
    if any(root.ref not in nodes or root.ref not in root.anchors for root in theorem.roots):
        raise ValueError("coverage closure must contain its known root")
    refs = np.asarray(tuple(roots), dtype=np.int64)
    goal_rows = _refs((state.goal for state in theorem.states), roots)
    hyp_rows = _refs((ref for state in theorem.states for ref in state.hyps), roots)
    goals = np.bincount(goal_rows, minlength=len(roots)).astype(np.int64, copy=False)
    hyps = np.bincount(hyp_rows, minlength=len(roots)).astype(np.int64, copy=False)
    if np.any(goals + hyps == 0):
        raise ValueError("coverage root has no observed goal/hypothesis occurrence")

    depths_set = set(depths)
    occs: list[Occurrence] | list[_CoverageOcc]
    if isinstance(theorem, CoverageCands):
        occs = [occ for occ in theorem.occs if occ.depth in depths_set]
        anchor_refs = (occ.anchor.ref for occ in occs)
    else:
        occs = [occ for occ in theorem.occs if occ.depth in depths_set]
        if any(not occ.nodes for occ in occs):
            raise ValueError("invalid coverage fragment occurrence")
        anchor_refs = (occ.nodes[0] for occ in occs)
    if any(not 0 <= occ.shape < len(theorem.shapes) for occ in occs):
        raise ValueError("invalid coverage fragment occurrence")
    anchors = _refs(anchor_refs, nodes)
    shapes, columns = np.unique(np.fromiter((occ.shape for occ in occs), dtype=np.int64), return_inverse=True)
    anchored = csr_array((np.ones(len(anchors), dtype=bool), (anchors, columns)), shape=(len(nodes), len(shapes)))
    counts = np.fromiter((len(root.anchors) for root in theorem.roots), dtype=np.int64)
    reachable = _refs((ref for root in theorem.roots for ref in root.anchors), nodes)
    # Rows are already grouped by root: build CSR directly without COO row IDs.
    row_offsets = np.empty(len(roots) + 1, dtype=np.int64)
    row_offsets[0] = 0
    np.cumsum(counts, out=row_offsets[1:])
    closures = csr_array((np.ones(len(reachable), dtype=bool), reachable, row_offsets), shape=(len(roots), len(nodes)))
    matches = closures @ anchored
    return RootCoverage(refs, matches, goals, hyps, shapes)


def build_coverage_idx(
    theorems: Iterable[Cands | CoverageCands], depths: tuple[int, ...], *, idx_mem_mib: int = DEFAULT_IDX_MEM_MIB
) -> CoverageIdx:
    """Stream decoded theorems into packed buffers, not Python sets per root.

    Only shapes reached at the requested depths are registered. Adjacency stays
    packed globally; decoded shape records belong to the current theorem. Each
    distinct identity is checked once; repeated identities must have the same
    adjacency. Root IDs never cross theorem boundaries. The guard
    guards estimated retained index/buffer bytes, NOT peak RSS: decoding,
    sparse products, dictionary resize and finalization can allocate additional
    memory. Final NumPy buffers view their array owners without copying.
    """
    if not depths or any(depth < 1 for depth in depths) or idx_mem_mib < 1:
        raise ValueError("coverage depths and index memory budget must be positive")
    shape_ids: dict[bytes, int] = {}
    idents, shape_data = bytearray(), bytearray()
    shape_offsets, node_counts = array("q", [0]), array("q")
    names: dict[str, None] = {}
    offsets = array("q", [0])
    refs, goals, hyps, cols, support = (array("q") for _ in range(5))
    indptr = array("q", [0])
    index_bytes = name_bytes = 0
    budget = idx_mem_mib * 2**20

    for theorem in theorems:
        if theorem.name in names:
            raise ValueError(f"duplicate coverage theorem: {theorem.name!r}")
        names[theorem.name] = None
        name_bytes += sys.getsizeof(theorem.name)
        roots = root_coverage(theorem, depths)
        active = np.unique(roots.matches.indices)
        local = np.full(len(roots.shapes), -1, dtype=np.int64)
        for idx in active:
            shape = theorem.shapes[roots.shapes[idx]]
            ident = (
                shape.ident if isinstance(shape.ident, bytes) else msgspec.convert(shape.ident, type=bytes, strict=True)
            )
            global_idx = shape_ids.get(ident)
            if global_idx is None:
                global_idx = len(support)
                _append_shape(ident, shape.edges, idents, shape_data, shape_offsets, node_counts)
                shape_ids[ident] = global_idx
                support.append(0)
                index_bytes += sys.getsizeof(ident) + sys.getsizeof(global_idx)
            else:
                lo, hi = shape_offsets[global_idx], shape_offsets[global_idx + 1]
                packed = (
                    memoryview(shape.edges)
                    if isinstance(shape, _CoverageShape)
                    else msgspec.msgpack.encode(shape.edges)
                )
                # Accept valid noncanonical MessagePack encodings just as the
                # full reader does; still reject different adjacency for an ID.
                if memoryview(shape_data)[lo:hi] != packed and (
                    not isinstance(shape, _CoverageShape)
                    or memoryview(shape_data)[lo:hi]
                    != msgspec.msgpack.encode(msgspec.msgpack.decode(shape.edges, type=Edges))
                ):
                    raise ValueError("conflicting coverage shapes for the same identity")
            local[idx] = global_idx
        # Binary rows contain a shape once even when several anchors/depths
        # matched. Support counts each participating theorem once as well.
        mapped = local[roots.matches.indices]
        for idx in np.unique(local[active]):
            support[int(idx)] += 1
        indptr.frombytes((roots.matches.indptr[1:].astype(np.int64, copy=False) + len(cols)).tobytes())
        cols.frombytes(mapped.tobytes())
        refs.frombytes(roots.refs.tobytes())
        goals.frombytes(roots.goals.tobytes())
        hyps.frombytes(roots.hyps.tobytes())
        offsets.append(len(refs))
        # Explicit estimate, not a promise that spilling/output batches bound RSS.
        shape_bytes = sum(map(sys.getsizeof, (idents, shape_data, shape_offsets, node_counts)))
        membership_bytes = sys.getsizeof(cols) + sys.getsizeof(indptr)
        root_bytes = sum(map(sys.getsizeof, (offsets, refs, goals, hyps, support)))
        lookup_bytes = index_bytes + name_bytes + sys.getsizeof(shape_ids) + sys.getsizeof(names)
        retained = shape_bytes + membership_bytes + root_bytes + lookup_bytes
        if retained > budget:
            raise MemoryError(
                f"coverage index exceeds its retained-memory guard after {len(names):,} theorems; "
                f"shapes={shape_bytes / 2**20:.1f} MiB, memberships={membership_bytes / 2**20:.1f} MiB, "
                f"roots/support={root_bytes / 2**20:.1f} MiB, lookups={lookup_bytes / 2**20:.1f} MiB; "
                "no index published (this guard is not a peak-RSS cap)"
            )

    if not names:
        raise ValueError("coverage input contains no theorems")
    arrs = [np.frombuffer(buf, dtype=np.int64) for buf in (offsets, refs, goals, hyps, cols, support, indptr)]
    offsets_arr, refs_arr, goals_arr, hyps_arr, col_arr, support_arr, row_arr = arrs
    matches = csr_array((np.ones(len(cols), dtype=bool), col_arr, row_arr), shape=(len(refs), len(support)), copy=False)
    matches.sum_duplicates()
    matches.sort_indices()
    return CoverageIdx(
        PackedShapes(
            np.frombuffer(idents, dtype="V32"),
            memoryview(shape_data),
            np.frombuffer(shape_offsets, dtype=np.int64),
            np.frombuffer(node_counts, dtype=np.int64),
        ),
        matches,
        goals_arr,
        hyps_arr,
        tuple(names),
        offsets_arr,
        refs_arr,
        support_arr,
    )


def eligible_shapes(idx: CoverageIdx, *, min_nodes: int = 2, min_support: int = 1) -> IntArray:
    """Eligibility for richer coverage; excluded roots remain in the population."""
    if min_nodes < 1 or min_support < 1:
        raise ValueError("coverage eligibility thresholds must be positive")
    return np.flatnonzero((idx.shapes.nodes >= min_nodes) & (idx.support >= min_support)).astype(np.int64, copy=False)


def shape_costs(idx: CoverageIdx, objective: Literal["entries", "dims"]) -> IntArray:
    if objective == "entries":
        return np.ones(len(idx.support), dtype=np.int64)
    if objective == "dims":
        if idx.shapes.nodes.size and idx.shapes.nodes.max() > np.iinfo(np.int64).max // (2 * len(NODE_NAMES)):
            raise OverflowError("coverage dimension cost exceeds int64")
        return idx.shapes.nodes * (2 * len(NODE_NAMES))
    raise ValueError("coverage objective must be entries or dims")


def _cols(selected: Sequence[int] | IntArray, width: int) -> IntArray:
    cols = np.asarray(selected)
    if cols.ndim != 1 or (cols.size and cols.dtype.kind not in "iu"):
        raise ValueError("selected coverage columns must be one-dimensional integers")
    if cols.size and (cols.min() < 0 or cols.max() >= width or len(np.unique(cols)) != len(cols)):
        raise ValueError("selected coverage columns must be distinct and in range")
    return cols.astype(np.int64, copy=False)


def coverage_counts(idx: CoverageIdx, selected: Sequence[int] | IntArray) -> IntArray:
    """Number of distinct selected matches per expression, not occurrence mass."""
    cols = _cols(selected, len(idx.support))
    return np.diff(idx.matches[:, cols].indptr).astype(np.int64, copy=False)


def _total(vals: IntArray) -> int:
    # Native sum for ordinary populations; preserve exactness for exceptional
    # archived weights without silently wrapping fixed-width arithmetic.
    if not vals.size or int(vals.max()) <= np.iinfo(np.int64).max // vals.size:
        return int(vals.sum())
    return sum(map(int, vals))


def coverage_report(idx: CoverageIdx, selected: Sequence[int] | IntArray) -> CoverageReport:
    cols = _cols(selected, len(idx.support))
    covered = np.diff(idx.matches[:, cols].indptr) > 0
    return CoverageReport(
        len(cols),
        _total(idx.shapes.nodes[cols]) * (2 * len(NODE_NAMES)),
        len(idx.refs),
        int(np.count_nonzero(covered)),
        _total(idx.goals),
        _total(idx.goals[covered]),
        _total(idx.hyps),
        _total(idx.hyps[covered]),
    )


def require_coverage(idx: CoverageIdx, selected: Sequence[int] | IntArray) -> None:
    """Independent post-selection feasibility check for every individual root."""
    uncovered = coverage_counts(idx, selected) == 0
    if np.any(uncovered):
        raise ValueError(
            f"vocabulary leaves {_total(idx.goals[uncovered])} goal occurrences and "
            f"{_total(idx.hyps[uncovered])} individual hypothesis occurrences uncovered"
        )


# Native heuristic adapter. Sparse preparation/reporting stays solver-independent.


def _native_cover(matches: csr_array, costs: IntArray, improvement_steps: int) -> tuple[IntArray, int]:
    """Bulk-import column memberships, then keep a feasible, non-worsening cover.

    No per-incidence Python/C++ calls, dense incidence table or exact optimizer.
    The proto and native model temporarily coexist; this is not a memory cap.
    OR-Tools uses float costs, so reject costs whose total cannot be exact there.
    """
    from ortools.set_cover.python import set_cover
    from ortools.set_cover.set_cover_pb2 import SetCoverProto

    shape = cast(tuple[int, int], matches.shape)
    if max(shape) >= 2**31 or _total(costs) > 2**53:
        raise OverflowError("set-cover model exceeds exact native index/cost limits")
    cols = matches.tocsc()
    proto = SetCoverProto()
    for col, cost in enumerate(costs):
        lo, hi = cols.indptr[col : col + 2]
        proto.subset.add(cost=float(cost), element=cols.indices[lo:hi].tolist())
    model = set_cover.SetCoverModel()
    model.import_model_from_proto(proto)
    del proto, cols
    if model.num_elements != shape[0] or not model.compute_feasibility():
        raise ValueError("native set-cover import lost requirements or is infeasible")
    state = set_cover.SetCoverInvariant(model)
    if not set_cover.GreedySolutionGenerator(state).next_solution():
        raise ValueError("native greedy search failed to construct a cover")
    greedy = np.asarray(state.is_selected(), dtype=bool)
    greedy_cost = _total(costs[greedy])
    if state.num_uncovered_elements() or np.any(np.diff(matches[:, np.flatnonzero(greedy)].indptr) == 0):
        raise ValueError("native greedy search returned an incomplete cover")
    if improvement_steps:
        search = set_cover.GuidedLocalSearch(state)
        search.set_max_iterations(improvement_steps)
        search.initialize()
        search.next_solution()
        improved = np.asarray(state.is_selected(), dtype=bool)
        if state.num_uncovered_elements():
            raise ValueError("native improvement returned an incomplete cover")
        if _total(costs[improved]) > greedy_cost:
            state.load_solution(greedy.tolist())
    # SteepestSearch removes redundant subsets, rather than exchanging away
    # richer coverage for the ubiquitous leaf shape (not in this model).
    prune = set_cover.SteepestSearch(state)
    prune.set_max_iterations(model.num_subsets)
    prune.next_solution()
    selected = np.flatnonzero(state.is_selected()).astype(np.int64, copy=False)
    if state.num_uncovered_elements() or _total(costs[selected]) > greedy_cost:
        raise ValueError("native pruning returned an incomplete or more expensive cover")
    # Check against SciPy memberships, independently of native invariants.
    if np.any(np.diff(matches[:, selected].indptr) == 0):
        raise ValueError("native cover fails independent requirement verification")
    state.recompute(set_cover.consistency_level.REDUNDANCY)
    if np.any(np.asarray(state.is_redundant(), dtype=bool)[selected]):
        raise ValueError("native pruning left redundant selected entries")
    return selected, greedy_cost


def select_coverage(
    idx: CoverageIdx, *, policy: CoverPolicy = DEFAULT_COVER_POLICY, min_nodes: int = 2, min_support: int = 1
) -> CoverSelection:
    """Cover each rich-eligible root, plus every individual root with a leaf.

    The leaf is fixed and excluded from optimization, which otherwise has the
    trivial one-entry solution. Roots without eligible richer matches remain in
    reports and retain fallback features. Observed coverage is not generalization
    or a proof of optimality. Occurrence weights do not bias the requirements.
    """
    if min_nodes < 2 or policy.improvement_steps < 0 or policy.improvement_steps >= 2**31:
        raise ValueError("rich coverage requires min_nodes >= 2 and bounded nonnegative improvement steps")
    costs = shape_costs(idx, policy.objective)
    leaf_ident = np.void(hashlib.sha256(msgspec.msgpack.encode(((),))).digest())
    leaves = np.flatnonzero(idx.shapes.idents == leaf_ident)
    if len(leaves) != 1:
        raise ValueError("coverage requires one observed real-leaf shape, not a wildcard frontier")
    leaf = leaves[0]
    if coverage_shape(idx.shapes, int(leaf)).edges != ((),):
        raise ValueError("invalid coverage leaf adjacency")
    require_coverage(idx, (leaf,))
    eligible = eligible_shapes(idx, min_nodes=min_nodes, min_support=min_support)
    # Fixed canonical order makes selection independent of worker completion.
    idents = idx.shapes.idents[eligible]
    eligible = eligible[np.argsort(idents)]
    matches = idx.matches[:, eligible]
    required = np.diff(matches.indptr) > 0
    if np.any(required):
        chosen, greedy_cost = _native_cover(matches[required], costs[eligible], policy.improvement_steps)
        rich = eligible[chosen]
    else:
        rich, greedy_cost = np.empty(0, dtype=np.int64), 0
    if np.any((coverage_counts(idx, rich) == 0) & required):
        raise ValueError("richer vocabulary leaves eligible individual expressions uncovered")
    cols = np.concatenate((np.asarray([leaf], dtype=np.int64), rich))
    require_coverage(idx, cols)
    summary = CoverSummary(
        policy,
        min_nodes,
        min_support,
        len(eligible),
        int(np.count_nonzero(required)),
        int(np.count_nonzero(~required)),
        greedy_cost,
        _total(costs[rich]),
        coverage_report(idx, rich),
        coverage_report(idx, cols),
    )
    return CoverSelection(cols, summary)
