"""Prepare expression graphs and optional derived views; no corpus I/O or aggregation."""

from __future__ import annotations

import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import msgspec
import numpy as np
from graph_tool import Graph, VertexPropertyMap
from graph_tool.topology import label_out_component, shortest_distance, topological_sort
from numba import njit
from numpy.typing import NDArray

from trustmebro.extraction import records as r
from trustmebro.preprocessing.records import Edges

type IntArray = NDArray[np.int64]

DEFAULT_REACH_BUDGET = 64 * 1024 * 1024


class _ReachedSearch(Protocol):
    """graph_tool's fixed return_reached form, including reusable property maps."""

    def __call__(
        self,
        graph: Graph,
        *,
        source: int,
        max_dist: int,
        return_reached: bool,
        dist_map: VertexPropertyMap,
        pred_map: VertexPropertyMap,
    ) -> tuple[Any, VertexPropertyMap, IntArray]: ...


class RootMeasure(msgspec.Struct, frozen=True):
    """Cached per-root measurements use compact, native-constructed records."""

    expanded: int
    distinct: int
    depth: int
    constrs: dict[str, int]


@dataclass(frozen=True)
class ExprGraph:
    """Ordered structural DAG; expression IDs remain those of the raw table."""

    exprs: tuple[r.Expr, ...]
    edges: list[tuple[int, ...]]
    graph: Graph
    order: tuple[int, ...]
    original: bool
    children: IntArray
    offsets: IntArray


@dataclass(frozen=True)
class GraphStats:
    """Exact implicit-tree sizes and depths, aligned to expression IDs."""

    sizes: list[int]
    depths: list[int]


@dataclass
class ReachBits:
    """LRU of retained integer bitmaps, shared only by simultaneously live graphs.

    The budget counts integer storage, not graphs, dictionary overhead, root
    measurements, temporary native masks, or total worker memory.
    """

    budget: int = DEFAULT_REACH_BUDGET
    used: int = 0
    entries: OrderedDict[tuple[int, int], int] = field(default_factory=OrderedDict)

    def get(self, graph: Graph, root: int) -> int:
        if self.budget < 0:
            raise ValueError("reachability bitmap budget must be nonnegative")
        key = (id(graph), root)
        if key in self.entries:
            self.entries.move_to_end(key)
            return self.entries[key]
        reachable = label_out_component(graph, graph.vertex(root))
        bits = int.from_bytes(np.packbits(reachable.a, bitorder="little").tobytes(), "little")
        size = sys.getsizeof(bits)
        if size <= self.budget:
            while self.used + size > self.budget:
                _, previous = self.entries.popitem(last=False)
                self.used -= sys.getsizeof(previous)
            self.entries[key] = bits
            self.used += size
        return bits


@dataclass
class ReachCache:
    """Scratch owned by one immutable graph, never shared between derived views."""

    graph: ExprGraph
    bitmaps: ReachBits = field(default_factory=ReachBits)
    measures: dict[tuple[int, ...], RootMeasure] = field(default_factory=dict)


@dataclass(frozen=True)
class ExprArrs:
    """Constructor-specific arrays for the exported expression graph."""

    children: IntArray
    offsets: IntArray
    names: tuple[str, ...]
    kinds: IntArray
    body_slots: IntArray
    binder_counts: IntArray
    var_bins: IntArray
    spines: IntArray
    ranks: IntArray


class AppHeads(msgspec.Struct, frozen=True):
    refs: IntArray
    arities: IntArray


class RootGraph(msgspec.Struct, frozen=True):
    """Root-local children/order/indegrees; nodes maps back to theorem IDs."""

    nodes: IntArray
    children: IntArray
    offsets: IntArray
    order: IntArray
    incoming: IntArray


def bitmap_nodes(bits: int) -> IntArray:
    """Decode only the occupied prefix, not a theorem-wide scratch array."""
    packed = bits.to_bytes((bits.bit_length() + 7) // 8, "little")
    return np.flatnonzero(np.unpackbits(np.frombuffer(packed, dtype=np.uint8), bitorder="little"))


def reachable_from(graph: Graph, roots: Sequence[int]) -> NDArray[np.bool_]:
    """Accumulate native root closures without a graph copy or Python bit shifts.

    A previously reached seed can be skipped because its descendants were
    included by the completed traversal that reached it. Return owned storage.
    """
    reached = graph.new_vertex_property("bool")
    for root in roots:
        if not reached[root]:
            label_out_component(graph, graph.vertex(root), label=reached)
    # graph_tool exposes Boolean properties as uint8; indexing needs bool dtype.
    return reached.a.astype(np.bool_, copy=True)


def build_graph(exprs: tuple[r.Expr, ...], *, edges: list[tuple[int, ...]] | None = None) -> ExprGraph:
    """Build and validate native topology, without preparing any measurements."""
    original = edges is None
    edges = [r.expr_refs(expr) for expr in exprs] if edges is None else edges
    adjacency = prepare_adjacency(exprs, edges=edges)
    # The exporter supplies post-order tables. Check that precondition before
    # using the cheap recurrence order; derived aliases can violate it.
    if original:
        if any(child >= node for node, refs in enumerate(edges) for child in refs):
            raise ValueError("original expression table is not a post-order DAG")
        order = tuple(range(len(exprs) - 1, -1, -1))
    else:
        order = tuple(int(node) for node in topological_sort(adjacency.graph))
    return ExprGraph(exprs, edges, adjacency.graph, order, original, adjacency.children, adjacency.offsets)


def graph_stats(graph: ExprGraph) -> GraphStats:
    """One bottom-up pass; expanded sizes may exceed native integer widths."""
    sizes, depths = [0] * len(graph.exprs), [0] * len(graph.exprs)
    for node in reversed(graph.order):
        sizes[node] = 1 + sum(sizes[child] for child in graph.edges[node])
        depths[node] = 1 + max((depths[child] for child in graph.edges[node]), default=0)
    return GraphStats(sizes, depths)


def constructor_masks(exprs: Sequence[r.Expr]) -> dict[str, int]:
    names = np.asarray([type(expr).__name__ for expr in exprs])
    return {
        str(kind): int.from_bytes(np.packbits(names == kind, bitorder="little").tobytes(), "little")
        for kind in sorted(set(names))
    }


def expr_arrays(graph: ExprGraph) -> ExprArrs:
    """Prepare descriptors once, only for theorems with selected structural roots."""
    if not graph.original:
        raise ValueError("constructor-specific descriptors require the original expression graph")
    # Ordered CSR avoids a theorem-size × maximum-arity padded matrix.
    children, offsets = graph.children, graph.offsets
    body_slots = np.full(len(graph.exprs), -1, dtype=np.int64)
    binder_counts = np.zeros(len(graph.exprs), dtype=np.int64)
    var_bins = np.full(len(graph.exprs), -1, dtype=np.int64)
    spines = np.zeros(len(graph.exprs), dtype=np.int64)
    ranks = np.empty(len(graph.exprs), dtype=np.int64)
    ranks[np.asarray(graph.order, dtype=np.int64)] = np.arange(len(graph.exprs))
    for node in reversed(graph.order):
        expr = graph.exprs[node]
        if isinstance(expr, (r.Lambda, r.Forall)):
            body_slots[node] = 1
            binder_counts[node] = len(expr.names)
        elif isinstance(expr, r.Let):
            body_slots[node] = 2
            binder_counts[node] = 1
        elif isinstance(expr, r.Bvar):
            var_bins[node] = min(4, expr.idx.bit_length())
        elif isinstance(expr, r.App):
            spines[node] = len(expr.args) + spines[expr.fn]
    names, codes = np.unique([type(expr).__name__ for expr in graph.exprs], return_inverse=True)
    return ExprArrs(
        children, offsets, tuple(str(name) for name in names), codes, body_slots, binder_counts, var_bins, spines, ranks
    )


def root_bitmap(cache: ReachCache, root: int) -> int:
    return cache.bitmaps.get(cache.graph.graph, root)


def root_union(cache: ReachCache, roots: Sequence[int]) -> int:
    bits = 0
    for root in roots:
        bits |= root_bitmap(cache, root)
    return bits


def reachable_nodes(cache: ReachCache, roots: Sequence[int]) -> IntArray:
    return bitmap_nodes(root_union(cache, roots))


def root_graph(cache: ReachCache, arrays: ExprArrs, root: int) -> RootGraph:
    nodes = reachable_nodes(cache, (root,))
    degrees = np.diff(arrays.offsets)[nodes]
    offsets = np.concatenate(([0], degrees.cumsum()))
    positions = np.repeat(arrays.offsets[nodes] - offsets[:-1], degrees) + np.arange(offsets[-1])
    children = np.searchsorted(nodes, arrays.children[positions])
    incoming = np.bincount(children, minlength=len(nodes))
    return RootGraph(nodes, children, offsets, np.argsort(arrays.ranks[nodes]), incoming)


def measure_roots(stats: GraphStats, masks: Mapping[str, int], cache: ReachCache, roots: Sequence[int]) -> RootMeasure:
    key = tuple(roots)
    if key not in cache.measures:
        bits = root_union(cache, roots)
        cache.measures[key] = RootMeasure(
            sum(stats.sizes[node] for node in roots),
            bits.bit_count(),
            max((stats.depths[node] for node in roots), default=0),
            {kind: (bits & mask).bit_count() for kind, mask in masks.items() if bits & mask},
        )
    return cache.measures[key]


@dataclass(frozen=True)
class Adjacency:
    graph: Graph
    children: IntArray
    offsets: IntArray


@dataclass(frozen=True)
class SearchScratch:
    dist: VertexPropertyMap
    pred: VertexPropertyMap
    positions: IntArray


# Preparation occurs once per theorem, not once per state or fragment.


def prepare_adjacency(exprs: tuple[r.Expr, ...], *, edges: list[tuple[int, ...]] | None = None) -> Adjacency:
    """Prepare validated post-order records; their references already ensure a DAG."""
    operands = [r.expr_refs(expr) for expr in exprs] if edges is None else edges
    counts = np.fromiter(map(len, operands), dtype=np.int64, count=len(exprs))
    offsets = np.empty(len(exprs) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    refs = np.fromiter((ref for items in operands for ref in items), dtype=np.int64, count=int(offsets[-1]))
    if refs.size and (refs.min() < 0 or refs.max() >= len(exprs)):
        raise ValueError("invalid expression operand reference")
    graph = Graph(directed=True)
    graph.add_vertex(len(exprs))
    if refs.size:
        graph.add_edge_list(np.column_stack((np.repeat(np.arange(len(exprs)), counts), refs)))
    return Adjacency(graph, refs, offsets)


def prepare_search(adjacency: Adjacency) -> SearchScratch:
    graph = adjacency.graph
    # Supply our own predecessor map to avoid the wrapper copying vertex_index
    # on every call. Both maps belong to this theorem and remain worker-local.
    return SearchScratch(
        graph.new_vp("int32_t"),
        graph.vertex_index.copy(value_type="int64_t"),
        np.full(graph.num_vertices(), -1, dtype=np.int64),
    )


@njit(cache=True)
def _canonical_region(
    children: IntArray,
    offsets: IntArray,
    dists: NDArray[np.int32],
    positions: IntArray,
    root: int,
    radius: int,
    count: int,
) -> tuple[IntArray, IntArray, IntArray, IntArray]:
    """Serialize already discovered vertices in operand-ordered BFS numbering.

    Each distinct node has one local position; repeated edges retain repeated
    references. Scratch positions are reset only for vertices touched here.
    """
    nodes = np.empty(count, np.int64)
    depths = np.empty(count, np.int64)
    local_offsets = np.empty(count + 1, np.int64)
    nodes[0], depths[0], positions[root] = root, 0, 0
    end = 1
    edge_count = 0
    for idx in range(count):
        node = nodes[idx]
        local_offsets[idx] = edge_count
        if depths[idx] >= radius:
            continue
        for slot in range(offsets[node], offsets[node + 1]):
            child = children[slot]
            if positions[child] == -1:
                positions[child] = end
                nodes[end] = child
                depths[end] = dists[child]
                end += 1
            edge_count += 1
    local_offsets[count] = edge_count
    refs = np.empty(edge_count, np.int64)
    for idx in range(count):
        node = nodes[idx]
        if depths[idx] < radius:
            start = local_offsets[idx]
            for slot in range(offsets[node], offsets[node + 1]):
                refs[start + slot - offsets[node]] = positions[children[slot]]
    for node in nodes:
        positions[node] = -1
    return nodes, depths, refs, local_offsets


def extract_fragments(
    adjacency: Adjacency, scratch: SearchScratch, root: int, depths: tuple[int, ...]
) -> dict[int, tuple[Edges, tuple[int, ...]]]:
    """One native search to the largest radius; smaller radii reuse its prefix."""
    radius = max(depths)
    if adjacency.offsets[root] == adjacency.offsets[root + 1]:
        return dict.fromkeys(depths, (((),), (root,)))
    _, _, reached = cast(_ReachedSearch, shortest_distance)(
        adjacency.graph, source=root, max_dist=radius, return_reached=True, dist_map=scratch.dist, pred_map=scratch.pred
    )
    # graph_tool's reached array omits the source. Its order is not our identity.
    nodes, distances, refs, offsets = _canonical_region(
        adjacency.children, adjacency.offsets, scratch.dist.a, scratch.positions, root, radius, len(reached) + 1
    )
    scratch.pred.a[reached] = reached
    scratch.pred.a[root] = root
    frags: dict[int, tuple[Edges, tuple[int, ...]]] = {}
    for depth in depths:
        end = int(np.searchsorted(distances, depth, side="right"))
        edges: Edges = tuple(
            None
            if distances[idx] == depth and adjacency.offsets[node] != adjacency.offsets[node + 1]
            else tuple(map(int, refs[offsets[idx] : offsets[idx + 1]]))
            for idx, node in enumerate(nodes[:end])
        )
        frags[depth] = edges, tuple(map(int, nodes[:end]))
    return frags
