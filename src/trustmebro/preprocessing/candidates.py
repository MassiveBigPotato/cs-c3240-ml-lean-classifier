"""Discover ordered, radius-bounded DAG fragments without selecting a vocabulary.

graph_tool owns reachability and bounded discovery. Numeric adjacency supplies
all operand edges, including those absent from a BFS discovery tree. The JIT
kernel only canonicalizes the discovered region; it does not search the graph.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from graph_tool.topology import label_out_component
from numpy.typing import NDArray

from trustmebro.artifacts import data_digest
from trustmebro.extraction import records as r
from trustmebro.graph import Adjacency, extract_fragments, prepare_adjacency, prepare_search
from trustmebro.preprocessing.records import (
    DEFAULT_DEPTHS,
    Cands,
    Edges,
    LocalInfo,
    Node,
    Occurrence,
    Root,
    Shape,
    State,
)

type IntArray = NDArray[np.int64]
type CountArray = NDArray[np.int64] | NDArray[np.object_]


def observe_roots(
    trns: tuple[r.Trn, ...], adjacency: Adjacency
) -> tuple[tuple[State, ...], tuple[Root, ...], CountArray]:
    states = tuple(
        State(
            step,
            trn.tactic,
            trn.state.target,
            tuple(local.type for local in trn.state.locals),
            tuple(LocalInfo(isinstance(local, r.LocalLet), local.is_instance) for local in trn.state.locals),
        )
        for step, trn in enumerate(trns)
    )
    goals = Counter(state.goal for state in states)
    hyps = Counter(ref for state in states for ref in state.hyps)
    # Each node is visited at most once per top-level occurrence. This bound
    # permits native batched sums without risking fixed-width overflow.
    upper_bound = max(sum(goals.values()), sum(hyps.values()))
    dtype = np.int64 if upper_bound <= np.iinfo(np.int64).max else object
    counts: CountArray = np.zeros((adjacency.graph.num_vertices(), 2), dtype=dtype)
    roots: list[Root] = []
    for ref in sorted(goals.keys() | hyps.keys()):
        mask = label_out_component(adjacency.graph, adjacency.graph.vertex(ref))
        anchors = np.flatnonzero(mask.a)
        roots.append(Root(ref, tuple(map(int, anchors))))
        counts[anchors] += (goals[ref], hyps[ref])
    return states, tuple(roots), counts


# Native bounded discovery, followed by compiled ordered canonicalization.


def extract_cands(theorem: r.Theorem, depths: tuple[int, ...] = DEFAULT_DEPTHS, *, validated: bool = False) -> Cands:
    """Extract every observed anchor; retain all candidates without filtering.

    validated=True is only for records already checked by validate_theorem;
    structural preparation still runs once here, irrespective of validation.
    """
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("candidate depths must be positive and nonempty")
    depths = tuple(sorted(set(depths)))
    if not validated:
        r.validate_theorem(theorem)
    adjacency = prepare_adjacency(theorem.exprs)
    scratch = prepare_search(adjacency)
    states, roots, counts = observe_roots(theorem.trns, adjacency)
    observed = np.flatnonzero(np.any(counts != 0, axis=1))
    nodes = tuple(Node(int(ref), theorem.exprs[ref], int(counts[ref, 0]), int(counts[ref, 1])) for ref in observed)
    shapes: list[Shape] = []
    shape_ids: dict[Edges, int] = {}
    ident_shapes: dict[bytes, Edges] = {}
    occs: list[Occurrence] = []
    for node in nodes:
        for depth, (edges, refs) in extract_fragments(adjacency, scratch, node.ref, depths).items():
            shape = shape_ids.get(edges)
            if shape is None:
                # Encode/hash each distinct theorem-local shape once, including
                # leaves and complete shapes repeated at several query depths.
                ident = data_digest(edges)
                previous = ident_shapes.get(ident)
                if previous is not None and previous != edges:
                    raise ValueError("conflicting fragments for the same shape identity")
                shape = len(shapes)
                shape_ids[edges] = shape
                ident_shapes[ident] = edges
                shapes.append(Shape(ident, edges))
            occs.append(Occurrence(shape, depth, refs))
    return Cands(theorem.name, nodes, tuple(shapes), tuple(occs), roots, states)
