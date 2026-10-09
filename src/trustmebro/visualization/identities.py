"""Distinct DAG, expression and local-pattern identities with scoped memoization."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator, Sequence
from functools import cache
from typing import Literal

import msgspec
import numpy as np
from graph_tool.search import bfs_iterator
from numpy.typing import NDArray

from trustmebro.artifacts import data_digest, encode_msgpack
from trustmebro.extraction import records as r
from trustmebro.graph import Adjacency, ExprGraph, SearchScratch, extract_fragments, prepare_search

from .views import TopoView, _pattern_label, resolve_root


class Var(msgspec.Struct, frozen=True):
    kind: Literal["free", "meta", "level"]
    id: int


type Data = str | int | Var | tuple[Data, ...]


def _root_order(graph: ExprGraph, root: int) -> tuple[list[int], dict[int, int]]:
    """Native BFS discovers each vertex once; convert its destination column in C.

    Only numbering uses the discovery tree. Descriptors retain the full ordered
    adjacency, including parallel edges and references to already-seen nodes.
    """
    tree_edges = bfs_iterator(graph.graph, root, array=True)
    order = [root, *tree_edges[:, 1].tolist()]
    return order, dict(zip(order, range(len(order))))


def shape_sig(graph: ExprGraph, root: int) -> bytes:
    """Constructor-labelled DAG ident for the bounded atlas; scalar data is ignored."""
    order, nums = _root_order(graph, root)
    descr = [(type(graph.exprs[node]).__name__, [nums[child] for child in graph.edges[node]]) for node in order]
    return data_digest(descr)


def topo_sig(graph: ExprGraph, root: int) -> tuple[bytes, int]:
    """Canonical rooted, ordered DAG topology; preserve sharing but erase labels.

    A breadth-first numbering is stable because child slots are ordered. Two
    different leaves stay different vertices even though both are unlabelled.
    """
    if not graph.edges[root]:
        return _leaf_sig("")[0], 1
    order, nums = _root_order(graph, root)
    descr = [[nums[child] for child in graph.edges[node]] for node in order]
    return data_digest(descr), len(order)


def _lvl(lvl: r.Lvl) -> Data:
    match lvl:
        case r.LvlMvar(id=id):
            return Var("level", id)
        case r.LvlZero():
            return ("zero",)
        case r.LvlSucc(level=child):
            return ("succ", _lvl(child))
        case r.LvlMax(left=a, right=b) | r.LvlIMax(left=a, right=b):
            return (type(lvl).__name__, _lvl(a), _lvl(b))
        case r.LvlParam(name=name):
            return ("param", name)


def _data(expr: r.Expr) -> Data:
    """Keep scalar information; child references become structural hashes."""
    match expr:
        case r.Fvar(id=id):
            fields = (Var("free", id),)
        case r.ExprMvar(id=id):
            fields = (Var("meta", id),)
        case r.Sort(lvl=lvl):
            fields = (_lvl(lvl),)
        case r.Const(name=name, universes=universes):
            fields = (name, tuple(_lvl(lvl) for lvl in universes))
        case r.Bvar(idx=idx):
            fields = (idx,)
        case r.Lambda(names=names, binder_info=info) | r.Forall(names=names, binder_info=info):
            fields = (names[0] if len(names) == 1 else names, info.value)
        case r.Let(name=name, nondep=nondep):
            fields = (name, nondep)
        case r.NatLiteral(val=val):
            fields = (val,)
        case r.StringLiteral(val=val):
            fields = (val,)
        case r.Metadata(data=data):
            fields = (data,)
        case r.Proj(type_name=name, idx=idx):
            fields = (name, idx)
        case r.App():
            fields = ()
    return (type(expr).__name__, *fields)


def _vars(data: Data) -> Iterator[Var]:
    if isinstance(data, Var):
        yield data
    elif isinstance(data, tuple):
        for val in data:
            yield from _vars(val)


def _rename(data: Data, names: dict[Var, int]) -> Data:
    if isinstance(data, Var):
        return (data.kind, names[data])
    if isinstance(data, tuple):
        return tuple(_rename(val, names) for val in data)
    return data


class Idents:
    """Root-relative renaming, preserving equality and variable kinds.

    Memo keys include the variable numbering inherited from the parent: reusing
    a child's *standalone* hash would incorrectly equate `f x x` and `f x y`.
    Ground subexpressions have no numbering and are hashed only once.
    """

    def __init__(self, graph: ExprGraph, active: NDArray[np.bool_] | None = None):
        self.graph = graph
        if active is not None and (active.shape != (len(graph.exprs),) or active.dtype != np.bool_):
            raise ValueError("identity population must be a graph-aligned Boolean mask")
        # An optional descendant-closed population avoids preparing discarded
        # auxiliary expressions. Missing children are rejected during hashing.
        self.data = [_data(expr) if active is None or active[node] else None for node, expr in enumerate(graph.exprs)]
        self.vars: list[tuple[Var, ...]] = [()] * len(graph.exprs)
        for node in reversed(graph.order):
            data = self.data[node]
            if data is None:
                continue
            vars = dict.fromkeys(_vars(data))
            for child in graph.edges[node]:
                vars.update(dict.fromkeys(self.vars[child]))
            self.vars[node] = tuple(vars)
        self.cache: dict[tuple[int, tuple[int, ...]], bytes] = {}

    def sig(self, root: int) -> bytes:
        key = (root, tuple(range(len(self.vars[root]))))
        pending = [key]
        while pending:
            current = pending[-1]
            if current in self.cache:
                pending.pop()
                continue
            node, numbering = current
            data = self.data[node]
            if data is None:
                raise ValueError("expression outside the prepared identity population")
            names = dict(zip(self.vars[node], numbering, strict=True))
            children = [(child, tuple(names[var] for var in self.vars[child])) for child in self.graph.edges[node]]
            missing = [child for child in children if child not in self.cache]
            if missing:
                pending.extend(missing)
                continue
            digest = hashlib.sha256(encode_msgpack(_rename(data, names)))
            for child in children:
                digest.update(self.cache[child])
            self.cache[current] = digest.digest()
            pending.pop()
        return self.cache[key]


@cache
def _leaf_sig(label: str) -> tuple[bytes, bytes]:
    topo = [()]
    return _pattern_digest(topo, [label])


def _pattern_digest(topo: object, labels: list[str]) -> tuple[bytes, bytes]:
    """Encode topology once; Raw embeds those exact bytes in the labelled identity."""
    encoded = msgspec.msgpack.encode(topo)
    return hashlib.sha256(encoded).digest(), data_digest((msgspec.Raw(encoded), labels))


def extract_patterns(
    view: TopoView, root: int, radii: Sequence[int], *, scratch: SearchScratch | None = None
) -> dict[int, tuple[bytes, bytes]]:
    """Share native region discovery while retaining local-pattern label identity.

    A consumer preparing several roots supplies one owned search scratch.
    Frontier wildcards and true leaves keep the original distinct identities.
    """
    root = resolve_root(view, root)
    if not radii:
        return {}
    if not view.graph.edges[root]:
        return dict.fromkeys(radii, _leaf_sig(type(view.graph.exprs[root]).__name__))
    graph = view.graph
    adjacency = Adjacency(graph.graph, graph.children, graph.offsets)
    scratch = prepare_search(adjacency) if scratch is None else scratch
    return {
        radius: _pattern_digest(
            list(edges),
            ["*" if refs is None else _pattern_label(view, node) for refs, node in zip(edges, nodes, strict=True)],
        )
        for radius, (edges, nodes) in extract_fragments(adjacency, scratch, root, tuple(radii)).items()
    }
