"""Prepare expression graphs and optional derived views; no corpus I/O or aggregation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from itertools import pairwise
from types import MappingProxyType

import msgspec
import numpy as np
from graph_tool import GraphView
from numpy.typing import NDArray

from trustmebro.extraction import records as r
from trustmebro.graph import AppHeads, ExprGraph, GraphStats, ReachCache, build_graph, reachable_from, reachable_nodes

from .measurements import GraphSize, IntArr, ViewMode

# Graph, view, and observation records.


@dataclass(frozen=True)
class TopoView:
    graph: ExprGraph
    # Source IDs retain each binder's original payload and scope.
    binders: tuple[tuple[int, ...], ...]
    aliases: tuple[int, ...] = ()
    markers: dict[int, Marker] = field(default_factory=dict)


class ViewArrs(msgspec.Struct, frozen=True):
    degrees: IntArr
    binders: IntArr


class SigReuse(msgspec.Struct, frozen=True):
    """Safe inheritance from the preceding view, indexed by resolved source IDs."""

    src: ViewMode
    topo: NDArray[np.bool_]
    labelled: NDArray[np.bool_]


type PatternSigs = dict[int, dict[int, tuple[bytes, bytes]]]


class Observations(msgspec.Struct, frozen=True):
    """Per-node uses: `top` counts root occurrences; `all` counts state DAGs.

    A shared node contributes once per state, unlike frequency `root_dag`,
    which counts once per top-level-root occurrence within that state.
    """

    states: Counter[tuple[int, ...]]
    top: IntArr
    all: IntArr


# Transformation rules and owned scratch data.


class Role(StrEnum):
    TYPE = "type"
    FAMILY = "type-family"
    INST = "instance"
    VAL = "value"


class Kind(StrEnum):
    OPERATOR = "operator"
    NUM = "numeral"
    FN_COE = "function-coercion"
    VAL_COE = "value-coercion"
    TYPE_COE = "type-coercion"
    SET_COE = "set-coercion"
    PROJ = "projection"
    INST_CONSTR = "instance-construction"


class OpName(StrEnum):
    TYPE = "type"
    INST = "instance"
    VAL = "value"
    LHS = "lhs"
    RHS = "rhs"
    SRC_TYPE = "source_type"
    DST_TYPE = "target_type"
    LHS_TYPE = "lhs_type"
    RHS_TYPE = "rhs_type"
    RES_TYPE = "result_type"
    ELEM_TYPE = "element_type"
    CONTAINER_TYPE = "container_type"
    CONTAINER = "container"
    ELEM = "element"
    NUM = "numeral"
    OBJ_TYPE = "object_type"
    DOMAIN = "domain"
    CODOMAIN = "codomain"
    OBJ = "object"
    PRED = "predicate"
    FN = "function"


class Policy(StrEnum):
    INSTS = "instances"
    COMPACT = "coercions-compact"
    ERASED = "coercions-erased"


class NodeDispo(StrEnum):
    RETAINED = "retained"
    ERASED = "erased"


@dataclass(frozen=True, slots=True)
class Slot:
    name: OpName
    role: Role


@dataclass(frozen=True, slots=True)
class FixedOp:
    name: OpName
    val: str


class OpRef(msgspec.Struct, frozen=True):
    name: OpName
    ref: int


@dataclass(frozen=True, slots=True)
class Rule:
    head: str
    kind: Kind
    slots: tuple[Slot, ...]
    # Fixed cast endpoints/numerals have no corresponding expression argument.
    fixed: tuple[FixedOp, ...] = ()


class Match(msgspec.Struct, frozen=True):
    rule: Rule
    root: int
    head: int
    args: tuple[int, ...]
    tail: tuple[int, ...]

    def arg(self, name: OpName) -> int:
        """Look up a semantic operand without depending on its numeric offset."""
        for slot, ref in zip(self.rule.slots, self.args, strict=True):
            if slot.name == name:
                return ref
        raise KeyError(name)


@dataclass(frozen=True, slots=True)
class RuleFamily:
    names: tuple[str, ...]
    kind: Kind
    slots: tuple[Slot, ...]
    fixed: tuple[FixedOp, ...] = ()

    def expand(self) -> tuple[Rule, ...]:
        return tuple(Rule(name, self.kind, self.slots, self.fixed) for name in self.names)


class Marker(msgspec.Struct, frozen=True):
    kind: Kind
    slots: tuple[OpRef, ...]
    fixed: tuple[FixedOp, ...]


@dataclass
class PlumbingState:
    """Mutable graph data owned by one view; transformations receive it explicitly."""

    edges: list[tuple[int, ...]]
    aliases: list[int]
    binders: list[tuple[int, ...]]
    markers: dict[int, Marker] = field(default_factory=dict)
    apps: set[int] = field(default_factory=set)


# Rule registry and processing defaults.


def _rules() -> Mapping[str, Rule]:
    """Declare shared layouts once, then expand aliases into immutable rules."""

    def family(*names: str, kind: Kind, slots: tuple[Slot, ...], fixed: tuple[FixedOp, ...] = ()) -> RuleFamily:
        return RuleFamily(names=names, kind=kind, slots=slots, fixed=fixed)

    type_ = Slot(OpName.TYPE, Role.TYPE)
    inst = Slot(OpName.INST, Role.INST)
    val = Slot(OpName.VAL, Role.VAL)
    binary = (Slot(OpName.LHS, Role.VAL), Slot(OpName.RHS, Role.VAL))
    coe = (Slot(OpName.SRC_TYPE, Role.TYPE), Slot(OpName.DST_TYPE, Role.TYPE), inst, val)
    het_binary = (
        Slot(OpName.LHS_TYPE, Role.TYPE),
        Slot(OpName.RHS_TYPE, Role.TYPE),
        Slot(OpName.RES_TYPE, Role.TYPE),
        inst,
        *binary,
    )
    # Elaborated membership places the container before the element
    membership = (
        Slot(OpName.ELEM_TYPE, Role.TYPE),
        Slot(OpName.CONTAINER_TYPE, Role.TYPE),
        inst,
        Slot(OpName.CONTAINER, Role.VAL),
        Slot(OpName.ELEM, Role.VAL),
    )
    # The fifth operand is the object; trailing operands are its call args
    dependent_fn_coe = (
        Slot(OpName.OBJ_TYPE, Role.TYPE),
        Slot(OpName.DOMAIN, Role.TYPE),
        Slot(OpName.CODOMAIN, Role.FAMILY),
        inst,
        Slot(OpName.OBJ, Role.VAL),
    )
    fn_coe = (type_, Slot(OpName.CODOMAIN, Role.FAMILY), inst, Slot(OpName.OBJ, Role.VAL))
    set_coe = (Slot(OpName.OBJ_TYPE, Role.TYPE), Slot(OpName.ELEM_TYPE, Role.TYPE), inst, Slot(OpName.OBJ, Role.VAL))
    subtype_proj = (type_, Slot(OpName.PRED, Role.FAMILY), Slot(OpName.OBJ, Role.VAL))
    cast = (Slot(OpName.DST_TYPE, Role.TYPE), inst, val)

    families: list[RuleFamily] = [
        # Operators
        family(
            "HAdd.hAdd",
            "HMul.hMul",
            "HSub.hSub",
            "HDiv.hDiv",
            "HPow.hPow",
            "HSMul.hSMul",
            kind=Kind.OPERATOR,
            slots=het_binary,
        ),
        family("Neg.neg", "Inv.inv", kind=Kind.OPERATOR, slots=(type_, inst, val)),
        family("LE.le", "LT.lt", kind=Kind.OPERATOR, slots=(type_, inst, *binary)),
        family("Membership.mem", kind=Kind.OPERATOR, slots=membership),
        # Numerals
        family("OfNat.ofNat", kind=Kind.NUM, slots=(type_, Slot(OpName.NUM, Role.VAL), inst)),
        family("Zero.zero", kind=Kind.NUM, slots=(type_, inst), fixed=(FixedOp(OpName.NUM, "0"),)),
        family("One.one", kind=Kind.NUM, slots=(type_, inst), fixed=(FixedOp(OpName.NUM, "1"),)),
        # Coercions and projections
        family("DFunLike.coe", kind=Kind.FN_COE, slots=dependent_fn_coe),
        family("CoeFun.coe", kind=Kind.FN_COE, slots=fn_coe),
        family("Coe.coe", "CoeTC.coe", "CoeHTCT.coe", kind=Kind.VAL_COE, slots=coe),
        family("CoeSort.coe", kind=Kind.TYPE_COE, slots=coe),
        family("SetLike.coe", kind=Kind.SET_COE, slots=set_coe),
        family("Subtype.val", kind=Kind.PROJ, slots=subtype_proj),
    ]
    families.extend(
        family(
            f"{src}.cast",
            f"{src}Cast.{src.lower()}Cast",
            kind=Kind.VAL_COE,
            slots=cast,
            fixed=(FixedOp(OpName.SRC_TYPE, src),),
        )
        for src in ("Nat", "Int")
    )
    families.extend(
        [
            family(
                "Int.ofNat",
                kind=Kind.VAL_COE,
                slots=(val,),
                fixed=(FixedOp(OpName.SRC_TYPE, "Nat"), FixedOp(OpName.DST_TYPE, "Int")),
            ),
            # Only these individually checked two-slot instance layouts are eligible
            family(
                "CommSemiring.toSemiring",
                "PartialOrder.toPreorder",
                "AddCommGroup.toAddCommMonoid",
                "Semiring.toNonAssocSemiring",
                "Preorder.toLE",
                "CommRing.toCommSemiring",
                "Zero.toOfNat0",
                "One.toOfNat1",
                "instHMul",
                "instHAdd",
                kind=Kind.INST_CONSTR,
                slots=(type_, inst),
            ),
        ]
    )
    expanded = tuple(rule for family in families for rule in family.expand())
    names = Counter(rule.head for rule in expanded)
    if dupes := [name for name, count in names.items() if count > 1]:
        raise ValueError(f"duplicate plumbing rules: {dupes}")
    return MappingProxyType({rule.head: rule for rule in expanded})


RULES = _rules()
POLICIES = tuple(Policy)


@dataclass(frozen=True)
class ProcessCfg:
    reg: Mapping[str, Rule] = field(default_factory=_rules)
    policies: tuple[Policy, ...] = POLICIES


DEFAULT_PROCESSING = ProcessCfg(RULES)


# Original graph preparation and root measurements.


# Derived-view measurements.


def view_arrs(view: TopoView) -> ViewArrs:
    """Prepare only for topology-size consumers, not patterns or original-state counts."""
    return ViewArrs(
        view.graph.graph.get_out_degrees(np.arange(len(view.graph.exprs))),
        np.fromiter((len(group) for group in view.binders), dtype=np.int64),
    )


def resolve_root(view: TopoView, root: int) -> int:
    return view.aliases[root] if view.aliases else root


def view_heads(view: TopoView) -> tuple[str, ...]:
    """Resolve heads once, respecting aliases and conversion boundaries."""
    graph = view.graph
    names = [""] * len(graph.exprs)
    for node in reversed(graph.order):
        expr = graph.exprs[node]
        if (marker := view.markers.get(node)) is not None:
            names[node] = f"<{marker.kind}>"
        elif isinstance(expr, r.Metadata):
            names[node] = names[resolve_root(view, expr.expr)]
        elif isinstance(expr, r.App) and graph.edges[node]:
            names[node] = names[graph.edges[node][0]]
        else:
            names[node] = expr.name if isinstance(expr, r.Const) else f"<{type(expr).__name__}>"
    return tuple(names)


def measure_view(
    view: TopoView, stats: GraphStats, cache: ReachCache, arrays: ViewArrs, roots: Sequence[int]
) -> GraphSize:
    """Union nodes/refs, but recount each root for the implicit expanded tree."""
    if not roots:
        raise ValueError("a topology measurement needs at least one root")
    roots = [resolve_root(view, root) for root in roots]
    nodes = reachable_nodes(cache, roots)
    return GraphSize(
        len(nodes),
        int(arrays.degrees[nodes].sum()),
        int(arrays.binders[nodes].sum()),
        max(stats.depths[root] for root in roots),
        sum(stats.sizes[root] for root in roots),
        int(arrays.degrees[nodes].max()) if len(nodes) else 0,
    )


# Application lookup and view transformations.


def app_spine(exprs: Sequence[r.Expr], root: int) -> tuple[int, tuple[int, ...]]:
    """Read an ordered application spine, ignoring metadata only for lookup.

    References still point into the unchanged raw table. Metadata is not deleted
    from that table, and callers retain the original root for provenance.
    """
    groups: list[tuple[int, ...]] = []
    head = root
    while True:
        match exprs[head]:
            case r.App(fn=fn, args=args):
                groups.append(args)
                head = fn
            case r.Metadata(expr=ref):
                head = ref
            case _:
                return head, tuple(ref for group in reversed(groups) for ref in group)


def app_heads(graph: ExprGraph) -> AppHeads:
    """One DAG pass resolves heads/arity, without materializing every spine."""
    refs = np.arange(len(graph.exprs), dtype=np.int64)
    arities = np.zeros(len(refs), dtype=np.int64)
    for node in reversed(graph.order):
        match graph.exprs[node]:
            case r.App(fn=fn, args=args):
                refs[node], arities[node] = refs[fn], arities[fn] + len(args)
            case r.Metadata(expr=expr):
                refs[node], arities[node] = refs[expr], arities[expr]
    return AppHeads(refs, arities)


def match_rule(
    exprs: Sequence[r.Expr],
    root: int,
    *,
    reg: Mapping[str, Rule] = RULES,
    inst_ctxt: bool = False,
    spine: tuple[int, tuple[int, ...]] | None = None,
) -> Match | None:
    """Match exact named heads; leave unknown/partially applied heads alone.

    Oversaturated heads retain all trailing arguments. Instance construction
    rules are diagnostic and only eligible within a verified instance slot.
    """
    head, args = app_spine(exprs, root) if spine is None else spine
    expr = exprs[head]
    if not isinstance(expr, r.Const):
        return None
    rule = reg.get(expr.name)
    if rule is None or (rule.kind == Kind.INST_CONSTR and not inst_ctxt):
        return None
    arity = len(rule.slots)
    if len(args) < arity:
        return None
    return Match(rule, root, head, args[:arity], args[arity:])


def find_matches(exprs: tuple[r.Expr, ...], heads: AppHeads, reg: Mapping[str, Rule] = RULES) -> dict[int, Match]:
    matches: dict[int, Match] = {}
    for node, expr in enumerate(exprs):
        head = exprs[heads.refs[node]]
        rule = reg.get(head.name) if isinstance(head, r.Const) else None
        if (
            isinstance(expr, r.App)
            and rule is not None
            and rule.kind != Kind.INST_CONSTR
            and heads.arities[node] >= len(rule.slots)
        ):
            matched = match_rule(exprs, node, reg=reg)
            if matched is not None:
                matches[node] = matched
    return matches


def _initial_state(base: TopoView) -> PlumbingState:
    return PlumbingState(
        edges=base.graph.edges.copy(), aliases=list(range(len(base.graph.exprs))), binders=list(base.binders)
    )


def _apply_rule(state: PlumbingState, node: int, match: Match | None, policy: Policy) -> NodeDispo:
    if match is None:
        return NodeDispo.RETAINED
    conv = match.rule.kind in (Kind.FN_COE, Kind.VAL_COE, Kind.TYPE_COE, Kind.SET_COE, Kind.PROJ)
    if conv and policy == Policy.ERASED:
        name = OpName.OBJ if match.rule.kind in (Kind.FN_COE, Kind.SET_COE, Kind.PROJ) else OpName.VAL
        val = match.arg(name)
        if not match.tail:
            state.aliases[node] = state.aliases[val]
            state.edges[node] = ()  # No unary proxy remains reachable.
            state.binders[node] = ()
            return NodeDispo.ERASED
        state.edges[node] = (val, *match.tail)
        state.markers[node] = Marker(Kind.OPERATOR, (OpRef(OpName.FN, state.aliases[val]),), ())
        return NodeDispo.RETAINED

    retained = tuple(
        OpRef(slot.name, ref) for slot, ref in zip(match.rule.slots, match.args, strict=True) if slot.role != Role.INST
    )
    operands = tuple(op.ref for op in retained)
    if conv and policy == Policy.COMPACT:
        state.edges[node] = (*operands, *match.tail)
        state.markers[node] = Marker(
            match.rule.kind, tuple(OpRef(op.name, state.aliases[op.ref]) for op in retained), match.rule.fixed
        )
    else:
        state.edges[node] = (match.head, *operands, *match.tail)
    return NodeDispo.RETAINED


def _redirect_children(state: PlumbingState, node: int, exprs: Sequence[r.Expr], policy: Policy) -> None:
    # Children are already processed; binders/lets need redirects too.
    state.edges[node] = tuple(state.aliases[ref] for ref in state.edges[node])
    expr = exprs[node]
    if isinstance(expr, r.Metadata) and policy == Policy.ERASED and state.aliases[expr.expr] != expr.expr:
        state.aliases[node] = state.aliases[expr.expr]
        state.edges[node] = ()


def _flatten_app(state: PlumbingState, node: int, exprs: Sequence[r.Expr]) -> None:
    marker = state.markers.get(node)
    is_app = isinstance(exprs[node], r.App) and (marker is None or marker.kind == Kind.OPERATOR)
    if not is_app or not state.edges[node]:
        return
    fn, *args = state.edges[node]
    # Erasure can expose a new spine. Conversion markers are not applications:
    # their first edge describes a type/value role, not a function.
    if fn in state.apps:
        state.edges[node] = (*state.edges[fn], *args)
    state.apps.add(node)


def _finish_view(base: TopoView, state: PlumbingState) -> TopoView:
    return TopoView(
        build_graph(base.graph.exprs, edges=state.edges),
        tuple(state.binders),
        tuple(state.aliases),
        markers=state.markers,
    )


def _plumbing_view(base: TopoView, policy: Policy, matches: Mapping[int, Match]) -> TopoView:
    state = _initial_state(base)
    for node in reversed(base.graph.order):
        if _apply_rule(state, node, matches.get(node), policy) == NodeDispo.ERASED:
            continue
        _redirect_children(state, node, base.graph.exprs, policy)
        _flatten_app(state, node, base.graph.exprs)
    return _finish_view(base, state)


def prepare_views(
    graph: ExprGraph, modes: Sequence[ViewMode], *, matches: Mapping[int, Match], cfg: ProcessCfg = DEFAULT_PROCESSING
) -> dict[ViewMode, TopoView]:
    """Build selected views and shared prerequisites once; no measurements or observations."""
    if not modes:
        return {}
    binders = tuple(
        (node,) * len(expr.names) if isinstance(expr, (r.Forall, r.Lambda)) else ()
        for node, expr in enumerate(graph.exprs)
    )
    base = TopoView(graph, binders)
    views = {ViewMode.ORIGINAL: base}
    for mode in modes:
        if mode == ViewMode.ORIGINAL:
            continue
        policy = Policy(mode)
        if policy not in cfg.policies:
            raise ValueError(f"disabled plumbing policy: {policy}")
        views[mode] = _plumbing_view(base, policy, matches)
    return {mode: views[mode] for mode in modes}


def expr_views(graph: ExprGraph, *, cfg: ProcessCfg = DEFAULT_PROCESSING) -> tuple[TopoView, ...]:
    modes = (ViewMode.ORIGINAL, *(ViewMode(policy) for policy in cfg.policies))
    matches = find_matches(graph.exprs, app_heads(graph), cfg.reg) if cfg.policies else {}
    return tuple(prepare_views(graph, modes, matches=matches, cfg=cfg).values())


# Observed occurrences and affected nodes.


def observe_roots(cache: ReachCache, trns: Sequence[r.Trn]) -> Observations:
    states = Counter((trn.state.target, *(local.type for local in trn.state.locals)) for trn in trns)
    top = np.zeros(len(cache.graph.exprs), dtype=np.int64)
    all_ = np.zeros_like(top)
    for roots, weight in states.items():
        np.add.at(top, np.asarray(roots, dtype=np.int64), weight)
        all_[reachable_nodes(cache, roots)] += weight
    return Observations(states, top, all_)


def affected_nodes(graph: ExprGraph, matches: Mapping[int, Match]) -> NDArray[np.bool_]:
    conv_kinds = (Kind.FN_COE, Kind.VAL_COE, Kind.TYPE_COE, Kind.SET_COE, Kind.PROJ)
    roots = [node for node, matched in matches.items() if matched.rule.kind in conv_kinds]
    return reachable_from(GraphView(graph.graph, reversed=True), roots)


def observed_heads(exprs: Sequence[r.Expr], heads: AppHeads, observations: Observations) -> dict[int, str]:
    result: dict[int, str] = {}
    for root in np.flatnonzero(observations.all):
        if isinstance(exprs[int(root)], r.App):
            expr = exprs[int(heads.refs[root])]
            result[int(root)] = expr.name if isinstance(expr, r.Const) else f"<{type(expr).__name__}>"
    return result


# Graph and expression identities.


def sig_reuse(views: Mapping[ViewMode, TopoView]) -> dict[ViewMode, SigReuse]:
    """Propagate changed edges/labels to ancestors once, not per signature root.

    Unchanged rooted regions retain identical ordered adjacency and node labels.
    Root aliases are resolved by consumers before lookup; measurement/binder
    caches are deliberately not shared by this identity-only preparation.
    """
    result: dict[ViewMode, SigReuse] = {}
    for (src, base), (mode, view) in pairwise(views.items()):
        changed = np.fromiter((a != b for a, b in zip(base.graph.edges, view.graph.edges, strict=True)), dtype=bool)
        reverse = GraphView(view.graph.graph, reversed=True)
        topo = ~reachable_from(reverse, np.flatnonzero(changed).tolist())
        # Constructors are shared by the source table. Only non-operator markers
        # change the labels used by pattern identities.
        labels = [
            node
            for node in base.markers.keys() | view.markers.keys()
            if _pattern_label(base, node) != _pattern_label(view, node)
        ]
        labelled = topo & ~reachable_from(reverse, labels) if labels else topo
        result[mode] = SigReuse(src, topo, labelled)
    return result


def _pattern_label(view: TopoView, node: int) -> str:
    marker = view.markers.get(node)
    return str(marker.kind) if marker and marker.kind != Kind.OPERATOR else type(view.graph.exprs[node]).__name__
