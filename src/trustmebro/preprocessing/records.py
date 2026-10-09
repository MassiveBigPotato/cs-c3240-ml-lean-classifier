"""Frozen preparation contracts; no discovery, graph search or fitting imports."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import msgspec
import numpy as np
from scipy.sparse import csr_array

from trustmebro.extraction import records as r

type Edges = tuple[tuple[int, ...] | None, ...]
DEFAULT_DEPTHS = (1, 2, 3)


# Passive archive records. Original references remain theorem-local.


class Selection(msgspec.Struct, frozen=True):
    db: str
    size: int
    modified_ns: int
    depths: tuple[int, ...]
    limit: int | None
    theorems: tuple[str, ...] | None
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class MinSupport:
    trns: int = 1
    theorems: int = 1


@dataclass(frozen=True, slots=True)
class SplitCfg:
    test_frac: float = 0.2
    seed: int = 0
    train_min: MinSupport = MinSupport()
    test_min: MinSupport = MinSupport()
    solver_sec: float = 30

    def __post_init__(self) -> None:
        if not 0 < self.test_frac < 1 or self.seed < 0:
            raise ValueError("test fraction must be between 0 and 1 and seed must be nonnegative")
        if not np.isfinite(self.solver_sec) or self.solver_sec <= 0:
            raise ValueError("solver time limit must be finite and positive")
        for min_ in (self.train_min, self.test_min):
            if any(type(val) is not int or not 1 <= val <= 2**53 for val in (min_.trns, min_.theorems)):
                raise ValueError("label minima must be positive integers representable exactly by the solver")


DEFAULT_SPLIT_CFG = SplitCfg()
DEFAULT_VALIDATION_CFG = SplitCfg(test_frac=0.1)


@dataclass(frozen=True, slots=True)
class LabelSupport:
    trns: int
    theorems: int


@dataclass(frozen=True, slots=True)
class SubsetStats:
    theorems: int
    trns: int
    labeled_trns: int
    dropped_trns: int
    labels: dict[str, LabelSupport]


class SplitManifest(msgspec.Struct, frozen=True):
    """Logical roles; no SQLite rows or feature columns are copied."""

    selection: Selection
    policy: LabelPolicy
    cfg: SplitCfg
    instance: int
    folds: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    train_stats: SubsetStats
    validation_stats: SubsetStats
    repair: str
    solver_optimal: bool | None


def split_id(split: SplitManifest) -> str:
    # Canonical typed round-trip also normalizes int inputs to float fields.
    normalized = msgspec.json.decode(msgspec.json.encode(split), type=SplitManifest)
    return hashlib.sha256(msgspec.json.encode(normalized)).hexdigest()


def read_split(path: Path) -> SplitManifest:
    split = msgspec.json.decode(path.read_bytes(), type=SplitManifest)
    if (
        not split.train
        or not split.validation
        or len(set(split.train)) != len(split.train)
        or len(set(split.validation)) != len(split.validation)
        or set(split.train) & set(split.validation)
    ):
        raise ValueError("split manifest must contain distinct, nonempty, theorem-disjoint roles")
    return split


class Shape(msgspec.Struct, frozen=True, array_like=True):
    ident: bytes
    edges: Edges


class Occurrence(msgspec.Struct, frozen=True, array_like=True):
    shape: int
    depth: int
    nodes: tuple[int, ...]


class Node(msgspec.Struct, frozen=True, array_like=True):
    ref: int
    expr: r.Expr
    goal_count: int
    hyp_count: int


class Root(msgspec.Struct, frozen=True, array_like=True):
    ref: int
    anchors: tuple[int, ...]


class LocalInfo(msgspec.Struct, frozen=True, array_like=True):
    is_let: bool
    is_instance: bool


class State(msgspec.Struct, frozen=True, array_like=True):
    step: int
    tactic: r.Tactic
    goal: int
    hyps: tuple[int, ...]
    locals: tuple[LocalInfo, ...] | None = None  # absent in older candidate archives


class Cands(msgspec.Struct, frozen=True, array_like=True):
    """One theorem's candidates, not fitted features or executable predictions.

    Root closures and states retain co-occurrence and repeated hypotheses without
    duplicating fragment records per state. Node counts use per-expression DAG
    anchors: one visit per goal or hypothesis occurrence, not per expanded path.
    Different query depths can share one shape; consumers must not count them as
    independent entries simply because both depths were requested.
    """

    name: str
    nodes: tuple[Node, ...]
    shapes: tuple[Shape, ...]
    occs: tuple[Occurrence, ...]
    roots: tuple[Root, ...]
    states: tuple[State, ...]


NODE_KINDS = (
    r.Bvar,
    r.Fvar,
    r.ExprMvar,
    r.Sort,
    r.Const,
    r.App,
    r.Lambda,
    r.Forall,
    r.Let,
    r.NatLiteral,
    r.StringLiteral,
    r.Metadata,
    r.Proj,
)
NODE_NAMES = tuple(kind.__name__ for kind in NODE_KINDS)
KIND_COLS = {kind: col for col, kind in enumerate(NODE_KINDS)}
MAX_COUNT = np.iinfo(np.int64).max


class AttrPolicy(msgspec.Struct, frozen=True):
    hyp_slots: int = 32
    name_dims: int = 100_000
    max_names: int = 512
    max_heads: int = 128
    min_support: int = 3
    memory_mib: int = 512  # retained evidence/index estimate, not peak process memory


DEFAULT_ATTR_POLICY = AttrPolicy()


class Representation(msgspec.Struct, frozen=True):
    policy: AttrPolicy
    heads: tuple[str, ...]
    names: tuple[tuple[bytes, int, str], ...]  # shape identity, canonical position, Const.name
    selection: Selection
    candidates: str
    size: int
    modified_ns: int
    label_policy_json: bytes


class Entry(msgspec.Struct, frozen=True, array_like=True):
    ident: bytes
    edges: Edges


class CoverPolicy(msgspec.Struct, frozen=True):
    objective: Literal["entries", "dims"] = "dims"
    improvement_steps: int = 0


DEFAULT_COVER_POLICY = CoverPolicy()

SUPERVISED_SCREENING = "role-count-correlation-balanced"


class SupervisedPolicy(msgspec.Struct, frozen=True):
    """Screening/enrichment settings, not classifier hyperparameters."""

    dims: int = 100_000  # additional dimensions; never charged against the cover
    shortlist: int = 256  # global association shortlist
    per_label: int = 32
    common: int = 64  # presence alone can miss informative positional node kinds
    min_support: int = 3  # distinct theorems with eligible labeled transitions
    redundancy: float = 0.5  # soft cosine penalty, not an exclusion threshold
    memory_mib: int = 1024  # retained score/pair buffers, not process-tree RSS


class Supervision(msgspec.Struct, frozen=True):
    policy: SupervisedPolicy
    label_policy_json: bytes  # canonical mapping contents, not just a mutable path
    candidates: str
    size: int
    modified_ns: int
    labels: tuple[str, ...]
    counts: tuple[int, ...]
    labeled_theorems: int
    shortlist_shapes: int
    added_shapes: int
    added_dims: int
    screening: Literal["role-presence-chi2", "role-count-correlation-balanced"] = SUPERVISED_SCREENING
    dimension_cost: Literal["structural", "complete"] = "structural"
    available_assoc: tuple[float, ...] = ()  # best supported squared count/label correlation, per family
    selected_assoc: tuple[float, ...] = ()  # best retained correlation, including the coverage backbone
    label_strength: tuple[float, ...] = ()  # accumulated normalized, redundancy-discounted entry utility


class DimBudget(msgspec.Struct, frozen=True):
    total: int
    baseline: int  # selected cover, fixed attributes, statistics and declaration slots
    name_share: float  # reserved fraction of the remaining budget; unused shape allocation transfers to names


class Vocab(msgspec.Struct, frozen=True):
    depths: tuple[int, ...]
    entries: tuple[Entry, ...]
    node_kinds: tuple[str, ...] = NODE_NAMES
    selection: Selection | None = None
    min_support: int = 1
    min_nodes: int = 2
    max_shapes: int | None = None
    coverage: CoverPolicy | None = None
    supervision: Supervision | None = None
    representation: Representation | None = None
    budget: DimBudget | None = None


@dataclass(frozen=True)
class FeatureRows:
    name: str
    steps: np.ndarray
    tactics: tuple[r.Tactic, ...]  # side information, not feature inputs
    matrix: csr_array


class LabelPolicy(msgspec.Struct, frozen=True, forbid_unknown_fields=True):
    """Exact exported parser-kind mapping, independent of source spelling.

    Treat the mapping as immutable during a query/experiment. Unmapped actions
    are rejected, excluded from learning rows, or assigned an explicit label;
    none of these choices alters the stored theorem or source tactics.
    """

    kinds: dict[str, str]
    unmapped: Literal["error", "drop", "other"]
    other_label: str | None = None

    def __post_init__(self) -> None:
        if self.unmapped not in ("error", "drop", "other"):
            raise ValueError("unmapped must be error, drop or other")
        if any(not isinstance(val, str) or not val.strip() for pair in self.kinds.items() for val in pair):
            raise ValueError("tactic kinds and labels must be nonempty strings")
        if self.unmapped == "other":
            if not isinstance(self.other_label, str) or not self.other_label.strip():
                raise ValueError("unmapped=other requires a nonempty other_label")
        elif self.other_label is not None:
            raise ValueError("other_label is only used with unmapped=other")
        if not self.labels:
            raise ValueError("policy declares no labels")

    @property
    def labels(self) -> tuple[str, ...]:
        labels = set(self.kinds.values())
        if self.other_label is not None:
            labels.add(self.other_label)
        return tuple(sorted(labels))

    @property
    def label_ids(self) -> dict[str, int]:
        """Stable IDs for every declared class, including locally absent ones."""
        return {label: idx for idx, label in enumerate(self.labels)}

    def label(self, tactic: r.Tactic) -> str | None:
        if (label := self.kinds.get(tactic.kind)) is not None:
            return label
        if self.unmapped == "error":
            raise ValueError(f"unmapped tactic kind: {tactic.kind!r}")
        return self.other_label if self.unmapped == "other" else None


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, val in pairs:
        if key in result:
            raise ValueError(f"duplicate label-policy key: {key!r}")
        result[key] = val
    return result


def read_label_policy(path: Path) -> LabelPolicy:
    data = json.loads(path.read_bytes(), object_pairs_hook=_unique_keys)
    return msgspec.convert(data, type=LabelPolicy, strict=True)


class _Ref(msgspec.Struct, frozen=True, array_like=True):
    ref: int


class _CoverageOcc(msgspec.Struct, frozen=True, array_like=True):
    shape: int
    depth: int
    anchor: _Ref  # first member of the archived occurrence's node list


class _CoverageShape(msgspec.Struct, frozen=True, array_like=True):
    # msgspec cannot decode a bytes|str union. Accept the historical large-Nat
    # encoder's base64 fallback here, then validate/normalize at registration.
    ident: object
    edges: msgspec.Raw  # repeated adjacency can be compared without decoding/repacking


class _CoverageState(msgspec.Struct, frozen=True, array_like=True):
    step: msgspec.Raw
    tactic: msgspec.Raw
    goal: int
    hyps: tuple[int, ...]


class CoverageCands(msgspec.Struct, frozen=True, array_like=True):
    """Only inputs consumed by coverage; omitted feature data is not validated."""

    name: str
    nodes: tuple[_Ref, ...]
    shapes: tuple[_CoverageShape, ...]
    occs: tuple[_CoverageOcc, ...]
    roots: tuple[Root, ...]
    states: tuple[_CoverageState, ...]
