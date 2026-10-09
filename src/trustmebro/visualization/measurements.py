"""Shared measurement records and column layouts; no graph preparation or analysis imports."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from typing import Literal, TypedDict

import msgspec
import numpy as np
from numpy.typing import NDArray

# Modes and shared type aliases.


class ViewMode(StrEnum):
    ORIGINAL = "original"
    INSTS = "instances"
    COMPACT = "coercions-compact"
    ERASED = "coercions-erased"


class PatternCol(IntEnum):
    FLAVOUR = 0
    HEAD = 1
    TOP = 2
    ALL = 3
    AFFECTED_TOP = 4
    AFFECTED_ALL = 5


type IntArr = NDArray[np.int64]
type FloatArr = NDArray[np.float64]
type ExprRole = Literal["goal", "ctxt"]
type Pop = Literal["top_level", "distinct_subexprs", "expanded_subexprs"]
type ComparisonLvl = Literal["expression", "state"]
type Pair = tuple[GraphSize, GraphSize]
type PairRow = tuple[Pair, int]
type PairHgram = Counter[tuple[GraphSize, GraphSize]] | PairCols
type PatternWeights = tuple[int, int, int, int]
type CountArray = NDArray[np.uint64] | NDArray[np.object_]
type HeadPatternRow = tuple[str, str, str, str, int, int, float]
type ShapeRow = tuple[bytes, int, int, int, int, tuple[tuple[int, int], ...]]
type ShapeKey = tuple[int, int, int]
type ShapeSample = tuple[ShapeKey, str, int, bytes]


# Measurement rows and column records.


@dataclass(slots=True)
class PatternBatch:
    """Unique packed keys and exact weights; head IDs belong to this head table.

    Normal batches contain two native buffers, not a Python object per row.
    Object weights are reserved for counts beyond uint64, never expanded sizes.
    """

    keys: NDArray[np.void]
    weights: CountArray
    heads: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.keys)


class PatternObs(msgspec.Struct, frozen=True):
    """Raw roots and their occurrence weights, shared across views and radii."""

    roots: IntArr
    weights: CountArray
    heads: tuple[str, ...]
    head_ids: NDArray[np.uint32]


class GraphSize(msgspec.Struct, frozen=True):
    nodes: int
    refs: int
    binders: int
    depth: int
    expanded: int
    max_children: int


class FreqRow(msgspec.Struct):
    """Exact identity counts; DAG occurrences are counted per top-level root."""

    digest: bytes
    distinct: int
    expanded: int
    top: int
    root_dag: int
    expanded_tree: int
    theorems: int = 1


class StateRow(msgspec.Struct, frozen=True):
    theorem: str
    step: int
    hyps: int
    goal_expanded: int
    goal_distinct: int
    goal_depth: int
    ctxt_expanded: int
    ctxt_distinct: int
    ctxt_depth: int
    state_expanded: int
    state_distinct: int
    largest_frac: float
    individual_distinct: int
    let_expanded: int
    mvar_expanded: int
    goal_constrs: dict[str, int]
    ctxt_constrs: dict[str, int]
    state_constrs: dict[str, int]


class ExprRow(msgspec.Struct, frozen=True):
    theorem: str
    role: ExprRole
    expanded: int
    distinct_nodes: int
    depth: int


class Descriptors(msgspec.Struct, frozen=True):
    """Sparse ordered-edge bins plus fixed motif/layer/sharing/binder groups."""

    edges: tuple[tuple[int, int, int], ...]
    edge_weights: tuple[float, ...]
    groups: tuple[float, ...]


class StralRow(msgspec.Struct, frozen=True):
    theorem: str
    root: int
    nodes: int
    depth: int
    expanded: int
    shared_frac: float
    constrs: dict[str, int]
    topo: Descriptors


class ReuseRow(msgspec.Struct, frozen=True):
    expanded: int
    local_dag: int
    reduced_tree: int
    reduced_dag: int
    novel: int
    nodes: int


@dataclass(frozen=True)
class SizeCols:
    nodes: IntArr
    refs: IntArr
    binders: IntArr
    depth: IntArr
    expanded: NDArray[np.object_]
    max_children: IntArr


@dataclass(frozen=True)
class PairCols:
    original: SizeCols
    transformed: SizeCols
    weights: FloatArr


@dataclass(slots=True)
class ShapeCount:
    nodes: int
    top: int
    all: int
    theorem: str
    root: int
    # Weight by exported root size, even when the transformed shape is smaller.
    bands: tuple[tuple[int, int], ...] = ()


class TopoMeasures(msgspec.Struct, frozen=True):
    name: str
    shapes: dict[ViewMode, list[ShapeRow]]
    pairs: dict[ViewMode, dict[ComparisonLvl, Counter[Pair]]]
    heads: dict[ViewMode, Counter[str]]


@dataclass(frozen=True, slots=True)
class Mdata:
    complete: bool
    src_db: str
    theorem_limit: int | None
    empty_ctxts: int
    conc: IntArr
    rotated_conc: IntArr
    stral_roots: int
    stral_sample_size: int
    atlas_min_nodes: int
    expr_idents: int


# Summary records.


class StateSummary(TypedDict):
    states: int
    theorems: int
    expr_idents: int
    pcentiles_expanded: dict[str, dict[float, int | None]]


class CoverageCurve(TypedDict):
    retained: list[int]
    coverage: list[float]


class FreqStats(TypedDict):
    pairs: list[tuple[int, int, list[int]]]
    coverage: dict[Pop, CoverageCurve]


@dataclass(frozen=True)
class Qtiles:
    median: int | float
    p95: int | float
    p99: int | float
    max_: int | float


@dataclass(frozen=True)
class LvlSummary:
    observations: int
    original: dict[str, Qtiles | None]
    transformed: dict[str, Qtiles | None]


class PatternDescr(TypedDict):
    patterns: int
    occs: int
    entropy_bits: float
    effective_patterns: float
    singletons: int
    rank: list[float]
    coverage: list[float]


@dataclass(frozen=True)
class TopoSummary:
    topos: int
    top_occs: int
    all_occs: int
    transformed_topos: dict[ViewMode, int]


@dataclass(frozen=True)
class TopoStats:
    summary: TopoSummary
    curves: dict[str, dict[ViewMode, dict[str, list[float]]]]
    baseline: dict[str, dict[str, dict[str, list[float]]]]


@dataclass(frozen=True)
class PatternStats:
    descrs: dict[str, PatternDescr]
    head_rows: list[HeadPatternRow]
    points: dict[int, list[tuple[int, int]]]


class HeadStats(TypedDict):
    counts: list[int]
    coverage: list[float]
    named_heads: int
    app_anchors: int


# View labels and packed layouts.


POLICY_TITLES = ("Instances hidden", "Compact conversions", "Conversion boundaries erased")
VIEW_MODES = tuple(ViewMode)
VIEW_TITLES = ("Exported baseline", *POLICY_TITLES)
POLICY_MODES = tuple(mode for mode in VIEW_MODES if mode != ViewMode.ORIGINAL)
PATTERN_MODES = VIEW_MODES
PATTERN_TITLES = VIEW_TITLES
SHAPE_SIZE_BANDS = 4
SHAPE_DTYPE = np.dtype([("nodes", "i8"), ("top", "i8"), ("all", "i8"), ("bands", "i8", (SHAPE_SIZE_BANDS, 2))])
SIZE_FIELDS = GraphSize.__struct_fields__


# Packed dictionary keys: SHA-256 digest, label flavour (0/1), interned head ID.
# NumPy views the same bytes for native sorting and selective column extraction.
# Explicit offsets avoid alignment padding and make the archive layout visible.
PATTERN_FLAVOUR_OFFSET = 32
PATTERN_HEAD_OFFSET = 33
PATTERN_BATCH_ROWS = 10_000
PATTERN_KEY_DTYPE = np.dtype(
    {
        "names": ["digest", "flavour", "head"],
        "formats": ["V32", "u1", "<u4"],
        "offsets": [0, PATTERN_FLAVOUR_OFFSET, PATTERN_HEAD_OFFSET],
        "itemsize": 37,
    }
)
PATTERN_BYTES_DTYPE = np.dtype((np.void, PATTERN_KEY_DTYPE.itemsize))
