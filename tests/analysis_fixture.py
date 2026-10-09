"""Publish small fixtures through the production collector and archive boundary."""

from collections import Counter
from collections.abc import Generator, Iterable, Iterator, Sequence
from contextlib import closing
from pathlib import Path

import numpy as np

from trustmebro.visualization import archives
from trustmebro.visualization.measurements import (
    PATTERN_BYTES_DTYPE,
    PATTERN_FLAVOUR_OFFSET,
    PATTERN_HEAD_OFFSET,
    PATTERN_KEY_DTYPE,
    ComparisonLvl,
    CountArray,
    GraphSize,
    Pair,
    PairRow,
    PatternBatch,
    PatternWeights,
    ViewMode,
)
from trustmebro.visualization.patterns import PATTERN_WEIGHT_COUNT
from trustmebro.visualization.products import Analysis, AnalysisCfg, analysis_set
from trustmebro.visualization.scan import collect_analysis

type PatternRow = tuple[bytes, int, str, PatternWeights]


def analyze(
    db_path: Path,
    out: Path,
    *,
    analyses: tuple[Analysis, ...] = ("metrics", "topology", "patterns"),
    limit: int | None = None,
    workers: int = 1,
    atlas_min_nodes: int = 20,
    depths: tuple[int, ...] = (1, 2, 3),
    aggr_mem: int = 512,
    replace: bool = False,
    timing_dir: Path | None = None,
) -> dict[str, int]:
    """Public in-process analysis with the same publication guarantee as the CLI."""
    cfg = AnalysisCfg(atlas_min_nodes, tuple(dict.fromkeys(depths)))
    with analysis_set(out, analyses, db_path, limit=limit, cfg=cfg, replace=replace) as pending:
        return collect_analysis(
            db_path,
            pending.root,
            analyses=analyses,
            limit=limit,
            workers=workers,
            atlas_min_nodes=atlas_min_nodes,
            depths=cfg.depths,
            aggr_mem=aggr_mem,
            timing_dir=timing_dir,
        )


def read_pairs(path: Path) -> dict[ViewMode, dict[ComparisonLvl, Counter[Pair]]]:
    result: dict[ViewMode, dict[ComparisonLvl, Counter[Pair]]] = {}
    for mode, lvl, batch in pair_batches(path):
        result.setdefault(mode, {}).setdefault(lvl, Counter()).update(dict(batch))
    return result


def pattern_batches(path: Path) -> Generator[list[PatternRow]]:
    with closing(archives.packed_pattern_batches(path)) as stream:
        for heads, batch in stream:
            yield [
                (
                    key[:PATTERN_FLAVOUR_OFFSET],
                    key[PATTERN_FLAVOUR_OFFSET],
                    heads[int.from_bytes(key[PATTERN_HEAD_OFFSET:], "little")],
                    weights,
                )
                for key, weights in zip(
                    batch.keys.view(PATTERN_BYTES_DTYPE).tolist(), map(tuple, batch.weights.tolist()), strict=True
                )
            ]


def read_pattern_cols(path: Path) -> tuple[list[str], np.ndarray]:
    chunks: list[np.ndarray] = []
    heads: list[str] = []
    with closing(archives.packed_pattern_batches(path)) as batches:
        for heads, rows in batches:
            chunks.append(archives._pattern_cols(rows))
    return heads, archives._concat_pattern_cols(chunks)


def pack_patterns(rows: Iterable[tuple[bytes, int, str, Sequence[int]]]) -> PatternBatch:
    """Construct artificial inputs for exact-count and aggregation boundary fixtures."""
    heads = {"": 0}
    keys: list[bytes] = []
    weights: list[Sequence[int]] = []
    for digest, flavour, head, vals in rows:
        head_id = heads.setdefault(head, len(heads))
        keys.append(pattern_key(digest, flavour, head_id))
        weights.append(vals)
    return PatternBatch(np.frombuffer(b"".join(keys), dtype=PATTERN_KEY_DTYPE), exact_weights(weights), tuple(heads))


def shape_counts(path: Path):
    return dict(item for batch in archives.shape_batches(path) for item in batch)


def pair_batches(
    path: Path, *, modes: set[ViewMode] | None = None, lvls: set[ComparisonLvl] | None = None
) -> Iterator[tuple[ViewMode, ComparisonLvl, list[PairRow]]]:
    with closing(archives._pair_batches(path, modes=modes, lvls=lvls)) as batches:
        for mode, lvl, rows in batches:
            batch: list[PairRow] = []
            for (a, b), weight in rows:
                batch.append(((GraphSize(*a), GraphSize(*b)), weight))
            yield mode, lvl, batch


def pattern_key(digest: bytes, flavour: int, head: int) -> bytes:
    """Pack a key matching PATTERN_KEY_DTYPE, without a per-key NumPy allocation."""
    return digest + bytes((flavour,)) + head.to_bytes(PATTERN_KEY_DTYPE["head"].itemsize, "little")


def exact_weights(rows: Sequence[Sequence[int]]) -> CountArray:
    """Keep the common path native without truncating exceptional naturals."""
    try:
        return np.asarray(rows, dtype=np.uint64).reshape(-1, PATTERN_WEIGHT_COUNT)
    except OverflowError:
        weights = np.asarray(rows, dtype=object).reshape(-1, PATTERN_WEIGHT_COUNT)
        if np.any(weights < 0):
            raise ValueError("local-pattern counts must be nonnegative")
        return weights
