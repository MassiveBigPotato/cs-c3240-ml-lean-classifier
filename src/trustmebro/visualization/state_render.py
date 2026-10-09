"""Render proof-state and expression-reuse measurements."""

from __future__ import annotations

import json
import math
import shutil
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from collections.abc import Set as AbstractSet
from contextlib import closing
from dataclasses import dataclass
from functools import partial
from hashlib import file_digest
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from trustmebro.visualization.archives import (
    freq_batches,
    freq_coverage,
    json_counts,
    mdata_val,
    mdata_vals,
    metric_cols,
    metric_summary,
)
from trustmebro.visualization.products import AnalysisPaths

from .drawing import (
    POINT_MEMORY_BUDGET,
    Axis,
    DensityCfg,
    DensityDrawer,
    DensityPlot,
    PointCols,
    actual_val_ticks,
    col_batches,
    col_bytes,
    count_ticks,
    density,
    log_counts,
    plot_style,
    save_single,
)
from .measurements import CoverageCurve, FloatArr

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# Fixed plot specifications.


STATE_COMPLEXITY = DensityPlot(
    "complexity-state.png",
    "Whole state: size and expansion",
    Axis("state_distinct_log", "Unique reachable expression nodes (log10 count)", "log10"),
    Axis("state_expanded_log", "Expanded tree nodes (log10 count)", "log10"),
    "Expanded size counts repeated subexpressions each time; unique size counts them once.",
    diag=True,
)
REUSE_NESTING = DensityPlot(
    "reuse-nesting.png",
    "Top-level expressions: reuse versus nesting",
    Axis("depth_log", "Maximum expression depth (log10 count)", "log10"),
    Axis("sharing", "Expanded / unique nodes (log10 ratio)", "log10"),
    "Goal and hypothesis types combined; each top-level root is an observation.",
)
CTXT_CONC = DensityPlot(
    "context-largest-to-average.png",
    "Context concentration: largest hypothesis versus average",
    Axis("hyps_log", "Local hypotheses (log10 count)", "log10"),
    Axis("largest_to_mean_log", "Largest / mean expanded hypothesis size (log10 ratio)", "log10"),
    "Each nonempty state contributes once; 1 means equally sized hypotheses.",
)
CTXT_COMP = DensityPlot(
    "context-composition.png",
    "Context composition: many hypotheses or large hypotheses?",
    Axis("hyps_log", "Local hypotheses (log10 count)", "log10"),
    Axis("mean_hyp_size", "Mean expanded hypothesis size (log10 nodes)", "log10"),
    "Colour shows how much of the context is occupied by its largest hypothesis; empty contexts excluded.",
)
CTXT_DENSITY = DensityPlot(
    "context-density.png",
    "Proof-state balance: goal versus context",
    Axis("goal_expanded_log", "Expanded goal size (log10 nodes)", "log10"),
    Axis("ctxt_expanded_log", "Expanded context size (log10 nodes)", "log10"),
    "Context size sums local hypothesis types; empty contexts excluded. Both axes use logarithmic scales.",
)
CROSS_ROOT_SHARING = DensityPlot(
    "cross-root-sharing.png",
    "Sharing between goal and hypotheses",
    Axis("hyps_log", "Local hypotheses (log10 count)", "log10"),
    Axis("cross_root_sharing", "Separate-root / union size (log10 ratio)", "log10"),
    "Counts distinct nodes per root, then compares their sum with the shared union; 1 means no overlap.",
)
SUBEXPR_REPETITION = DensityPlot(
    "subexpression-repetition.png",
    "Where does tree expansion amplify occurrence counts?",
    Axis("distinct_log", "Unique reachable expression nodes (log10 count)", "log10"),
    Axis("expanded_log", "Expanded tree nodes (log10 count)", "log10"),
    "Expanded occurrences / distinct-per-root occurrences, aggregated at each exact size pair.",
    diag=True,
)
FREQ_POPS = (
    ("top_level", "Top-level goals and hypothesis types"),
    ("distinct_subexprs", "Subexpressions: once per root occurrence"),
    ("expanded_subexprs", "Subexpressions: repeated tree occurrences"),
)


# Parameterized plot specifications.


def _global_reuse_plot(lvl: str, baseline: str, reduced: str) -> DensityPlot:
    return DensityPlot(
        f"global-reuse-{lvl}-{baseline.replace('_', '-')}.png",
        f"Global pattern reuse: {lvl}s versus {baseline.replace('_', ' ')} size",
        Axis(baseline + "_log", "Original size (log10 nodes)", "log10"),
        Axis(reduced + "_log", "Residual size (log10 nodes/reference tokens)", "log10"),
        "Largest-first cross-theorem patterns; references include variable arguments. Colour counts observations.",
    )


def _global_novelty_plot(lvl: str) -> DensityPlot:
    return DensityPlot(
        f"global-novelty-{lvl}.png",
        f"Cross-theorem novelty: {lvl}s",
        Axis("nodes_log", "Local DAG size (log10 nodes)", "log10"),
        Axis("novel_log", "Novel nodes + 1 (log10 count)", "log10"),
        "Adding 1 includes zero novelty; colour shows mean novel fraction. Reuse requires another theorem.",
    )


def _sharing_nesting_plot(root_count: int, stral_roots: int) -> DensityPlot:
    return DensityPlot(
        "sharing-nesting.png",
        "Shared DAG nodes versus nesting",
        Axis("depth_log", "Maximum expression depth (log10 count)", "log10"),
        Axis("shared", "Fraction of nodes with multiple incoming references"),
        f"Uniform hash sample: {root_count:,} distinct (theorem, root) pairs out of {stral_roots:,}; reference multiplicity includes parallel edges.",
    )


def _embedding_plot(mode: str, field: str, title: str, root_count: int) -> DensityPlot:
    return DensityPlot(
        f"structural-umap-{'' if mode == 'size-aware' else 'shape-only-'}{field}.png",
        f"Structural UMAP ({mode}): {title}",
        Axis("x", "UMAP coordinate 1 (arbitrary units)"),
        Axis("y", "UMAP coordinate 2 (arbitrary units)"),
        f"{root_count:,} roots; equally weighted constructor, edge, motif, depth-profile, spine, sharing, and binder distributions. Size-aware adds log size/depth.",
    )


def _reuse_breadth_plot(filename: str) -> DensityPlot:
    return DensityPlot(
        filename,
        "Broad reuse versus internal repetition",
        Axis("breadth", "Distinct theorems containing expression (log10 count)", "log10"),
        Axis("repetition", "Expanded / distinct-per-root occurrences (log10 ratio)", "log10"),
        "Includes subexpressions; theorem breadth counts each identity once per theorem. Coordinates rounded to 0.01 log units.",
    )


def _freq_plot(name: str, title: str, stat: str, subtitle: str) -> DensityPlot:
    return DensityPlot(
        f"freq-{name}-{stat}.png",
        f"{title}\n{subtitle}",
        Axis("distinct_log", "Unique reachable expression nodes (log10 count)", "log10"),
        Axis("expanded_log", "Expanded tree nodes (log10 count)", "log10"),
        "Variable IDs are consistently renamed; constants and other scalar information remain exact.",
        diag=True,
    )


def _freq_dom_plot(name: str, title: str) -> DensityPlot:
    return DensityPlot(
        f"freq-{name}-dom.png",
        f"{title}\nDoes one expression dominate its size pair?",
        Axis("distinct_log", "Unique reachable expression nodes (log10 count)", "log10"),
        Axis("expanded_log", "Expanded tree nodes (log10 count)", "log10"),
        "Largest individual frequency / total exact size-pair frequency; pairs with fewer than 10 occurrences omitted.",
        diag=True,
    )


# Column sources and display projections.


def _metric_src(
    path: Path,
    scope: Literal["state", "expression"] | None,
    cols: tuple[str, ...],
    logs: AbstractSet[str] = frozenset(),
    transform: Callable[[PointCols], PointCols] | None = None,
    *,
    kind: Literal["theorem", "global_reuse", "stral"] = "theorem",
    prepared: Callable[[], Iterator[PointCols]] | None = None,
) -> Callable[[], Iterator[PointCols]]:
    def batches() -> Iterator[PointCols]:
        source = prepared() if prepared is not None else metric_cols(path, kind, scope, cols)
        for raw in col_batches(source):
            batch = {name: raw[name] for name in cols}
            points = {
                f"{name}_log" if name in logs else name: raw[f"{name}_log"]
                if name in logs and f"{name}_log" in raw
                else log_counts(vals)
                if name in logs
                else vals
                for name, vals in batch.items()
            }
            yield transform(points) if transform else points

    return batches


def root_reuse(cols: PointCols) -> PointCols:
    return {**cols, "sharing": cols["expanded_log"] - cols["distinct_nodes_log"]}


def context_concentration(cols: PointCols) -> PointCols:
    return {
        **cols,
        "hyps_log": log_counts(cols["hyps"]),
        "largest_to_mean_log": log_counts(cols["largest_frac"] * cols["hyps"]),
    }


def context_composition(cols: PointCols) -> PointCols:
    hyps = log_counts(cols["hyps"])
    return {**cols, "hyps_log": hyps, "mean_hyp_size": cols["ctxt_expanded_log"] - hyps}


def cross_root_sharing(cols: PointCols) -> PointCols:
    return {
        **cols,
        "hyps_log": log_counts(cols["hyps"]),
        "cross_root_sharing": np.log10(cols["individual_distinct"] / cols["state_distinct"]),
    }


def novelty(cols: PointCols) -> PointCols:
    return {
        **cols,
        "nodes_log": log_counts(cols["nodes"]),
        "novel_log": log_counts(cols["novel"] + 1),
        "novel_frac": cols["novel"] / cols["nodes"],
    }


def reuse_breadth(batches: Iterable[Mapping[str, np.ndarray]]) -> pd.DataFrame:
    """Retain one aggregate per display bin, never one DataFrame row per identity.

    Rounding and accumulation order preserve the original plot's geometric mean.
    The aggregate index scales with occupied bins; this is not a total-memory bound.
    """
    bins: dict[tuple[float, float], int] = {}
    counts = np.empty(0, dtype=np.int64)
    sizes = np.empty(0, dtype=np.float64)
    for cols in batches:
        length = len(cols["theorems"])
        if not length:
            continue
        # Keep CPython's scalar log/round semantics at bin boundaries, including
        # arbitrary-size integers. Grouping and accumulation are native; ordered
        # add.at preserves each bin's floating sum across source batches.
        coords = np.fromiter(
            (
                val
                for breadth, expanded, dag in zip(
                    cols["theorems"], cols["expanded_tree"], cols["root_dag"], strict=True
                )
                for val in (round(math.log10(breadth), 2), round(math.log10(expanded) - math.log10(dag), 2))
            ),
            dtype=np.float64,
            count=2 * length,
        ).reshape(-1, 2)
        keys, first, inverse, weights = np.unique(
            coords, axis=0, return_index=True, return_inverse=True, return_counts=True
        )
        ids = np.empty(len(keys), dtype=np.intp)
        # New bins retain first-occurrence order, not NumPy's sorted key order.
        for idx in np.argsort(first):
            key = tuple(keys[idx])
            ids[idx] = bins.setdefault(key, len(bins))
        if len(bins) > len(counts):
            capacity = max(len(bins), 2 * len(counts), 256)
            counts = np.pad(counts, (0, capacity - len(counts)))
            sizes = np.pad(sizes, (0, capacity - len(sizes)))
        counts[ids] += weights
        logs = np.fromiter(map(math.log10, cols["distinct"]), dtype=np.float64, count=length)
        np.add.at(sizes, ids[inverse], logs)
    return pd.DataFrame(
        [(x, y, sizes[idx] / counts[idx], counts[idx]) for (x, y), idx in bins.items()],
        columns=["breadth", "repetition", "size", "identities"],
    )


def frequency_frame(
    pairs: Iterable[tuple[int, int, list[int]]], pop: int, stat: Literal["total", "dom", "repetition"]
) -> pd.DataFrame:
    """Project only the requested measure from exact size-pair aggregates."""
    rows: list[tuple[float, float, float]] = []
    for distinct, expanded, counts in pairs:
        total, largest = counts[2 * pop : 2 * pop + 2]
        if total:
            match stat:
                case "total":
                    val = math.log10(total)
                case "dom":
                    val = largest / total if total >= 10 else float("nan")
                case "repetition":
                    val = math.log10(total) - math.log10(counts[2])
            rows.append((math.log10(distinct), math.log10(expanded), val))
    return pd.DataFrame(rows, columns=["distinct_log", "expanded_log", stat])


# Prepared-data figure builders.


@dataclass(frozen=True)
class ConstructorCols:
    sizes: np.ndarray
    counts: dict[str, np.ndarray]


def constructor_cols(batches: Iterable[Mapping[str, np.ndarray]]) -> Iterator[ConstructorCols]:
    """Compact constructor maps into count columns, without retaining their dicts."""
    for cols in col_batches(batches):
        maps = cols["state_constrs"]
        kinds = sorted({name for row in maps for name in row})
        counts: dict[str, np.ndarray] = {}
        for name in kinds:
            try:
                counts[name] = np.fromiter((row.get(name, 0) for row in maps), dtype=np.int64, count=len(maps))
            except OverflowError:
                counts[name] = np.fromiter((row.get(name, 0) for row in maps), dtype=object, count=len(maps))
        yield ConstructorCols(cols["state_expanded"], counts)


def constructor_composition(sizes: np.ndarray, batches: Iterable[ConstructorCols]) -> tuple[list[str], np.ndarray]:
    # Exact percentile membership needs raw sizes, not every constructor map.
    sizes = np.sort(sizes)
    bands = np.array([0, 0.5, 0.9, 0.95, 0.99, 0.999, 1])
    counts: list[Counter[str]] = [Counter() for _ in range(6)]
    for cols in batches:
        vals = cols.sizes
        ranks = (np.searchsorted(sizes, vals, side="left") + np.searchsorted(sizes, vals, side="right") + 1) / (
            2 * len(sizes)
        )
        idxs = np.clip(np.searchsorted(bands[1:], ranks, side="left"), 0, 5)
        for idx, counter in enumerate(counts):
            mask = idxs == idx
            if not mask.any():
                continue
            for name, vals in cols.counts.items():
                # Object reduction preserves exact totals even when the sum of
                # individually representable counts exceeds int64.
                counter[name] += vals[mask].sum(dtype=object)
    kinds = sorted({key for counter in counts for key in counter})
    totals = [sum(counter.values()) for counter in counts]
    img = np.array(
        [
            [counter[kind] / total if total else np.nan for kind in kinds]
            for counter, total in zip(counts, totals, strict=True)
        ]
    )
    return kinds, img


@plot_style
def constructor_figure(kinds: Sequence[str], img: np.ndarray) -> Figure | None:
    """Build from percentile-band shares; the caller saves/closes the figure."""
    if not kinds:
        return None
    fig, ax = plt.subplots(figsize=(12, 5))
    artist = ax.imshow(img, aspect="auto", vmin=0, vmax=1)
    ax.set(
        xlabel="Expression constructor",
        ylabel="Whole-state expanded-size percentile band",
        xticks=range(len(kinds)),
        xticklabels=kinds,
        yticks=range(6),
        yticklabels=["0–50%", "50–90%", "90–95%", "95–99%", "99–99.9%", "99.9–100%"],
        title="Constructor composition by whole-state size percentile",
    )
    ax.tick_params(axis="x", labelrotation=45)
    fig.colorbar(artist, ax=ax, label="Share of distinct reachable nodes")
    return fig


@plot_style
def concentration_figure(concentration: np.ndarray, empty_ctxts: int) -> Figure:

    grid = concentration.T
    fig, ax = plt.subplots(figsize=(8, 7))
    artist = ax.imshow(
        np.where(grid > 0, np.log10(np.maximum(grid, 1)), np.nan), origin="lower", extent=(0, 1, 0, 1), aspect="equal"
    )
    ax.plot([0, 1], [0, 1], "w--")
    # Conditional quantiles make the distribution legible without discarding
    # its density: at each fraction, how concentrated are typical contexts?
    quants: list[FloatArr | list[float]] = []
    for col in grid.T:
        cum = np.cumsum(col)
        quants.append(
            np.interp(np.array([0.1, 0.5, 0.9]) * cum[-1], cum, (np.arange(grid.shape[0]) + 0.5) / grid.shape[0])
            if cum[-1]
            else [np.nan] * 3
        )
    lo, mid, hi = np.array(quants).T
    x = (np.arange(grid.shape[1]) + 0.5) / grid.shape[1]
    ax.plot(x, mid, color="cyan", label="Median")
    ax.plot(x, lo, color="cyan", linestyle=":", label="10th–90th percentiles")
    ax.plot(x, hi, color="cyan", linestyle=":")

    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()
    ax.set(
        xlabel="Fraction of hypotheses (largest first)",
        ylabel="Fraction of expanded context size",
        title=f"Context concentration; {empty_ctxts} empty contexts excluded",
    )
    colorbar = fig.colorbar(artist, ax=ax, label="States per bin (log scale)")
    actual_val_ticks(colorbar.ax.yaxis)
    return fig


@plot_style
def concentration_excess_figure(rotated: np.ndarray) -> Figure:
    grid = rotated.T
    fig, ax = plt.subplots(figsize=(12, 7))
    data = np.ma.masked_where(grid == 0, np.log10(np.maximum(grid, 1)))
    artist = ax.imshow(data, origin="lower", extent=(0, 1, 0, 0.5), interpolation="nearest", aspect="equal")
    xlabel = "Position along equality baseline: (hypothesis fraction + size fraction) / 2"
    ylabel = "Excess above equality: (size fraction − hypothesis fraction) / 2"
    title = "Context concentration: rotated Lorenz-style view"
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=(0, 1), ylim=(0, 0.5))
    ax.plot([0, 0.5, 1], [0, 0.5, 0], "w--", linewidth=0.8)
    ax.axvline(0.5, color="white", linestyle=":", linewidth=0.6)
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    actual_val_ticks(fig.colorbar(artist, ax=ax, label="States per pixel (log scale)").ax.yaxis)
    return fig


@plot_style
def vocabulary_figure(coverage: Mapping[str, CoverageCurve]) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, title in FREQ_POPS:
        if name in coverage:
            curve = coverage[name]
            ax.plot(curve["retained"], curve["coverage"], label=title)
    ax.set(
        xscale="log",
        xlabel="Most frequent expression identities retained",
        ylabel="Share of expression occurrences covered",
        ylim=(0, 1),
        title="Vocabulary coverage and the long tail",
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    count_ticks(ax.xaxis)
    ax.legend()
    fig.text(0.1, 0.01, "Occurrence coverage is not compression savings: subexpressions overlap.", fontsize=9)
    return fig


# Graph-family rendering.


def _plot_complexity(path: Path, out: Path, draw: DensityDrawer) -> None:
    cols = (
        "state_distinct",
        "state_expanded",
        "hyps",
        "largest_frac",
        "ctxt_expanded",
        "goal_expanded",
        "individual_distinct",
    )
    states = partial(
        _metric_src,
        path,
        "state",
        prepared=prepare_metric_cols(
            path,
            "theorem",
            "state",
            cols,
            logs=frozenset(("state_distinct", "state_expanded", "ctxt_expanded", "goal_expanded")),
        ),
    )
    exprs = partial(_metric_src, path, "expression")
    draw(
        out, states(("state_distinct", "state_expanded"), {"state_distinct", "state_expanded"}, None), STATE_COMPLEXITY
    )
    draw(
        out,
        exprs(("distinct_nodes", "expanded", "depth"), {"distinct_nodes", "expanded", "depth"}, root_reuse),
        REUSE_NESTING,
    )
    draw(out, states(("hyps", "largest_frac"), transform=context_concentration), CTXT_CONC)
    draw(
        out,
        states(("hyps", "ctxt_expanded", "largest_frac"), {"ctxt_expanded"}, context_composition),
        CTXT_COMP,
        DensityCfg(median="largest_frac", colour_range=(0, 1)),
    )
    draw(out, states(("goal_expanded", "ctxt_expanded"), {"goal_expanded", "ctxt_expanded"}), CTXT_DENSITY)
    draw(
        out, states(("hyps", "individual_distinct", "state_distinct"), transform=cross_root_sharing), CROSS_ROOT_SHARING
    )


def _plot_constrs(path: Path, out: Path) -> None:
    def source():
        return constructor_cols(metric_cols(path, "theorem", "state", ("state_expanded", "state_constrs")))

    size_chunks: list[np.ndarray] = []
    retained: list[ConstructorCols] | None = []
    nbytes = 0
    for cols in source():
        size_chunks.append(cols.sizes)
        if retained is not None:
            nbytes += col_bytes({"sizes": cols.sizes, **cols.counts})
            if nbytes > POINT_MEMORY_BUDGET:
                retained = None
            else:
                retained.append(cols)
    sizes = np.concatenate(size_chunks) if size_chunks else np.empty(0, dtype=object)
    size_chunks.clear()
    batches = iter(retained) if retained is not None else source()
    fig = constructor_figure(*constructor_composition(sizes, batches))
    if fig is not None:
        save_single(fig, out / "constructors-state.png")


def _plot_global_reuse(path: Path, out: Path, draw: DensityDrawer) -> None:
    for lvl in ("state", "expression"):
        cols = ("expanded", "reduced_tree", "local_dag", "reduced_dag", "nodes", "novel")
        src = partial(
            _metric_src,
            path,
            lvl,
            kind="global_reuse",
            prepared=prepare_metric_cols(
                path,
                "global_reuse",
                lvl,
                cols,
                logs=frozenset(("expanded", "reduced_tree", "local_dag", "reduced_dag")),
            ),
        )
        for base, reduced in (("expanded", "reduced_tree"), ("local_dag", "reduced_dag")):
            draw(out, src((base, reduced), {base, reduced}), _global_reuse_plot(lvl, base, reduced))
        cfg = DensityCfg(
            colour="novel_frac",
            reduce_color="mean",
            colour_proj="linear",
            colour_range=(0, 1),
            label="Mean novel-node fraction",
        )
        draw(out, src(("nodes", "novel"), transform=novelty), _global_novelty_plot(lvl), cfg)


def _draw_embedding(frame: pd.DataFrame, dir: Path, mode: str, draw: DensityDrawer) -> None:
    for field, title, log in (
        ("nodes_log", "Distinct DAG nodes", True),
        ("depth_log", "Maximum depth", True),
        ("reuse", "Expanded / distinct node ratio", True),
        ("shared", "Fraction of shared nodes", False),
    ):
        label = f"Mean {title.lower()} per pixel" + (" (log scale)" if log else "")
        cfg = DensityCfg(
            colour=field,
            label=label,
            colour_proj="log10" if log else "linear",
            colour_range=None if log else (0, 1),
            reduce_color="mean",
        )
        draw(dir, frame, _embedding_plot(mode, field, title, len(frame)), cfg)


def _plot_stral(path: Path, stral_roots: int, dir: Path, stats: Path, draw: DensityDrawer) -> None:
    """Population hash sample; embedding depicts descriptors, not exact DAGs."""

    chunks = list(_metric_src(path, None, ("nodes", "depth", "expanded", "shared_frac"), {"expanded"}, kind="stral")())
    if not chunks:
        return
    frame = pd.DataFrame({name: np.concatenate([cols[name] for cols in chunks]) for name in chunks[0]}).rename(
        columns={"shared_frac": "shared"}
    )
    chunks.clear()
    frame["depth_log"] = np.log10(frame.depth)
    frame["nodes_log"] = np.log10(frame.nodes)
    frame["reuse"] = frame.expanded_log - frame.nodes_log
    draw(dir, frame, _sharing_nesting_plot(len(frame), stral_roots))
    src = AnalysisPaths(stats)
    shared_path = src.embedding_shared
    shared_id = None
    if shared_path.exists():
        with shared_path.open("rb") as shared_stream:
            shared_id = file_digest(shared_stream, "sha256").hexdigest()
    if shared_path.exists() and (dir / shared_path.name).resolve() != shared_path.resolve():
        shutil.copyfile(shared_path, dir / shared_path.name)
    for mode in ("size-aware", "shape-only"):
        path = src.embedding(mode)
        if not path.exists():
            if len(frame) >= 4:
                raise ValueError("missing embeddings; run graphs --graphs embedding to prepare them")
            continue
        with np.load(path, allow_pickle=False) as data:
            if str(data["shared_sha256"]) != shared_id or int(data["rows"]) != len(frame):
                raise ValueError("embedding coordinates and shared observations disagree")
            frame["x"], frame["y"] = data["coordinates"].T
            _draw_embedding(frame, dir, mode, draw)
            dst = dir / path.name  # plot exports never write into a published generation
            if dst.resolve() != path.resolve():
                shutil.copyfile(path, dst)


def _plot_reuse(path: Path, dir: Path, draw: DensityDrawer) -> None:
    frame = reuse_breadth(metric_cols(path, "freqs", None, ("theorems", "expanded_tree", "root_dag", "distinct")))
    for color, label, filename in (
        ("size", "Geometric mean distinct-node size per bin; maximum across bins (log scale)", "reuse-breadth.png"),
        (None, None, "reuse-breadth-density.png"),
    ):
        draw(
            dir,
            frame,
            _reuse_breadth_plot(filename),
            DensityCfg(colour=color, label=label, weight=None if color else "identities"),
        )


def _render_freqs(path: Path, out: Path, draw: DensityDrawer) -> None:
    for pop, (name, title) in enumerate(FREQ_POPS):

        def src(stat: Literal["total", "dom", "repetition"], pop: int = pop):
            return lambda: (frequency_frame(rows, pop, stat) for rows in freq_batches(path))

        if pop < 2:
            subtitle = "Total size-pair frequency"
            draw(
                out,
                src("total"),
                _freq_plot(name, title, "total", subtitle),
                DensityCfg(colour="total", label=f"Max {subtitle.lower()} per bin (log scale)"),
            )
        if pop == 0:
            draw(
                out,
                src("dom"),
                _freq_dom_plot(name, title),
                DensityCfg(
                    colour="dom", label="Largest individual share per bin", colour_proj="linear", colour_range=(0, 1)
                ),
            )
        if pop == 2:
            draw(
                out,
                src("repetition"),
                SUBEXPR_REPETITION,
                DensityCfg(colour="repetition", label="Largest repetition multiplier per bin (log scale)"),
            )
    coverage = freq_coverage(path)
    save_single(vocabulary_figure(coverage), out / "vocabulary-coverage.png")
    (out / "vocabulary-coverage.json").write_text(json.dumps(coverage, indent=2))


# State rendering entry point.


def render(
    metrics_path: Path, out: Path, *, atlas_min_nodes: int = 20, dot_diam: int = 1, families: set[str] | None = None
) -> None:
    """Each consumer loads only its required records and releases them on return."""
    default_families = {"complexity", "constructors", "context", "reuse", "frequencies", "embedding", "expr-atlas"}
    selected = families if families is not None else default_families
    draw = partial(density, dot_diam=dot_diam)
    out.mkdir(parents=True, exist_ok=True)
    if "complexity" in selected:
        _plot_complexity(metrics_path, out, draw)
    if "constructors" in selected:
        _plot_constrs(metrics_path, out)
    if "reuse" in selected:
        _plot_global_reuse(metrics_path, out, draw)
        _plot_reuse(metrics_path, out, draw)
    if "frequencies" in selected:
        _render_freqs(metrics_path, out, draw)
    if "context" in selected:
        ctxt = mdata_vals(metrics_path, {"conc": np.ndarray, "rotated_conc": np.ndarray, "empty_ctxts": int})
        save_single(concentration_figure(ctxt["conc"], ctxt["empty_ctxts"]), out / "context-concentration.png")
        save_single(concentration_excess_figure(ctxt["rotated_conc"]), out / "context-concentration-excess.png")
    if "embedding" in selected:
        _plot_stral(metrics_path, mdata_val(metrics_path, "stral_roots", int), out, metrics_path.parent, draw)
    if "expr-atlas" in selected:
        from .atlas import render_expr_atlas

        render_expr_atlas(metrics_path, out, atlas_min_nodes)
    summary = metric_summary(metrics_path)
    (out / "summary.json").write_text(json.dumps(json_counts(summary), indent=2))


def prepare_metric_cols(
    path: Path,
    kind: Literal["theorem", "global_reuse", "stral"],
    scope: Literal["state", "expression"] | None,
    cols: tuple[str, ...],
    *,
    budget: int = POINT_MEMORY_BUDGET,
    logs: AbstractSet[str] = frozenset(),
) -> Callable[[], Iterator[PointCols]]:
    """Read a related figure group's raw columns once when they fit the budget.

    Chunk retention avoids concatenation copies. The source is reopened on
    overflow, preserving whole populations and the streaming raster fallback.
    """
    chunks: list[PointCols] = []
    used = 0
    with closing(metric_cols(path, kind, scope, cols)) as batches:
        for batch in batches:
            batch.update((f"{name}_log", log_counts(batch[name])) for name in logs)
            used += col_bytes(batch)
            if used > budget:
                chunks.clear()
                return partial(metric_cols, path, kind, scope, cols)
            chunks.append(batch)
    return lambda: iter(chunks)
