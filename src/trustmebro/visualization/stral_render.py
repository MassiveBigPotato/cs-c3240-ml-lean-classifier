"""Numeric, phase-isolated rendering of durable topology measurements.

No SQLite caches or expanded observation lists: archives are decoded in batches,
and only requested numeric columns and bounded raster groups survive decoding.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from compression import zstd
from dataclasses import dataclass
from functools import partial
from itertools import batched
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

import datashader as ds
import numpy as np
import pandas as pd

from trustmebro.visualization.archives import (
    json_counts,
    pair_col_batches,
    pattern_points,
    read_pattern_stats,
    read_summary,
    shape_cols,
)
from trustmebro.visualization.products import AnalysisPaths

from .drawing import (
    POINT_MEMORY_BUDGET,
    Axis,
    DensityPlot,
    PanelCfg,
    PointRaster,
    col_bytes,
    count_ticks,
    density,
    density_panels,
    log_counts,
    panel_figure,
    plot_style,
    prepare_point_raster,
    save_fig,
)
from .measurements import PATTERN_MODES, PATTERN_TITLES, ComparisonLvl, HeadStats, PairCols, ViewMode
from .measurements import VIEW_MODES as MODES
from .measurements import VIEW_TITLES as TITLES
from .measurements import PatternDescr as Description

if TYPE_CHECKING:
    from matplotlib.figure import Figure


# Render types and comparison specifications.


type CoverageCurves = dict[str, dict[str, dict[str, list[float]]]]
type ComparisonSrc = Callable[[], Iterator[tuple[ViewMode, ComparisonLvl, PairCols]]]
type PlotBounds = dict[str, dict[int, tuple[np.ndarray, np.ndarray]]]


class RenderSummary(TypedDict):
    topos: int
    top_occs: int
    all_occs: int
    transformed_topos: dict[str, int]


@dataclass(frozen=True)
class ComparisonPlot:
    name: str
    x: Axis
    y: Axis
    diag: bool = False

    def panels(self, caption: str = "") -> PanelCfg:
        return PanelCfg(self.x.label, self.y.label, self.diag, self.diag, self.y.proj, caption, self.x.proj)


PAIRED_PLOTS = (
    ComparisonPlot(
        "nodes",
        Axis("nodes_src_log", "Exported unique nodes", "log10"),
        Axis("nodes_dst_log", "Transformed unique nodes", "log10"),
        True,
    ),
    ComparisonPlot(
        "refs",
        Axis("refs_src_log", "1 + exported ref entries", "log10"),
        Axis("refs_dst_log", "1 + transformed ref entries", "log10"),
        True,
    ),
    ComparisonPlot(
        "binders",
        Axis("binders_src_log", "1 + exported binders (including grouped)", "log10"),
        Axis("binders_dst_log", "1 + retained binders (including grouped)", "log10"),
        True,
    ),
    ComparisonPlot(
        "depth",
        Axis("depth_src_log", "Exported maximum depth", "log10"),
        Axis("depth_dst_log", "Transformed max depth", "log10"),
        True,
    ),
    ComparisonPlot(
        "node-reduction",
        Axis("nodes_src_log", "Exported unique nodes", "log10"),
        Axis("node_reduction", "Node reduction"),
    ),
    ComparisonPlot(
        "depth-reduction",
        Axis("depth_src_log", "Exported maximum depth", "log10"),
        Axis("depth_reduction", "Depth reduction"),
    ),
)
SIZE_PLOTS = (
    ComparisonPlot(
        "complexity",
        Axis("nodes_log", "Unique reachable nodes", "log10"),
        Axis("expanded_log", "Expanded tree nodes", "log10"),
        True,
    ),
    ComparisonPlot(
        "reuse-nesting",
        Axis("depth_log", "Maximum depth", "log10"),
        Axis("reuse_log", "Expanded / unique nodes", "log10"),
    ),
    ComparisonPlot(
        "depth-width",
        Axis("depth_log", "Maximum depth", "log10"),
        Axis("width_log", "1 + maximum children per node", "log10"),
    ),
)


# At most twelve 1000x800 float grids (~74 MiB), plus up to 1 GiB of
# prepared coordinates. These are product bounds, not a total-memory guarantee.
COMPARISON_GRID_BUDGET = 12


# Comparison projections and bounded raster preparation.


def comparison_cols(
    cols: PairCols,
    plot: ComparisonPlot,
    *,
    original: bool = False,
    coords: dict[tuple[str, bool], np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Project two requested coordinates, not every measure on every replay."""
    old, new = cols.original, cols.transformed
    sizes = old if original else new

    def coordinate(field: str) -> np.ndarray:
        match field:
            case "node_reduction":
                return 100 * (1 - new.nodes / old.nodes)
            case "depth_reduction":
                return 100 * (1 - new.depth / old.depth)
            case "reuse_log":
                return log_counts(sizes.expanded) - log_counts(sizes.nodes)
            case "width_log":
                return log_counts(1 + sizes.max_children)
        name = field.removesuffix("_log")
        side = old if name.endswith("_src") else new if name.endswith("_dst") else sizes
        name = name.removesuffix("_src").removesuffix("_dst")
        vals = getattr(side, name)
        return log_counts(vals + 1 if name in {"refs", "binders"} else vals)

    coords = {} if coords is None else coords
    fields = (plot.x.field, plot.y.field)
    for field in fields:
        key = field, original
        if key not in coords:
            coords[key] = coordinate(field)
    vals = {"x": coords[fields[0], original], "y": coords[fields[1], original], "weight": cols.weights}
    valid = np.isfinite(vals["x"]) & np.isfinite(vals["y"])
    valid &= (
        ~pd.isna(cols.weights) & (cols.weights != np.inf) & (cols.weights != -np.inf)
        if cols.weights.dtype.hasobject
        else np.isfinite(cols.weights)
    )
    return vals if valid.all() else {name: col[valid] for name, col in vals.items()}


def _comparison_batches(
    src: ComparisonSrc, modes: tuple[ViewMode, ...], plots: Sequence[ComparisonPlot]
) -> Iterator[tuple[ComparisonPlot, int, dict[str, np.ndarray]]]:
    for mode, _, pair in src():
        coords: dict[tuple[str, bool], np.ndarray] = {}
        for plot in plots:
            if plot in SIZE_PLOTS and mode == modes[1]:
                yield plot, 0, comparison_cols(pair, plot, original=True, coords=coords)
            idx = modes.index(mode) - (plot in PAIRED_PLOTS)
            yield plot, idx, comparison_cols(pair, plot, coords=coords)


def _prepare_comparisons(
    batches: Iterator[tuple[ComparisonPlot, int, dict[str, np.ndarray]]],
) -> tuple[PlotBounds, list[tuple[ComparisonPlot, int, dict[str, np.ndarray]]] | None]:
    """Collect family coordinates once; replay groups only above the 1 GiB budget."""
    bounds: PlotBounds = {}
    chunks: list[tuple[ComparisonPlot, int, dict[str, np.ndarray]]] | None = []
    nbytes = 0
    for plot, idx, cols in batches:
        if not len(cols["x"]):
            continue
        lo = np.array([cols["x"].min(), cols["y"].min()])
        hi = np.array([cols["x"].max(), cols["y"].max()])
        panels = bounds.setdefault(plot.name, {})
        prev = panels.get(idx)
        panels[idx] = (lo, hi) if prev is None else (np.minimum(prev[0], lo), np.maximum(prev[1], hi))
        if chunks is not None:
            nbytes += col_bytes(cols)
            if nbytes > POINT_MEMORY_BUDGET:
                chunks = None
            else:
                chunks.append((plot, idx, cols))
    return bounds, chunks


def _comparison_canvas(bounds: Mapping[int, tuple[np.ndarray, np.ndarray]], equal: bool) -> ds.Canvas:
    padded = []
    for lo, hi in bounds.values():
        pad = np.where(lo == hi, 0.1, (hi - lo) * 0.015)
        padded.append((lo - pad, hi + pad))
    lo = np.min([p[0] for p in padded], axis=0)
    hi = np.max([p[1] for p in padded], axis=0)
    if equal:
        lo[:] = lo.min()
        hi[:] = hi.max()
    return ds.Canvas(plot_width=1000, plot_height=800, x_range=(lo[0], hi[0]), y_range=(lo[1], hi[1]))


def comparison_grids(
    src: ComparisonSrc, modes: tuple[ViewMode, ...], plots: Sequence[ComparisonPlot]
) -> Iterator[tuple[ComparisonPlot, list[np.ndarray], tuple[float, float], tuple[float, float]]]:
    """Share one source pass across plots, while allocating only a bounded raster group."""
    per_group = max(1, COMPARISON_GRID_BUDGET // len(modes))
    bounds, chunks = _prepare_comparisons(_comparison_batches(src, modes, plots))
    for group in batched(plots, per_group):
        canvases = {
            plot.name: _comparison_canvas(bounds[plot.name], plot in PAIRED_PLOTS and plot.diag)
            for plot in group
            if plot.name in bounds
        }
        reductions: dict[str, list[PointRaster | None]] = {
            plot.name: [None for _ in range(len(modes) - (plot in PAIRED_PLOTS))]
            for plot in group
            if plot.name in canvases
        }
        names = {plot.name for plot in group}
        batches = (
            (chunk for chunk in chunks if chunk[0].name in names)
            if chunks is not None
            else _comparison_batches(src, modes, group)
        )
        for plot, idx, cols in batches:
            if plot.name not in canvases or not len(cols["x"]):
                continue
            frame = pd.DataFrame(cols, copy=False)
            reduction = reductions[plot.name][idx]
            if reduction is None:
                reduction = reductions[plot.name][idx] = prepare_point_raster(
                    frame, canvases[plot.name], "x", "y", ds.sum("weight")
                )
            reduction.append(frame)
        for plot in group:
            if plot.name not in canvases:
                continue
            canvas = canvases[plot.name]
            imgs = [
                reduction.finish().values if reduction is not None else np.full((800, 1000), np.nan)
                for reduction in reductions.pop(plot.name)
            ]
            for grid in imgs:
                positive = grid > 0
                np.log10(grid, out=grid, where=positive)
                grid[~positive] = np.nan
            yield plot, imgs, cast(tuple[float, float], canvas.x_range), cast(tuple[float, float], canvas.y_range)


# Prepared-data figure builders.


@plot_style
def topology_coverage_figure(baseline_curves: Mapping[str, Mapping[str, Mapping[str, list[float]]]]) -> Figure:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, field in zip(axes, ("top", "all"), strict=True):
        for label, curve in baseline_curves[field].items():
            ax.plot(curve["rank"], curve["coverage"], label=label)
        ax.set(xscale="log", ylim=(0, 1.02), xlabel="Most frequent topology shapes retained", title=field)
        count_ticks(ax.xaxis)
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Fraction of shape occurrences covered")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


@plot_style
def head_vocabulary_figure(stats: Mapping[str, HeadStats]) -> Figure:
    """Frequency ranks and vocabulary coverage, plus the baseline's top heads."""
    import matplotlib.pyplot as plt

    modes = MODES
    titles = TITLES
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for mode, title in zip(modes, titles, strict=True):
        # Pseudo-head markers remain in the frequency plot, but are excluded
        # from the reported number of distinct named Lean constants.
        counts = np.asarray(stats[mode]["counts"])
        if not len(counts):
            continue
        ranks = np.arange(1, len(counts) + 1)
        axes[0].plot(ranks, counts, label=title)
        axes[1].plot(ranks, stats[mode]["coverage"], label=title)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="Application-head frequency rank",
        ylabel="State-DAG occurrences",
        title="Head frequencies, including conversion markers",
    )
    axes[1].set(
        xscale="log",
        xlabel="Most frequent heads retained",
        ylabel="Fraction of application anchors covered",
        ylim=(0, 1.02),
        title="Head vocabulary coverage",
    )
    for ax in axes:
        count_ticks(ax.xaxis)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    count_ticks(axes[0].yaxis)
    fig.tight_layout()
    return fig


@plot_style
def common_heads_figure(heads: Mapping[str, int]) -> Figure | None:
    import matplotlib.pyplot as plt

    common = sorted(heads.items(), key=lambda item: (-item[1], item[0]))[:25]
    if common:
        fig, ax = plt.subplots(figsize=(16, 8))
        names, counts = zip(*reversed(common), strict=True)
        ax.barh(names, counts)
        ax.set(xscale="log", xlabel="Occurrences, once per state DAG", title="Most common baseline application heads")
        count_ticks(ax.xaxis)
        fig.tight_layout()
        return fig
    return None


@plot_style
def coverage_comparison_figure(summary: CoverageCurves) -> Figure:
    """Redraw cached coverage curves without decoding millions of shape records."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(5, 2, figsize=(14, 18), sharey=True)
    for band, label in enumerate(
        (
            "All original sizes",
            "Original 1–7 nodes",
            "Original 8–31 nodes",
            "Original 32–127 nodes",
            "Original 128+ nodes",
        )
    ):
        for col, field in enumerate(("top", "all")):
            ax = axes[band, col]
            for mode, title in zip(MODES, TITLES, strict=True):
                curve = summary[f"{field}:{band}"][mode]
                rank, coverage = curve["rank"], curve["coverage"]
                if len(rank):
                    ax.plot(rank, coverage, label=title)
            ax.set(
                xscale="log",
                ylim=(0, 1.02),
                xlabel="Most frequent topology shapes retained",
                title=label + (" · top-level roots" if col == 0 else " · original rooted nodes"),
            )
            count_ticks(ax.xaxis)
            ax.grid(alpha=0.2)
            if col == 0:
                ax.set_ylabel("Fraction of original occurrences covered")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Topology vocabulary coverage · fixed original roots and original size bands")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


@plot_style
def pattern_coverage_figure(curves: Mapping[str, Description], title: str) -> Figure:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    for mode, label in zip(PATTERN_MODES, PATTERN_TITLES, strict=True):
        curve = curves[mode]
        if curve["rank"]:
            ax.plot(curve["rank"], curve["coverage"], label=label)
    ax.set(
        xscale="log",
        ylim=(0, 1.02),
        xlabel="Most frequent local patterns retained",
        ylabel="Fraction of original occurrences covered",
        title=title,
    )
    ax.grid(alpha=0.2)
    count_ticks(ax.xaxis)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# Comparison rendering.


def _plot_comparisons(pairs: Path, out: Path, dot_diam: int) -> None:
    for lvl in ("expression", "state"):
        pair_lvl = cast(ComparisonLvl, lvl)
        src = partial(pair_col_batches, pairs, modes=set(MODES[1:]), lvls={pair_lvl})
        for plot, imgs, xb, yb in comparison_grids(src, MODES, (*PAIRED_PLOTS, *SIZE_PLOTS)):
            paired = plot in PAIRED_PLOTS
            titles = list(TITLES[1:] if paired else TITLES)
            axes = (
                plot.panels()
                if paired
                else PanelCfg(plot.x.label, plot.y.label, diag=plot.diag, y_proj=plot.y.proj, x_proj=plot.x.proj)
            )
            fig = panel_figure(imgs, xb, yb, titles, axes, dot_diam=dot_diam)
            save_fig(fig, out / f"plumbing-{lvl}-{plot.name}.png", tight=True)


# Structural rendering entry points.


def render_shapes(stats_path: Path, out: Path, dot_diam: int) -> RenderSummary:

    stats = read_summary(AnalysisPaths(stats_path).stats("topology"))
    curves = stats["curves"]
    summary = stats["summary"]
    paths = AnalysisPaths(stats_path)

    def src(mode, field):
        def batches():
            for cols in shape_cols(paths.topo(mode), field):
                yield (
                    pd.DataFrame({"x": log_counts(cols["nodes"]), "y": log_counts(cols["occurrences"]), "weight": 1.0})
                )

        return batches

    for field in ("top", "all"):
        frames = [src(mode, field) for mode in MODES]
        prepared = density_panels(
            frames,
            list(TITLES),
            out / f"plumbing-topology-frequency-{field}.png",
            PanelCfg(
                "Distinct DAG nodes",
                "Occurrences of this topology",
                caption="Each observation is one topology; colours count topologies, not root occurrences.",
            ),
            dot_diam=dot_diam,
        )
        density(
            out,
            frames[0],
            DensityPlot(
                f"topology-frequency-{field}.png",
                "Complete unlabeled DAG topologies",
                Axis("x", "Distinct DAG nodes", "log10"),
                Axis("y", "Occurrences", "log10"),
                "Each point is a distinct topology; colours count topologies.",
            ),
            dot_diam=dot_diam,
            prepared=prepared[0],
        )
    save_fig(coverage_comparison_figure(curves), out / "plumbing-topology-coverage.png", dpi=110)
    (out / "plumbing-topology-coverage.json").write_text(json.dumps(curves))
    # Baseline size-band coverage has its own existing output contract.
    baseline_curves = stats["baseline"]
    save_fig(topology_coverage_figure(baseline_curves), out / "topology-coverage.png", dpi=130)
    (out / "topology-coverage.json").write_text(json.dumps(baseline_curves))
    (out / "topology-render-summary.json").write_text(json.dumps(summary))
    return summary


def render_pairs(stats: Path, out: Path, dot_diam: int) -> None:
    pairs = AnalysisPaths(stats).comparisons
    _plot_comparisons(pairs, out, dot_diam)
    (out / "plumbing-summary.json").write_text(
        json.dumps(json_counts(read_summary(AnalysisPaths(stats).stats("comparison"))), indent=2)
    )


def render_heads(stats: Path, out: Path) -> None:
    paths = AnalysisPaths(stats)
    measured = read_summary(paths.stats("head"))
    save_fig(head_vocabulary_figure(measured), out / "plumbing-head-vocabulary.png", dpi=130)
    common = common_heads_figure(read_summary(paths.heads)[ViewMode.ORIGINAL])
    if common is not None:
        save_fig(common, out / "plumbing-common-heads.png", dpi=120)
    summary = {mode: {field: measured[mode][field] for field in ("named_heads", "app_anchors")} for mode in MODES}
    (out / "plumbing-head-summary.json").write_text(json.dumps(summary, indent=2))


def render_patterns(stats: Path, out: Path, dot_diam: int = 1) -> None:
    summary: dict[str, dict[str, Description]] = {mode: {} for mode in PATTERN_MODES}
    curves: dict[tuple[str, str, str, int], dict[str, Description]] = {}
    conditioned: set[tuple[int, int]] = set()
    path = AnalysisPaths(stats).stats("pattern")
    with zstd.open(out / "local-pattern-heads.csv.zst", "wb", level=3) as compressed:
        text = io.TextIOWrapper(compressed, encoding="utf-8", newline="")
        csv_writer = csv.writer(text)
        csv_writer.writerow(
            ("mode", "depth", "flavour", "scope", "cohort", "head", "patterns", "occurrences", "entropy_bits")
        )
        for mode, depth, measured in read_pattern_stats(path, include_points=False):
            descrs, head_rows, points = measured.descrs, measured.head_rows, measured.points
            for key, descr in descrs.items():
                summary[mode][f"{depth}:{key}"] = descr
                label, scope, cohort = key.split(":")
                curves.setdefault((label, scope, cohort, depth), {})[mode] = descr
            csv_writer.writerows((mode, depth, *row) for row in head_rows)
            conditioned.update((depth, flavour) for flavour in points)
        text.flush()
        text.detach()
    for (label, scope, cohort, depth), modes in curves.items():
        title = f"{label} · radius {depth} · {scope} roots · {cohort} expressions"
        fig = pattern_coverage_figure(modes, title)
        save_fig(fig, out / f"local-pattern-{label}-{scope}-{cohort}-depth{depth}.png", dpi=120)

    def src(mode: ViewMode, depth: int, flavour: int):
        def batches():
            for rows in pattern_points(path, mode, depth, flavour):
                yield pd.DataFrame({"x": log_counts(rows[:, 0]), "y": log_counts(rows[:, 1]), "weight": 1.0})

        return batches

    for depth, flavour in sorted(conditioned):
        density_panels(
            [src(mode, depth, flavour) for mode in PATTERN_MODES],
            list(PATTERN_TITLES),
            out / f"local-pattern-heads-{flavour}-depth{depth}.png",
            PanelCfg(
                "Baseline-head occurrences in affected rooted expressions",
                "Distinct local patterns for that head",
                diag=True,
                eq_ranges=True,
                caption="Every application-head group is included, including rare heads. Colours count head groups.",
            ),
            dot_diam=dot_diam,
        )
    (out / "local-pattern-summary.json").write_text(json.dumps(summary, indent=2))
