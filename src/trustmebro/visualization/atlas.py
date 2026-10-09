"""Optional selected-expression layouts and atlases; numerical renderers do not import this module."""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import groupby
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from graph_tool import GraphView
from graph_tool.topology import label_out_component, topological_sort

from trustmebro.extraction.records import Expr
from trustmebro.extraction.storage import BlobCodec, decode_exprs
from trustmebro.graph import ExprGraph, GraphStats, ReachCache, build_graph, graph_stats, root_bitmap
from trustmebro.visualization.archives import mdata_val, read_examples, read_stats, shape_batches
from trustmebro.visualization.identities import shape_sig
from trustmebro.visualization.metrics import atlas_examples
from trustmebro.visualization.products import AnalysisPaths
from trustmebro.visualization.views import TopoView, expr_views, measure_view, resolve_root, view_arrs

from .drawing import GraphStyle, actual_val_ticks, draw_graph, plot_style, save_fig
from .measurements import VIEW_MODES as MODES
from .measurements import VIEW_TITLES as TITLES
from .measurements import GraphSize, ShapeCount, ShapeKey, ShapeSample

# Selected-expression layout records.


@dataclass
class ShapeLayout:
    graph: GraphView
    metrics: ExprGraph
    order: list[int]
    lvls: dict[int, int]
    layers: dict[int, list[int]]
    positions: dict[int, float]
    parents: dict[int, list[int]]


@dataclass
class ShapePanel:
    key: ShapeKey
    theorem: str
    root: int
    layout: ShapeLayout
    expanded: int
    width: np.ndarray
    shared: np.ndarray


# Selected-expression preparation.


def expr_layout(metrics: ExprGraph, root: int) -> ShapeLayout:
    """Native reachable subgraph and topological order; no tree expansion."""
    reachable = label_out_component(metrics.graph, metrics.graph.vertex(root))
    graph = GraphView(metrics.graph, vfilt=reachable)
    order = [int(v) for v in topological_sort(graph)]
    lvls: dict[int, int] = {root: 0}
    parents: dict[int, list[int]] = {node: [] for node in order}
    for node in order:
        for child in metrics.edges[node]:
            lvls[child] = max(lvls.get(child, 0), lvls[node] + 1)
            parents[child].append(node)
    layers: dict[int, list[int]] = {}
    for node in order:
        layers.setdefault(lvls[node], []).append(node)
    positions: dict[int, float] = {}
    # Barycentric ordering within fixed ranks reduces crossings without changing
    # depth. Repeated child edges are preserved, including App(x, x).
    for _, nodes in sorted(layers.items()):
        nodes.sort(key=lambda n: (np.mean([positions[p] for p in parents[n]]) if parents[n] else 0, n))
        positions.update((node, (i + 0.5) / len(nodes)) for i, node in enumerate(nodes))
    return ShapeLayout(graph, metrics, order, lvls, layers, positions, parents)


def _prepare_shapes(samples: list[ShapeSample], min_nodes: int) -> list[ShapePanel]:
    """Decode and prepare each selected expression once for all atlas figures."""
    panels: list[ShapePanel] = []
    seen: set[bytes] = set()
    distinct: dict[tuple[str, int], ShapeSample] = {(sample[1], sample[2]): sample for sample in samples}
    # One graph per selected theorem, never a corpus-wide preparation cache.
    for theorem, roots in groupby(sorted(distinct.values(), key=lambda row: row[1]), key=lambda row: row[1]):
        samples = list(roots)
        metrics = build_graph(decode_exprs(samples[0][3]))
        stats, reach = graph_stats(metrics), ReachCache(metrics)
        for key, _, root, _ in samples:
            if root_bitmap(reach, root).bit_count() < min_nodes:
                continue
            sig = shape_sig(metrics, root)
            if sig in seen:
                continue
            seen.add(sig)
            layout = expr_layout(metrics, root)
            lvls = np.array([layout.lvls[node] for node in layout.order])
            incoming = np.array([len(layout.parents[node]) for node in layout.order])
            width = np.bincount(lvls)
            shared = np.bincount(lvls, weights=incoming > 1) / width
            panels.append(ShapePanel(key, theorem, root, layout, stats.sizes[root], width, shared))
    return panels


def _panel_title(panel: ShapePanel, idx: int) -> str:
    count = len(panel.layout.order)
    title = f"E{idx + 1}: {count:,} nodes; depth {len(panel.width)}\nexpansion ×10^{math.log10(panel.expanded) - math.log10(count):.1f}"
    if panel.key[0] == -2:
        title += f"; {panel.key[2]:,} root occurrences"
    return title


def _shape_descr(panel: ShapePanel, idx: int) -> dict:
    return {
        "panel": idx + 1,
        "stratum": panel.key,
        "selection": ("largest DAG", "deepest DAG", "greatest expansion ratio")[panel.key[1]]
        if panel.key[0] == 4
        else "frequent canonical expression (topology-deduplicated)",
        "root_occs": panel.key[2] if panel.key[0] == -2 else None,
        "theorem": panel.theorem,
        "root": panel.root,
        "nodes": len(panel.layout.order),
        "depth": len(panel.width),
        "expanded_log10": math.log10(panel.expanded),
        "layer_width": panel.width.tolist(),
        "shared_frac": panel.shared.tolist(),
    }


# Atlas panel drawing.


def _draw_dag_panel(ax, panel: ShapePanel, idx: int) -> None:
    # Drawing is optional: defer Cairo-dependent imports until an atlas is requested.

    palette = {"App": "#4c9be8", "Const": "#45c565", "Forall": "#bd71dd", "Lambda": "#edb44b", "Let": "#f27464"}
    graph, metrics, order, lvls, layers, positions, parents = (
        panel.layout.graph,
        panel.layout.metrics,
        panel.layout.order,
        panel.layout.lvls,
        panel.layout.layers,
        panel.layout.positions,
        panel.layout.parents,
    )
    pos = graph.new_vertex_property("vector<double>")
    colors = graph.new_vertex_property("string")
    sizes = graph.new_vertex_property("double")
    edge_colors = graph.new_edge_property("vector<double>")
    for node in order:
        pos[node] = [positions[node], lvls[node] / max(1, len(layers) - 1)]
        colors[node] = palette.get(type(metrics.exprs[node]).__name__, "#aaaaaa")
        sizes[node] = 0.016 if len(parents[node]) > 1 else 0.009
        # Outgoing edges retain insertion order, including parallel edges.
        # Slots distinguish fn/type from arg/body without verbose labels.
        for slot, edge in enumerate(graph.vertex(node).out_edges()):
            edge_colors[edge] = [0.35, 0.65, 0.95, 0.65] if slot == 0 else [0.9, 0.75, 0.4, 0.65]
    draw_graph(graph, pos, ax, GraphStyle(colors, sizes, edge_colors, 0.0015, 0.008))
    ax.set(xlim=(-0.05, 1.05), ylim=(1.05, -0.05))
    ax.set_title(_panel_title(panel, idx), fontsize=9)
    ax.set_axis_off()


def _draw_adj_panel(matrix_ax, panel: ShapePanel, idx: int) -> None:
    layout = panel.layout
    metrics = layout.metrics
    order = layout.order
    lvls = layout.lvls
    layers = layout.layers
    positions = layout.positions
    # Group nodes by longest-path layer, then by the DAG drawing's ordering.
    # Arbitrary topological orders previously obscured branching/merging.
    matrix_order = sorted(order, key=lambda node: (lvls[node], positions[node]))
    rank: dict[int, int] = {node: i for i, node in enumerate(matrix_order)}
    edges: list[tuple[int, int]] = [(rank[node], rank[child]) for node in order for child in metrics.edges[node]]
    if edges:
        src, dst = np.array(edges).T
        raster, _, _ = np.histogram2d(src, dst, bins=min(768, len(order)), range=((0, len(order)), (0, len(order))))
        matrix_ax.imshow(
            np.ma.masked_equal(np.log1p(raster), 0),
            vmin=0,
            vmax=max(math.log(3), float(np.log1p(raster).max())),
            origin="upper",
            extent=(0, len(order), len(order), 0),
            interpolation="nearest",
        )
    matrix_ax.set_title(_panel_title(panel, idx), fontsize=9)
    if not edges:
        matrix_ax.text(0.5, 0.5, "Leaf expr\n(no child refs)", ha="center", va="center", transform=matrix_ax.transAxes)
    bounds = np.cumsum([len(layers[i]) for i in range(len(layers))])[:-1]
    for bound in bounds[:: max(1, len(bounds) // 6)]:
        matrix_ax.axhline(bound, color="white", linewidth=0.3, alpha=0.3)
        matrix_ax.axvline(bound, color="white", linewidth=0.3, alpha=0.3)
    matrix_ax.set(xlabel="Child index, grouped by depth", ylabel="Parent index, grouped by depth")


@plot_style
def _draw_atlases(panels: list[ShapePanel], dir: Path) -> None:
    cols = 4
    rows = math.ceil(len(panels) / cols)
    atlas, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), squeeze=False)
    matrices, matrix_axes = plt.subplots(rows, cols, figsize=(16, 4 * rows), squeeze=False)
    for idx, panel in enumerate(panels):
        _draw_dag_panel(axes.flat[idx], panel, idx)
        _draw_adj_panel(matrix_axes.flat[idx], panel, idx)
    for ax in [*axes.flat, *matrix_axes.flat]:
        if not ax.has_data():
            ax.set_axis_off()
    atlas.suptitle(
        "Complete expression DAGs: root above children; larger nodes have multiple incoming references\nNodes: blue App · green Const · purple Forall · gold Lambda · red Let · grey other; edges: blue first child, gold later children"
    )
    matrices.suptitle(
        "Expression adjacency atlas: nodes grouped by depth; grid lines delimit selected layers; colour = log(1 + references per pixel)"
    )
    for fig, name in ((atlas, "expr-dag-atlas.png"), (matrices, "expr-adj-atlas.png")):
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        save_fig(fig, dir / name, dpi=180)


@plot_style
def _plot_layer_profiles(panels: list[ShapePanel], dir: Path) -> None:
    profiles = [(panel.width, panel.shared) for panel in panels]
    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, len(panels) * 0.25)))
    for ax, field, title in zip(
        axes, (0, 1), ("Layer width (log scale)", "Fraction of nodes with multiple incoming refs"), strict=True
    ):
        img = np.array(
            [
                np.interp(
                    np.linspace(0, 1, 100),
                    np.linspace(0, 1, len(p[field])),
                    np.log10(p[field]) if field == 0 else p[field],
                )
                for p in profiles
            ]
        )
        artist = ax.imshow(
            img, aspect="auto", extent=(0, 100, len(panels) + 0.5, 0.5), vmin=0, vmax=1 if field else None
        )
        ax.set(
            xlabel="Normalized longest-path depth (%)",
            ylabel="Expression ID",
            title=title,
            yticks=range(1, len(panels) + 1),
            yticklabels=[f"E{i}" for i in range(1, len(panels) + 1)],
        )
        bar = fig.colorbar(artist, ax=ax)
        if field == 0:
            actual_val_ticks(bar.ax.yaxis)
    fig.suptitle("Representative and extreme expression profiles—not population frequencies")
    fig.tight_layout()
    save_fig(fig, dir / "expr-layer-profiles.png", dpi=220)


@plot_style
def _plot_abs_depth(panels: list[ShapePanel], dir: Path) -> None:
    profiles = [(panel.width, panel.shared) for panel in panels]
    # Absolute depth complements normalized profiles: retain maximum width in
    # fixed-resolution bins, never allocate a matrix proportional to DAG size.
    depth = max(len(width) for width, _ in profiles)
    resolution = min(1024, depth)
    img = np.full((len(profiles), resolution), np.nan)
    for row, (width, _) in enumerate(profiles):
        bins = np.minimum(np.arange(len(width)) * resolution // depth, resolution - 1)
        maxima = np.zeros(resolution)
        np.maximum.at(maxima, bins, width)
        positive = maxima > 0
        img[row, positive] = np.log10(maxima[positive])
    fig, ax = plt.subplots(figsize=(12, max(4, len(panels) * 0.25)))
    artist = ax.imshow(img, aspect="auto", extent=(0, depth, len(panels) + 0.5, 0.5), interpolation="nearest")
    xlabel = "Longest root-to-node path (edges)"
    ylabel = "Expression ID"
    yticklabels = [f"E{i}" for i in range(1, len(panels) + 1)]
    title = "Layer width at absolute depth; maximum width per bin"
    ax.set(xlabel=xlabel, ylabel=ylabel, yticks=range(1, len(panels) + 1), yticklabels=yticklabels, title=title)
    actual_val_ticks(fig.colorbar(artist, ax=ax, label="Distinct nodes per layer (log scale)").ax.yaxis)
    fig.tight_layout()
    save_fig(fig, dir / "expr-layer-depth.png", dpi=220)


# Expression and comparison atlas rendering.


def _plot_shapes(samples: list[ShapeSample], dir: Path, min_nodes: int = 20) -> None:
    panels = _prepare_shapes(samples, min_nodes)
    if panels:
        _draw_atlases(panels, dir)
        _plot_layer_profiles(panels, dir)
        _plot_abs_depth(panels, dir)
    else:
        for name in ("expr-dag-atlas.png", "expr-adj-atlas.png", "expr-layer-profiles.png", "expr-layer-depth.png"):
            (dir / name).unlink(missing_ok=True)
    descrs = [_shape_descr(panel, idx) for idx, panel in enumerate(panels)]
    (dir / "expr-shapes.json").write_text(json.dumps(descrs, indent=2))


def _plot_atlas(counts: dict[bytes, ShapeCount], exprs: dict[str, tuple[Expr, ...]], out: Path) -> None:
    import matplotlib.pyplot as plt

    rows: list[ShapeCount] = []
    for lower, upper in ((3, 8), (8, 32), (32, 128)):
        group = (item for item in counts.values() if lower <= item.nodes < upper)
        rows.extend(heapq.nsmallest(4, group, key=lambda item: (-item.all, item.theorem, item.root)))
    if not rows:
        return
    with plt.style.context("dark_background"):
        fig, axes = plt.subplots(math.ceil(len(rows) / 4), 4, figsize=(16, 4 * math.ceil(len(rows) / 4)), squeeze=False)
        cached: dict[str, ExprGraph] = {}
        for ax, item in zip(axes.flat, rows, strict=False):
            if item.theorem not in cached:
                cached[item.theorem] = build_graph(exprs[item.theorem])
            layout = expr_layout(cached[item.theorem], item.root)
            graph = layout.graph
            pos = graph.new_vertex_property("vector<double>")
            sizes = graph.new_vertex_property("double")
            for node in layout.order:
                pos[node] = [layout.positions[node], layout.lvls[node] / max(1, len(layout.layers) - 1)]
                sizes[node] = 0.018 if len(layout.parents[node]) > 1 else 0.01
            draw_graph(graph, pos, ax, GraphStyle("#57bde9", sizes, "#a8bdc9", 0.004))
            title = f"{item.nodes} nodes\n{item.all:,} rooted · {item.top:,} top-level"
            ax.set(xlim=(-0.05, 1.05), ylim=(1.05, -0.05), title=title)
            ax.set_axis_off()
        for ax in list(axes.flat)[len(rows) :]:
            ax.set_axis_off()
        fig.suptitle("Frequent complete DAG topologies · node types and values omitted")
        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2.5)
        save_fig(fig, out / "topology-atlas.png", dpi=180)


@plot_style
def _plot_atlas_comparison(counts: dict[bytes, ShapeCount], exprs: dict[str, tuple[Expr, ...]], out: Path) -> None:
    """Keep expressions fixed across columns; include common and large examples."""
    import matplotlib.pyplot as plt

    selected: list[ShapeCount] = []
    for lower, upper in ((3, 8), (8, 32), (32, 128)):
        selected.extend(
            heapq.nsmallest(
                2,
                (item for item in counts.values() if lower <= item.nodes < upper),
                key=lambda item: (-item.top, -item.all, item.theorem, item.root),
            )
        )
    cache: dict[str, tuple[TopoView, ...]] = {}
    sizes: dict[str, tuple[GraphStats, ...]] = {}
    measures: dict[str, tuple[Callable[[Sequence[int]], GraphSize], ...]] = {}

    def views(item: ShapeCount) -> tuple[TopoView, ...]:
        if item.theorem not in cache:
            cache[item.theorem] = expr_views(build_graph(exprs[item.theorem]))
            sizes[item.theorem] = tuple(graph_stats(view.graph) for view in cache[item.theorem])
            measures[item.theorem] = tuple(
                partial(measure_view, view, stats, ReachCache(view.graph), view_arrs(view))
                for view, stats in zip(cache[item.theorem], sizes[item.theorem], strict=True)
            )
        return cache[item.theorem]

    def reduction(item: ShapeCount) -> tuple[int, int, str, int]:
        views(item)
        raw, *transformed = sizes[item.theorem]
        depth = raw.depths[item.root]
        reduced = min(stats.depths[item.root] for stats in transformed)
        return reduced - depth, -depth, item.theorem, item.root

    # Score a bounded set of large candidates instead of rereading the corpus
    # during rendering. These are illustrative examples, not global extrema.
    cands = heapq.nsmallest(
        16,
        (item for item in counts.values() if item.top and item.nodes >= 128),
        key=lambda item: (-item.nodes, item.theorem, item.root),
    )
    selected.extend(heapq.nsmallest(2, cands, key=reduction))
    if not selected:
        return
    fig, axes = plt.subplots(len(selected), len(MODES), figsize=(6 * len(MODES), 4 * len(selected)), squeeze=False)
    for row, item in zip(axes, selected, strict=True):
        for ax, view, measure, title in zip(row, views(item), measures[item.theorem], TITLES, strict=True):
            layout = expr_layout(view.graph, resolve_root(view, item.root))
            graph = layout.graph
            pos = graph.new_vertex_property("vector<double>")
            for node in layout.order:
                pos[node] = [layout.positions[node], layout.lvls[node] / max(1, len(layout.layers) - 1)]
            draw_graph(graph, pos, ax, GraphStyle("#57bde9", 0.01, "#a8bdc9", 0.004))
            size = measure([item.root])
            ax_title = (
                f"{title}\n{size.nodes:,} nodes · depth {size.depth} · {size.refs:,} refs · {size.binders} binders"
            )
            ax.set(xlim=(-0.05, 1.05), ylim=(1.05, -0.05), title=ax_title)
            ax.set_axis_off()
        text = f"{item.theorem}\nroot {item.root}; {item.top:,} top-level occurrences"
        row[0].text(0, -0.12, text, transform=row[0].transAxes, fontsize=8)
    fig.suptitle(
        "Same expression across coercion/instance views · common shapes and depth reductions among large examples"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98), h_pad=3)
    save_fig(fig, out / "plumbing-topology-atlas.png", dpi=130)


# Atlas entry points.


def render_atlas(stats: Path, out: Path) -> None:
    examples: dict[bytes, ShapeCount] = {}
    for batch in shape_batches(AnalysisPaths(stats).topo()):
        examples = atlas_examples(examples | dict(batch))
    with read_examples(AnalysisPaths(stats).examples) as (dicts, rows):
        codec = BlobCodec(dicts)
        exprs = {name: decode_exprs(blob, codec) for name, blob in rows}
    _plot_atlas(examples, exprs, out)
    _plot_atlas_comparison(examples, exprs, out)


def render_expr_atlas(path: Path, out: Path, min_nodes: int) -> None:
    if min_nodes < mdata_val(path, "atlas_min_nodes", int):
        raise ValueError("atlas minimum is below the analyzed selection threshold; rerun analysis")
    samples = [record[1:] for record in read_stats(path, {"shape"}) if record[0] == "shape"]
    _plot_shapes(samples, out, min_nodes)
