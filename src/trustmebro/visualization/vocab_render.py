"""Selected frozen vocabulary adjacency and actual column allocation; no invented support."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from graph_tool import Graph, PropertyMap
from graph_tool.draw import graph_draw

from trustmebro.preprocessing.archives import ARRAY_DTYPE, read_diagnostics, read_vocab
from trustmebro.preprocessing.layout import compile_vocab
from trustmebro.preprocessing.records import Vocab

from .drawing import count_ticks, plot_style, save_fig, save_single

ATLAS_MAX_NODES = 128
ATLAS_MAX_EDGES = 512
ATLAS_PANELS = 18


def _dag_positions(graph: Graph, links: list[tuple[int, int, int]]) -> PropertyMap:
    """Use the installed Graphviz dot engine, not a custom layout algorithm.

    graph_tool 3.9 no longer exposes graphviz_draw. Only bounded atlas shapes
    cross this CLI boundary; ordering=out preserves the declared operand order.
    """
    nodes = (f"v{idx};" for idx in range(graph.num_vertices()))
    edges = (f"v{src} -> v{dst};" for src, dst, _ in links)
    data = "digraph { graph [ordering=out]; " + " ".join((*nodes, *edges)) + " }"
    result = subprocess.run(["dot", "-Tjson"], input=data, capture_output=True, text=True, check=True)
    positions = graph.new_vp("vector<double>")
    for node in json.loads(result.stdout)["objects"]:
        positions[int(node["name"][1:])] = tuple(map(float, node["pos"].split(",")))
    return positions


@plot_style
def _atlas(vocab: Vocab, output: Path, min_nodes: int) -> None:
    nodes = np.asarray([len(entry.edges) for entry in vocab.entries])
    edge_counts = np.asarray([sum(len(refs) for refs in entry.edges if refs is not None) for entry in vocab.entries])
    eligible = np.flatnonzero((nodes >= min_nodes) & (nodes <= ATLAS_MAX_NODES) & (edge_counts <= ATLAS_MAX_EDGES))
    # Canonical vocabulary order is meaningful; no discarded-candidate support is fabricated.
    selected = eligible[:ATLAS_PANELS]
    if not len(selected):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            f"No selected shapes with {min_nodes}–{ATLAS_MAX_NODES} positions and ≤{ATLAS_MAX_EDGES} operand edges",
            ha="center",
        )
    else:
        rows = (len(selected) + 2) // 3
        fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows), squeeze=False)
        for ax in axes.flat:
            ax.axis("off")
        for ax, idx in zip(axes.flat, selected):
            edges = vocab.entries[idx].edges
            graph = Graph(directed=True)
            graph.add_vertex(len(edges))
            links = [
                (src, dst, slot) for src, refs in enumerate(edges) if refs is not None for slot, dst in enumerate(refs)
            ]
            slots = graph.new_ep("int")
            if links:
                graph.add_edge_list(np.asarray(links, dtype=np.int64), eprops=[slots])
            labels = graph.new_vp("string", vals=[str(pos) for pos in range(len(edges))])
            colours = graph.new_vp("string", vals=["#ffaf45" if refs is None else "#57bde9" for refs in edges])
            positions = _dag_positions(graph, links)
            # Labels expose canonical positions; slot labels preserve ordered
            # operands and parallel edges, rather than merely a spanning tree.
            graph_draw(
                graph,
                pos=positions,
                mplfig=ax,
                vertex_text=labels,
                vertex_fill_color=colours,
                vertex_size=18,
                vertex_font_size=10,
                edge_text=slots,
                edge_font_size=8,
                edge_color="#a8bdc9",
                vertex_pen_width=0,
            )
            ax.set_title(f"Entry {idx + 1}: {int(nodes[idx])} positions; {int(edge_counts[idx])} edges", fontsize=10)
    fig.suptitle("Selected nontrivial vocabulary shapes (node kinds and values omitted)")
    fig.text(
        0.08,
        0.01,
        f"Up to {ATLAS_PANELS} shapes with {min_nodes}–{ATLAS_MAX_NODES} positions and ≤{ATLAS_MAX_EDGES} edges, "
        "in vocabulary order. "
        "Orange: wildcard frontier; blue terminal: true leaf.\nNode numbers are fragment-local IDs; "
        "edge numbers are operand slots; repeated IDs/edges retain sharing.",
        fontsize=10,
    )
    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.35)
    save_fig(fig, output / "candidate-atlas.png", tight=True)


def render_inventory(src: Path, output: Path, *, min_nodes: int = 20) -> None:
    """Render selected adjacency and actual column allocations, never guessed coverage."""
    vocab = read_vocab(src)
    layout = compile_vocab(vocab)
    output.mkdir(parents=True, exist_ok=True)
    _atlas(vocab, output, min_nodes)
    with plt.rc_context():
        fig, ax = plt.subplots(figsize=(12, 8))
        shape_dims = 2 * layout.block_width
        ax.bar(
            ("Graph-position channels", "Statistics / attributes / names"),
            (shape_dims, layout.width - shape_dims),
            color=("#57bde9", "#ffaf45"),
        )
        ax.set(
            ylabel="Allocated feature dimensions",
            title=f"Frozen vocabulary: {len(vocab.entries):,} entries; {layout.width:,} total dimensions",
        )
        fig.text(
            0.1,
            0.015,
            "Column allocation is not empirical activation or corpus coverage; use feature diagnostics.",
            fontsize=9,
        )
        save_single(fig, output / "vocabulary-dimensions.png")


@plot_style
def render_features(src: Path, output: Path) -> None:
    report = read_diagnostics(src)
    output.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    counts = (report.covered_theorems, report.covered_states, report.goal_covered, report.hyp_covered)
    totals = (report.theorems, report.states, report.states, report.states)
    rates = [count / total if total else 0 for count, total in zip(counts, totals, strict=True)]
    labels = ("Theorems\n(any state)", "States\n(either role)", "States\ngoal", "States\nhypotheses")
    bars = ax.bar(labels, rates, color="#57bde9")
    ax.bar_label(bars, labels=[f"{count:,}/{total:,}" for count, total in zip(counts, totals, strict=True)])
    ax.set(
        ylim=(0, 1.12),
        ylabel="Fraction with at least one nontrivial selected pattern",
        title="Actual representation coverage",
    )
    fig.text(
        0.1,
        0.015,
        "Nontrivial means more than one canonical position; this does not measure predictive usefulness.",
        fontsize=9,
    )
    save_single(fig, output / "feature-coverage.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    zeros: list[str] = []
    for name, hgram in (
        ("Active feature dimensions", report.active_dims),
        ("Active role/pattern pairs", report.active_patterns),
    ):
        vals = np.asarray([val for val, _ in hgram], dtype=np.int64)
        counts_array = np.asarray([count for _, count in hgram], dtype=np.int64)
        cumulative = np.cumsum(counts_array) / max(1, report.states)
        positive = vals > 0
        ax.step(vals[positive], cumulative[positive], where="post", label=name)
        zeros.append(f"{name}: {dict(hgram).get(0, 0):,} zero states")
    ax.set(
        xscale="log",
        ylim=(0, 1.01),
        xlabel="Nonzero columns per state (actual counts)",
        ylabel="Fraction of all states at or below this count",
        title="Actual sparse activation distributions",
    )
    count_ticks(ax.xaxis)
    ax.legend()
    fig.text(
        0.1, 0.015, "; ".join(zeros) + "\nZero states remain in the CDF denominator; no state sampling.", fontsize=9
    )
    save_single(fig, output / "feature-activation.png")

    fig, ax = plt.subplots(figsize=(12, 8))
    if report.pair_limit:
        pairs = np.frombuffer(report.pairs, dtype=ARRAY_DTYPE).reshape(report.pair_limit, report.pair_limit)
        support = np.diag(pairs).astype(float)
        union = support[:, None] + support[None, :] - pairs
        jaccard = np.divide(pairs, union, out=np.zeros_like(pairs, dtype=float), where=union > 0)
        img = ax.imshow(jaccard, vmin=0, vmax=1, cmap="turbo", interpolation="nearest", origin="lower")
        fig.colorbar(img, ax=ax, label="Fraction of union states containing both patterns (Jaccard)")
        ax.set(xlabel="Vocabulary entry index", ylabel="Vocabulary entry index")
    else:
        ax.text(0.5, 0.5, "Co-occurrence diagnostic disabled", ha="center", transform=ax.transAxes)
    ax.set_title("State-level pattern co-occurrence (goal/hypothesis roles unioned)")
    fig.text(
        0.1,
        0.015,
        f"First {report.pair_limit:,} entries only; all {report.states:,} converted states. "
        "All other entries remain in the features.\nRaw pair counts are retained; colour normalization happens here.",
        fontsize=9,
    )
    save_single(fig, output / "feature-cooccurrence.png")
