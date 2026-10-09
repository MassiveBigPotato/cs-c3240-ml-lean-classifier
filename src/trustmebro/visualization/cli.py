"""Analyze the corpus, render saved measurements, or run both phases."""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from compression import zstd
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, TypedDict, assert_never, cast

from trustmebro.runtime import Phase
from trustmebro.visualization.archives import read_summary
from trustmebro.visualization.products import (
    GRAPH_FAMILIES,
    Analysis,
    AnalysisCfg,
    AnalysisPaths,
    RenderStage,
    SrcStamp,
    analysis_set,
)

# Command types and records.


type Cmd = Literal["analyze", "graphs"]


class StageReport(TypedDict):
    dur_sec: float
    peak_rss_mib: float


# Supported analyses and parsed defaults.


ANALYSES: tuple[Analysis, ...] = ("metrics", "topology", "patterns")


@dataclass(frozen=True)
class Args:
    """Settings for graph requirements and private isolated stages."""

    stats: Path = Path("data/analysis")
    output: Path = Path("data/graphs")
    db: Path | None = None
    workers: int = 2
    atlas_min_nodes: int = 20
    dot_diam: int = 1
    limit: int | None = None
    replace: bool = False
    aggregation_memory_mib: int = 512
    timing_dir: Path | None = None
    skip_embedding: bool = False
    pattern_depths: tuple[int, ...] = (1, 2, 3)
    graphs: tuple[str, ...] = ("all",)
    analyses: tuple[Analysis, ...] = ANALYSES
    inventory: Path | None = None
    feature_stats: Path | None = None

    @classmethod
    def from_namespace(cls, namespace: argparse.Namespace) -> Args:
        vals = vars(namespace)
        choices: dict[str, Any] = {
            name: tuple(vals[name]) for name in ("pattern_depths", "graphs", "analyses") if name in vals
        }
        return cls(**(vals | choices))


DEFAULT_ARGS = Args()


# Argument parsing and artifact requirements.


def _pos(val: str) -> int:
    num = int(val)
    if num < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return num


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reuse or compute selected measurements, then render graphs.",
        epilog="Families: " + "; ".join(f"{name}: {family.descr}" for name, family in GRAPH_FAMILIES.items()),
    )
    parser.add_argument("--stats", type=Path, default=DEFAULT_ARGS.stats)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARGS.output)
    parser.add_argument(
        "--db", type=Path, default=DEFAULT_ARGS.db, help="required only for missing/regenerated analyses"
    )
    parser.add_argument("--workers", type=_pos, default=DEFAULT_ARGS.workers)
    parser.add_argument("--atlas-min-nodes", type=_pos, default=DEFAULT_ARGS.atlas_min_nodes)
    parser.add_argument("--dot-diameter", dest="dot_diam", type=_pos, default=DEFAULT_ARGS.dot_diam)
    parser.add_argument("--limit", type=_pos, help="first N theorems for a smoke test")
    parser.add_argument("--replace", action="store_true", help="regenerate prerequisites of selected families")
    parser.add_argument("--aggregation-memory-mib", type=_pos, default=DEFAULT_ARGS.aggregation_memory_mib)
    parser.add_argument("--timing-dir", type=Path)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--pattern-depths", type=_pos, nargs="+", default=DEFAULT_ARGS.pattern_depths)
    parser.add_argument("--graphs", choices=("all", *GRAPH_FAMILIES), nargs="+", default=DEFAULT_ARGS.graphs)
    parser.add_argument("--inventory", type=Path, help="frozen vocabulary; enables selected-shape plots")
    parser.add_argument("--feature-stats", type=Path, help="empirical feature diagnostics")
    return parser


def _require_graphs(
    paths: AnalysisPaths, graphs: Sequence[str], inventory: Path | None = None, feature_stats: Path | None = None
) -> None:
    reqs: dict[Analysis, set[str]] = {}
    for name in graphs:
        family = GRAPH_FAMILIES[name]
        if family.analysis is None:
            src = feature_stats if name == "features" else inventory
            flag = "--feature-stats" if name == "features" else "--inventory"
            if src is None or not src.is_file():
                raise ValueError(f"{name} graphs require {flag} pointing to their saved input")
            continue
        reqs.setdefault(family.analysis, set()).update(path.name for path in family.artifacts(paths))
    paths.require(reqs, reqs)
    if "embedding" in graphs:
        modes = read_summary(paths.embedding_info, tuple[str, ...])
        paths.require(("metrics",), {"metrics": (paths.embedding(mode).name for mode in modes)})


# Isolated stage dispatch and reporting.


def _analyze(stage: str, args: Args) -> dict[str, int]:
    from .scan import analyze_embeddings, collect_analysis

    if stage == "embedding":
        analyze_embeddings(args.stats, args.workers)
        return {}
    if args.db is None:
        raise ValueError("analysis requires a source database")
    return collect_analysis(
        args.db,
        args.stats,
        analyses=tuple(args.analyses),
        limit=args.limit,
        workers=args.workers,
        atlas_min_nodes=args.atlas_min_nodes,
        depths=tuple(dict.fromkeys(args.pattern_depths)),
        aggr_mem=args.aggregation_memory_mib,
        timing_dir=args.timing_dir,
    )


def _render(stage: RenderStage, args: Args) -> None:
    # Heavy visualization imports remain in the isolated renderer process.
    import matplotlib

    matplotlib.use("Agg")
    match stage:
        case "metrics":
            from .state_render import render

            render(
                AnalysisPaths(args.stats).analysis("metrics"),
                args.output,
                atlas_min_nodes=args.atlas_min_nodes,
                dot_diam=args.dot_diam,
                families={name for name in args.graphs if GRAPH_FAMILIES[name].analysis == "metrics"},
            )
        case "topology":
            from . import stral_render

            stral_render.render_shapes(args.stats, args.output, args.dot_diam)
        case "comparisons":
            from . import stral_render

            stral_render.render_pairs(args.stats, args.output, args.dot_diam)
        case "heads":
            from . import stral_render

            stral_render.render_heads(args.stats, args.output)
        case "topology-atlas":
            from .atlas import render_atlas

            render_atlas(args.stats, args.output)
        case "local-patterns":
            from . import stral_render

            stral_render.render_patterns(args.stats, args.output, args.dot_diam)
        case "candidates":
            from .vocab_render import render_inventory

            if args.inventory is None:
                raise ValueError("candidate graphs require --inventory")
            render_inventory(args.inventory, args.output, min_nodes=args.atlas_min_nodes)
        case "features":
            from .vocab_render import render_features

            if args.feature_stats is None:
                raise ValueError("feature graphs require --feature-stats")
            render_features(args.feature_stats, args.output)
        case _:
            assert_never(stage)


def _isolated() -> None:
    """Private stdin protocol, not another public argparse command surface."""
    cmd, stage, vals = json.load(sys.stdin)
    for name in ("stats", "output", "db", "timing_dir", "inventory", "feature_stats"):
        if vals.get(name) is not None:
            vals[name] = Path(vals[name])
    args = Args.from_namespace(argparse.Namespace(**vals))
    started = time.perf_counter()
    report = _analyze(stage, args) if cmd == "analyze" else (_render(stage, args) or {})
    print(
        json.dumps(
            report
            | {
                "dur_sec": time.perf_counter() - started,
                "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            }
        )
    )


def _stage_report(cmd: list[str], data: str) -> StageReport:
    """Own the stage's process group so cancellation also stops its workers."""
    with subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, start_new_session=True
    ) as process:
        try:
            stdout, _ = process.communicate(data)
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            except ProcessLookupError:
                pass
            raise
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, cmd, stdout)
    return cast(StageReport, json.loads(stdout))


def _run_stages(cmd: Cmd, stages: Sequence[str], args: Args) -> dict[str, StageReport]:
    reports: dict[str, StageReport] = {}
    paths = AnalysisPaths(args.stats)
    if cmd == "graphs":
        _require_graphs(paths, args.graphs, args.inventory, args.feature_stats)
    for stage in stages:
        # Pin an immutable generation before launching each renderer. Writers
        # may publish another manifest without changing this request's inputs.
        stage_args = args
        if cmd == "graphs":
            analysis = "metrics" if stage == "metrics" else GRAPH_FAMILIES[stage].analysis
            if analysis is not None:
                stage_args = replace(args, stats=paths.dir(analysis))
        # Shared analysis owns live progress in its subprocess; two refresh
        # threads on inherited stderr would repeatedly overwrite each other.
        with Phase(f"{cmd.capitalize()}: {stage}", refresh=not (cmd == "analyze" and stage == "shared")) as progress:
            report = _stage_report(
                [sys.executable, "-c", "from trustmebro.visualization.cli import _isolated; _isolated()"],
                json.dumps((cmd, stage, stage_values(cmd, stage, stage_args)), default=str),
            )
            reports[stage] = report
            progress.details = f"{report['dur_sec']:.2f}s; peak process RSS {report['peak_rss_mib']:,.0f} MiB"
    return reports


# Command entry points.


def _missing(paths: AnalysisPaths, graphs: Sequence[str], args: Args) -> tuple[list[Analysis], bool]:
    """Discover current products through their manifest without reading the source DB."""
    analyses: list[Analysis] = []
    for name in graphs:
        kind = GRAPH_FAMILIES[name].analysis
        if kind is not None and kind not in analyses:
            analyses.append(kind)
    if args.replace:
        return analyses, False
    missing: list[Analysis] = []
    embedding_only = False
    srcs = set()
    for kind in analyses:
        gen = paths.manifest.get(kind)
        if gen is None:
            if paths.analysis(kind).exists():
                raise ValueError(f"unpublished/obsolete {kind} statistics; regenerate with --replace")
            missing.append(kind)
            continue
        srcs.add(gen.src)
        if args.db is not None and args.db.is_file() and gen.src != SrcStamp.from_path(args.db, args.limit):
            raise ValueError("statistics do not match the source selection; regenerate with --replace")
        if (kind == "patterns" and gen.cfg.depths != tuple(dict.fromkeys(args.pattern_depths))) or (
            any(name in graphs for name in ("expr-atlas", "topology-atlas"))
            and gen.cfg.atlas_min_nodes != args.atlas_min_nodes
        ):
            raise ValueError(f"{kind} analysis settings differ; regenerate with --replace")
        names = {
            p.name
            for name in graphs
            if GRAPH_FAMILIES[name].analysis == kind
            for p in GRAPH_FAMILIES[name].artifacts(paths)
        }
        # Embeddings can be prepared from saved descriptors without another corpus scan.
        if kind == "metrics" and "embedding" in graphs and not gen.cfg.embedding:
            names.discard(paths.embedding_info.name)
            embedding_only = True
        paths.require((kind,), {kind: names})
    if len(srcs) > 1:
        raise ValueError("selected statistics describe inconsistent source populations; regenerate with --replace")
    if missing and srcs and args.db is not None and SrcStamp.from_path(args.db, args.limit) not in srcs:
        raise ValueError("new analyses would mix inconsistent source populations")
    return missing, embedding_only


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = Args.from_namespace(parser.parse_args(argv))
    if args.atlas_min_nodes < 2:
        parser.error("--atlas-min-nodes must be at least 2")
    graphs = list(GRAPH_FAMILIES) if "all" in args.graphs else list(dict.fromkeys(args.graphs))
    if "all" in args.graphs:
        if args.inventory is None:
            graphs.remove("candidates")
        if args.feature_stats is None:
            graphs.remove("features")
    if args.skip_embedding:
        graphs = [name for name in graphs if name != "embedding"]
    if not graphs:
        parser.error("no graph families selected")
    selected = replace(args, graphs=tuple(graphs))
    try:
        paths = AnalysisPaths(args.stats)
        db = args.db
        missing, embedding_only = _missing(paths, graphs, args)
        if missing or embedding_only:
            if args.db is None:
                existing = (
                    [paths.manifest[kind].src.path for kind in ANALYSES if kind in paths.manifest]
                    if not args.replace
                    else []
                )
                args = replace(args, db=Path(existing[0]) if existing else Path("data/mathlib.db"))
                db = args.db
                selected = replace(selected, db=args.db)
                missing, embedding_only = _missing(paths, graphs, args)
            if db is None or not db.is_file():
                raise ValueError(f"missing prerequisites require a source database: {args.db}")
        reports: dict[str, object] = {}
        started = time.perf_counter()
        if missing or embedding_only:
            if db is None:
                raise ValueError("missing prerequisites require a source database")
            kinds = tuple(dict.fromkeys((*missing, *(("metrics",) if embedding_only else ()))))
            embedding = "embedding" in graphs and "metrics" in kinds
            cfg = AnalysisCfg(args.atlas_min_nodes, tuple(dict.fromkeys(args.pattern_depths)), embedding)
            with analysis_set(args.stats, kinds, db, limit=args.limit, cfg=cfg, replace=True) as pending:
                if embedding_only and "metrics" not in missing:
                    old = paths.manifest["metrics"]
                    for name in old.artifacts:
                        shutil.copyfile(paths.artifact("metrics", name), pending.root / name)
                stages = (["shared"] if missing else []) + (["embedding"] if embedding else [])
                reports["analysis"] = _run_stages(
                    "analyze", stages, replace(selected, stats=pending.root, analyses=tuple(missing))
                )
        _require_graphs(AnalysisPaths(args.stats), graphs, args.inventory, args.feature_stats)
        args.output.mkdir(parents=True, exist_ok=True)
        stages = list(dict.fromkeys(GRAPH_FAMILIES[name].stage for name in graphs))
        reports["rendering"] = _run_stages("graphs", stages, selected)
        print(json.dumps({"duration_sec": time.perf_counter() - started, "stages": reports}))
    except (OSError, ValueError, zstd.ZstdError, subprocess.CalledProcessError) as error:
        parser.error(f"{error}; unusable statistics require regeneration with --replace")
    return 0


def stage_values(cmd: Cmd, stage: str, cfg: Args) -> dict[str, object]:
    """Narrow private requests; no parser namespace or unrelated controls cross stages."""
    if cmd == "analyze":
        fields = (
            ("stats", "workers")
            if stage == "embedding"
            else (
                "stats",
                "workers",
                "db",
                "limit",
                "analyses",
                "atlas_min_nodes",
                "pattern_depths",
                "aggregation_memory_mib",
                "timing_dir",
            )
        )
    elif stage == "metrics":
        fields = ("stats", "output", "graphs", "atlas_min_nodes", "dot_diam")
    elif stage == "candidates":
        fields = ("output", "inventory")
    elif stage == "features":
        fields = ("output", "feature_stats")
    else:
        fields = ("stats", "output", "atlas_min_nodes", "dot_diam")
    return {name: getattr(cfg, name) for name in fields}
