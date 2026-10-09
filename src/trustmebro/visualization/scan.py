"""Bounded scan workers and shared-memory reuse orchestration."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Generator, Iterable, Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from functools import lru_cache, partial
from hashlib import file_digest
from itertools import batched
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, cast
from zipfile import ZipFile

import msgspec
import numpy as np
from graph_tool import openmp_set_num_threads
from numpy.typing import NDArray

from trustmebro.artifacts import report_file_sizes
from trustmebro.extraction import records as r
from trustmebro.extraction.storage import BlobCodec, Dicts, decode_exprs, decode_trns, stored_dicts
from trustmebro.graph import AppHeads, ExprGraph, ReachCache, build_graph, graph_stats
from trustmebro.runtime import Phase, TimingLog, ready_results, timed
from trustmebro.visualization import patterns
from trustmebro.visualization.archives import (
    StatsSink,
    packed_pattern_batches,
    pattern_heads,
    stats_writer,
    write_examples,
    write_pairs,
    write_pattern_batches,
    write_pattern_stats,
    write_patterns,
    write_shapes,
    write_summary,
)
from trustmebro.visualization.identities import shape_sig
from trustmebro.visualization.metrics import (
    ExampleCounts,
    FreqTotals,
    MetricCounts,
    ReuseMeasures,
    TheoremMetrics,
    add_freq_totals,
    add_metrics,
    add_topo,
    common_cands,
    comparison_summary,
    finish_samples,
    freq_stats,
    head_stats,
    measure_reuse,
    measure_theorem,
    select_common,
    state_summary,
    topo_stats,
)
from trustmebro.visualization.patterns import merge_patterns, pattern_stats, remap_pattern_heads
from trustmebro.visualization.products import Analysis, AnalysisPaths
from trustmebro.visualization.views import (
    Match,
    Observations,
    PatternSigs,
    SigReuse,
    TopoView,
    affected_nodes,
    app_heads,
    find_matches,
    observe_roots,
    observed_heads,
    prepare_views,
    sig_reuse,
)

from . import metrics
from .measurements import PATTERN_MODES, VIEW_MODES, Mdata, PatternBatch, PatternStats, TopoMeasures, ViewMode

# Scan inputs, configuration, and worker records.


class TheoremSrc(msgspec.Struct, frozen=True):
    name: str
    exprs: bytes
    trns: bytes


class ReuseJob(msgspec.Struct, frozen=True):
    src: TheoremSrc
    roots: Sequence[int]


@dataclass(frozen=True)
class MeasureCfg:
    analyses: tuple[Analysis, ...]
    depths: tuple[int, ...]
    timing_dir: Path | None = None
    min_nodes: int = 20


@dataclass(frozen=True)
class SrcSelection:
    path: Path
    limit: int | None = None

    def rows(self, db: sqlite3.Connection) -> Iterator[TheoremSrc]:
        query = "SELECT name,exprs,trns FROM theorems ORDER BY id"
        params = (self.limit,) if self.limit is not None else ()
        yield from (TheoremSrc(*row) for row in db.execute(query + (" LIMIT ?" if params else ""), params))


@dataclass(frozen=True)
class ScanPlan:
    src: SrcSelection
    analysis: MeasureCfg
    workers: int


@dataclass(frozen=True)
class SharedIdx:
    name: str
    count: int


@dataclass(frozen=True)
class WorkerCfg:
    dicts: Dicts | None
    analysis: MeasureCfg | None = None
    reuse: SharedIdx | None = None
    timing_dir: Path | None = None


@dataclass
class WorkerCtxt:
    codec: BlobCodec
    analysis: MeasureCfg | None
    shared: SharedPatterns | None = None
    memory: SharedMemory | None = None
    timings: TimingLog | None = None

    @classmethod
    def create(cls, cfg: WorkerCfg) -> WorkerCtxt:
        codec = BlobCodec(cfg.dicts)
        timings = TimingLog(cfg.timing_dir, "worker") if cfg.timing_dir is not None else None
        if cfg.reuse is None:
            return cls(codec, cfg.analysis, timings=timings)
        memory = SharedMemory(name=cfg.reuse.name, track=False)
        shared = SharedPatterns(np.ndarray((cfg.reuse.count,), dtype="V32", buffer=memory.buf))
        return cls(codec, cfg.analysis, shared, memory, timings)


@dataclass
class Measured:
    general: TheoremMetrics | None
    topo: TopoMeasures | None
    patterns: dict[tuple[ViewMode, int], PatternBatch]
    name: str = ""


@dataclass(frozen=True)
class TheoremInput:
    name: str
    graph: ExprGraph
    trns: tuple[r.Trn, ...]


PATTERN_MERGE_FAN_IN = 32


# Per-theorem preparation and measurement.


def _prepare(row: TheoremSrc, codec: BlobCodec, timings: TimingLog | None = None) -> TheoremInput:
    call = partial(timed, timings, row.name)
    exprs = call("decode_exprs", decode_exprs, row.exprs, codec)
    graph = call("build_graph", build_graph, exprs)
    trns = call("decode_trns", decode_trns, row.trns, codec)
    return TheoremInput(row.name, graph, trns)


def _measure(row: TheoremSrc, codec: BlobCodec, cfg: MeasureCfg, timings: TimingLog | None = None) -> Measured:
    call = partial(timed, timings, row.name)
    data = _prepare(row, codec, timings)
    graph, cache = data.graph, ReachCache(data.graph)
    general = "metrics" in cfg.analyses
    topo = "topology" in cfg.analyses
    local_patterns = "patterns" in cfg.analyses
    stats = call("graph_stats", graph_stats, graph) if general or topo else None
    local: dict[tuple[ViewMode, int], PatternBatch] = {}
    observations: Observations | None = None
    heads: AppHeads | None = None
    matches: dict[int, Match] = {}
    views: dict[ViewMode, TopoView] = {}
    reuse: dict[ViewMode, SigReuse] = {}
    if topo or local_patterns:
        heads = call("app_heads", app_heads, graph)
        matches = call("find_matches", find_matches, graph.exprs, heads)
        modes = VIEW_MODES if topo else PATTERN_MODES
        views = call("prepare_views", prepare_views, graph, modes, matches=matches)
        observations = call("observe_roots", observe_roots, cache, data.trns)
        reuse = call("signature_reuse", sig_reuse, views)
    if local_patterns:
        assert observations is not None and heads is not None
        affected = call("affected_nodes", affected_nodes, graph, matches)
        named_heads = call("observed_heads", observed_heads, graph.exprs, heads, observations)
        pattern_obs = call("pattern_observations", patterns.pattern_observations, observations, affected, named_heads)
        previous_sigs: PatternSigs = {}
        previous_mode: ViewMode | None = None
        for mode in PATTERN_MODES:
            inherited = reuse.get(mode)
            cached = None
            if inherited is not None and inherited.src == previous_mode:
                cached = inherited.labelled, previous_sigs
            batches, previous_sigs = call(
                f"patterns.{mode}",
                patterns.measure_patterns,
                views[mode],
                cfg.depths,
                pattern_obs,
                mode=mode,
                timings=timings,
                name=data.name,
                inherited=cached,
            )
            previous_mode = mode
            local.update(((mode, depth), batch) for depth, batch in batches.items())
        del previous_sigs
    general_result = None
    topo_result = None
    if general:
        assert stats is not None
        general_result = call(
            "measure_theorem", measure_theorem, data.name, graph, stats, cache, data.trns, min_nodes=cfg.min_nodes
        )
    if topo:
        assert stats is not None and observations is not None
        topo_result = call(
            "measure_topo", metrics.measure_topo, data.name, views, observations, stats, cache, timings, reuse
        )
    return Measured(general_result, topo_result, local, data.name)


def measure_src(db: sqlite3.Connection, plan: ScanPlan, timings: TimingLog | None = None) -> Iterator[Measured]:
    """One bounded decode/prepare/measure pass shared by all selected analyses."""
    rows = plan.src.rows(db)
    worker_cfg = WorkerCfg(stored_dicts(db), analysis=plan.analysis, timing_dir=plan.analysis.timing_dir)
    if plan.workers == 1:
        openmp_set_num_threads(1)
        codec = BlobCodec(worker_cfg.dicts)
        for row in rows:
            yield timed(timings, row.name, "worker_total", _measure, row, codec, plan.analysis, timings)
    else:
        with _worker_pool(plan.workers, worker_cfg) as pool:
            yield from ready_results(pool, rows, plan.workers, _run_analysis, timings)


# Process-local adapters and bounded scheduling.


# ProcessPoolExecutor initializers cannot pass their result to jobs. Keep its
# process-local lookup ONLY in these adapters; calculations take explicit inputs.
_worker_ctxt: WorkerCtxt | None = None


def _init_worker(cfg: WorkerCfg) -> None:
    global _worker_ctxt
    openmp_set_num_threads(1)
    _worker_ctxt = WorkerCtxt.create(cfg)


def _run_analysis(row: TheoremSrc) -> Measured:
    if _worker_ctxt is None or _worker_ctxt.analysis is None:
        raise RuntimeError("analysis worker has not been initialized")
    timings = _worker_ctxt.timings
    try:
        return timed(
            timings, row.name, "worker_total", _measure, row, _worker_ctxt.codec, _worker_ctxt.analysis, timings
        )
    finally:
        if timings is not None:
            timings.flush()


def _run_reuse(row: ReuseJob) -> ReuseMeasures:
    if _worker_ctxt is None or _worker_ctxt.shared is None:
        raise RuntimeError("reuse worker has not been initialized")
    timings = _worker_ctxt.timings
    try:
        return timed(
            timings,
            row.src.name,
            "reuse_total",
            _measure_global_reuse,
            row,
            _worker_ctxt.codec,
            _worker_ctxt.shared,
            timings,
        )
    finally:
        if timings is not None:
            timings.flush()


def _worker_pool(workers: int, cfg: WorkerCfg) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(
        max_workers=workers, mp_context=get_context("spawn"), initializer=_init_worker, initargs=(cfg,)
    )


# Shared cross-theorem reuse pass.


class SharedPatterns:
    """Sorted full SHA-256 keys: compact exact-key lookup, no Python set.

    Workers map the same shared-memory bytes and batch their binary searches.
    Full fingerprints are compared, not truncated integer hashes.
    """

    def __init__(self, keys: NDArray[np.void]):
        self.keys = keys
        self.keys.flags.writeable = False

    def find(self, digests: Iterable[bytes]) -> NDArray[np.bool_]:
        queries = np.frombuffer(b"".join(digests), dtype="V32")
        if not len(self.keys):
            return np.zeros(len(queries), dtype=bool)
        positions = self.keys.searchsorted(queries)
        return (positions < len(self.keys)) & (self.keys[np.minimum(positions, len(self.keys) - 1)] == queries)


def _measure_global_reuse(
    row: ReuseJob, codec: BlobCodec, shared: SharedPatterns, timings: TimingLog | None = None
) -> ReuseMeasures:
    data = _prepare(row.src, codec, timings)
    return timed(
        timings, data.name, "measure_reuse", measure_reuse, data.name, data.graph, data.trns, shared, row.roots
    )


@contextmanager
def _shared_pattern_memory(keys: NDArray[np.void]) -> Generator[str]:
    memory = SharedMemory(create=True, size=max(1, keys.nbytes))
    try:
        view = np.ndarray(keys.shape, dtype=keys.dtype, buffer=memory.buf)
        view[:] = keys
        del view
        yield memory.name
    finally:
        memory.close()
        memory.unlink()


def _write_global_reuse(
    db: sqlite3.Connection,
    plan: ScanPlan,
    counts: MetricCounts,
    write: StatsSink,
    progress: Phase | None,
    timings: TimingLog | None = None,
) -> None:
    """Second streaming pass: cross-theorem eligibility requires corpus counts.

    Workers retain graphs locally; only completed plot rows cross process boundaries.
    """
    freqs = counts.freqs
    workers, dicts = plan.workers, stored_dicts(db)
    sampled = {(sample.name, sample.root): sample for sample in counts.examples.stral}
    roots_by_theorem: dict[str, list[int]] = {}
    for name, root in sampled:
        roots_by_theorem.setdefault(name, []).append(root)
    shared_count = sum(count.theorems > 1 for count in freqs.values())
    keys = np.empty(shared_count, dtype="V32")
    for idx, digest in enumerate(digest for digest, counts in freqs.items() if counts.theorems > 1):
        keys[idx] = np.void(digest)
    keys.sort()
    rows: Iterable[ReuseJob] = (ReuseJob(row, roots_by_theorem.get(row.name, [])) for row in plan.src.rows(db))

    def collect(results: Iterable[ReuseMeasures]) -> None:
        for idx, result in enumerate(results, 1):
            for root, row in result.stral.items():
                sampled[result.name, root].row = row
            timed(
                timings, result.name, "write_reuse", write, ("global_reuse", result.name, result.states, result.exprs)
            )
            if timings is not None:
                timings.flush()
            if progress is not None:
                progress.details = (
                    f"Global reuse: {idx:,} theorems; {shared_count:,} shared patterns; {workers} workers"
                )

    if workers == 1:
        codec = BlobCodec(dicts)
        shared = SharedPatterns(keys)
        collect(_measure_global_reuse(row, codec, shared, timings) for row in rows)
    else:
        with _shared_pattern_memory(keys) as memory_name:
            # Shared mappings outlive all workers, including failed/cancelled jobs.
            del keys
            cfg = WorkerCfg(dicts, reuse=SharedIdx(memory_name, shared_count), timing_dir=plan.analysis.timing_dir)
            with _worker_pool(workers, cfg) as pool:
                collect(ready_results(pool, rows, workers, _run_reuse, timings))


# Bounded pattern spills and merging.


class PatternBatches:
    """Aggregate in memory; spill compressed counts only when the budget is hit.

    Merging spills needs no expressions or graph reconstruction. Only one final
    merged row stream is consumed at a time; no final key index is rebuilt.
    """

    def __init__(self, dir: Path, memory_mib: int, timings: TimingLog | None = None):
        self.dir = dir
        self.timings = timings
        self.budget = memory_mib * 1024**2
        self.counts: dict[tuple[ViewMode, int], patterns.PatternCounts] = {}
        self.heads: dict[str, int] = {"": 0}
        self.parts: dict[tuple[ViewMode, int], list[Path]] = {}
        self.spills = 0

    def add(self, measured: dict[tuple[ViewMode, int], PatternBatch], label: str = "") -> None:
        remaps: dict[tuple[str, ...], NDArray[np.uint32]] = {}
        for key, rows in measured.items():
            if key not in self.counts:
                self.counts[key] = patterns.PatternCounts(self.heads)
            phase = f"add_patterns.{key[0]}.{key[1]}"
            timed(self.timings, label, phase, self.counts[key].add, rows, self.timings, label, remaps)
        # Budgets below one count-buffer allocation deliberately favor immediate
        # spilling. Pending batching must not defeat these small-budget workflows.
        if self.budget < patterns.PATTERN_CAP_ROWS * patterns.PATTERN_WEIGHT_COUNT * 8:
            for count in self.counts.values():
                count.flush(self.timings, label)
        if sum(count.memory_bytes for count in self.counts.values()) >= self.budget:
            self.spills += 1
            for (mode, depth), count in self.counts.items():
                path = self.dir / f"{mode}-{depth}-{self.spills}.msgpack.zst"
                self.parts.setdefault((mode, depth), []).append(path)
                phase = f"spill_patterns.{mode}.{depth}"
                timed(self.timings, label, phase, write_patterns, path, count, {}, self.timings, report_sizes=False)
                if self.timings is not None:
                    self.timings.flush()
            self.counts.clear()
            self.heads = {"": 0}

    def finish(
        self, keys: Iterable[tuple[ViewMode, int]]
    ) -> Iterator[tuple[ViewMode, int, list[str], Iterator[PatternBatch]]]:
        for mode, depth in keys:
            count = self.counts.pop((mode, depth), None)
            label = f"{mode}-{depth}"
            parts = _compact_pattern_runs(self.parts.get((mode, depth), []), self.dir, label, self.timings)
            with _pattern_runs(parts, count, self.timings, label) as (heads, rows):
                yield mode, depth, heads, rows
            for path in parts:
                path.unlink()
            del count


@contextmanager
def _pattern_runs(
    paths: Sequence[Path],
    pending: patterns.PatternCounts | None = None,
    timings: TimingLog | None = None,
    label: str = "",
) -> Generator[tuple[list[str], Generator[PatternBatch]]]:
    heads = {""} if pending is None else set(pending.heads)
    src_heads = [pattern_heads(path) for path in paths]
    heads.update(head for group in src_heads for head in group)
    dst_heads = sorted(heads)
    streams = [
        remap_pattern_heads(
            src, dst_heads, (batch for _, batch in packed_pattern_batches(path, timings)), timings, label
        )
        for path, src in zip(paths, src_heads, strict=True)
    ]
    if pending is not None:
        streams.append(
            remap_pattern_heads(list(pending.heads), dst_heads, pending.batches(timings, label), timings, label)
        )
    rows = merge_patterns(streams, timings, label)
    try:
        yield dst_heads, rows
    finally:
        rows.close()
        for stream in streams:
            stream.close()


def _compact_pattern_runs(paths: list[Path], dir: Path, stem: str, timings: TimingLog | None = None) -> list[Path]:
    """Bound open files and decoded spill batches, not just buffered counters."""
    round_ = 0
    while len(paths) > PATTERN_MERGE_FAN_IN:
        round_ += 1
        merged: list[Path] = []
        for idx, group in enumerate(batched(paths, PATTERN_MERGE_FAN_IN)):
            path = dir / f"{stem}-merge-{round_}-{idx}.msgpack.zst"
            with _pattern_runs(group, timings=timings, label=stem) as (heads, rows):
                write_pattern_batches(path, heads, rows, {}, timings=timings, report_sizes=False)
            merged.append(path)
            for old in group:
                old.unlink()
        paths = merged
    return paths


# Archive finalization.


def _finish_general(
    db: sqlite3.Connection, codec: BlobCodec, counts: MetricCounts, write: StatsSink, src: SrcSelection
) -> None:
    @lru_cache(maxsize=4)
    def lookup(name: str) -> ExprGraph:
        (blob,) = db.execute("SELECT exprs FROM theorems WHERE name=?", (name,)).fetchone()
        return build_graph(decode_exprs(blob, codec))

    examples = counts.examples
    select_common(
        examples.shapes,
        counts.freqs,
        (
            (digest, ref, shape_sig(lookup(ref.theorem), ref.root))
            for digest, ref in common_cands(examples, counts.freqs)
        ),
    )
    lookup.cache_clear()

    @lru_cache(maxsize=4)
    def example_blob(name: str) -> bytes:
        (blob,) = db.execute("SELECT exprs FROM theorems WHERE name=?", (name,)).fetchone()
        return codec.compress(codec.decompress(blob, codec.exprs), None)

    stral = finish_samples(examples.stral)
    states = counts.states
    mdata = Mdata(
        complete=True,
        src_db=str(src.path.resolve()),
        theorem_limit=src.limit,
        empty_ctxts=states.empty,
        conc=states.conc,
        rotated_conc=states.rotated_conc,
        stral_roots=examples.stral_roots,
        stral_sample_size=len(stral),
        atlas_min_nodes=examples.min_nodes,
        expr_idents=len(counts.freqs),
    )
    write(("stral", stral))
    freqs = FreqTotals()
    for batch in batched(counts.freqs.values(), 50_000):
        add_freq_totals(freqs, batch)
        write(("freqs", list(batch)))
    for key, cand in sorted(examples.shapes.items(), key=lambda item: (item[1].theorem, item[0])):
        name, root = cand.theorem, cand.root
        write(("shape", key, name, root, example_blob(name)))
    example_blob.cache_clear()
    write(("mdata", mdata))
    write(("stats", state_summary(states, len(counts.freqs)), freq_stats(freqs)))


def _finish_topo(db: sqlite3.Connection, paths: AnalysisPaths, counts: metrics.TopoCounts, src: SrcSelection) -> None:
    for mode, shapes in counts.shapes.items():
        write_shapes(paths.topo(mode), {"db": str(src.path.resolve()), "limit": src.limit}, shapes)
    write_pairs(paths.comparisons, counts.pairs)
    write_summary(paths.heads, counts.heads)
    stats: dict[Literal["topology", "comparison", "head"], object] = {
        "topology": topo_stats(counts.shapes),
        "comparison": comparison_summary(counts.pairs),
        "head": head_stats(counts.heads),
    }
    for kind, data in stats.items():
        write_summary(paths.stats(kind), data)
    names = sorted({item.theorem for item in metrics.atlas_examples(counts.shapes[ViewMode.ORIGINAL]).values()})
    rows = ((name, db.execute("SELECT exprs FROM theorems WHERE name=?", (name,)).fetchone()[0]) for name in names)
    write_examples(paths.examples, stored_dicts(db), rows)


def _finish_patterns(paths: AnalysisPaths, counts: PatternBatches, plan: ScanPlan) -> None:
    """Merge each count index independently; save statistics for cheap redraws."""
    mdata = {"db": str(plan.src.path.resolve()), "limit": plan.src.limit}
    depths = plan.analysis.depths
    keys = ((mode, depth) for mode in PATTERN_MODES for depth in depths)

    def summaries() -> Iterator[tuple[ViewMode, int, PatternStats]]:
        for mode, depth, heads, rows in counts.finish(keys):
            path = paths.patterns(mode, depth)
            cols = write_pattern_batches(
                path, heads, rows, {**mdata, "mode": mode, "depth": depth}, with_cols=True, timings=counts.timings
            )
            assert cols is not None
            yield mode, depth, timed(counts.timings, path.name, "pattern.statistics", pattern_stats, heads, cols)
            del cols

    write_pattern_stats(paths.stats("pattern"), summaries())
    write_summary(paths.analysis("patterns"), {"depths": depths, "modes": PATTERN_MODES, "exact": True})


# Analysis orchestration.


def _require_pending(path: Path) -> None:
    if path.parent.name != ".generations" or not path.name.startswith(".pending-") or not path.is_dir():
        raise ValueError("analysis producers require a pending generation; use analyze() or the public CLI")


def collect_analysis(
    db_path: Path,
    out: Path,
    *,
    analyses: tuple[Analysis, ...] = ("metrics", "topology", "patterns"),
    limit: int | None = None,
    workers: int = 1,
    atlas_min_nodes: int = 20,
    depths: tuple[int, ...] = (1, 2, 3),
    aggr_mem: int = 512,
    timing_dir: Path | None = None,
) -> dict[str, int]:
    """Produce unpublished artifacts inside an orchestration-owned generation."""
    _require_pending(out)
    plan = ScanPlan(SrcSelection(db_path, limit), MeasureCfg(analyses, depths, timing_dir, atlas_min_nodes), workers)
    paths = AnalysisPaths(out)
    metric_counts = MetricCounts(ExampleCounts(atlas_min_nodes)) if "metrics" in analyses else None
    topo_counts = metrics.TopoCounts() if "topology" in analyses else None
    with ExitStack() as ctxts:
        timings = ctxts.enter_context(closing(TimingLog(timing_dir, "coordinator"))) if timing_dir is not None else None
        src = ctxts.enter_context(closing(sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True)))
        write = ctxts.enter_context(stats_writer(paths.analysis("metrics"))) if metric_counts is not None else None
        pattern_counts = None
        if "patterns" in analyses:
            tmp = ctxts.enter_context(TemporaryDirectory(prefix="pattern-counts-", dir=out))
            pattern_counts = PatternBatches(Path(tmp), aggr_mem, timings)
        theorems = 0
        with Phase("Prepare graphs and collect selected metrics") as progress:
            for result in measure_src(src, plan, timings):
                theorems += 1
                label = result.name
                if result.general is not None:
                    assert metric_counts is not None and write is not None
                    timed(timings, label, "add_metrics", add_metrics, metric_counts, result.general)
                    timed(
                        timings,
                        label,
                        "write_metrics",
                        write,
                        ("theorem", result.general.name, result.general.states, result.general.exprs),
                    )
                if result.topo is not None:
                    assert topo_counts is not None
                    timed(timings, label, "add_topo", add_topo, topo_counts, result.topo)
                if pattern_counts is not None:
                    timed(timings, label, "aggregate_patterns", pattern_counts.add, result.patterns, label)
                if timings is not None:
                    timings.flush()
                spills = pattern_counts.spills if pattern_counts is not None else 0
                progress.details = f"{theorems:,} theorems; {workers} workers; {spills} count spills"
        if metric_counts is not None:
            assert write is not None
            with Phase("Apply the completed global reuse index") as progress:
                timed(
                    timings, "", "reuse_pass", _write_global_reuse, src, plan, metric_counts, write, progress, timings
                )
            timed(
                timings,
                "",
                "finish_metrics",
                _finish_general,
                src,
                BlobCodec(stored_dicts(src)),
                metric_counts,
                write,
                plan.src,
            )
            metric_counts = None
        if topo_counts is not None:
            timed(timings, "", "finish_topo", _finish_topo, src, paths, topo_counts, plan.src)
            topo_counts = None
        if pattern_counts is not None:
            timed(timings, "", "finish_patterns", _finish_patterns, paths, pattern_counts, plan)
    return {
        "theorems": theorems,
        "source_passes": 2 if "metrics" in analyses else 1,
        "pattern_spills": pattern_counts.spills if pattern_counts is not None else 0,
    }


def analyze_embeddings(stats: Path, workers: int) -> None:
    from .archives import read_stats
    from .metrics import embeddings

    _require_pending(stats)
    paths = AnalysisPaths(stats)
    with closing(read_stats(paths.analysis("metrics"), {"stral"})) as stream:
        rows = next(record[1] for record in stream if record[0] == "stral")
    shared, results = embeddings(rows, workers)
    # All keys here name arrays, never NumPy's allow_pickle control argument.
    save_arrays = cast(Callable[..., None], np.savez_compressed)
    save_arrays(paths.embedding_shared, **shared)
    with paths.embedding_shared.open("rb") as shared_stream:
        shared_id = file_digest(shared_stream, "sha256").hexdigest()
    for mode, data in results.items():
        path = paths.embedding(mode)
        save_arrays(path, **data, shared_sha256=np.asarray(shared_id), rows=np.asarray(len(rows)))
        with ZipFile(path) as archive:
            report_file_sizes(path, sum(item.file_size for item in archive.infolist()), kind="members")
    write_summary(paths.embedding_info, tuple(results))
