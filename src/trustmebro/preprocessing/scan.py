"""Schedule shared candidate extraction, training-only fitting and feature conversion.

Extraction/conversion workers prepare one theorem per job; selection workers
reduce byte/count-bounded theorem batches. Results arrive in completion order.
Computation lives in candidates/coverage/features/supervised; archives owns serialization
and publication. The source DB is read-only and must remain completed/immutable.
"""

from __future__ import annotations

import sqlite3
import sys
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, closing
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from random import Random
from time import monotonic
from typing import cast

import msgspec
from graph_tool import openmp_set_num_threads

from trustmebro.artifacts import decode_nat_ext, encode_msgpack
from trustmebro.extraction import records as r
from trustmebro.extraction.storage import THEOREM_COLS, BlobCodec, Dicts, TheoremRow, decode_theorem_row, stored_dicts
from trustmebro.preprocessing.candidates import extract_cands
from trustmebro.preprocessing.features import FeatureStats, add_stats, encode_cands, encode_theorem, prepare_stats
from trustmebro.preprocessing.layout import Layout, compile_vocab
from trustmebro.preprocessing.records import (
    DEFAULT_DEPTHS,
    Cands,
    FeatureRows,
    LabelPolicy,
    Selection,
    Vocab,
    read_label_policy,
)
from trustmebro.runtime import Phase, ready_results

from .archives import (
    ARRAY_DTYPE,
    Diagnostics,
    FeatureHeader,
    Footer,
    candidate_frames,
    decode_candidates,
    first_frame,
    pack_rows,
    publish,
    read_frames,
    read_selection,
    read_vocab,
    unpack_rows,
)
from .supervised import SelectionBatch

SELECTION_BATCH_BYTES = 2 * 2**20
SELECTION_BATCH_THEOREMS = 16


class CandCounts(msgspec.Struct):
    theorems: int = 0
    nodes: int = 0
    shape_records: int = 0
    occs: int = 0
    states: int = 0
    stored_bytes: int = 0
    uncompressed_stream_bytes: int = 0


@dataclass(frozen=True)
class CandWorkerCfg:
    codec: BlobCodec
    depths: tuple[int, ...]


@dataclass(frozen=True)
class FeatureWorkerCfg:
    layout: Layout
    codec: BlobCodec
    decoder: msgspec.msgpack.Decoder


@dataclass(frozen=True, slots=True)
class SelectionWorkerCfg:
    fn: Callable[[Iterable[Cands]], SelectionBatch]
    decoder: msgspec.msgpack.Decoder


# Fixed configuration is initialized once per persistent child, never per job.
# Each child owns its decoder/layout; no shared corpus aggregation lives here.
_cand_cfg: CandWorkerCfg | None = None
_feature_cfg: FeatureWorkerCfg | None = None
_selection_cfg: SelectionWorkerCfg | None = None


# Bounded scheduling and read-only source selection.


def _sample_theorems(db: sqlite3.Connection, limit: int, seed: int | None) -> tuple[str, ...]:
    # Sample the small name index, not decoded graphs or SQLite RANDOM() rows.
    # Stable input ordering makes seeded selection independent of worker count.
    with closing(db.execute("SELECT name FROM theorems ORDER BY id")) as rows:
        names = [row[0] for row in rows]
    return tuple(Random(seed).sample(names, min(limit, len(names))))


def _src_rows(db: sqlite3.Connection, selection: Selection) -> Generator[TheoremRow]:
    if selection.theorems is not None:
        for name in selection.theorems:
            row = db.execute(f"SELECT {THEOREM_COLS} FROM theorems WHERE name = ?", (name,)).fetchone()
            if row is None:
                raise ValueError(f"selected theorem not found: {name!r}")
            yield row
    else:
        with closing(db.execute(f"SELECT {THEOREM_COLS} FROM theorems ORDER BY id")) as rows:
            yield from rows


def _decode_src_row(row: TheoremRow, codec: BlobCodec) -> r.Theorem:
    """Shared DB boundary: check blob counts/spans before theorem preparation."""
    # Discovery/encoding validates spans and DAG references immediately after
    # this shared storage decode, so do not traverse the expression table twice.
    theorem = decode_theorem_row(row, codec, check_refs=False)
    if not theorem.trns:
        raise ValueError(f"theorem {theorem.name!r} has no transitions")
    return theorem


# Persistent theorem workers; discovery and conversion remain separate tasks.


def _init_cand_worker(dicts: Dicts | None, depths: tuple[int, ...]) -> None:
    global _cand_cfg
    openmp_set_num_threads(1)
    _cand_cfg = CandWorkerCfg(BlobCodec(dicts), depths)


def _extract_row(row: TheoremRow, cfg: CandWorkerCfg) -> tuple[bytes, CandCounts]:
    cands = extract_cands(_decode_src_row(row, cfg.codec), cfg.depths)
    counts = CandCounts(1, len(cands.nodes), len(cands.shapes), len(cands.occs), len(cands.states))
    return encode_msgpack(cands), counts


def _cand_worker(row: TheoremRow) -> tuple[bytes, CandCounts]:
    if _cand_cfg is None:
        raise RuntimeError("candidate worker has not been initialized")
    return _extract_row(row, _cand_cfg)


def _init_feature_worker(vocab: Vocab, dicts: Dicts | None) -> None:
    global _feature_cfg
    openmp_set_num_threads(1)
    _feature_cfg = FeatureWorkerCfg(
        compile_vocab(vocab), BlobCodec(dicts), msgspec.msgpack.Decoder(ext_hook=decode_nat_ext)
    )


def _convert_row(row: TheoremRow | bytes, cfg: FeatureWorkerCfg) -> FeatureRows:
    if isinstance(row, bytes):
        return encode_cands(decode_candidates(row, cfg.decoder), cfg.layout)
    return encode_theorem(_decode_src_row(row, cfg.codec), cfg.layout)


def _feature_worker(row: TheoremRow | bytes) -> bytes:
    if _feature_cfg is None:
        raise RuntimeError("feature worker has not been initialized")
    return pack_rows(_convert_row(row, _feature_cfg))


def _conversion_results(
    rows: Iterable[TheoremRow | bytes], layout: Layout, workers: int, dicts: Dicts | None = None
) -> Generator[FeatureRows | bytes]:
    if workers == 1:
        cfg = FeatureWorkerCfg(layout, BlobCodec(dicts), msgspec.msgpack.Decoder(ext_hook=decode_nat_ext))
        for row in rows:
            yield _convert_row(row, cfg)
    else:
        with (
            ProcessPoolExecutor(
                workers,
                mp_context=get_context("spawn"),
                initializer=_init_feature_worker,
                initargs=(layout.vocab, dicts),
            ) as pool,
            closing(ready_results(pool, rows, workers, _feature_worker)) as results,
        ):
            yield from results


def _init_selection_worker(fn: Callable[[Iterable[Cands]], SelectionBatch]) -> None:
    global _selection_cfg
    openmp_set_num_threads(1)
    _selection_cfg = SelectionWorkerCfg(fn, msgspec.msgpack.Decoder(ext_hook=decode_nat_ext))


def _selection_worker(frames: tuple[bytes, ...]) -> SelectionBatch:
    if _selection_cfg is None:
        raise RuntimeError("selection worker has not been initialized")
    cfg = _selection_cfg
    return cfg.fn(decode_candidates(frame, cfg.decoder) for frame in frames)


def _selection_jobs(frames: Iterable[bytes]) -> Iterator[tuple[bytes, ...]]:
    """Bound encoded jobs by bytes/count; an oversized theorem travels alone."""
    batch: list[bytes] = []
    size = 0
    for frame in frames:
        if batch and size + len(frame) > SELECTION_BATCH_BYTES:
            yield tuple(batch)
            batch, size = [], 0
        batch.append(frame)
        size += len(frame)
        if len(batch) == SELECTION_BATCH_THEOREMS or size >= SELECTION_BATCH_BYTES:
            yield tuple(batch)
            batch, size = [], 0
    if batch:
        yield tuple(batch)


def _selection_results[Result: SelectionBatch](
    fn: Callable[[Iterable[Cands]], Result],
    phase: str,
    *,
    candidates: Path,
    names: tuple[str, ...] | None,
    workers: int,
    total: int | None,
) -> Generator[Result]:
    """A persistent pool per pass; only encoded input/reduced arrays cross IPC.

    ready_results bounds outstanding jobs and replenishes before aggregation.
    Each child owns one immutable pass configuration; the corpus index remains
    in the coordinator. Queue bounds exclude decoder/native scratch and config
    copies, and are not a process-tree memory limit.
    """
    with ExitStack() as stack:
        progress = stack.enter_context(Phase(phase))
        frames = stack.enter_context(closing(candidate_frames(candidates, names)))
        jobs = _selection_jobs(frames)
        count = 0
        denom = "" if total is None else f"/{total:,}"
        worker_label = "worker" if workers == 1 else "workers"
        if workers == 1:
            decoder = msgspec.msgpack.Decoder(ext_hook=decode_nat_ext)
            results = (fn(decode_candidates(frame, decoder) for frame in job) for job in jobs)
        else:
            pool = stack.enter_context(
                ProcessPoolExecutor(
                    workers, mp_context=get_context("spawn"), initializer=_init_selection_worker, initargs=(fn,)
                )
            )
            results = ready_results(pool, jobs, workers, _selection_worker)
        stack.enter_context(closing(results))
        for batch in results:
            count += len(batch.scope)
            rate = count / max(monotonic() - progress.started, 1e-9)
            progress.details = f"{count:,}{denom} theorems; {rate:.1f}/s; {workers} {worker_label}"
            yield cast(Result, batch)


# Coordinator progress and diagnostics.


def _show_progress(count: int, total: int, elapsed: float) -> None:
    print(f"\r\033[2KCandidates: {elapsed:.1f}s; theorems: {count:,}/{total:,}", end="", file=sys.stderr, flush=True)


def _cand_frames(
    results: Iterable[tuple[bytes, CandCounts]], counts: CandCounts, started: float, total: int
) -> Iterator[bytes]:
    terminal = sys.stderr.isatty()
    last_update = monotonic()
    if terminal:
        _show_progress(counts.theorems, total, last_update - started)
    for data, batch in results:
        counts.theorems += batch.theorems
        counts.nodes += batch.nodes
        counts.shape_records += batch.shape_records
        counts.occs += batch.occs
        counts.states += batch.states
        yield data
        if terminal:
            now = monotonic()
            if now - last_update >= 0.25:
                _show_progress(counts.theorems, total, now - started)
                last_update = now
    if terminal:
        _show_progress(counts.theorems, total, monotonic() - started)


def _report(stats: FeatureStats, header: FeatureHeader, nnz: int) -> Diagnostics:
    return Diagnostics(
        header,
        stats.theorems,
        stats.covered_theorems,
        stats.states,
        stats.covered_states,
        stats.goal_covered,
        stats.hyp_covered,
        nnz,
        tuple(sorted(stats.active_dims.items())),
        tuple(sorted(stats.active_patterns.items())),
        stats.support.astype(ARRAY_DTYPE, copy=False).tobytes(),
        stats.pair_limit,
        stats.pairs.astype(ARRAY_DTYPE, copy=False).tobytes(),
    )


# Publication workflows.


def scan_cands(
    db_path: Path,
    output: Path,
    *,
    depths: tuple[int, ...] = DEFAULT_DEPTHS,
    workers: int = 1,
    limit: int | None = None,
    seed: int | None = None,
    theorems: tuple[str, ...] | None = None,
    replace: bool = False,
) -> CandCounts:
    """Publish only after all selected rows succeed; never modify the source DB.

    At most 2*workers jobs are buffered, each with one theorem. This bounds the
    job queue, not theorem size, decoded graphs, native search scratch, encoded
    frames, compression buffers, or total process-tree memory.
    """
    if not depths or any(depth < 1 for depth in depths):
        raise ValueError("candidate depths must be positive and nonempty")
    if workers < 1 or (limit is not None and limit < 1):
        raise ValueError("workers and limit must be positive")
    if seed is not None and limit is None:
        raise ValueError("seed requires a random sample selected with limit")
    if theorems is not None and (limit is not None or not theorems):
        raise ValueError("a nonempty theorem selection cannot be combined with a limit")
    if theorems is not None and len(set(theorems)) != len(theorems):
        raise ValueError("the theorem selection contains duplicates")
    db_path = db_path.resolve()
    if output.resolve() == db_path or (output.exists() and output.samefile(db_path)):
        raise ValueError("candidate output cannot replace the source database")
    if output.exists() and not replace:
        raise FileExistsError(f"candidate archive exists: {output}; use --replace explicitly")
    stat = db_path.stat()
    sel = Selection(str(db_path), stat.st_size, stat.st_mtime_ns, tuple(sorted(set(depths))), limit, theorems, seed)
    counts = CandCounts()
    started = monotonic()
    try:
        with closing(sqlite3.connect(db_path.as_uri() + "?mode=ro", uri=True)) as db:
            dicts = stored_dicts(db)
            if limit is not None:
                sel = msgspec.structs.replace(sel, theorems=_sample_theorems(db, limit, seed))
            total = (
                len(sel.theorems)
                if sel.theorems is not None
                else db.execute("SELECT COUNT(*) FROM theorems").fetchone()[0]
            )

            def frames(rows: Iterable[TheoremRow]) -> Iterator[bytes]:
                yield encode_msgpack(sel)
                if workers == 1:
                    cfg = CandWorkerCfg(BlobCodec(dicts), sel.depths)
                    results = (_extract_row(row, cfg) for row in rows)
                    yield from _cand_frames(results, counts, started, total)
                else:
                    with (
                        ProcessPoolExecutor(
                            workers,
                            mp_context=get_context("spawn"),
                            initializer=_init_cand_worker,
                            initargs=(dicts, sel.depths),
                        ) as pool,
                        closing(ready_results(pool, rows, workers, _cand_worker)) as results,
                    ):
                        yield from _cand_frames(results, counts, started, total)

            with closing(_src_rows(db, sel)) as rows:
                counts.uncompressed_stream_bytes = publish(output, frames(rows), sources=(db_path,), replace=replace)
        counts.stored_bytes = output.stat().st_size
    finally:
        if sys.stderr.isatty():
            print(file=sys.stderr)
    return counts


def convert_corpus(
    src: Path,
    vocab_path: Path,
    output: Path,
    stats_path: Path,
    *,
    cands: bool = False,
    workers: int = 1,
    limit: int | None = None,
    seed: int | None = None,
    theorems: tuple[str, ...] | None = None,
    pair_limit: int = 128,
    labels: Path | None = None,
    replace: bool = False,
    split_id: str | None = None,
    expected_theorems: tuple[str, ...] | None = None,
) -> dict[str, int | float]:
    """Stream results, preserving all rows and writing no temporary SQLite data.

    Both sources bound outstanding jobs to 2*workers, not total memory. Each
    child retains its own compiled vocabulary and theorem scratch. Candidate
    frames are decoded in children and avoid discovery entirely. Sparse products
    can still grow with a single large theorem. No theorem or state is lost.
    """
    if workers < 1 or (limit is not None and limit < 1) or (seed is not None and limit is None):
        raise ValueError("workers/limit must be positive; seed requires limit")
    if theorems is not None and (not theorems or len(set(theorems)) != len(theorems) or limit is not None):
        raise ValueError("explicit theorem names must be nonempty, distinct and cannot accompany limit")
    if cands and (limit is not None or theorems is not None):
        raise ValueError("candidate reuse streams its existing selection; sampling applies only to DB input")
    srcs = (src, vocab_path) if labels is None else (src, vocab_path, labels)
    label_policy = read_label_policy(labels) if labels is not None else None
    frozen_labels = msgspec.json.encode(label_policy) if label_policy is not None else None
    if output.resolve() == stats_path.resolve():
        raise ValueError("feature and diagnostics outputs must be different paths")
    for path in (output, stats_path):
        if any(path.resolve() == source.resolve() or (path.exists() and path.samefile(source)) for source in srcs):
            raise ValueError("output cannot replace a source artifact")
        if path.exists() and not replace:
            raise FileExistsError(f"output exists: {path}; use --replace")
    layout = compile_vocab(read_vocab(vocab_path))
    repr = layout.vocab.representation
    if repr is not None:
        frozen = msgspec.json.decode(repr.label_policy_json, type=LabelPolicy)
        if label_policy is not None and label_policy != frozen:
            raise ValueError("explicit label policy differs from the frozen vocabulary")
        label_policy = frozen
        frozen_labels = msgspec.json.encode(frozen)
    stats = prepare_stats(layout, pair_limit)
    stamp = src.stat()
    src_selection = (
        read_selection(src)
        if cands
        else Selection(str(src.resolve()), stamp.st_size, stamp.st_mtime_ns, layout.vocab.depths, limit, theorems, seed)
    )
    if cands and not set(layout.vocab.depths).issubset(src_selection.depths):
        raise ValueError("candidate archive lacks requested vocabulary discovery depths")
    started = monotonic()
    nnz = 0

    def frames(results: Iterable[FeatureRows | bytes], header: FeatureHeader) -> Iterator[bytes]:
        nonlocal nnz
        expected = None if expected_theorems is None else set(expected_theorems)
        seen: set[str] = set()
        yield encode_msgpack(header)
        with Phase("Convert proof states") as progress:
            for result in results:
                rows = (
                    unpack_rows(result, layout.width, real=layout.vocab.representation is not None)
                    if isinstance(result, bytes)
                    else result
                )
                if expected is not None:
                    if rows.name not in expected or rows.name in seen:
                        raise ValueError("conversion theorem population disagrees with split")
                    seen.add(rows.name)
                if label_policy is not None:
                    for tactic in rows.tactics:
                        label_policy.label(tactic)  # reject unmapped=error before publishing the archive
                add_stats(stats, rows)
                nnz += rows.matrix.nnz
                progress.details = f"{stats.theorems:,} theorems; {stats.states:,} states; {nnz:,} nonzeros"
                yield result if isinstance(result, bytes) else pack_rows(rows)
        if expected is not None and seen != expected:
            raise ValueError("candidate archive is missing development theorems")
        yield encode_msgpack(Footer(stats.theorems, stats.states))

    if cands:
        header = FeatureHeader(
            layout.vocab, src_selection, str(src.resolve()), stamp.st_size, stamp.st_mtime_ns, frozen_labels, split_id
        )
        with closing(read_frames(src)) as observations:
            msgspec.msgpack.decode(first_frame(observations), type=Selection)
            with closing(_conversion_results(observations, layout, workers)) as results:
                raw_bytes = publish(output, frames(results, header), sources=srcs, replace=replace)
    else:
        with closing(sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True)) as db:
            if limit is not None:
                src_selection = msgspec.structs.replace(src_selection, theorems=_sample_theorems(db, limit, seed))
            header = FeatureHeader(
                layout.vocab,
                src_selection,
                str(src.resolve()),
                stamp.st_size,
                stamp.st_mtime_ns,
                frozen_labels,
                split_id,
            )
            with (
                closing(_src_rows(db, src_selection)) as observations,
                closing(_conversion_results(observations, layout, workers, stored_dicts(db))) as results,
            ):
                raw_bytes = publish(output, frames(results, header), sources=srcs, replace=replace)
    # The rows artifact is independently complete. A failure publishing the
    # diagnostics leaves it usable; these two files are not an atomic pair.
    report = _report(stats, header, nnz)
    stat_bytes = publish(stats_path, (encode_msgpack(report),), sources=srcs, replace=replace)
    return {
        "theorems": stats.theorems,
        "states": stats.states,
        "dimensions": layout.width,
        "shapes": len(layout.vocab.entries),
        "nontrivial_covered_theorems": stats.covered_theorems,
        "nontrivial_covered_states": stats.covered_states,
        "nnz": nnz,
        "cooccurrence_shapes": stats.pair_limit,
        "stored_bytes": output.stat().st_size,
        "uncompressed_stream_bytes": raw_bytes,
        "diagnostics_stored_bytes": stats_path.stat().st_size,
        "diagnostics_uncompressed_stream_bytes": stat_bytes,
        "dur_sec": monotonic() - started,
    }
