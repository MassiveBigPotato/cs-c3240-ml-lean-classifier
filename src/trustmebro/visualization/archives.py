"""Atomic, framed MessagePack/Zstandard measurement archives; no database caches."""

from __future__ import annotations

import sys
from collections import Counter
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from compression import zstd
from contextlib import closing, contextmanager
from dataclasses import fields
from functools import cache, partial
from itertools import batched, pairwise
from pathlib import Path
from struct import Struct
from tempfile import NamedTemporaryFile
from types import GenericAlias
from typing import TYPE_CHECKING, Any, Literal, cast, overload

import msgspec
import numpy as np

from trustmebro.artifacts import decode_nat_ext, encode_msgpack, report_file_sizes
from trustmebro.artifacts import read_frames as read_binary_frames
from trustmebro.artifacts import write_frame as write_binary_frame
from trustmebro.extraction.storage import Dicts
from trustmebro.runtime import TimingLog, checkpoint, timed, timed_batches

from .measurements import *

if TYPE_CHECKING:
    from .patterns import PatternCounts


# Archive types, paths, and framing defaults.


type Analysis = Literal["metrics", "topology", "patterns"]
type MetricRow = StateRow | ExprRow | StralRow | ReuseRow | FreqRow | GraphSize
type StoredShape = tuple[bytes, int, int, int, str, int, tuple[tuple[int, int], ...]]
type StoredPair = tuple[tuple[list[int], list[int]], int]
type StatsRecord = (
    tuple[Literal["theorem"], str, list[StateRow], list[ExprRow]]
    | tuple[Literal["stral"], list[StralRow]]
    | tuple[Literal["mdata"], Mdata]
    | tuple[Literal["freqs"], list[FreqRow]]
    | tuple[Literal["shape"], ShapeKey, str, int, bytes]
    | tuple[Literal["global_reuse"], str, list[ReuseRow], list[ReuseRow]]
    | tuple[Literal["stats"], StateSummary, FreqStats]
)
type StatsSink = Callable[[StatsRecord], None]


LENGTH = Struct("<Q")
decode_record = partial(msgspec.msgpack.decode, ext_hook=decode_nat_ext)


# Record and array encoding.


@cache
def row_fields(cls: type[MetricRow]) -> tuple[str, ...]:
    return cls.__struct_fields__


def row_vals(row: MetricRow) -> tuple[object, ...]:
    return tuple(getattr(row, name) for name in row_fields(type(row)))


def array_data(array: np.ndarray) -> tuple[str, tuple[int, ...], bytes]:
    return array.dtype.str, array.shape, array.tobytes()


def decode_array(data: object) -> np.ndarray:
    dtype, shape, buffer = msgspec.convert(data, type=tuple[str, tuple[int, ...], bytes])
    if np.dtype(dtype) != np.dtype(np.int64) or len(shape) != 2 or any(dim < 0 for dim in shape):
        raise ValueError("invalid concentration-grid dtype or shape")
    if len(buffer) != shape[0] * shape[1] * np.dtype(dtype).itemsize:
        raise ValueError("concentration-grid byte length does not match its shape")
    array = np.frombuffer(buffer, dtype=dtype).reshape(shape)
    if np.any(array < 0):
        raise ValueError("concentration-grid counts must be nonnegative")
    return array


def _encode_mdata(data: Mdata) -> dict[str, object]:
    vals = {field.name: getattr(data, field.name) for field in fields(Mdata)}
    for name in ("conc", "rotated_conc"):
        vals[name] = array_data(getattr(data, name))
    return vals


def _decode_mdata(vals: dict) -> Mdata:
    vals = vals.copy()
    for name in ("conc", "rotated_conc"):
        vals[name] = decode_array(vals[name])
    if vals["complete"] is not True or not isinstance(vals["src_db"], str):
        raise ValueError("invalid analysis metadata")
    for name in ("theorem_limit", "empty_ctxts", "stral_roots", "stral_sample_size", "atlas_min_nodes", "expr_idents"):
        val = vals[name]
        if (name != "theorem_limit" or val is not None) and (type(val) is not int or val < 0):
            raise ValueError(f"invalid analysis metadata: {name}")
    return Mdata(**vals)


def _decode_stats(kind: str, parts: list[msgspec.Raw]) -> StatsRecord:
    """Reconstruct passive records only for consumers that actually need them.

    Block buffers stay encoded until selected, never decode then re-encode.
    Figure population readers use metric_cols instead of this bounded row view.
    """
    match kind, len(parts):
        case "theorem", 4:
            name = msgspec.msgpack.decode(parts[1], type=str)
            return "theorem", name, block_rows(parts[2], StateRow), block_rows(parts[3], ExprRow)
        case "global_reuse", 4:
            name = msgspec.msgpack.decode(parts[1], type=str)
            return "global_reuse", name, block_rows(parts[2], ReuseRow), block_rows(parts[3], ReuseRow)
        case "stral", 2:
            return "stral", block_rows(parts[1], StralRow)
        case "freqs", 2:
            return "freqs", block_rows(parts[1], FreqRow)
        case "mdata", 2:
            return "mdata", _decode_mdata(decode_record(parts[1]))
        case "shape", 5:
            vals = [decode_record(part) for part in parts[1:]]
            key, name, root, blob = msgspec.convert(vals, type=tuple[ShapeKey, str, int, bytes])
            return "shape", key, name, root, blob
        case "stats", 3:
            return (
                "stats",
                msgspec.convert(decode_record(parts[1]), type=StateSummary),
                msgspec.convert(decode_record(parts[2]), type=FreqStats),
            )
        case _:
            raise ValueError("invalid metrics record; regenerate analysis artifacts")


# Framed I/O and atomic analysis publication.


@overload
def read_summary(path: Path) -> Any: ...


@overload
def read_summary[T](path: Path, cls: type[T]) -> T: ...


def read_summary(path: Path, cls: type[Any] = object) -> Any:
    with closing(records(path)) as stream:
        val = msgspec.convert(decode_record(next(stream)), type=cls)
        if next(stream, None) is not None:
            raise ValueError(f"unexpected records after archive summary: {path}")
        return val


def write_summary(path: Path, val: object) -> None:
    with writer(path) as write:
        write(val)


@contextmanager
def writer(
    path: Path, *, timings: TimingLog | None = None, report_sizes: bool = True
) -> Generator[Callable[[object], None]]:
    """Publish the archive only after its producer finishes successfully."""
    with NamedTemporaryFile(dir=path.parent, suffix=".part", delete=False) as tmp:
        pending = Path(tmp.name)
        try:
            with zstd.open(tmp, "wb", level=3) as stream:

                def write(item: object) -> None:
                    encoded = timed(timings, path.name, "archive.encode", encode_msgpack, item)
                    with checkpoint(timings, path.name, "archive.compress_write"):
                        write_binary_frame(stream, encoded, LENGTH)

                yield write
                uncompressed_bytes = stream.tell()
            pending.replace(path)
            if report_sizes:
                report_file_sizes(path, uncompressed_bytes)
        finally:
            pending.unlink(missing_ok=True)


def records(path: Path) -> Generator[bytes]:
    """Stream records and reject truncation; frame lengths are not memory-capped."""
    with zstd.open(path, "rb") as stream:
        try:
            yield from read_binary_frames(stream, LENGTH)
        except ValueError as error:
            raise ValueError(f"{error}: {path}") from error


def write_batches(write: Callable[[object], None], rows: Iterable[object], size: int = 50_000) -> None:
    for batch in batched(rows, size):
        write(batch)


# Metric archives and selected column readers.


@contextmanager
def stats_writer(path: Path) -> Generator[StatsSink]:
    """The completion marker distinguishes finished statistics from incomplete runs."""
    if path.exists():
        raise FileExistsError(f"statistics already exist: {path}; run graphs --replace")
    path.parent.mkdir(parents=True, exist_ok=True)

    with writer(path) as write:

        def write_stats(record: StatsRecord) -> None:
            match record:
                case ("theorem", name, states, exprs) | ("global_reuse", name, states, exprs):
                    write(
                        (
                            record[0],
                            name,
                            col_blocks(states, StateRow if record[0] == "theorem" else ReuseRow),
                            col_blocks(exprs, ExprRow if record[0] == "theorem" else ReuseRow),
                        )
                    )
                case ("stral" | "freqs", rows):
                    write((record[0], col_blocks(rows, StralRow if record[0] == "stral" else FreqRow)))
                case ("mdata", data):
                    write(("mdata", _encode_mdata(data)))
                case _:
                    write(record)

        yield write_stats
        write(("complete",))


def selected_records(path: Path, kinds: set[str] | None = None) -> Generator[tuple[str, list[msgspec.Raw]]]:
    """Select framed payloads before decoding their rows, including completion checks."""
    with closing(records(path)) as stream:
        for encoded in stream:
            parts = msgspec.msgpack.decode(encoded, type=list[msgspec.Raw])
            kind = msgspec.msgpack.decode(parts[0], type=str)
            if kind == "complete":
                if len(parts) != 1 or next(stream, None) is not None:
                    raise ValueError("unexpected data after completed statistics")
                return
            if kinds is None or kind in kinds:
                yield kind, parts
        raise ValueError("statistics scan is incomplete")


def metric_cols(
    path: Path,
    kind: Literal["theorem", "global_reuse", "stral", "freqs"],
    scope: Literal["state", "expression"] | None,
    cols: tuple[str, ...],
    *,
    size: int = 10_000,
) -> Generator[dict[str, np.ndarray]]:
    """Validate aligned blocks; decode only requested columns and exact exceptions."""
    cls = metric_row_type(kind, scope)
    with closing(selected_records(path, {kind})) as batches:
        for _, parts in batches:
            data = parts[1 if scope is None else 2 if scope == "state" else 3]
            start = 0
            for encoded in msgspec.msgpack.decode(data, type=list[msgspec.Raw]):
                block, columns = decode_cols(encoded, cls, cols, start=start)
                start += block.rows
                rows = len(next(iter(columns.values()))) if columns else 0
                for offset in range(0, rows, size):
                    yield {name: val[offset : offset + size] for name, val in columns.items()}


def read_stats(path: Path, kinds: set[str] | None = None) -> Generator[StatsRecord]:
    with closing(selected_records(path, kinds)) as stream:
        for kind, parts in stream:
            yield _decode_stats(kind, parts)


def metric_summary(path: Path) -> StateSummary:
    result = None
    with closing(selected_records(path, {"stats"})) as batches:
        for _, parts in batches:
            result = msgspec.convert(decode_record(parts[1]), type=StateSummary)
    if result is None:
        raise ValueError("no statistics found")
    return result


def mdata_vals(path: Path, cols: Mapping[str, type]) -> dict[str, Any]:
    """One selected-mdata pass; unrelated grids and records stay encoded."""
    result = None
    with closing(selected_records(path, {"mdata"})) as batches:
        for _, parts in batches:
            fields_ = msgspec.msgpack.decode(parts[1], type=dict[str, msgspec.Raw])
            result = {
                name: decode_array(decode_record(fields_[name]))
                if cls is np.ndarray
                else msgspec.convert(decode_record(fields_[name]), type=cls)
                for name, cls in cols.items()
            }
    if result is None:
        raise ValueError("no mdata found")
    return result


def mdata_val[T](path: Path, field: str, cls: type[T]) -> T:
    return cast(T, mdata_vals(path, {field: cls})[field])


def freq_batches(path: Path, size: int = 10_000) -> Iterator[list[tuple[int, int, list[int]]]]:
    """Only the chosen aggregate field is decoded, in bounded plotting batches."""
    with closing(selected_records(path, {"stats"})) as batches:
        for _, parts in batches:
            fields_ = msgspec.msgpack.decode(parts[2], type=dict[str, msgspec.Raw])
            rows = msgspec.msgpack.decode(fields_["pairs"], type=list[msgspec.Raw])
            for chunk in batched(rows, size):
                yield [msgspec.convert(decode_record(row), type=tuple[int, int, list[int]]) for row in chunk]


def freq_coverage(path: Path) -> dict:
    result = None
    with closing(selected_records(path, {"stats"})) as batches:
        for _, parts in batches:
            fields_ = msgspec.msgpack.decode(parts[2], type=dict[str, msgspec.Raw])
            result = decode_record(fields_["coverage"])
    if result is None:
        raise ValueError("no frequency coverage found")
    return result


# Topology and comparison archives.


def write_shapes(path: Path, mdata: Mapping[str, str | int | None], counts: Mapping[bytes, ShapeCount]) -> None:
    with writer(path) as write:
        write(mdata)
        write_batches(
            write, ((digest, c.nodes, c.top, c.all, c.theorem, c.root, c.bands) for digest, c in counts.items())
        )


def shape_batches(path: Path) -> Iterator[list[tuple[bytes, ShapeCount]]]:
    stream = records(path)
    with closing(stream) as batches:
        msgspec.convert(decode_record(next(stream)), type=dict[str, str | int | None])
        for encoded in batches:
            rows = msgspec.convert(decode_record(encoded), type=list[StoredShape])
            yield [
                (digest, ShapeCount(size, top, all_, name, root, bands))
                for digest, size, top, all_, name, root, bands in rows
            ]


def shape_cols(path: Path, count: Literal["top", "all"]) -> Iterator[dict[str, np.ndarray]]:
    """Population plots need only node/occurrence columns, not names or bands."""
    stream = records(path)
    with closing(stream) as batches:
        msgspec.convert(decode_record(next(stream)), type=dict[str, str | int | None])
        for encoded in batches:
            rows = msgspec.msgpack.decode(encoded, type=list[list[msgspec.Raw]])
            nodes, counts = [], []
            for row in rows:
                occs = decode_record(row[2 if count == "top" else 3])
                if occs > 0:
                    nodes.append(decode_record(row[1]))
                    counts.append(occs)
            yield {"nodes": _col(nodes), "occurrences": _col(counts)}


def write_pairs(path: Path, pairs: Mapping[ViewMode, Mapping[ComparisonLvl, Counter[Pair]]]) -> None:
    with writer(path) as write:
        for mode, lvls in pairs.items():
            for lvl, counts in lvls.items():
                for batch in batched(counts.items(), 10_000):
                    write((mode, lvl, [((row_vals(a), row_vals(b)), weight) for (a, b), weight in batch]))


def _pair_batches(
    path: Path, *, modes: set[ViewMode] | None = None, lvls: set[ComparisonLvl] | None = None
) -> Generator[tuple[ViewMode, ComparisonLvl, list[StoredPair]]]:
    with closing(records(path)) as batches:
        for encoded in batches:
            parts = msgspec.msgpack.decode(encoded, type=list[msgspec.Raw])
            mode = msgspec.convert(decode_record(parts[0]), type=ViewMode)
            lvl = msgspec.convert(decode_record(parts[1]), type=ComparisonLvl)
            if (modes is not None and mode not in modes) or (lvls is not None and lvl not in lvls):
                continue
            rows = msgspec.convert(decode_record(parts[2]), type=list[StoredPair])
            for (a, b), weight in rows:
                if len(a) != len(SIZE_FIELDS) or len(b) != len(SIZE_FIELDS):
                    raise ValueError("invalid graph-size row length")
                if weight < 0 or any(val < 0 for val in (*a, *b)):
                    raise ValueError("comparison counts must be nonnegative")
            yield mode, lvl, rows


def _size_cols(rows: list[StoredPair], idx: int) -> SizeCols:
    return SizeCols(
        *(
            np.asarray([pair[idx][col] for pair, _ in rows], dtype=object)
            if name == "expanded"
            else _col([pair[idx][col] for pair, _ in rows])
            for col, name in enumerate(SIZE_FIELDS)
        )
    )


def pair_col_batches(
    path: Path, *, modes: set[ViewMode], lvls: set[ComparisonLvl]
) -> Iterator[tuple[ViewMode, ComparisonLvl, PairCols]]:
    """Decode native plotting cols without allocating GraphSize records."""
    with closing(_pair_batches(path, modes=modes, lvls=lvls)) as batches:
        for mode, lvl, rows in batches:
            yield (
                mode,
                lvl,
                PairCols(
                    _size_cols(rows, 0),
                    _size_cols(rows, 1),
                    np.asarray([weight for _, weight in rows], dtype=np.float64),
                ),
            )


# Pattern archives and column readers.


def write_patterns(
    path: Path,
    counts: PatternCounts,
    mdata: Mapping[str, object],
    timings: TimingLog | None = None,
    *,
    report_sizes: bool = True,
) -> None:
    write_pattern_batches(
        path, list(counts.heads), counts.batches(timings, path.name), mdata, timings=timings, report_sizes=report_sizes
    )


def write_pattern_batches(
    path: Path,
    heads: list[str],
    batches: Iterable[PatternBatch],
    mdata: Mapping[str, object],
    *,
    with_cols: bool = False,
    timings: TimingLog | None = None,
    report_sizes: bool = True,
) -> np.ndarray | None:
    """Persist native buffers directly; only exceptional weights are boxed.

    Keys use PATTERN_KEY_DTYPE and ordinary weights use little-endian uint64.
    The tuple's weight member is a binary buffer or arbitrary-natural rows.
    No tuple/integers per ordinary row are created for encoding or reading.
    """
    chunks: list[np.ndarray] = []
    with writer(path, timings=timings, report_sizes=report_sizes) as write:
        write({**mdata, "heads": heads})
        for batch in timed_batches(timings, path.name, "pattern.produce_batch", batches):
            weights = batch.weights
            data = weights.tolist() if weights.dtype == object else weights.astype("<u8", copy=False).tobytes()
            write((batch.keys.tobytes(), data))
            if with_cols:
                chunks.append(timed(timings, path.name, "pattern.cols", _pattern_cols, batch))
    return timed(timings, path.name, "pattern.concat_cols", _concat_pattern_cols, chunks) if with_cols else None


def pattern_heads(path: Path) -> list[str]:
    stream = records(path)
    try:
        return msgspec.convert(decode_record(next(stream))["heads"], type=list[str])
    finally:
        stream.close()


def packed_pattern_batches(path: Path, timings: TimingLog | None = None) -> Generator[tuple[list[str], PatternBatch]]:
    with closing(records(path)) as source:
        stream = timed_batches(timings, path.name, "archive.read", source)
        heads = msgspec.convert(decode_record(next(stream))["heads"], type=list[str])
        empty = PatternBatch(np.empty(0, dtype=PATTERN_KEY_DTYPE), np.empty((0, 4), dtype=np.uint64), tuple(heads))
        yield heads, empty  # Preserve the header even for an empty count stream.
        for encoded in stream:
            with checkpoint(timings, path.name, "archive.decode_validate"):
                data, vals = msgspec.convert(decode_record(encoded), type=tuple[bytes, bytes | list[PatternWeights]])
                if len(data) % PATTERN_KEY_DTYPE.itemsize:
                    raise ValueError("invalid local-pattern key buffer length")
                keys = np.frombuffer(data, dtype=PATTERN_KEY_DTYPE)
                if np.any(keys["flavour"] > 1) or np.any(keys["head"] >= len(heads)):
                    raise ValueError("invalid local-pattern key or head reference")
                if isinstance(vals, bytes):
                    if len(vals) != len(keys) * 4 * 8:
                        raise ValueError("invalid local-pattern weight buffer length")
                    weights = np.frombuffer(vals, dtype="<u8").reshape(-1, 4)
                else:
                    weights = np.asarray(vals, dtype=object).reshape(-1, 4)
                    if len(weights) != len(keys) or np.any(weights < 0):
                        raise ValueError("invalid local-pattern counts")
            yield heads, PatternBatch(keys, weights, tuple(heads))


def _pattern_cols(batch: PatternBatch) -> np.ndarray:
    if not len(batch):
        return np.empty((0, len(PatternCol)), dtype=np.uint64)
    return np.column_stack((batch.keys["flavour"], batch.keys["head"], batch.weights))


def _concat_pattern_cols(chunks: list[np.ndarray]) -> np.ndarray:
    return np.concatenate(chunks) if chunks else np.empty((0, len(PatternCol)), dtype=np.uint64)


def write_pattern_stats(path: Path, rows: Iterable[tuple[ViewMode, int, PatternStats]]) -> None:
    with writer(path) as write:
        for mode, depth, stats in rows:
            write((mode, depth, stats))


def read_pattern_stats(path: Path, *, include_points: bool = True) -> Iterator[tuple[ViewMode, int, PatternStats]]:
    with closing(records(path)) as batches:
        for encoded in batches:
            if include_points:
                yield msgspec.convert(decode_record(encoded), type=tuple[ViewMode, int, PatternStats])
                continue
            parts = msgspec.msgpack.decode(encoded, type=list[msgspec.Raw])
            fields_ = msgspec.msgpack.decode(parts[2], type=dict[str, msgspec.Raw])
            # Preserve the point-series keys for orchestration, not their populations.
            flavours = msgspec.msgpack.decode(fields_["points"], type=dict[int, msgspec.Raw])
            stats = {name: decode_record(val) for name, val in fields_.items() if name != "points"}
            stats["points"] = {flavour: [] for flavour in flavours}
            yield msgspec.convert(
                (decode_record(parts[0]), decode_record(parts[1]), stats), type=tuple[ViewMode, int, PatternStats]
            )


def pattern_points(path: Path, mode: ViewMode, depth: int, flavour: int) -> Iterator[np.ndarray]:
    with closing(records(path)) as batches:
        for encoded in batches:
            parts = msgspec.msgpack.decode(encoded, type=list[msgspec.Raw])
            if decode_record(parts[0]) != mode or decode_record(parts[1]) != depth:
                continue
            fields_ = msgspec.msgpack.decode(parts[2], type=dict[str, msgspec.Raw])
            points = msgspec.msgpack.decode(fields_["points"], type=dict[int, msgspec.Raw])
            if flavour not in points:
                continue
            rows = msgspec.msgpack.decode(points[flavour], type=list[msgspec.Raw])
            for chunk in batched(rows, 10_000):
                vals = [decode_record(row) for row in chunk]
                yield np.column_stack((_col([row[0] for row in vals]), _col([row[1] for row in vals])))


# Examples and human-facing summaries.


def write_examples(path: Path, dicts: Dicts | None, rows: Iterable[tuple[str, bytes]]) -> None:
    with writer(path) as write:
        write(dicts)
        for row in rows:
            write(row)


@contextmanager
def read_examples(path: Path) -> Generator[tuple[Dicts | None, Iterator[tuple[str, bytes]]]]:
    with closing(records(path)) as stream:
        dicts = msgspec.convert(decode_record(next(stream)), type=Dicts | None)
        yield dicts, (msgspec.convert(decode_record(row), type=tuple[str, bytes]) for row in stream)


def json_counts(val: object) -> object:
    """Presentation only: exceptional naturals use a lossless hexadecimal string.

    Binary archives always retain integers. Avoid changing the interpreter's
    decimal-conversion guard just to produce a human-facing JSON summary.
    """
    match val:
        case bool():
            return val
        case int() as number:
            limit = sys.get_int_max_str_digits()
            return hex(number) if limit and number.bit_length() > limit * 3 else number
        case Mapping():
            return {key: json_counts(item) for key, item in val.items()}
        case list() | tuple():
            return [json_counts(item) for item in val]
        case _:
            return val


# Numerical archive tuning, not a bound on total worker/coordinator memory.
COL_BLOCK_BYTES = 2 * 2**20
COL_BLOCK_ROWS = 4096
COUNT_DTYPE = np.dtype("<u8")
FLOAT_DTYPE = np.dtype("<f8")
MAX_COUNT = np.iinfo(np.uint64).max


class CountCol(msgspec.Struct, frozen=True):
    dtype: Literal["<u8"]
    data: bytes
    indices: tuple[int, ...]
    exceptions: tuple[int, ...]


class FloatCol(msgspec.Struct, frozen=True):
    dtype: Literal["<f8"]
    data: bytes


class PackedCol(msgspec.Struct, frozen=True):
    dtype: Literal["msgpack"]
    rows: int
    data: bytes


class ColBlock(msgspec.Struct, frozen=True):
    schema: tuple[str, ...]
    rows: int
    ordinal: bytes
    cols: dict[str, msgspec.Raw]


def metric_row_type(kind: str, scope: str | None) -> type[MetricRow]:
    if (kind in {"stral", "freqs"}) != (scope is None):
        raise ValueError("structural/frequency measurements have no state/expression scope")
    return (
        StralRow
        if kind == "stral"
        else FreqRow
        if kind == "freqs"
        else ReuseRow
        if kind == "global_reuse"
        else StateRow
        if scope == "state"
        else ExprRow
    )


def _column(vals: list, field_type: object) -> CountCol | FloatCol | PackedCol:
    if field_type is int:
        if any(type(val) is not int or val < 0 for val in vals):
            raise ValueError("metric counts must be nonnegative integers")
        indices = tuple(idx for idx, val in enumerate(vals) if val > MAX_COUNT)
        base = np.fromiter((val if val <= MAX_COUNT else 0 for val in vals), dtype=COUNT_DTYPE, count=len(vals))
        return CountCol("<u8", base.tobytes(), indices, tuple(vals[idx] for idx in indices))
    if field_type is float:
        return FloatCol("<f8", np.asarray(vals, dtype=FLOAT_DTYPE).tobytes())
    return PackedCol("msgpack", len(vals), encode_msgpack(vals))


def col_blocks(rows: Sequence[MetricRow], cls: type[MetricRow], *, start: int = 0) -> list[dict[str, object]]:
    """Consume passive results into columns, without a second canonical row store.

    Blocks cap rows and use an encoded-byte soft target. Larger individual
    observations travel alone; bytes and scratch do not constitute an RSS cap.
    """
    fields_ = msgspec.structs.fields(cls)
    blocks: list[dict[str, object]] = []
    chunk: list[MetricRow] = []
    estimated = 0

    def flush() -> None:
        nonlocal estimated, start
        if chunk:
            cols = {field.name: _column([getattr(row, field.name) for row in chunk], field.type) for field in fields_}
            # Frame kind/name scope these keys to a population and theorem.
            # Ordinals retain each observation, including duplicate roots/states.
            block = {
                "schema": row_fields(cls),
                "rows": len(chunk),
                "ordinal": np.arange(start, start + len(chunk), dtype=COUNT_DTYPE).tobytes(),
                "cols": cols,
            }
            if len(chunk) > 1 and len(encode_msgpack(block)) > COL_BLOCK_BYTES:
                mid = len(chunk) // 2
                blocks.extend(col_blocks(chunk[:mid], cls, start=start))
                blocks.extend(col_blocks(chunk[mid:], cls, start=start + mid))
            else:
                blocks.append(block)
            start += len(chunk)
            chunk.clear()
            estimated = 0

    for row in rows:
        # Numerical buffers dominate regular rows; strings/maps can be larger.
        # Actual encoded size is checked at the coherent block boundary.
        chunk.append(row)
        estimated += 8 * len(fields_)
        if len(chunk) >= COL_BLOCK_ROWS or estimated >= COL_BLOCK_BYTES:
            flush()
    flush()
    return blocks


def decode_cols(
    encoded: bytes | msgspec.Raw, cls: type[MetricRow], requested: tuple[str, ...], *, start: int
) -> tuple[ColBlock, dict[str, np.ndarray]]:
    block = msgspec.msgpack.decode(encoded, type=ColBlock)
    fields_ = msgspec.structs.fields(cls)
    if block.rows < 0 or block.schema != row_fields(cls) or tuple(block.cols) != block.schema:
        raise ValueError("invalid numerical block schema/alignment; regenerate statistics")
    if len(block.ordinal) != block.rows * 8 or not np.array_equal(
        np.frombuffer(block.ordinal, dtype=COUNT_DTYPE), np.arange(start, start + block.rows, dtype=COUNT_DTYPE)
    ):
        raise ValueError("invalid numerical observation keys")
    if any(name not in block.schema for name in requested):
        raise ValueError("unknown numerical measurement column")
    result: dict[str, np.ndarray] = {}
    for field in fields_:
        name, raw = field.name, block.cols[field.name]
        if field.type is int:
            col = msgspec.convert(decode_record(raw), type=CountCol)
            if len(col.data) != block.rows * 8 or len(col.indices) != len(col.exceptions):
                raise ValueError("count column length differs from its row count")
            base = np.frombuffer(col.data, dtype=COUNT_DTYPE)
            if (
                any(idx < 0 or idx >= block.rows for idx in col.indices)
                or any(a >= b for a, b in pairwise(col.indices))
                or any(val <= MAX_COUNT for val in col.exceptions)
                or any(base[idx] != 0 for idx in col.indices)
            ):
                raise ValueError("invalid exact count exceptions")
            if name in requested:
                if col.indices:
                    vals = base.astype(object)
                    vals[np.asarray(col.indices, dtype=np.intp)] = col.exceptions
                    result[name] = vals
                else:
                    result[name] = base
        elif field.type is float:
            col = msgspec.msgpack.decode(raw, type=FloatCol)
            if len(col.data) != block.rows * 8:
                raise ValueError("float column length differs from its row count")
            if name in requested:
                result[name] = np.frombuffer(col.data, dtype=FLOAT_DTYPE)
        else:
            col = msgspec.msgpack.decode(raw, type=PackedCol)
            if col.rows != block.rows:
                raise ValueError("packed column length differs from its row count")
            if name in requested:
                vals = msgspec.convert(decode_record(col.data), type=GenericAlias(list, field.type))
                if len(vals) != block.rows:
                    raise ValueError("packed column length differs from its row count")
                result[name] = np.empty(block.rows, dtype=object)
                result[name][:] = vals
    return block, result


def block_rows[Row: MetricRow](encoded: msgspec.Raw, cls: type[Row]) -> list[Row]:
    """Passive row view for bounded embedding and atlas consumers, not persisted row codecs."""
    rows: list[Row] = []
    start = 0
    fields_ = msgspec.structs.fields(cls)
    # decode_cols validates each field; only the heterogeneous call signature
    # is dynamic here, not the returned row type.
    construct = cast(Callable[..., Row], cls)
    for block in msgspec.msgpack.decode(encoded, type=list[msgspec.Raw]):
        info, cols = decode_cols(block, cls, row_fields(cls), start=start)
        start += info.rows
        rows.extend(
            construct(
                *(
                    int(val) if field.type is int else float(val) if field.type is float else val
                    for field, val in zip(fields_, vals, strict=True)
                )
            )
            for vals in zip(*cols.values(), strict=True)
        )
    return rows


def _col(vals: list) -> np.ndarray:
    """Native numeric columns when representable, exact objects otherwise."""
    if vals and isinstance(vals[0], float):
        return np.asarray(vals, dtype=np.float64)
    max_int = np.iinfo(np.int64).max
    if vals and type(vals[0]) is int and all(0 <= val <= max_int for val in vals):
        return np.asarray(vals, dtype=np.int64)
    return np.asarray(vals, dtype=object)
