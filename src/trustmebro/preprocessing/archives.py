"""Compressed preprocessing archives: framing, codecs and atomic publication.

Readers do not query the source DB or start workers. All producers share the
same framing and source-preserving publication boundary; each archive keeps
its existing header, records and completion contract.
"""

from __future__ import annotations

from collections.abc import Generator, Iterable, Iterator
from compression import zstd
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from struct import Struct
from tempfile import NamedTemporaryFile
from typing import cast

import msgspec
import numpy as np
from scipy.sparse import csr_array

from trustmebro.artifacts import BinaryWriter, decode_nat_ext, encode_msgpack, report_file_sizes
from trustmebro.artifacts import read_frames as read_binary_frames
from trustmebro.artifacts import write_frame as write_binary_frame
from trustmebro.extraction import records as r
from trustmebro.preprocessing.layout import Layout, compile_vocab
from trustmebro.preprocessing.records import Cands, CoverageCands, FeatureRows, Selection, Vocab

_LENGTH = Struct("!Q")
ARRAY_DTYPE = np.dtype("<i8")
REAL_DTYPE = np.dtype("<f8")
BLOCK_ROWS = 4096
type PackedRows = tuple[str, bytes, tuple[r.Tactic, ...], bytes, bytes, bytes]


class _CandidateName(msgspec.Struct, frozen=True, array_like=True):
    name: str  # trailing candidate fields are skipped, not reconstructed


class FeatureHeader(msgspec.Struct, frozen=True):
    vocab: Vocab
    selection: Selection
    src: str
    src_size: int
    src_modified_ns: int
    label_policy: bytes | None = None  # frozen JSON; no external policy needed for new training archives
    split_id: str | None = None  # binds a development archive to its logical split manifest


@dataclass(frozen=True, slots=True)
class FeatureStream:
    """Header and decoded rows owned by an open archive context."""

    header: FeatureHeader
    width: int
    rows: Iterator[FeatureRows]


class Footer(msgspec.Struct, frozen=True):
    theorems: r.Nat
    states: r.Nat


class Diagnostics(msgspec.Struct, frozen=True):
    header: FeatureHeader
    theorems: r.Nat
    covered_theorems: r.Nat
    states: r.Nat
    covered_states: r.Nat
    goal_covered: r.Nat
    hyp_covered: r.Nat
    nnz: r.Nat
    active_dims: tuple[tuple[r.Nat, r.PosNat], ...]
    active_patterns: tuple[tuple[r.Nat, r.PosNat], ...]
    support: bytes  # role-specific state support, int64
    pair_limit: r.Nat
    pairs: bytes  # role-unioned state co-occurrence, int64 square


def write_frame(stream: BinaryWriter, data: bytes) -> int:
    return write_binary_frame(stream, data, _LENGTH)


def read_frames(path: Path, *, max_frame_bytes: int | None = None) -> Generator[bytes]:
    """Shared archive framing; each producer defines its own header/records."""
    with zstd.open(path, "rb") as stream:
        yield from read_binary_frames(stream, _LENGTH, max_frame_bytes=max_frame_bytes)


def publish(path: Path, frames: Iterable[bytes], *, sources: tuple[Path, ...], replace: bool) -> int:
    """An artifact commits only after its producer and immutable-source checks succeed."""
    for src in sources:
        if path.resolve() == src.resolve() or (path.exists() and src.exists() and path.samefile(src)):
            raise ValueError("output cannot replace a source artifact")
    if path.exists() and not replace:
        raise FileExistsError(f"output exists: {path}; use --replace")
    stamps = [(src.stat().st_size, src.stat().st_mtime_ns) for src in sources]
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".archive-", delete=False) as tmp:
        pending = Path(tmp.name)
    try:
        raw_bytes = 0
        with zstd.open(pending, "wb") as stream:
            for data in frames:
                raw_bytes += write_frame(stream, data)
        if [(src.stat().st_size, src.stat().st_mtime_ns) for src in sources] != stamps:
            raise ValueError("source changed during archive publication")
        pending.replace(path)
        report_file_sizes(path, raw_bytes)
        return raw_bytes
    finally:
        pending.unlink(missing_ok=True)


def first_frame(frames: Iterator[bytes]) -> bytes:
    data = next(frames, None)
    if data is None:
        raise ValueError("archive lacks its header")
    return data


def decode_candidates(data: bytes, decoder: msgspec.msgpack.Decoder) -> Cands:
    """Keep extension-aware natural decoding identical for readers and workers."""
    return msgspec.convert(decoder.decode(data), type=Cands, strict=True)


def candidate_frames(path: Path, names: tuple[str, ...] | None = None) -> Generator[bytes]:
    """Select by name before payload decoding; retain archive order and scope checks."""
    wanted = None if names is None else set(names)
    if names is not None and (not wanted or len(wanted) != len(names)):
        raise ValueError("candidate selection must be nonempty and distinct")
    seen: set[str] = set()
    name_decoder = msgspec.msgpack.Decoder(type=_CandidateName)
    with closing(read_frames(path)) as frames:
        header = next(frames, None)
        if header is None:
            raise ValueError("candidate archive lacks its selection header")
        msgspec.convert(msgspec.msgpack.decode(header, ext_hook=decode_nat_ext), type=Selection, strict=True)
        for data in frames:
            if wanted is not None:
                name = name_decoder.decode(data).name
                if name not in wanted:
                    continue
                if name in seen:
                    raise ValueError(f"duplicate candidate theorem: {name!r}")
                seen.add(name)
            yield data
    if wanted is not None and seen != wanted:
        raise ValueError("candidate archive lacks requested training theorems")


def read_coverage_candidates(path: Path, *, names: tuple[str, ...] | None = None) -> Generator[CoverageCands]:
    """Typed native projection: references/memberships only, not feature validation."""
    decoder = msgspec.msgpack.Decoder(type=CoverageCands)
    with closing(candidate_frames(path, names)) as frames:
        for data in frames:
            yield decoder.decode(data)


def read_selection(path: Path) -> Selection:
    with closing(read_frames(path)) as frames:
        try:
            return msgspec.msgpack.decode(next(frames), type=Selection)
        except StopIteration as error:
            raise ValueError("candidate archive lacks its selection header") from error


def read_vocab(path: Path) -> Vocab:
    with closing(read_frames(path)) as frames:
        vocab = msgspec.msgpack.decode(first_frame(frames), type=Vocab)
        if next(frames, None) is not None:
            raise ValueError("unexpected trailing vocabulary frames")
    return vocab


def pack_rows(rows: FeatureRows) -> bytes:
    matrix = rows.matrix
    steps, indices, indptr = (
        array.astype(ARRAY_DTYPE, copy=False).tobytes() for array in (rows.steps, matrix.indices, matrix.indptr)
    )
    dtype = REAL_DTYPE if matrix.dtype.kind == "f" else ARRAY_DTYPE
    data = matrix.data.astype(dtype, copy=False).tobytes()
    return encode_msgpack((rows.name, steps, rows.tactics, data, indices, indptr))


def unpack_rows(data: bytes, width: int, *, real: bool = False) -> FeatureRows:
    name, steps, tactics, vals, indices, indptr = msgspec.msgpack.decode(data, type=PackedRows)
    step_array, col_array, row_array = (np.frombuffer(blob, dtype=ARRAY_DTYPE) for blob in (steps, indices, indptr))
    counts = np.frombuffer(vals, dtype=REAL_DTYPE if real else ARRAY_DTYPE)
    if (
        len(tactics) != len(step_array)
        or len(row_array) != len(step_array) + 1
        or len(counts) != len(col_array)
        or row_array[0] != 0
        or row_array[-1] != len(counts)
        or np.any(np.diff(row_array) < 0)
        or np.any(step_array < 0)
        or len(np.unique(step_array)) != len(step_array)
        or np.any(counts <= 0)
        or np.any(~np.isfinite(counts))
        or np.any(col_array < 0)
        or np.any(col_array >= width)
    ):
        raise ValueError("invalid sparse feature row buffers/provenance")
    matrix = csr_array((counts, col_array, row_array), shape=(len(step_array), width))
    if not matrix.has_canonical_format:
        raise ValueError("feature rows must have sorted, distinct column indices")
    return FeatureRows(name, step_array, tactics, matrix)


@contextmanager
def open_features(
    path: Path, *, max_frame_bytes: int | None = None, prepared: FrozenFeatures | None = None
) -> Generator[FeatureStream]:
    """Decode the header once; closing the context closes the compressed input.

    Frame limits apply before allocating encoded records. Consuming all rows
    also checks the footer; stopping early does not certify archive completion.
    """
    with closing(read_frames(path, max_frame_bytes=max_frame_bytes)) as frames:
        encoded = first_frame(frames)
        if prepared is None:
            header = msgspec.msgpack.decode(encoded, type=FeatureHeader)
            width = compile_vocab(header.vocab).width
        else:
            if encoded != prepared.encoded:
                raise ValueError("feature header changed since preparation")
            header, width = prepared.header, prepared.layout.width
        with closing(_feature_records(frames, width, real=header.vocab.representation is not None)) as rows:
            yield FeatureStream(header, width, rows)


def _feature_records(frames: Iterator[bytes], width: int, *, real: bool) -> Generator[FeatureRows]:
    states = 0
    for theorems, data in enumerate(frames):
        if not data:
            raise ValueError("empty feature row frame")
        # Arrays contain rows; maps contain the completion footer. Dispatch
        # does not generically decode/copy embedded CSR buffers twice.
        if data[0] & 0xF0 == 0x80 or data[0] in (0xDE, 0xDF):
            footer = msgspec.msgpack.decode(data, type=Footer)
            if (footer.theorems, footer.states) != (theorems, states) or next(frames, None) is not None:
                raise ValueError("feature completion counters/trailing frames disagree")
            return
        rows = unpack_rows(data, width, real=real)
        states += cast(tuple[int, int], rows.matrix.shape)[0]
        yield rows
    raise ValueError("feature archive lacks its completion footer")


def read_diagnostics(path: Path) -> Diagnostics:
    with closing(read_frames(path)) as frames:
        report = msgspec.msgpack.decode(first_frame(frames), type=Diagnostics)
        if next(frames, None) is not None:
            raise ValueError("unexpected trailing diagnostics frames")
    width = len(report.header.vocab.entries) * 2
    if (
        report.pair_limit > min(2048, width // 2)
        or len(report.support) != width * 8
        or len(report.pairs) != report.pair_limit**2 * 8
    ):
        raise ValueError("diagnostic array dimensions disagree with vocabulary/prefix")
    if (
        report.covered_theorems > report.theorems
        or report.covered_states > report.states
        or max(report.goal_covered, report.hyp_covered) > report.covered_states
        or any(
            sum(count for _, count in hgram) != report.states for hgram in (report.active_dims, report.active_patterns)
        )
        or sum(val * count for val, count in report.active_dims) != report.nnz
    ):
        raise ValueError("diagnostic coverage/histogram totals disagree")
    return report


@dataclass(frozen=True, slots=True)
class FrozenFeatures:
    """Validated immutable header/layout; stream row/footer checks remain independent."""

    encoded: bytes
    header: FeatureHeader
    layout: Layout


def prepare_feature_source(path: Path, *, max_frame_bytes: int | None = None) -> FrozenFeatures:
    with closing(read_frames(path, max_frame_bytes=max_frame_bytes)) as frames:
        encoded = first_frame(frames)
    header = msgspec.msgpack.decode(encoded, type=FeatureHeader)
    return FrozenFeatures(encoded, header, compile_vocab(header.vocab))
