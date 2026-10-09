"""Store Lean theorem records as compressed MessagePack in SQLite."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from compression import zstd
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import msgspec

from trustmebro.artifacts import BlobSizes, decode_nat_ext, encode_msgpack

from .records import Expr, SchemaError, Theorem, Trn, iter_theorems, validate_theorem

DEFAULT_DICT_SIZE = 100 * 1024
TRN_PREFIX_BYTES = 250 * 1024


class Dicts(msgspec.Struct, frozen=True):
    exprs: bytes
    trns: bytes


class BlobCodec:
    """Compress independent theorem blobs using dictionaries embedded in the DB."""

    def __init__(self, dicts: Dicts | None = None):
        self.exprs = zstd.ZstdDict(dicts.exprs) if dicts is not None else None
        self.trns = zstd.ZstdDict(dicts.trns) if dicts is not None else None

    @staticmethod
    def compress(data: bytes, dict: zstd.ZstdDict | None) -> bytes:
        return zstd.compress(data, level=3, zstd_dict=dict.as_digested_dict if dict is not None else None)

    @staticmethod
    def decompress(data: bytes, dict: zstd.ZstdDict | None) -> bytes:
        try:
            return zstd.decompress(data, zstd_dict=dict)
        except zstd.ZstdError as error:
            raise SchemaError(f"invalid compressed blob: {error}") from error


_PLAIN_CODEC = BlobCodec()


def _decode[T](data: bytes | bytearray | memoryview, schema: type[T]) -> T:
    try:
        val = _decoder.decode(data)
        return msgspec.convert(val, type=schema, strict=True)
    except (msgspec.DecodeError, msgspec.ValidationError) as error:
        raise SchemaError(f"invalid MessagePack: {error}") from error


def encode_exprs(exprs: tuple[Expr, ...], codec: BlobCodec = _PLAIN_CODEC) -> bytes:
    """Encode a theorem's expression table independently of its trns."""
    return codec.compress(encode_msgpack(exprs), codec.exprs)


def decode_exprs(data: bytes, codec: BlobCodec = _PLAIN_CODEC) -> tuple[Expr, ...]:
    return _decode(codec.decompress(data, codec.exprs), tuple[Expr, ...])


def encode_trns(trns: tuple[Trn, ...], codec: BlobCodec = _PLAIN_CODEC) -> bytes:
    """Encode a theorem's trns independently of its expression table."""
    return codec.compress(encode_msgpack(trns), codec.trns)


def decode_trns(data: bytes, codec: BlobCodec = _PLAIN_CODEC) -> tuple[Trn, ...]:
    return _decode(codec.decompress(data, codec.trns), tuple[Trn, ...])


_CREATE_THEOREMS_TABLE = """
CREATE TABLE theorems (
    id                 INTEGER PRIMARY KEY,
    name               TEXT NOT NULL UNIQUE,
    module             TEXT NOT NULL,
    src_path           TEXT,
    src_start          INTEGER,
    src_end            INTEGER,
    expr_count         INTEGER NOT NULL CHECK (expr_count >= 0),
    trn_count          INTEGER NOT NULL CHECK (trn_count > 0),
    exprs              BLOB NOT NULL,
    trns               BLOB NOT NULL,
    CHECK (
        (src_start IS NULL AND src_end IS NULL) OR
        (src_start IS NOT NULL AND src_end IS NOT NULL AND src_start >= 0 AND src_end >= src_start)
    )
)
"""

_CREATE_COMPLETED_FILES_TABLE = """
CREATE TABLE completed_files (
    src_path TEXT PRIMARY KEY,
    theorem_count INTEGER NOT NULL CHECK (theorem_count >= 0),
    expr_count INTEGER NOT NULL CHECK (expr_count >= 0),
    trn_count INTEGER NOT NULL CHECK (trn_count >= 0)
)
"""

_CREATE_DICTS_TABLE = """
CREATE TABLE compression_dicts (
    kind TEXT PRIMARY KEY,
    dict BLOB NOT NULL
)
"""

THEOREM_COLS = "name, module, src_start, src_end, expr_count, trn_count, exprs, trns"


@dataclass(slots=True)
class DatasetSummary:
    theorems: int = 0
    exprs: int = 0
    trns: int = 0


type TheoremRow = tuple[str, str, int | None, int | None, int, int, bytes, bytes]


@dataclass(slots=True)
class EncodedFile:
    rows: list[TheoremRow]
    summary: DatasetSummary


def encode_file(stream: BinaryIO, codec: BlobCodec = _PLAIN_CODEC) -> EncodedFile:
    """Validate and encode one Lean output stream without writing to SQLite."""
    rows: list[TheoremRow] = []
    names: set[str] = set()
    summary = DatasetSummary()
    for theorem in iter_theorems(stream):
        if theorem.name in names:
            raise SchemaError(f"duplicate theorem name {theorem.name!r}")
        names.add(theorem.name)
        start, end = theorem.src_span or (None, None)
        exprs = encode_exprs(theorem.exprs, codec)
        trns = encode_trns(theorem.trns, codec)
        rows.append((theorem.name, theorem.module, start, end, len(theorem.exprs), len(theorem.trns), exprs, trns))
        summary.theorems += 1
        summary.exprs += len(theorem.exprs)
        summary.trns += len(theorem.trns)
    return EncodedFile(rows, summary)


def _create_extraction_tables(db: sqlite3.Connection, dicts: Dicts | None) -> None:
    db.execute("BEGIN IMMEDIATE")
    try:
        db.execute(_CREATE_THEOREMS_TABLE)
        db.execute(_CREATE_COMPLETED_FILES_TABLE)
        db.execute(_CREATE_DICTS_TABLE)
        if dicts is not None:
            db.executemany(
                "INSERT INTO compression_dicts VALUES (?, ?)", (("exprs", dicts.exprs), ("trns", dicts.trns))
            )
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


def check_extraction_schema(db: sqlite3.Connection, path: Path) -> None:
    tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if not {"theorems", "completed_files"} <= tables:
        raise SchemaError(f"database is not an extraction database: {path}")
    if "compression_dicts" not in tables:
        raise SchemaError(f"database uses an older extraction schema; create a fresh database: {path}")
    cols = {row[1] for row in db.execute("PRAGMA table_info(theorems)")}
    if "src_path" not in cols:
        raise SchemaError(f"database uses an older extraction schema; create a fresh database: {path}")


def stored_dicts(db: sqlite3.Connection) -> Dicts | None:
    rows = dict(db.execute("SELECT kind, dict FROM compression_dicts"))
    if not rows:
        return None
    if set(rows) != {"exprs", "trns"}:
        raise SchemaError("database has an incomplete compression dictionary set")
    return Dicts(rows["exprs"], rows["trns"])


def open_extraction_db(path: Path, dicts: Dicts | None = None) -> sqlite3.Connection:
    """Open or initialize the corpus database; the caller chooses pending files."""
    # Reject invalid dictionary bytes before creating a resumable database.
    BlobCodec(dicts)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    db = sqlite3.connect(path, isolation_level=None)
    try:
        if existed:
            check_extraction_schema(db, path)
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='partition_info' AND type='table'").fetchone():
                raise SchemaError("partition databases cannot be resumed as extraction databases")
            if dicts is not None and dicts != stored_dicts(db):
                raise SchemaError("provided dictionaries differ from those in the database")
        else:
            _create_extraction_tables(db, dicts)
        return db
    except BaseException:
        db.close()
        raise


def commit_extracted_file(db: sqlite3.Connection, src: Path, encoded: EncodedFile) -> None:
    """Commit a file's records and completion marker as one transaction."""
    db.execute("BEGIN IMMEDIATE")
    try:
        db.executemany(
            f"INSERT INTO theorems ({THEOREM_COLS}, src_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ((*row, str(src)) for row in encoded.rows),
        )
        db.execute(
            "INSERT INTO completed_files VALUES (?, ?, ?, ?)",
            (str(src), encoded.summary.theorems, encoded.summary.exprs, encoded.summary.trns),
        )
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise


def completed_paths(db: sqlite3.Connection) -> set[Path]:
    """Return source files already committed in a resumable database."""
    return {Path(row[0]) for row in db.execute("SELECT src_path FROM completed_files")}


def db_summary(db: sqlite3.Connection) -> DatasetSummary:
    """Summarize all committed theorem rows."""
    counts = db.execute(
        "SELECT COUNT(*), COALESCE(SUM(expr_count), 0), COALESCE(SUM(trn_count), 0) FROM theorems"
    ).fetchone()
    assert counts is not None
    return DatasetSummary(*counts)


def db_blob_sizes(db: sqlite3.Connection) -> dict[str, BlobSizes]:
    """Read encoded blob sizes from headers, without decoding theorem records.

    BlobCodec emits one content-size-bearing Zstandard frame per blob, even with
    dictionary compression. Missing content sizes remain unknown, not estimates.
    These component sizes exclude SQLite pages, indexes, and dictionary storage.
    """
    sizes = {col: BlobSizes() for col in ("exprs", "trns")}
    query = "SELECT length(exprs), substr(exprs,1,18), length(trns), substr(trns,1,18) FROM theorems"
    for expr_bytes, expr_header, trn_bytes, trn_header in db.execute(query):
        for col, stored, header in (("exprs", expr_bytes, expr_header), ("trns", trn_bytes, trn_header)):
            raw = zstd.get_frame_info(header).decompressed_size
            total = sizes[col]
            total.stored_bytes += stored
            if raw is None or total.uncompressed_bytes is None:
                total.uncompressed_bytes = total.largest_uncompressed_bytes = None
            else:
                total.uncompressed_bytes += raw
                total.largest_uncompressed_bytes = max(total.largest_uncompressed_bytes or 0, raw)
    return sizes


def _checked_count(actual: int, stored: int, kind: str, name: str) -> None:
    if actual != stored:
        raise SchemaError(f"theorem {name!r}: {kind} count is {stored}, but blob contains {actual}")


def decode_theorem_row(row: TheoremRow, codec: BlobCodec, *, check_refs: bool = True) -> Theorem:
    """Decode one THEOREM_COLS row, checking schema and stored blob counts.

    References/spans are checked by default. A preparation stage that performs
    validate_theorem itself can defer that traversal, but must not omit it.
    """
    name, module, start, end, n_exprs, n_trns, expr_blob, trn_blob = row
    exprs = decode_exprs(expr_blob, codec)
    trns = decode_trns(trn_blob, codec)
    _checked_count(len(exprs), n_exprs, "expression", name)
    _checked_count(len(trns), n_trns, "transition", name)
    span = None if start is None and end is None else (start, end)
    try:
        theorem = msgspec.convert([name, module, span, exprs, trns], type=Theorem, strict=True)
    except msgspec.ValidationError as error:
        raise SchemaError(f"theorem {name!r}: invalid stored metadata: {error}") from error
    return validate_theorem(theorem) if check_refs else theorem


def train_dicts(src: Path, dict_size: int = DEFAULT_DICT_SIZE, trn_prefix_bytes: int = TRN_PREFIX_BYTES) -> Dicts:
    """Train from raw or compressed blobs, bounding each transition sample."""
    if dict_size < 1 or trn_prefix_bytes < 1:
        raise ValueError("dictionary size and transition prefix must be positive")
    with closing(sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "theorems" not in tables:
            raise ValueError("training source has no theorem table")
        codec = BlobCodec(stored_dicts(db)) if "compression_dicts" in tables else None
        exprs = [
            codec.decompress(row[0], codec.exprs) if codec else row[0]
            for row in db.execute("SELECT exprs FROM theorems ORDER BY rowid")
        ]
        expr_dict = zstd.train_dict(exprs, dict_size)
        del exprs
        if codec:
            trns = [
                zstd.ZstdDecompressor(zstd_dict=codec.trns).decompress(row[0], max_length=trn_prefix_bytes)
                for row in db.execute("SELECT trns FROM theorems ORDER BY rowid")
            ]
        else:
            trns = [
                row[0]
                for row in db.execute("SELECT substr(trns, 1, ?) FROM theorems ORDER BY rowid", (trn_prefix_bytes,))
            ]
        trn_dict = zstd.train_dict(trns, dict_size)
    return Dicts(expr_dict.dict_content, trn_dict.dict_content)


def recompress_db(db: sqlite3.Connection, dicts: Dicts) -> None:
    """Atomically replace blobs and their dictionaries, then reclaim free pages."""
    old = BlobCodec(stored_dicts(db))
    new = BlobCodec(dicts)
    db.execute("BEGIN IMMEDIATE")
    try:
        for id, exprs, trns in db.execute("SELECT id, exprs, trns FROM theorems ORDER BY id"):
            encoded: list[bytes] = []
            for blob, old_dict, new_dict in ((exprs, old.exprs, new.exprs), (trns, old.trns, new.trns)):
                raw = old.decompress(blob, old_dict)
                compressed = new.compress(raw, new_dict)
                if new.decompress(compressed, new_dict) != raw:
                    raise SchemaError("recompression changed a theorem blob")
                encoded.append(compressed)
            db.execute("UPDATE theorems SET exprs=?, trns=? WHERE id=?", (*encoded, id))
        db.execute("DELETE FROM compression_dicts")
        db.executemany("INSERT INTO compression_dicts VALUES (?, ?)", (("exprs", dicts.exprs), ("trns", dicts.trns)))
        db.execute("COMMIT")
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("VACUUM")


@dataclass(frozen=True, slots=True)
class Corpus:
    """Own the completed corpus connection and embedded dictionary decoder for splitting."""

    db: sqlite3.Connection
    codec: BlobCodec


@contextmanager
def open_corpus(path: Path) -> Generator[Corpus]:
    """Open an existing, completed corpus without creating or modifying it."""
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        check_extraction_schema(db, path)
        yield Corpus(db, BlobCodec(stored_dicts(db)))


_decoder = msgspec.msgpack.Decoder(ext_hook=decode_nat_ext)
