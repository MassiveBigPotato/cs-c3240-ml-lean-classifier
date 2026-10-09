"""Schedule bounded Lean extraction and commit completed files to SQLite."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from compression import zstd
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TextIO

import re2 as re

from trustmebro.extraction.storage import (
    DEFAULT_DICT_SIZE,
    BlobCodec,
    DatasetSummary,
    EncodedFile,
    commit_extracted_file,
    completed_paths,
    db_blob_sizes,
    db_summary,
    encode_file,
    open_extraction_db,
    recompress_db,
    stored_dicts,
    train_dicts,
)
from trustmebro.runtime import Phase

from .records import SchemaError

DEFAULT_ROOT = Path(".lake/packages/mathlib/Mathlib")
DEFAULT_LEAN_EXE = Path(".lake/build/bin/trustmebro-extract-state")
DEFAULT_WORKERS = 1
DEFAULT_SAMPLE_SEED = 2

_LOG_FIELDS = [
    "recorded_at_utc",
    "src",
    "status",
    "dur_sec",
    "theorems",
    "exprs",
    "trns",
    "setup_ms",
    "imports_ms",
    "elaboration_ms",
    "selection_ms",
    "json_build_ms",
    "json_encode_ms",
    "total_ms",
    "error",
    "diagns",
]
_SETUP_TIMING = re.compile(r"TIMING setup: (?P<setup_ms>\d+)ms")
_FILE_TIMING = re.compile(
    r"TIMING imports=(?P<imports_ms>\d+)ms "
    r"elaboration=(?P<elaboration_ms>\d+)ms "
    r"selection=(?P<selection_ms>\d+)ms "
    r"json_build=(?P<json_build_ms>\d+)ms "
    r"json_encode=(?P<json_encode_ms>\d+)ms "
    r"total=(?P<total_ms>\d+)ms"
)


@dataclass(slots=True)
class ExtractedFile:
    src: Path
    encoded: EncodedFile
    dur_sec: float
    diagns: str


class ExtractionFailure(RuntimeError):
    def __init__(self, src: Path, dur_sec: float, diagns: str, reason: str):
        super().__init__(f"{src}: {reason}; see the per-file log")
        self.src = src
        self.dur_sec = dur_sec
        self.diagns = diagns
        self.reason = reason


class WorkerPool:
    def __init__(self, exe: Path, timing: bool, lean_env: dict[str, str], codec: BlobCodec):
        self.exe = exe
        self.timing = timing
        self.lean_env = lean_env
        self.codec = codec
        self.stopping = threading.Event()
        self._lock = threading.Lock()
        self._active: set[subprocess.Popen[bytes]] = set()

    def stop(self) -> None:
        self.stopping.set()
        with self._lock:
            for proc in self._active:
                self._signal_group(proc, signal.SIGTERM)

    @staticmethod
    def _signal_group(proc: subprocess.Popen[bytes], sig: signal.Signals) -> None:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, sig)
            except ProcessLookupError:
                pass

    def _run_worker(self, src: Path) -> tuple[bytes, bytes, int]:
        """Run Lean once and drain both output pipes without deadlocking."""

        if self.stopping.is_set():
            raise RuntimeError("extraction stopped")
        cmd = [str(self.exe), str(src)]
        if self.timing:
            cmd.append("--timing")
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
            env=self.lean_env,
        )
        with self._lock:
            self._active.add(proc)
            if self.stopping.is_set():
                self._signal_group(proc, signal.SIGTERM)
        try:
            stdout, stderr = proc.communicate()
            assert stdout is not None and stderr is not None
            assert proc.returncode is not None
            return stdout, stderr, proc.returncode
        finally:
            if proc.poll() is None:
                self._signal_group(proc, signal.SIGKILL)
                proc.wait()
            with self._lock:
                self._active.discard(proc)

    def extract(self, src: Path) -> ExtractedFile:
        started = time.monotonic()
        stderr = b""
        try:
            stdout, stderr, code = self._run_worker(src)
            if code != 0:
                raise RuntimeError(f"extractor exited with code {code}")
            encoded = encode_file(io.BytesIO(stdout), self.codec)
            # The CSV log uses surrogateescape to preserve malformed stderr bytes.
            diagns = stderr.decode("utf-8", errors="surrogateescape")
            return ExtractedFile(src, encoded, time.monotonic() - started, diagns)
        except Exception as error:
            raise ExtractionFailure(
                src, time.monotonic() - started, stderr.decode("utf-8", errors="surrogateescape"), str(error)
            ) from error


def _lake_env() -> dict[str, str]:
    """Resolve Lake's worker environment once, without printing its variables."""

    result = subprocess.run(["lake", "env", "env", "-0"], capture_output=True, check=False)
    if result.returncode != 0:
        raise RuntimeError("could not obtain the Lake environment")
    return {
        os.fsdecode(key): os.fsdecode(val)
        for entry in result.stdout.split(b"\0")
        if entry
        for key, val in [entry.split(b"=", 1)]
    }


def _timing_and_diagns(stderr: str) -> tuple[dict[str, int], str]:
    timing: dict[str, int] = {}
    other: list[str] = []
    for line in stderr.splitlines(keepends=True):
        text = line.rstrip("\r\n")
        if not text.isascii():
            other.append(line)
        elif match := _SETUP_TIMING.fullmatch(text):
            val = match["setup_ms"]
            if val is None:
                raise ValueError("setup timing did not match")
            timing["setup_ms"] = int(val)
        elif match := _FILE_TIMING.fullmatch(text):
            for key, val in match.groupdict().items():
                if val is None:
                    raise ValueError(f"timing field {key!r} did not match")
                timing[key] = int(val)
        else:
            other.append(line)
    return timing, "".join(other)


def _write_event(
    log: TextIO,
    src: Path,
    status: str,
    dur_sec: float,
    diagns: str,
    summary: DatasetSummary | None = None,
    reason: str | None = None,
) -> None:
    timing, other = _timing_and_diagns(diagns)
    entry: dict[str, str | int | float] = {
        "recorded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "src": str(src),
        "status": status,
        "dur_sec": round(dur_sec, 3),
        **timing,
    }
    if summary is not None:
        entry.update(theorems=summary.theorems, exprs=summary.exprs, trns=summary.trns)
    if other:
        entry["diagns"] = other
    if reason is not None:
        entry["error"] = reason
    csv.DictWriter(log, fieldnames=_LOG_FIELDS).writerow(entry)
    log.flush()


def _print_summary(summary: DatasetSummary, db: Path, start: float) -> None:
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        sizes = {col: asdict(size) for col, size in db_blob_sizes(connection).items()}
    print(
        json.dumps(
            {
                **asdict(summary),
                "dur_sec": round(time.monotonic() - start, 3),
                "db": str(db.resolve()),
                "db_bytes": db.stat().st_size,
                "blob_sizes": sizes,
            },
            sort_keys=True,
        )
    )


def _commit_worker_result(future: Future[ExtractedFile], src: Path, db: sqlite3.Connection, log: TextIO) -> None:
    """Commit and log a successful file, or log the worker's failure."""

    try:
        extracted = future.result()
    except ExtractionFailure as error:
        _write_event(log, src, "failed", error.dur_sec, error.diagns, reason=error.reason)
        raise
    try:
        commit_extracted_file(db, src, extracted.encoded)
    except Exception as error:
        _write_event(log, src, "failed", extracted.dur_sec, extracted.diagns, reason=str(error))
        raise
    else:
        _write_event(log, src, "completed", extracted.dur_sec, extracted.diagns, extracted.encoded.summary)


def _run_pending_files(
    remaining: list[Path], workers: int, pool: WorkerPool, db: sqlite3.Connection, log: TextIO, phase: Phase
) -> None:
    """Keep at most `workers` files in flight, committing each finished file."""

    pending = iter(remaining)
    finished = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        try:
            futures = {executor.submit(pool.extract, src): src for src in islice(pending, workers)}
            phase.details = f"0/{len(remaining):,} files; {workers} workers"
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    src = futures.pop(future)
                    _commit_worker_result(future, src, db, log)
                    finished += 1
                    phase.details = f"{finished:,}/{len(remaining):,} files; {workers} workers"
                    next_src = next(pending, None)
                    if next_src is not None:
                        futures[executor.submit(pool.extract, next_src)] = next_src
        except BaseException:
            pool.stop()
            raise


def run_pipeline(
    file_list: Path,
    db_path: Path,
    exe: Path,
    workers: int,
    timing: bool,
    log_file: Path | None = None,
    phase_name: str = "Extract proof states",
) -> DatasetSummary:
    if workers < 1:
        raise ValueError("workers must be a positive integer")
    file_list = file_list.resolve()
    srcs = read_file_list(file_list)
    if not exe.is_file():
        raise FileNotFoundError(f"build the Lean extractor first: {exe}")
    with closing(open_extraction_db(db_path)) as db:
        completed = completed_paths(db)
        remaining = [src for src in srcs if src not in completed]
        if not remaining:
            print(f"{phase_name}: nothing pending; {len(srcs):,} files already completed", file=sys.stderr)
            return db_summary(db)
        pool = WorkerPool(exe, timing, _lake_env(), BlobCodec(stored_dicts(db)))
        log_path = log_file or db_path.with_suffix(".log.csv")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a+", encoding="utf-8", errors="surrogateescape", newline="") as log:
            log.seek(0)
            header = next(csv.reader(log), None)
            if header is None:
                csv.writer(log).writerow(_LOG_FIELDS)
            elif header != _LOG_FIELDS:
                raise ValueError(f"log file has an unexpected CSV header: {log_path}")
            log.seek(0, os.SEEK_END)
            with Phase(phase_name) as phase:
                _run_pending_files(remaining, workers, pool, db, log, phase)
                phase.details += f"; {len(srcs) - len(remaining):,} already completed"
                summary = db_summary(db)
                phase.details += f"; database: {summary.theorems:,} theorems, {summary.trns:,} transitions"
        return db_summary(db)


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--files-from", required=True, type=Path)
    parser.add_argument("--db", dest="db", required=True, type=Path)
    parser.add_argument("--workers", type=_positive_int, default=DEFAULT_WORKERS)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument(
        "--log-file", type=Path, help="append per-file CSV records here (default: DATABASE stem + .log.csv)"
    )
    parser.add_argument("--lean-exe", type=Path, default=DEFAULT_LEAN_EXE)
    return parser


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    args = _parser().parse_args(argv)
    try:
        summary = run_pipeline(
            args.files_from, args.db, args.lean_exe.resolve(), args.workers, args.timing, args.log_file
        )
    except (OSError, sqlite3.Error, SchemaError, RuntimeError, ValueError, zstd.ZstdError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    _print_summary(summary, args.db, started)
    return 0


# Skip non-code tokens in RE2; bytes avoid re-encoding on each search.
_LEAN_TOKEN = re.compile(
    rb"""--[^\n]*|/-|r(#+)"|"(?:\\[\s\S]|[^"\\])*"|"""
    rb"""\x{00AB}[^\x{00BB}]*\x{00BB}|'(?:\\(?:x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|.)|[^'\\\n])'|"""
    rb"""\b(theorem|lemma|example|by)\b"""
)
_COMMENT_DELIM = re.compile(rb"/-|-/")


def _proof_keywords(src: str) -> set[str]:
    """Find proof keywords outside comments, literals, and quoted names.

    This remains a candidate filter, not a Lean parser: the declaration and
    tactic block need not belong to the same declaration.
    """
    data = src.encode("utf-8")
    found: set[str] = set()
    pos = 0
    while token := _LEAN_TOKEN.search(data, pos):
        word = token[0]
        pos = token.end()
        if word == b"/-":
            depth = 1
            for delim in _COMMENT_DELIM.finditer(data, pos):
                depth += 1 if delim[0] == b"/-" else -1
                if depth == 0:
                    pos = delim.end()
                    break
            else:
                break  # Unterminated comment consumes the remaining source.
        elif hashes := token[1]:
            closing = b'"' + hashes
            end = data.find(closing, pos)
            if end < 0:
                break
            pos = end + len(closing)
        elif keyword := token[2]:
            # RE2's word boundaries are ASCII. Check Unicode identifiers and
            # Lean's identifier apostrophes using at most one UTF-8 character.
            before = data[max(0, token.start() - 4) : token.start()].decode("utf-8", errors="ignore")[-1:]
            after = data[pos : pos + 4].decode("utf-8", errors="ignore")[:1]
            if not any(char and (char.isalnum() or char in "_'") for char in (before, after)):
                found.add(keyword.decode("ascii"))
    return found


def find_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ValueError(f"not a Lean source directory: {root}")
    cands = []
    for path in sorted(root.rglob("*.lean")):
        keywords = _proof_keywords(path.read_text(encoding="utf-8"))
        if "by" in keywords and keywords & {"theorem", "lemma", "example"}:
            cands.append(path.resolve())
    return cands


def read_file_list(path: Path) -> list[Path]:
    srcs: list[Path] = []
    seen: set[Path] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#") or not line.strip():
            continue
        src = Path(line.strip()).resolve(strict=True)
        if not src.is_file():
            raise ValueError(f"not a source file: {src}")
        if src in seen:
            raise ValueError(f"duplicate source file in {path}: {src}")
        seen.add(src)
        srcs.append(src)
    return srcs


def sample_files(files: list[Path], count: int, seed: int) -> list[Path]:
    if count < 1:
        raise ValueError("sample size must be positive")
    return random.Random(seed).sample(files, count)


def write_file_list(path: Path, files: list[Path]) -> None:
    """Never silently replace an existing selection."""
    text = "".join(f"{file}\n" for file in files)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise ValueError(f"file list already exists with different contents: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False) as temp:
        temp.write(text)
        pending = Path(temp.name)
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def find_main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        with Phase("Discover candidate files") as phase:
            files = find_files(args.root)
            write_file_list(args.output, files)
            phase.details = f"{len(files):,} candidates; list: {args.output}"
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    print(f"found {len(files):,} candidate files: {args.output}")
    return 0


def _pipeline_phase(db: sqlite3.Connection, cfg: str) -> str:
    exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='extraction_pipeline'").fetchone()
    if not exists:
        if (
            db.execute("SELECT COUNT(*) FROM completed_files").fetchone()[0]
            or db.execute("SELECT COUNT(*) FROM theorems").fetchone()[0]
        ):
            raise ValueError("existing database was not created by the full pipeline")
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute(
                "CREATE TABLE extraction_pipeline ("
                "id INTEGER PRIMARY KEY CHECK(id=1), config TEXT NOT NULL, "
                "phase TEXT NOT NULL CHECK(phase IN ('sample','rest','done')))"
            )
            db.execute("INSERT INTO extraction_pipeline VALUES (1, ?, 'sample')", (cfg,))
            db.execute("COMMIT")
        except BaseException:
            db.execute("ROLLBACK")
            raise
        return "sample"
    row = db.execute("SELECT config, phase FROM extraction_pipeline WHERE id=1").fetchone()
    if row is None or row[0] != cfg:
        raise ValueError("pipeline configuration or candidate files changed; use a new database")
    return row[1]


def _train_and_recompress(db_path: Path, dict_size: int, phase: str) -> None:
    label = "Sample" if phase == "rest" else "Full corpus"
    with Phase(f"{label}: train dictionaries") as progress:
        dicts = train_dicts(db_path, dict_size)
        progress.details = f"{len(dicts.exprs):,} expression bytes; {len(dicts.trns):,} transition bytes"
    with closing(open_extraction_db(db_path)) as db:
        before = db_path.stat().st_size
        with Phase(f"{label}: recompress database") as progress:
            recompress_db(db, dicts)
            after = db_path.stat().st_size
            progress.details = f"{before / 2**20:.2f} → {after / 2**20:.2f} MiB"
        # A crash before this marker only repeats safe recompression on resume.
        db.execute("UPDATE extraction_pipeline SET phase=? WHERE id=1", (phase,))


def run_full_pipeline(
    root: Path, db_path: Path, sample_size: int, seed: int, dict_size: int, exe: Path, workers: int, timing: bool
) -> DatasetSummary:
    if workers < 1 or dict_size < 1:
        raise ValueError("worker count and dictionary size must be positive")
    if not exe.is_file():
        raise FileNotFoundError(f"build the Lean extractor first: {exe}")
    with Phase("Discover candidate files") as progress:
        files = find_files(root)
        progress.details = f"{len(files):,} candidates"
    with Phase("Select initial sample") as progress:
        sample = sample_files(files, sample_size, seed)
        progress.details = f"{len(sample):,} files; seed {seed}"
    cfg = json.dumps(
        {"files": [str(path) for path in files], "sample_size": sample_size, "seed": seed, "dict_size": dict_size},
        sort_keys=True,
    )
    with closing(open_extraction_db(db_path)) as db:
        phase = _pipeline_phase(db, cfg)
        if phase == "done":
            return db_summary(db)
    cands_path = db_path.with_suffix(".files.txt")
    sample_path = db_path.with_suffix(".sample.txt")
    write_file_list(cands_path, files)
    write_file_list(sample_path, sample)
    if phase == "sample":
        run_pipeline(sample_path, db_path, exe, workers, timing, phase_name="Extract initial sample")
        _train_and_recompress(db_path, dict_size, "rest")
    run_pipeline(cands_path, db_path, exe, workers, timing, phase_name="Extract remaining files")
    _train_and_recompress(db_path, dict_size, "done")
    with closing(open_extraction_db(db_path)) as db:
        return db_summary(db)


def pipeline_main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=750)
    parser.add_argument("--seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--dictionary-size", type=int, default=DEFAULT_DICT_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--lean-exe", type=Path, default=DEFAULT_LEAN_EXE)
    args = parser.parse_args(argv)
    try:
        summary = run_full_pipeline(
            args.root,
            args.db,
            args.sample_size,
            args.seed,
            args.dictionary_size,
            args.lean_exe.resolve(),
            args.workers,
            args.timing,
        )
    except (OSError, sqlite3.Error, RuntimeError, ValueError, zstd.ZstdError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    _print_summary(summary, args.db, started)
    return 0
