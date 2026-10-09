"""Process-owned timing and consumer-replenished bounded execution."""

from __future__ import annotations

import csv
import os
import sys
import threading
import time
from collections.abc import Callable, Generator, Iterable
from concurrent.futures import FIRST_COMPLETED, Executor, wait
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from typing import Self, cast


class TimingLog:
    """Buffered spans distinguish wall, process CPU, and calling-thread CPU time.

    CPU includes executor management threads in the coordinator. Nested spans
    overlap, so totals must be compared by category, not summed indiscriminately.
    Flush at theorem boundaries to retain completed work after forced shutdown.
    """

    def __init__(self, dir: Path, role: str):
        dir.mkdir(parents=True, exist_ok=True)
        self.stream = (dir / f"{role}-{os.getpid()}.csv").open("x", newline="", buffering=65536)
        self.writer = csv.writer(self.stream)
        self.writer.writerow(("label", "phase", "start_ns", "wall_ns", "cpu_ns", "thread_ns", "ok"))
        self.flush()

    def flush(self) -> None:
        self.stream.flush()

    def close(self) -> None:
        self.stream.close()


@contextmanager
def checkpoint(log: TimingLog | None, label: str, phase: str) -> Generator[None]:
    """Time a complete operation, never a generator suspension across consumer work."""
    if log is None:
        yield
        return
    start, cpu, thread = time.perf_counter_ns(), time.process_time_ns(), time.thread_time_ns()
    ok = False
    try:
        yield
        ok = True
    finally:
        elapsed, used = time.perf_counter_ns() - start, time.process_time_ns() - cpu
        thread_used = time.thread_time_ns() - thread
        log.writer.writerow((label, phase, start, elapsed, used, thread_used, int(ok)))


def timed[**P, T](
    log: TimingLog | None, label: str, phase: str, fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs
) -> T:
    """Disabled diagnostics perform no clock reads, encoding, or file I/O."""
    if log is None:
        return fn(*args, **kwargs)
    with checkpoint(log, label, phase):
        return fn(*args, **kwargs)


def timed_batches[T](log: TimingLog | None, label: str, phase: str, batches: Iterable[T]) -> Generator[T]:
    """Measure producing each batch; serialization/consumer time belongs to other spans."""
    if log is None:
        yield from batches
        return
    iterator = iter(batches)
    sentinel = object()
    while (batch := timed(log, label, phase, next, iterator, sentinel)) is not sentinel:
        yield cast(T, batch)


class Phase:
    """One timed stage; refresh a single terminal line without spamming logs."""

    def __init__(self, name: str, *, refresh: bool = True):
        self.name = name
        self.refresh = refresh
        self.details = ""
        self.started = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _render(self) -> None:
        elapsed = time.monotonic() - self.started
        print(f"\r\033[2K{self.name}: {elapsed:.1f}s {self.details}", end="", file=sys.stderr, flush=True)

    def _refresh(self) -> None:
        while not self._stop.wait(0.5):
            self._render()

    def __enter__(self) -> Self:
        self.started = time.monotonic()
        print(f"{self.name}...", file=sys.stderr, flush=True)
        if self.refresh and sys.stderr.isatty():
            self._thread = threading.Thread(target=self._refresh, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, kind, error, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            print("\r\033[2K", end="", file=sys.stderr)
        status = "completed" if kind is None else "interrupted" if kind is KeyboardInterrupt else "failed"
        elapsed = time.monotonic() - self.started
        suffix = f" — {self.details}" if self.details else ""
        print(f"{self.name} {status} in {elapsed:.2f}s{suffix}", file=sys.stderr, flush=True)


def ready_results[Job, Result](
    pool: Executor, rows: Iterable[Job], workers: int, task: Callable[[Job], Result], timings: TimingLog | None = None
) -> Generator[Result]:
    """At most 2*workers pending jobs and one completed batch of at most 2*workers.

    Replenishment happens only on consumer demand, not executor callbacks.
    Refill freed slots before aggregation; release futures as results are consumed.
    This bounds outstanding jobs, not aggregate indexes or total process RSS.
    """
    if workers < 1:
        raise ValueError("workers must be positive")
    pending = iter(rows)
    futures = {timed(timings, "", "submit", pool.submit, task, row) for row in islice(pending, workers * 2)}
    try:
        while futures:
            finished, _ = timed(timings, "", "wait_results", wait, futures, return_when=FIRST_COMPLETED)
            futures.difference_update(finished)
            # Do not launch replacements after a completed worker failure.
            for future in finished:
                if future.exception() is not None:
                    future.result()
            for row in islice(pending, len(finished)):
                futures.add(timed(timings, "", "submit", pool.submit, task, row))
            while finished:
                future = finished.pop()
                result = timed(timings, "", "get_result", future.result)
                del future
                yield result
                del result
    finally:
        for future in futures:
            future.cancel()
