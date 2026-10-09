"""Publish two theorem-disjoint databases with explicit label support minima."""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from contextlib import ExitStack, closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic
from typing import Any, cast

import msgspec
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_array, vstack
from sklearn.model_selection import StratifiedGroupKFold

from trustmebro.artifacts import BlobSizes
from trustmebro.extraction import records as r
from trustmebro.extraction.storage import (
    Corpus,
    db_blob_sizes,
    decode_trns,
    open_corpus,
    open_extraction_db,
    stored_dicts,
)
from trustmebro.preprocessing.records import (
    DEFAULT_SPLIT_CFG,
    LabelPolicy,
    LabelSupport,
    MinSupport,
    Selection,
    SplitCfg,
    SplitManifest,
    SubsetStats,
    read_label_policy,
)
from trustmebro.runtime import Phase


@dataclass(frozen=True, slots=True)
class SplitReport:
    src: str
    src_size: int
    src_modified_ns: int
    policy: LabelPolicy
    cfg: SplitCfg
    train: SubsetStats
    test: SubsetStats
    repair: str
    solver_optimal: bool | None
    blob_sizes: dict[str, dict[str, BlobSizes]] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class LabelCounts:
    ids: np.ndarray
    raw_trns: np.ndarray
    labels: tuple[str, ...]
    trns: csr_array  # theorem × label; no expression blobs or graphs retained


def logical_splits(src: Path, policy: LabelPolicy, cfg: SplitCfg, *, folds: int = 1) -> tuple[SplitManifest, ...]:
    """One quota-repaired holdout or disjoint stratified grouped K-folds.

    K-fold stratification attempts label balance, not guaranteed quotas. Verify
    every fold's hard minima and fail explicitly if it cannot satisfy them.
    Only transition blobs are decoded; graphs and feature vectors are untouched.
    """
    if folds < 1:
        raise ValueError("fold count must be positive; one selects holdout mode")
    if folds > 1:
        cfg = replace(cfg, test_frac=1 / folds)
    stamp = src.stat()
    with open_corpus(src) as corpus:
        counts = _label_counts(corpus, policy)
        names = np.asarray(
            [row[0] for row in corpus.db.execute("SELECT name FROM theorems ORDER BY name")], dtype=object
        )
    if folds == 1:
        mask, repair, optimal = _assign(counts, cfg)
        masks = [mask]
    else:
        if folds > len(names):
            raise ValueError("fold count exceeds theorem count")
        support = np.asarray(cast(csr_array, counts.trns > 0).sum(axis=0)).ravel()
        if np.any(support < folds * cfg.test_min.theorems):
            raise ValueError("insufficient distinct-theorem label support for all validation folds")
        observed = counts.trns.tocoo()
        repeats = cast(NDArray[np.int64], observed.data)
        y = np.repeat(observed.col, repeats)
        groups = np.repeat(observed.row, repeats)
        assignment = np.full(len(names), -1, dtype=np.int64)
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=cfg.seed)
        for idx, (_, held_out) in enumerate(splitter.split(np.empty((len(y), 0)), y, groups)):
            assignment[np.unique(groups[held_out])] = idx
        # Theorems whose actions are all dropped still belong to exactly one fold.
        sizes = np.bincount(assignment[assignment >= 0], minlength=folds)
        for row in np.random.default_rng(cfg.seed).permutation(np.flatnonzero(assignment < 0)):
            idx = int(sizes.argmin())
            assignment[row] = idx
            sizes[idx] += 1
        masks = [assignment == idx for idx in range(folds)]
        repair, optimal = "stratified-group", None
    selection = Selection(str(src.resolve()), stamp.st_size, stamp.st_mtime_ns, (), None, None)
    manifests = []
    for idx, mask in enumerate(masks):
        train, validation = _subset_stats(counts, ~mask), _subset_stats(counts, mask)
        for stats, minimum in ((train, cfg.train_min), (validation, cfg.test_min)):
            if not stats.theorems or any(
                val.trns < minimum.trns or val.theorems < minimum.theorems for val in stats.labels.values()
            ):
                raise ValueError(f"split {idx + 1} fails label-support minima; adjust the split settings")
        manifests.append(
            SplitManifest(
                selection,
                policy,
                cfg,
                idx + 1,
                folds,
                tuple(names[~mask]),
                tuple(names[mask]),
                train,
                validation,
                repair,
                optimal,
            )
        )
    if (src.stat().st_size, src.stat().st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
        raise ValueError("source changed during logical splitting")
    return tuple(manifests)


type StoredRow = tuple[int, str, str, str | None, int | None, int | None, int, int, bytes, bytes]


def _label_counts(corpus: Corpus, policy: LabelPolicy) -> LabelCounts:
    labels = policy.labels
    cols = {label: idx for idx, label in enumerate(labels)}
    ids: list[int] = []
    raw_trns: list[int] = []
    refs: list[int] = []
    vals: list[int] = []
    bounds = [0]
    for ident, name, count, blob in corpus.db.execute("SELECT id,name,trn_count,trns FROM theorems ORDER BY name"):
        trns = decode_trns(blob, corpus.codec)
        if count != len(trns):
            raise r.SchemaError(f"transition count disagrees for {name!r}")
        labeled = Counter(label for trn in trns if (label := policy.label(trn.tactic)) is not None)
        ids.append(ident)
        raw_trns.append(count)
        for label, amount in labeled.items():
            refs.append(cols[label])
            vals.append(amount)
        bounds.append(len(refs))
    counts = csr_array(
        (np.asarray(vals, dtype=np.int64), np.asarray(refs, dtype=np.int64), np.asarray(bounds, dtype=np.int64)),
        shape=(len(ids), len(labels)),
    )
    return LabelCounts(np.asarray(ids, dtype=np.int64), np.asarray(raw_trns, dtype=np.int64), labels, counts)


def _assign(counts: LabelCounts, cfg: SplitCfg) -> tuple[np.ndarray, str, bool | None]:
    """Repair only when random assignment violates hard minima.

    Binary variables select test theorems. The objective minimizes changes to
    the initial seeded assignment, not label-ratio matching. Native sparse
    constraints bound both transition and distinct-theorem support per label.
    """
    n, labels = cast(tuple[int, int], counts.trns.shape)
    if n < 2:
        raise ValueError("partitioning requires at least two theorems")
    constraints = vstack((counts.trns.T, cast(csr_array, counts.trns > 0).T), format="csc")
    totals = np.asarray(constraints.sum(axis=1)).reshape(-1)
    if np.any(totals > 2**53):
        raise OverflowError("label support exceeds exact solver arithmetic")
    lo = np.repeat((cfg.test_min.trns, cfg.test_min.theorems), labels)
    hi = totals - np.repeat((cfg.train_min.trns, cfg.train_min.theorems), labels)
    if np.any(lo > hi):
        lacking = sorted({counts.labels[idx % labels] for idx in np.flatnonzero(lo > hi)})
        raise ValueError(f"insufficient corpus support for label minima: {', '.join(lacking)}")
    test = np.zeros(n, dtype=bool)
    test[np.random.default_rng(cfg.seed).permutation(n)[: max(1, min(n - 1, round(n * cfg.test_frac)))]] = True

    def meets_min(mask: np.ndarray) -> bool:
        support = constraints @ mask.astype(np.int64)
        return bool(np.all(support >= lo) and np.all(support <= hi))

    if meets_min(test):
        return test, "none", None
    result = milp(
        np.where(test, -1.0, 1.0),
        integrality=np.ones(n, dtype=np.uint8),
        bounds=Bounds(0, 1),
        constraints=LinearConstraint(constraints, cast(Any, lo), cast(Any, hi)),  # SciPy accepts array bounds.
        options={"time_limit": cfg.solver_sec, "mip_rel_gap": 0},
    )
    if result.x is None:
        reason = (
            "label constraints are infeasible" if result.status == 2 else "no feasible split found within solver limits"
        )
        raise ValueError(f"{reason}: {result.message}")
    rounded = np.rint(result.x)
    if not np.all(np.isfinite(result.x)) or not np.all(np.abs(result.x - rounded) < 1e-6):
        raise ValueError("solver did not return an integer theorem assignment")
    if not np.all((rounded == 0) | (rounded == 1)) or not meets_min(rounded.astype(bool)):
        raise ValueError("solver assignment fails exact label-support checks")
    return rounded.astype(bool), "minimum-move", result.status == 0


def _subset_stats(counts: LabelCounts, selected: np.ndarray) -> SubsetStats:
    trns = counts.trns[selected]
    amounts = np.asarray(trns.sum(axis=0)).reshape(-1)
    theorems = np.asarray((trns > 0).sum(axis=0)).reshape(-1)
    raw = int(counts.raw_trns[selected].sum())
    retained = int(amounts.sum())
    return SubsetStats(
        int(selected.sum()),
        raw,
        retained,
        raw - retained,
        {
            label: LabelSupport(int(amount), int(proofs))
            for label, amount, proofs in zip(counts.labels, amounts, theorems, strict=True)
        },
    )


def _copy_partitions(corpus: Corpus, root: Path, test_ids: set[int], report: SplitReport) -> SplitReport:
    """Copy compressed rows verbatim; never mark a partial source file complete.

    Two buffers retain at most 256 compressed theorem rows apiece. This bounds
    row retention, not their bytes or total process memory.
    """
    cols = "id,name,module,src_path,src_start,src_end,expr_count,trn_count,exprs,trns"
    insert = f"INSERT INTO theorems ({cols}) VALUES ({','.join(['?'] * 10)})"
    with ExitStack() as stack:
        dicts = stored_dicts(corpus.db)
        dbs = {
            role: stack.enter_context(closing(open_extraction_db(root / f"{role}.db", dicts)))
            for role in ("train", "test")
        }
        buffers: dict[str, list[StoredRow]] = {role: [] for role in dbs}
        for db in dbs.values():
            db.execute("BEGIN")
            db.execute("CREATE TABLE partition_info (role TEXT PRIMARY KEY, report TEXT NOT NULL)")
        for row in corpus.db.execute(f"SELECT {cols} FROM theorems ORDER BY id"):
            role = "test" if row[0] in test_ids else "train"
            buffers[role].append(row)
            if len(buffers[role]) == 256:
                dbs[role].executemany(insert, buffers[role])
                buffers[role].clear()
        for role, db in dbs.items():
            db.executemany(insert, buffers[role])
        report = replace(report, blob_sizes={role: db_blob_sizes(db) for role, db in dbs.items()})
        data = msgspec.json.encode(report).decode()
        for role, db in dbs.items():
            db.execute("INSERT INTO partition_info VALUES (?,?)", (role, data))
            db.execute("COMMIT")
    return report


def partition_corpus(src: Path, output: Path, policy: LabelPolicy, cfg: SplitCfg = DEFAULT_SPLIT_CFG) -> SplitReport:
    """Publish train.db + test.db atomically as a new directory, without refitting.

    The source must remain completed and immutable. All raw rows, including
    unmapped actions, remain in exactly one partition; policy filtering happens
    when querying labels/features. Existing output is never replaced.
    """
    if output.exists():
        raise FileExistsError(f"partition output already exists: {output}")
    stamp = (src.stat().st_size, src.stat().st_mtime_ns)
    with open_corpus(src) as corpus:
        with Phase("Count theorem labels") as phase:
            counts = _label_counts(corpus, policy)
            phase.details = f"{len(counts.ids):,} theorems; {len(counts.labels):,} labels"
        with Phase("Choose theorem split") as phase:
            test, repair, optimal = _assign(counts, cfg)
            phase.details = f"{int((~test).sum()):,} train; {int(test.sum()):,} test; repair: {repair}"
        train_stats = _subset_stats(counts, ~test)
        test_stats = _subset_stats(counts, test)
        report = SplitReport(str(src.resolve()), *stamp, policy, cfg, train_stats, test_stats, repair, optimal)
        output.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=output.parent, prefix=".partition-") as tmp:
            pending = Path(tmp)
            with Phase("Copy partition databases"):
                report = _copy_partitions(corpus, pending, set(map(int, counts.ids[test])), report)
            if (src.stat().st_size, src.stat().st_mtime_ns) != stamp:
                raise ValueError("source database changed during partitioning")
            if output.exists():
                raise FileExistsError(f"partition output already exists: {output}")
            pending.rename(output)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument(
        "--labels", type=Path, required=True, help="JSON policy mapping exported tactic kinds to labels"
    )
    parser.add_argument("--output", type=Path, required=True, help="new directory for train.db and test.db")
    parser.add_argument(
        "--test-frac",
        type=float,
        default=DEFAULT_SPLIT_CFG.test_frac,
        help="target fraction of theorems, before quota repair",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SPLIT_CFG.seed)
    for role, support in (("train", DEFAULT_SPLIT_CFG.train_min), ("test", DEFAULT_SPLIT_CFG.test_min)):
        parser.add_argument(f"--{role}-min-trns", type=int, default=support.trns)
        parser.add_argument(f"--{role}-min-theorems", type=int, default=support.theorems)
    parser.add_argument("--solver-seconds", type=float, default=DEFAULT_SPLIT_CFG.solver_sec)
    args = parser.parse_args(argv)
    start = monotonic()
    try:
        cfg = SplitCfg(
            args.test_frac,
            args.seed,
            MinSupport(args.train_min_trns, args.train_min_theorems),
            MinSupport(args.test_min_trns, args.test_min_theorems),
            args.solver_seconds,
        )
        report = partition_corpus(args.db, args.output, read_label_policy(args.labels), cfg)
    except (ValueError, OSError, sqlite3.Error, msgspec.DecodeError) as error:
        parser.exit(2, f"{error}\n")
    print(
        msgspec.json.encode(
            {
                "output": str(args.output),
                "train": report.train,
                "test": report.test,
                "repair": report.repair,
                "blob_sizes": report.blob_sizes,
                "dur_sec": monotonic() - start,
            }
        ).decode()
    )
    return 0
