"""Read-only corpus queries and training batches from precomputed feature archives."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import file_digest
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, Literal, cast

import msgspec
import numpy as np
from scipy.sparse import csr_array, vstack

from trustmebro.preprocessing.records import LabelPolicy

from .preprocessing.archives import FeatureStream, FrozenFeatures, open_features, prepare_feature_source

if TYPE_CHECKING:
    from .preprocessing.records import SplitManifest


@dataclass(frozen=True, slots=True)
class FeatureSpace:
    width: int
    classes: tuple[str, ...]
    vocab_id: str


@dataclass(frozen=True, slots=True)
class Data:
    """One model-input contract; cached loading keeps transition indices separately."""

    matrix: csr_array
    label_ids: np.ndarray
    theorems: tuple[str, ...]
    space: FeatureSpace
    count_cols: np.ndarray | None = None  # explicit layout metadata, required only for log-count scaling

    @property
    def classes(self) -> tuple[str, ...]:
        return self.space.classes


@dataclass(frozen=True, slots=True)
class Evaluation:
    rows: int
    ce: float
    accuracy: float
    top3_acc: float
    top5_acc: float
    macro_f1: float
    confusion: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray
    weighted_ce: float | None = None


@dataclass(frozen=True, slots=True)
class TrainingCfg:
    """Bounds output batches and encoded theorem/header frames, not total RSS.

    float32 is an explicit approximate training representation. The canonical
    archive remains unchanged. Sparse indices use int32 when they fit.
    """

    rows: int = 8192
    nnz: int = 1_000_000
    frame_bytes: int = 64 * 2**20
    dtype: Literal["float32", "float64"] = "float64"


@dataclass(frozen=True, slots=True)
class TrainingBatch:
    matrix: csr_array
    label_ids: np.ndarray
    classes: tuple[str, ...]
    theorems: tuple[str, ...]  # row-aligned; repeated names share their string object
    steps: np.ndarray  # original transition indices, not indices after filtering


DEFAULT_TRAINING_CFG = TrainingCfg()
DEFAULT_MAX_BYTES = 4 * 2**30


def _check_training_cfg(cfg: TrainingCfg) -> None:
    if min(cfg.rows, cfg.nnz, cfg.frame_bytes) < 1 or cfg.dtype not in ("float32", "float64"):
        raise ValueError("training batch/frame limits must be positive and dtype float32 or float64")


@dataclass(frozen=True, slots=True)
class LearningLabels:
    """Resolved frozen policy and one stable class mapping per loading operation."""

    policy: LabelPolicy
    classes: tuple[str, ...]
    class_ids: dict[str, int]


def _training_policy(src: FeatureStream | FrozenFeatures, policy: LabelPolicy | None) -> LearningLabels:
    if src.header.label_policy is not None:
        frozen = msgspec.json.decode(src.header.label_policy, type=LabelPolicy)
        if policy is not None and policy != frozen:
            raise ValueError("label policy differs from the frozen training archive")
        policy = frozen
    if policy is None:
        raise ValueError("archive has no frozen labels; supply a label policy or convert with --labels")
    classes = policy.labels
    return LearningLabels(policy, classes, {label: idx for idx, label in enumerate(classes)})


def _index_dtype(width: int, rows: int, nnz: int) -> type[np.int32 | np.int64]:
    return np.int32 if max(width, rows, nnz) <= np.iinfo(np.int32).max else np.int64


def _join_training(parts: list[TrainingBatch]) -> TrainingBatch:
    if len(parts) == 1:
        return parts[0]
    return TrainingBatch(
        cast(csr_array, vstack([part.matrix for part in parts], format="csr")),
        np.concatenate([part.label_ids for part in parts]),
        parts[0].classes,
        tuple(chain.from_iterable(part.theorems for part in parts)),
        np.concatenate([part.steps for part in parts]),
    )


def _labeled_cached_rows(
    src: FeatureStream, labels: LearningLabels, names: set[str] | None = None, pop: set[str] | None = None
) -> Iterator[TrainingBatch]:
    classes, class_ids, policy = labels.classes, labels.class_ids, labels.policy
    seen: set[str] = set()
    for rows in src.rows:
        if pop is not None:
            if rows.name not in pop or rows.name in seen:
                raise ValueError("feature theorem population disagrees with split manifest")
            seen.add(rows.name)
        if names is not None and rows.name not in names:
            continue
        ids = np.fromiter(
            (class_ids.get(label, -1) if label is not None else -1 for label in map(policy.label, rows.tactics)),
            dtype=np.int64,
        )
        keep = np.flatnonzero(ids >= 0)
        matrix, steps = rows.matrix, rows.steps
        if len(keep) != len(ids):
            matrix, steps, ids = matrix[keep], steps[keep], ids[keep]
        if len(ids):
            yield TrainingBatch(matrix, ids, classes, (rows.name,) * len(ids), steps)
    if pop is not None and seen != pop:
        raise ValueError("feature archive is missing split theorems")


def _training_batches(
    src: FeatureStream,
    labels: LearningLabels,
    cfg: TrainingCfg,
    names: set[str] | None = None,
    pop: set[str] | None = None,
) -> Iterator[TrainingBatch]:
    parts: list[TrainingBatch] = []
    used_rows = used_nnz = 0
    for rows in _labeled_cached_rows(src, labels, names, pop):
        matrix, ids, steps = rows.matrix, rows.label_ids, rows.steps
        start = 0
        while start < len(ids):
            end = min(
                len(ids),
                start + cfg.rows - used_rows,
                int(np.searchsorted(matrix.indptr, matrix.indptr[start] + cfg.nnz - used_nnz, side="right")) - 1,
            )
            if end == start:
                if not parts:
                    raise MemoryError("one feature row exceeds the batch nonzero limit; increase TrainingCfg.nnz")
                yield _join_training(parts)
                parts.clear()
                used_rows = used_nnz = 0
                continue
            chunk = matrix if start == 0 and end == len(ids) else matrix[start:end]
            idx_dtype = _index_dtype(src.width, end - start, chunk.nnz)
            with np.errstate(over="raise", invalid="raise"):
                chunk = csr_array(
                    (
                        chunk.data.astype(cfg.dtype, copy=False),
                        chunk.indices.astype(idx_dtype, copy=False),
                        chunk.indptr.astype(idx_dtype, copy=False),
                    ),
                    shape=chunk.shape,
                )
            parts.append(TrainingBatch(chunk, ids[start:end], rows.classes, rows.theorems[start:end], steps[start:end]))
            used_rows += end - start
            used_nnz += chunk.nnz
            start = end
            if used_rows == cfg.rows or used_nnz == cfg.nnz:
                yield _join_training(parts)
                parts.clear()
                used_rows = used_nnz = 0
    if parts:
        yield _join_training(parts)


def read_training_batches(
    path: Path,
    policy: LabelPolicy | None = None,
    *,
    cfg: TrainingCfg = DEFAULT_TRAINING_CFG,
    split: SplitManifest | None = None,
    subset: Literal["train", "validation"] | None = None,
) -> Iterator[TrainingBatch]:
    """Decompress stored CSR rows; never query a DB or construct graph features.

    Limits bound each encoded frame, output rows and output nonzeros. A frame
    and decoded theorem coexist with accumulated chunks and assembly scratch;
    these are not process-RSS limits. Retaining batches is the caller's choice.
    Labels come from the frozen header unless an older archive needs a policy.
    Consume the iterator completely to verify footer counters; close it if you
    stop early. Batches can cross theorem boundaries; original provenance and
    input selection are preserved. A bound manifest selects a logical role;
    without one all labeled rows are returned. No split or vocabulary is fitted.
    """
    _check_training_cfg(cfg)
    with open_features(path, max_frame_bytes=cfg.frame_bytes) as source:
        labels = _training_policy(source, policy)
        names, population = _training_subset(source, split, subset, labels)
        yield from _training_batches(source, labels, cfg, names, population)


def _training_subset(
    src: FeatureStream | FrozenFeatures,
    split: SplitManifest | None,
    subset: Literal["train", "validation"] | None,
    labels: LearningLabels,
) -> tuple[set[str] | None, set[str] | None]:
    if subset not in (None, "train", "validation") or (subset is None) != (split is None):
        raise ValueError("provide both a split manifest and train/validation subset, or neither")
    if split is None:
        return None, None
    from .preprocessing.records import split_id

    header, fitted = src.header, src.header.vocab.selection
    if (
        header.split_id != split_id(split)
        or fitted is None
        or set(fitted.theorems or ()) != set(split.train)
        or labels.policy != split.policy
        or (header.selection.db, header.selection.size, header.selection.modified_ns)
        != (split.selection.db, split.selection.size, split.selection.modified_ns)
    ):
        raise ValueError("split manifest differs from the frozen feature archive")
    return set(split.train if subset == "train" else split.validation), set(split.train) | set(split.validation)


def _training_bytes(width: int, rows: int, nnz: int, cfg: TrainingCfg) -> int:
    idx_size = np.dtype(_index_dtype(width, rows, nnz)).itemsize
    return nnz * (np.dtype(cfg.dtype).itemsize + idx_size) + (rows + 1) * idx_size + 16 * rows


def load_training_matrix(
    path: Path,
    policy: LabelPolicy | None = None,
    *,
    cfg: TrainingCfg = DEFAULT_TRAINING_CFG,
    max_bytes: int = DEFAULT_MAX_BYTES,
    split: SplitManifest | None = None,
    subset: Literal["train", "validation"] | None = None,
    prepared: FrozenFeatures | None = None,
) -> TrainingBatch:
    """Two sequential decompression passes, one allocation of the full CSR.

    First count selected rows/nonzeros; then fill exactly allocated buffers.
    This avoids keeping all batch matrices plus a second full concatenation.
    max_bytes limits final numeric buffers (CSR, labels, steps), not total RSS:
    bounded decoder/batch scratch and theorem-name references also remain.
    Requires an immutable, completed archive; it does not rebuild features.
    """
    _check_training_cfg(cfg)
    if max_bytes < 1:
        raise ValueError("training numeric-buffer budget must be positive")
    prepared = prepare_feature_source(path, max_frame_bytes=cfg.frame_bytes) if prepared is None else prepared
    labels = _training_policy(prepared, policy)
    names, population = _training_subset(prepared, split, subset, labels)
    return _load_matrix(path, prepared, labels, cfg, max_bytes, names, population)


def _load_matrix(
    path: Path,
    prepared: FrozenFeatures,
    labels: LearningLabels,
    cfg: TrainingCfg,
    max_bytes: int,
    role_names: set[str] | None,
    pop: set[str] | None,
) -> TrainingBatch:
    """Count and fill owned CSR buffers; all frozen preparation is supplied."""
    stamp = path.stat().st_size, path.stat().st_mtime_ns
    rows = nnz = 0
    with open_features(path, max_frame_bytes=cfg.frame_bytes, prepared=prepared) as source:
        width = source.width
        # Counting needs no concatenation, precision conversion, or batch copies.
        for batch in _labeled_cached_rows(source, labels, role_names, pop):
            if np.diff(batch.matrix.indptr).max(initial=0) > cfg.nnz:
                raise MemoryError("one feature row exceeds the batch nonzero limit; increase TrainingCfg.nnz")
            rows += len(batch.label_ids)
            nnz += batch.matrix.nnz
            if _training_bytes(width, rows, nnz, cfg) > max_bytes:
                raise MemoryError("full training matrix exceeds its numeric-buffer budget; stream batches instead")
    if (path.stat().st_size, path.stat().st_mtime_ns) != stamp:
        raise ValueError("feature archive changed during counting")
    if _training_bytes(width, rows, nnz, cfg) > max_bytes:
        raise MemoryError("empty training matrix exceeds its numeric-buffer budget")
    idx_dtype = _index_dtype(width, rows, nnz)
    data, idxs = np.empty(nnz, dtype=cfg.dtype), np.empty(nnz, dtype=idx_dtype)
    indptr = np.empty(rows + 1, dtype=idx_dtype)
    indptr[0] = 0
    ids, steps = np.empty(rows, dtype=np.int64), np.empty(rows, dtype=np.int64)
    names: list[str] = []
    row_pos = nnz_pos = 0
    with open_features(path, max_frame_bytes=cfg.frame_bytes, prepared=prepared) as source:
        batches = _training_batches(source, labels, cfg, role_names, pop)
        for batch in batches:
            end_row, end_nnz = row_pos + len(batch.label_ids), nnz_pos + batch.matrix.nnz
            if end_row > rows or end_nnz > nnz:
                raise ValueError("feature archive counters changed between passes")
            data[nnz_pos:end_nnz], idxs[nnz_pos:end_nnz] = batch.matrix.data, batch.matrix.indices
            indptr[row_pos + 1 : end_row + 1] = batch.matrix.indptr[1:] + nnz_pos
            ids[row_pos:end_row], steps[row_pos:end_row] = batch.label_ids, batch.steps
            names.extend(batch.theorems)
            row_pos, nnz_pos = end_row, end_nnz
    if (row_pos, nnz_pos) != (rows, nnz) or (path.stat().st_size, path.stat().st_mtime_ns) != stamp:
        raise ValueError("feature archive changed between loading passes")
    return TrainingBatch(csr_array((data, idxs, indptr), shape=(rows, width)), ids, labels.classes, tuple(names), steps)


@dataclass(frozen=True, slots=True)
class LearningData:
    train: TrainingBatch
    validation: TrainingBatch
    source: FrozenFeatures


def load_learning_data(
    path: Path,
    split: SplitManifest,
    policy: LabelPolicy | None = None,
    *,
    cfg: TrainingCfg = DEFAULT_TRAINING_CFG,
    max_bytes: int = DEFAULT_MAX_BYTES,
    prepared: FrozenFeatures | None = None,
) -> LearningData:
    """One frozen preparation shared by four exact-allocation stream passes."""
    prepared = prepare_feature_source(path, max_frame_bytes=cfg.frame_bytes) if prepared is None else prepared
    _check_training_cfg(cfg)
    if max_bytes < 1:
        raise ValueError("training numeric-buffer budget must be positive")
    labels = _training_policy(prepared, policy)
    train_names, pop = _training_subset(prepared, split, "train", labels)
    assert train_names is not None and pop is not None
    train, validation = (
        _load_matrix(path, prepared, labels, cfg, max_bytes, names, pop) for names in (train_names, pop - train_names)
    )
    return LearningData(train, validation, prepared)


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    accuracy: float
    macro_f1: float
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    support: np.ndarray


def classification_metrics(confusion: np.ndarray) -> ClassificationMetrics:
    """Declared-class metrics, including zero support; model owners accumulate CE."""
    support, predicted = confusion.sum(axis=1), confusion.sum(axis=0)
    correct = confusion.diagonal()
    precision = np.divide(correct, predicted, out=np.zeros(len(correct)), where=predicted != 0)
    recall = np.divide(correct, support, out=np.zeros(len(correct)), where=support != 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(len(correct)), where=denom != 0)
    return ClassificationMetrics(float(correct.sum() / support.sum()), float(f1.mean()), precision, recall, f1, support)


def publish_model_report(checkpoint: Path, report: dict[str, object], *, replace: bool = False) -> None:
    """A small report associated with an already-published, trusted checkpoint.

    Missing/failed report publication cannot alter the checkpoint. Readers check
    its content hash without loading predictor weights; no fitting is triggered.
    """
    with checkpoint.open("rb") as stream:
        ident = file_digest(stream, "sha256").hexdigest()
    data = {**report, "checkpoint_sha256": ident}
    path = checkpoint.with_suffix(checkpoint.suffix + ".report.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".report-", delete=False) as stream:
        pending = Path(stream.name)
    try:
        pending.write_bytes(msgspec.json.encode(data))
        if replace:
            pending.replace(path)
        else:
            path.hardlink_to(pending)
    finally:
        pending.unlink(missing_ok=True)
