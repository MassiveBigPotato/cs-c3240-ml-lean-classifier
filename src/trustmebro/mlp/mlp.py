"""Sparse-input MLP training, independent of corpus access and feature construction."""

import json
import math
import os
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Literal, cast

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.sparse import csr_array, csr_matrix
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from torch import Tensor, nn

from trustmebro.learning import Data, Evaluation, FeatureSpace, classification_metrics, publish_model_report

type CSR = csr_array | csr_matrix
type FloatArray = NDArray[np.floating]
type BoolArray = NDArray[np.bool_]
type Scaling = Literal["none", "maxabs", "log_maxabs", "log_standard"]
SCALING_POLICIES: tuple[Scaling, ...] = ("none", "maxabs", "log_maxabs", "log_standard")


@dataclass(frozen=True, slots=True)
class PreparedData:
    """Validated, caller-owned immutable partitions and their train-fitted scale."""

    train: Data
    validation: Data
    scaling: Scaling
    scale: FloatArray | None
    class_count: NDArray[np.int64]
    log_cols: BoolArray | None = None


@dataclass(frozen=True, slots=True)
class ModelCfg:
    hidden: tuple[int, ...] = (256,)
    dropout: float = 0.0


DEFAULT_MODEL_CFG = ModelCfg()


@dataclass(frozen=True, slots=True)
class FitCfg:
    model: ModelCfg = DEFAULT_MODEL_CFG
    batch_rows: int = 8192
    max_epochs: int = 50
    patience: int = 10
    min_delta: float = 1e-4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 0
    device: str = "cuda"
    scaling: Scaling = "maxabs"
    class_weight: Literal["none", "balanced", "inverse_sqrt"] = "none"
    label_smoothing: float = 0.0


DEFAULT_FIT_CFG = FitCfg()


@dataclass(frozen=True, slots=True)
class Epoch:
    epoch: int
    train_ce: float
    validation: Evaluation
    dur_sec: float


@dataclass(frozen=True, slots=True)
class Predictor:
    model: SparseMLP
    scale: FloatArray | None
    space: FeatureSpace
    class_weights: FloatArray | None = None
    log_cols: BoolArray | None = None


@dataclass(frozen=True, slots=True)
class _DeviceScaling:
    """Pass-owned device metadata and deferred input-error flags; no corpus values are retained."""

    scale: Tensor | None
    log_cols: Tensor | None
    errors: Tensor


@dataclass(frozen=True, slots=True)
class CheckpointState:
    """An owned CPU snapshot; coincident minima may share the same immutable snapshot."""

    epoch: int
    state: dict[str, Tensor]


@dataclass(frozen=True, slots=True)
class FitResult:
    predictor: Predictor
    cfg: FitCfg
    history: tuple[Epoch, ...]
    best_epoch: int
    train_split: dict[str, int | str]
    val_split: dict[str, int | str]
    weighted_best: CheckpointState | None = None


class SparseMLP(nn.Module):
    """Only the input is sparse; parameters, gradients and AdamW state are dense."""

    def __init__(self, width: int, classes: int, cfg: ModelCfg = DEFAULT_MODEL_CFG) -> None:
        super().__init__()
        if width < 1 or classes < 2 or not cfg.hidden or any(size < 1 for size in cfg.hidden):
            raise ValueError("positive input/hidden widths and at least two classes are required")
        if not 0 <= cfg.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.cfg = cfg
        # Store this in multiplication order, avoiding a huge transposed weight per minibatch.
        self.weight = nn.Parameter(torch.empty(width, cfg.hidden[0], dtype=torch.float32))
        self.bias = nn.Parameter(torch.empty(cfg.hidden[0], dtype=torch.float32))
        bound = 1 / math.sqrt(width)
        nn.init.uniform_(self.weight, -bound, bound)
        nn.init.uniform_(self.bias, -bound, bound)
        layers: list[nn.Module] = [nn.ReLU(), nn.Dropout(cfg.dropout)]
        for src, dst in zip(cfg.hidden, cfg.hidden[1:]):
            layers.extend((nn.Linear(src, dst, dtype=torch.float32), nn.ReLU(), nn.Dropout(cfg.dropout)))
        layers.append(nn.Linear(cfg.hidden[-1], classes, dtype=torch.float32))
        self.tail = nn.Sequential(*layers)

    def forward(self, data: Tensor) -> Tensor:
        return self.tail(torch.sparse.mm(data, self.weight) + self.bias)


def _validate_space(space: FeatureSpace) -> None:
    if space.width < 1 or len(space.classes) < 2 or len(set(space.classes)) != len(space.classes):
        raise ValueError("feature space needs a positive width and distinct ordered classes")
    if not space.vocab_id or any(not isinstance(name, str) or not name for name in space.classes):
        raise ValueError("nonempty vocabulary identity and class names are required")


def _validate_matrix(matrix: CSR, space: FeatureSpace) -> None:
    _validate_space(space)
    if (
        not isinstance(matrix, (csr_array, csr_matrix))
        or cast(tuple[int, int], matrix.shape)[1] != space.width
        or cast(tuple[int, int], matrix.shape)[0] < 1
    ):
        raise ValueError("expected a nonempty CSR matrix with the declared feature width")
    matrix.check_format(full_check=True)
    if not matrix.has_canonical_format:
        raise ValueError("CSR rows must have sorted, unique column indices")
    if matrix.dtype.kind not in "fiu" or not np.isfinite(matrix.data).all():
        raise ValueError("feature values must be real and finite")


def _validate_data(data: Data, expected: FeatureSpace | None = None) -> None:
    if expected is not None and data.space != expected:
        raise ValueError("feature width, class order or vocabulary identity differs")
    _validate_matrix(data.matrix, data.space)
    labels = data.label_ids
    if not isinstance(labels, np.ndarray) or labels.ndim != 1 or labels.dtype.kind not in "iu":
        raise ValueError("label_ids must be a one-dimensional integer array")
    if len(labels) != cast(tuple[int, int], data.matrix.shape)[0] or len(data.theorems) != len(labels):
        raise ValueError("features, labels and theorem identifiers must be row-aligned")
    if labels.min() < 0 or labels.max() >= len(data.space.classes):
        raise ValueError("label id outside declared class order")
    if any(not isinstance(name, str) or not name for name in data.theorems):
        raise ValueError("each row requires a nonempty theorem identifier")


def _device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("this draft supports CPU or CUDA only")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use device='cpu' explicitly for CPU execution")
    return device


def validate_cfg(cfg: FitCfg) -> torch.device:
    """Reject invalid settings/devices before allocating a network or starting a run queue."""
    if not cfg.model.hidden or any(size < 1 for size in cfg.model.hidden) or not 0 <= cfg.model.dropout < 1:
        raise ValueError("positive hidden widths and dropout in [0, 1) are required")
    if min(cfg.batch_rows, cfg.max_epochs, cfg.patience) < 1:
        raise ValueError("batch_rows, max_epochs and patience must be positive")
    if not math.isfinite(cfg.lr) or cfg.lr <= 0:
        raise ValueError("learning rate must be finite and positive")
    if any(not math.isfinite(val) or val < 0 for val in (cfg.weight_decay, cfg.min_delta)):
        raise ValueError("weight decay and minimum improvement must be finite and nonnegative")
    if cfg.seed < 0 or cfg.scaling not in SCALING_POLICIES:
        raise ValueError(f"expected a nonnegative seed and scaling in {SCALING_POLICIES}")
    if cfg.class_weight not in ("none", "balanced", "inverse_sqrt"):
        raise ValueError("expected class_weight='none', 'balanced' or 'inverse_sqrt'")
    if not math.isfinite(cfg.label_smoothing) or not 0 <= cfg.label_smoothing <= 1:
        raise ValueError("label_smoothing must be finite and in [0, 1]")
    device = _device(cfg.device)
    if device.type == "cuda" and device.index is not None and device.index >= torch.cuda.device_count():
        raise ValueError("requested CUDA device does not exist")
    return device


def _scaled_values(
    vals: np.ndarray, cols: np.ndarray, log_cols: BoolArray | None, scale: FloatArray | None
) -> np.ndarray:
    """Transform stored entries only; never center, densify, or mutate caller-owned data."""
    vals = np.array(vals, dtype=np.float64, copy=True)
    if log_cols is not None:
        selected = log_cols[cols]
        if np.any(selected & (vals < 0)):
            raise ValueError("log-count scaling requires nonnegative count features")
        np.log1p(vals, out=vals, where=selected)
    if scale is not None:
        vals /= scale[cols]
    return vals


def _device_scaling(predictor: Predictor) -> _DeviceScaling:
    device = predictor.model.weight.device
    return _DeviceScaling(
        None if predictor.scale is None else torch.tensor(predictor.scale, dtype=torch.float64, device=device),
        None if predictor.log_cols is None else torch.tensor(predictor.log_cols, dtype=torch.bool, device=device),
        torch.zeros(2, dtype=torch.bool, device=device),
    )


def _check_inputs(scaling: _DeviceScaling) -> None:
    # Synchronize once per pass/epoch, not once per minibatch.
    negative, nonfinite = scaling.errors.cpu().tolist()
    if negative:
        raise ValueError("log-count scaling requires nonnegative count features")
    if nonfinite:
        raise ValueError("scaled features cannot be represented as finite float32 values")


def _input(matrix: CSR, rows: NDArray[np.integer] | slice, scaling: _DeviceScaling) -> Tensor:
    batch = matrix[rows]
    device = scaling.errors.device
    cols = torch.from_numpy(batch.indices).to(device)
    vals = torch.from_numpy(batch.data).to(device)
    if scaling.scale is not None or scaling.log_cols is not None:
        # Keep float64 until after scaling: large raw counts can overflow float32 but have finite scaled values.
        vals = vals.to(torch.float64)
        if scaling.log_cols is not None:
            selected = scaling.log_cols[cols]
            scaling.errors[0].logical_or_((selected & (vals < 0)).any())
            logged = torch.log1p(torch.where(selected, vals, 0))
            vals = torch.where(selected, logged, vals)
        if scaling.scale is not None:
            vals.div_(scaling.scale[cols])
    vals = vals.to(torch.float32)
    scaling.errors[1].logical_or_(~torch.isfinite(vals).all())
    return torch.sparse_csr_tensor(
        torch.from_numpy(batch.indptr).to(device), cols, vals, size=batch.shape, device=device, check_invariants=False
    )


def _batches(rows: int, batch_rows: int) -> Iterator[slice]:
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    for start in range(0, rows, batch_rows):
        yield slice(start, min(start + batch_rows, rows))


def _evaluation(
    confusion: NDArray[np.int64], loss: float, top_hits: NDArray[np.int64], weighted_loss: float | None
) -> Evaluation:
    measured = classification_metrics(confusion)
    support, precision, recall, f1 = measured.support, measured.precision, measured.recall, measured.f1
    rows = int(support.sum())
    return Evaluation(
        rows,
        loss / rows,
        measured.accuracy,
        float(top_hits[0] / rows),
        float(top_hits[1] / rows),
        measured.macro_f1,
        confusion,
        precision,
        recall,
        f1,
        support,
        weighted_loss,
    )


def _evaluate(
    predictor: Predictor,
    data: Data,
    batch_rows: int,
    weights: Tensor | None = None,
    *,
    scaling: _DeviceScaling | None = None,
) -> Evaluation:
    predictor.model.eval()
    scaling = _device_scaling(predictor) if scaling is None else scaling
    device = predictor.model.weight.device
    count = len(predictor.space.classes)
    confusion = torch.zeros(count * count, dtype=torch.int64, device=device)
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    top_hits = torch.zeros(2, dtype=torch.int64, device=device)
    weighted_loss = torch.zeros((), dtype=torch.float64, device=device)
    weighted_mass = torch.zeros((), dtype=torch.float64, device=device)
    with torch.inference_mode():
        for rows in _batches(len(data.label_ids), batch_rows):
            labels = torch.from_numpy(data.label_ids[rows].astype(np.int64, copy=False)).to(device)
            logits = predictor.model(_input(data.matrix, rows, scaling))
            losses = nn.functional.cross_entropy(logits, labels, reduction="none").double()
            total_loss += losses.sum()
            if weights is not None:
                mass = weights[labels].double()
                weighted_loss += (losses * mass).sum()
                weighted_mass += mass.sum()
            ranks = logits.argsort(dim=1, descending=True, stable=True)[:, : min(5, count)]
            confusion += torch.bincount(labels * count + ranks[:, 0], minlength=count * count)
            matches = ranks == labels[:, None]
            top_hits[0] += matches[:, : min(3, count)].any(dim=1).sum()
            top_hits[1] += matches.any(dim=1).sum()
    _check_inputs(scaling)
    loss = float(total_loss.item())
    if not math.isfinite(loss):
        raise FloatingPointError("nonfinite evaluation cross-entropy")
    weighted_mean = None
    if weights is not None:
        if weighted_mass.item() <= 0:
            raise ValueError("validation labels have no positive training-derived class weight")
        weighted_mean = float((weighted_loss / weighted_mass).item())
        if not math.isfinite(weighted_mean):
            raise FloatingPointError("nonfinite weighted evaluation cross-entropy")
    return _evaluation(confusion.reshape(count, count).cpu().numpy(), loss, top_hits.cpu().numpy(), weighted_mean)


def evaluate(predictor: Predictor, data: Data, *, batch_rows: int = 8192) -> Evaluation:
    """Aggregate row-based metrics and optionally train-weighted cross-entropy."""
    _validate_data(data, predictor.space)
    weights = (
        None
        if predictor.class_weights is None
        else torch.tensor(predictor.class_weights, dtype=torch.float32, device=predictor.model.weight.device)
    )
    return _evaluate(predictor, data, batch_rows, weights)


def _split_info(data: Data) -> dict[str, int | str]:
    names = sorted(set(data.theorems))
    ident = sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return {"rows": len(data.label_ids), "theorems": len(names), "theorem_ids_sha256": ident}


def prepare_training(train: Data, val: Data, *, scaling: Scaling = "maxabs") -> PreparedData:
    """Validate once and fit scaling once for sequential fits of immutable input data."""
    _validate_data(train)
    _validate_data(val, train.space)
    if set(train.theorems).intersection(val.theorems):
        raise ValueError("training and validation contain overlapping theorems")
    if scaling not in SCALING_POLICIES:
        raise ValueError(f"scaling must be one of {SCALING_POLICIES}")
    log_cols = None
    fit_matrix = train.matrix
    if scaling.startswith("log_"):
        if train.count_cols is None:
            raise ValueError("log-count scaling requires an explicit count_cols mask from the feature layout")
        if (
            not isinstance(train.count_cols, np.ndarray)
            or train.count_cols.dtype != np.bool_
            or train.count_cols.shape != (train.space.width,)
        ):
            raise ValueError("count_cols must be a boolean mask of the feature width")
        log_cols = train.count_cols.copy()
        log_cols.setflags(write=False)
        # One transient sparse copy during fitting
        fit_matrix = type(train.matrix)(
            (
                _scaled_values(train.matrix.data, train.matrix.indices, log_cols, None),
                train.matrix.indices,
                train.matrix.indptr,
            ),
            shape=train.matrix.shape,
            copy=False,
        )
    scaler = StandardScaler(with_mean=False) if scaling == "log_standard" else MaxAbsScaler()
    # Both configured scalers expose a fitted scale array; StandardScaler's
    # optional scale applies only when with_std=False, which we do not use.
    scale = None if scaling == "none" else cast(FloatArray, scaler.fit(fit_matrix).scale_).astype(np.float64)
    if log_cols is not None and scale is not None:
        scale[~log_cols] = 1  # leave indicators and already-bounded fractions unchanged
    if scale is not None and (not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("training scaler must have finite positive scales")
    counts = np.bincount(train.label_ids.astype(np.int64, copy=False), minlength=len(train.space.classes))
    return PreparedData(train, val, scaling, scale, counts, log_cols)


def fit_prepared(
    data: PreparedData, cfg: FitCfg = DEFAULT_FIT_CFG, *, on_epoch: Callable[[Epoch], None] | None = None
) -> FitResult:
    """Start a fresh model/optimizer.

    data must come from prepare_training and remain unmodified between fits.
    """
    device = validate_cfg(cfg)
    if cfg.scaling != data.scaling:
        raise ValueError("configuration differs from the prepared scaling policy")
    train, val, scale = data.train, data.validation, data.scale
    class_weights = None
    weights = None
    if cfg.class_weight != "none":
        observed = data.class_count > 0
        class_weights = np.zeros(len(data.class_count), dtype=np.float64)
        # N/(K*n_c), using only observed training classes. Absent classes get zero, never infinity.
        class_weights[observed] = data.class_count.sum() / (observed.sum() * data.class_count[observed])
        if cfg.class_weight == "inverse_sqrt":
            np.sqrt(class_weights, out=class_weights)
            # Preserve mean sample weight one and zero weights for unobserved classes.
            class_weights *= data.class_count.sum() / np.dot(data.class_count, class_weights)
        weights = torch.tensor(class_weights, dtype=torch.float32, device=device)
    rng = np.random.default_rng(cfg.seed)
    devices = (
        [] if device.type == "cpu" else [device.index if device.index is not None else torch.cuda.current_device()]
    )
    with torch.random.fork_rng(devices=devices):
        torch.default_generator.manual_seed(cfg.seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(cfg.seed)
        predictor = Predictor(
            SparseMLP(train.space.width, len(train.space.classes), cfg.model).to(device),
            scale,
            train.space,
            class_weights,
            data.log_cols,
        )
        scaling = _device_scaling(predictor)
        # foreach=False avoids AdamW's additional parameter-sized tensor-list intermediates.
        optimizer = torch.optim.AdamW(
            predictor.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, foreach=False
        )
        history: list[Epoch] = []
        criteria = 1 if weights is None else 2  # unweighted CE, then train-weighted CE
        best_states: dict[int, CheckpointState] = {}
        best_losses = [math.inf] * criteria
        stop_losses = [math.inf] * criteria
        stale = [0] * criteria
        for epoch in range(1, cfg.max_epochs + 1):
            started = perf_counter()
            predictor.model.train()
            order = rng.permutation(len(train.label_ids))
            train_loss = torch.zeros((), dtype=torch.float64, device=device)
            train_mass = torch.zeros((), dtype=torch.float64, device=device)
            for batch in _batches(len(order), cfg.batch_rows):
                rows = order[batch]
                labels = torch.from_numpy(train.label_ids[rows].astype(np.int64, copy=False)).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = predictor.model(_input(train.matrix, rows, scaling))
                loss = nn.functional.cross_entropy(logits, labels, weight=weights, label_smoothing=cfg.label_smoothing)
                loss.backward()
                optimizer.step()
                mass = len(rows) if weights is None else weights[labels].sum()
                train_loss += loss.detach().double() * mass
                train_mass += mass
            _check_inputs(scaling)
            mean_loss = float((train_loss / train_mass).item())
            if not math.isfinite(mean_loss):
                raise FloatingPointError("nonfinite training cross-entropy")
            metrics = _evaluate(predictor, val, cfg.batch_rows, weights, scaling=scaling)
            record = Epoch(epoch, mean_loss, metrics, perf_counter() - started)
            history.append(record)
            snapshot = None
            losses = (metrics.ce,) if weights is None else (metrics.ce, metrics.weighted_ce)
            for idx, val_loss in enumerate(losses):
                if val_loss is None:
                    raise RuntimeError("weighted fitting requires weighted validation cross-entropy")
                if val_loss < best_losses[idx]:
                    best_losses[idx] = val_loss
                    if snapshot is None:
                        snapshot = CheckpointState(
                            epoch,
                            {
                                name: tensor.detach().cpu().clone()
                                for name, tensor in predictor.model.state_dict().items()
                            },
                        )
                    best_states[idx] = snapshot
                if val_loss < stop_losses[idx] - cfg.min_delta:
                    stop_losses[idx], stale[idx] = val_loss, 0
                else:
                    stale[idx] += 1
            if on_epoch is not None:
                on_epoch(record)
            if all(count >= cfg.patience for count in stale):
                break
        predictor.model.load_state_dict(best_states[0].state)
        predictor.model.eval()
    return FitResult(
        predictor, cfg, tuple(history), best_states[0].epoch, _split_info(train), _split_info(val), best_states.get(1)
    )


def checkpoint_epoch(result: FitResult, crit: Literal["unweighted", "weighted"] = "unweighted") -> int:
    if crit == "unweighted":
        return result.best_epoch
    if crit != "weighted" or result.weighted_best is None:
        raise ValueError("a weighted checkpoint requires a class-weighted fit")
    return result.weighted_best.epoch


def epoch_history(result: FitResult) -> list[dict[str, float | int | None]]:
    """Report only epoch metrics; tensor state remains checkpoint-owned."""
    return [
        {
            "epoch": epoch.epoch,
            "train_cross_entropy": epoch.train_ce,
            "val_cross_entropy": epoch.validation.ce,
            "val_weighted_cross_entropy": epoch.validation.weighted_ce,
            "val_accuracy": epoch.validation.accuracy,
            "val_top3_accuracy": epoch.validation.top3_acc,
            "val_top5_accuracy": epoch.validation.top5_acc,
            "val_macro_f1": epoch.validation.macro_f1,
            "duration_s": epoch.dur_sec,
        }
        for epoch in result.history
    ]


def checkpoint_summary(result: FitResult, criterion: Literal["unweighted", "weighted"] = "unweighted") -> dict:
    """Metrics at that criterion's selected epoch, not unrelated maxima over the run."""
    epoch = checkpoint_epoch(result, criterion)
    best = result.history[epoch - 1].validation
    return {
        "criterion": criterion,
        "best_epoch": epoch,
        "val_cross_entropy": best.ce,
        "val_weighted_cross_entropy": best.weighted_ce,
        "val_accuracy": best.accuracy,
        "val_top3_accuracy": best.top3_acc,
        "val_top5_accuracy": best.top5_acc,
        "val_macro_f1": best.macro_f1,
        "precision": best.precision.tolist(),
        "recall": best.recall.tolist(),
        "f1": best.f1.tolist(),
        "support": best.support.tolist(),
        "confusion": best.confusion.tolist(),
    }


def save_checkpoint(
    path: Path, result: FitResult, *, replace: bool = False, crit: Literal["unweighted", "weighted"] = "unweighted"
) -> None:
    """Publish an inference checkpoint"""
    predictor = result.predictor
    epoch = checkpoint_epoch(result, crit)
    state = (
        result.weighted_best.state
        if crit == "weighted" and result.weighted_best is not None
        else predictor.model.state_dict()
    )
    data = {
        "space": asdict(predictor.space),
        "cfg": asdict(result.cfg),
        "state": {name: tensor.detach().cpu() for name, tensor in state.items()},
        "scale": None if predictor.scale is None else torch.from_numpy(predictor.scale),
        "log_cols": None if predictor.log_cols is None else torch.from_numpy(predictor.log_cols.copy()),
        "class_weights": None if predictor.class_weights is None else torch.from_numpy(predictor.class_weights),
        "criterion": crit,
        "best_epoch": epoch,
        "train_split": result.train_split,
        "val_split": result.val_split,
        "history": [
            {
                "epoch": epoch.epoch,
                "train_cross_entropy": epoch.train_ce,
                "val_cross_entropy": epoch.validation.ce,
                "val_weighted_cross_entropy": epoch.validation.weighted_ce,
                "accuracy": epoch.validation.accuracy,
                "top3_accuracy": epoch.validation.top3_acc,
                "top5_accuracy": epoch.validation.top5_acc,
                "macro_f1": epoch.validation.macro_f1,
                "confusion": torch.from_numpy(epoch.validation.confusion),
                "duration_s": epoch.dur_sec,
            }
            for epoch in result.history
        ],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not replace and path.with_suffix(path.suffix + ".report.json").exists():
        raise FileExistsError("checkpoint report already exists; choose a new checkpoint or use replacement")
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        tmp = Path(stream.name)
    try:
        torch.save(data, tmp)
        if replace:
            os.replace(tmp, path)
        else:
            os.link(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    publish_model_report(
        path,
        {
            "vocab_id": predictor.space.vocab_id,
            "width": predictor.space.width,
            "classes": predictor.space.classes,
            "cfg": asdict(result.cfg),
            "train": result.train_split,
            "validation": result.val_split,
            "epochs": len(result.history),
            **checkpoint_summary(result, crit),
            "history": epoch_history(result),
        },
        replace=replace,
    )


def load_checkpoint(path: Path, *, device: str = "cpu", expected_space: FeatureSpace | None = None) -> Predictor:
    """Load only tensors/primitives and optionally verify vocabulary and label identity."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    space = FeatureSpace(**data["space"])
    _validate_space(space)
    if expected_space is not None and space != expected_space:
        raise ValueError("checkpoint feature width, class order or vocabulary identity differs")
    cfg = ModelCfg(**data["cfg"]["model"])
    with torch.random.fork_rng(devices=[]):
        model = SparseMLP(space.width, len(space.classes), cfg)
    model.load_state_dict(data["state"])
    scale = None if data["scale"] is None else data["scale"].numpy().copy()
    if scale is not None and (scale.shape != (space.width,) or not np.isfinite(scale).all() or (scale <= 0).any()):
        raise ValueError("invalid checkpoint scale")
    model.to(_device(device)).eval()
    class_weights = None if data.get("class_weights") is None else data["class_weights"].numpy().copy()
    if class_weights is not None and (
        class_weights.shape != (len(space.classes),)
        or not np.isfinite(class_weights).all()
        or (class_weights < 0).any()
        or not (class_weights > 0).any()
    ):
        raise ValueError("invalid checkpoint class weights")
    log_cols = None if data.get("log_cols") is None else data["log_cols"].numpy().copy()
    if log_cols is not None and (log_cols.dtype != np.bool_ or log_cols.shape != (space.width,)):
        raise ValueError("invalid checkpoint log-count mask")
    if data["cfg"]["scaling"].startswith("log_") and (log_cols is None or scale is None):
        raise ValueError("log-scaled checkpoint is missing its fitted transformation")
    return Predictor(model, scale, space, class_weights, log_cols)
