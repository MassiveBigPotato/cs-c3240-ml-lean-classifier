"""Cached sparse features → seeded one-vs-rest SGD fits → selected checkpoints.

Epochs cover every training row once. Validation selects checkpoints, never
fits scaling; neither this command nor its reports evaluate the test corpus.
"""

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Literal, Protocol, cast

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MaxAbsScaler
from sklearn.utils.sparsefuncs import inplace_column_scale

from trustmebro.learning import (
    DEFAULT_MAX_BYTES,
    DEFAULT_TRAINING_CFG,
    Data,
    Evaluation,
    FeatureSpace,
    TrainingCfg,
    classification_metrics,
    load_learning_data,
    publish_model_report,
)
from trustmebro.preprocessing.archives import prepare_feature_source, read_vocab
from trustmebro.preprocessing.records import read_label_policy, read_split, split_id


class _FittedClasses(Protocol):
    """SGDClassifier exposes this attribute after partial_fit, not in its inferred type."""

    classes_: np.ndarray


@dataclass(frozen=True, slots=True)
class FitCfg:
    max_epochs: int = 5
    batch_rows: int = DEFAULT_TRAINING_CFG.rows
    patience: int = 5
    min_delta: float = 1e-4
    alpha: float = 1e-4
    seed: int = 0
    workers: int = -1
    scaling: Literal["maxabs", "none"] = "maxabs"
    max_bytes: int = DEFAULT_MAX_BYTES


DEFAULT_FIT_CFG = FitCfg()
VARIANTS: dict[str, Literal["l2"] | None] = {"no-regularization": None, "default-regularization": "l2"}


def _check_cfg(cfg: FitCfg) -> None:
    if min(cfg.max_epochs, cfg.batch_rows, cfg.patience, cfg.max_bytes) < 1:
        raise ValueError("epoch, batch, patience, and numeric-buffer limits must be positive")
    if not math.isfinite(cfg.alpha) or cfg.alpha <= 0:
        raise ValueError("alpha must be finite and positive for the optimal learning-rate schedule")
    if not math.isfinite(cfg.min_delta) or cfg.min_delta < 0:
        raise ValueError("min_delta must be finite and nonnegative")
    if cfg.seed < 0 or cfg.seed >= 2**32 or cfg.workers == 0:
        raise ValueError("seed must fit uint32 and workers must be nonzero")
    if cfg.scaling not in ("maxabs", "none"):
        raise ValueError("unknown scaling policy")


def _split_info(data: Data) -> dict[str, int | str]:
    names = sorted(set(data.theorems))
    ident = hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    return {"rows": len(data.label_ids), "theorems": len(names), "theorem_ids_sha256": ident}


def _scale_owned(train: Data, validation: Data, cfg: FitCfg) -> MaxAbsScaler | None:
    """Mutate only newly loaded, owned CSR buffers; retain a non-mutating predictor scaler."""
    if cfg.scaling == "none":
        return None
    scaler = MaxAbsScaler().fit(train.matrix)
    inverse = 1 / cast(np.ndarray, scaler.scale_)
    for data in (train, validation):
        inplace_column_scale(data.matrix, inverse)
    return scaler


def _probabilities(model: SGDClassifier, data: Data, start: int, end: int) -> np.ndarray:
    probs = model.predict_proba(data.matrix[start:end])
    if not np.isfinite(probs).all() or np.any(probs.sum(axis=1) <= 0):
        raise ValueError("nonfinite or undefined predicted probabilities; adjust scaling or the SGD settings")
    return probs


def _cross_entropy(model: SGDClassifier, data: Data, batch_rows: int) -> float:
    # Predictive multiclass CE, not SGD's sum of independently optimized binary losses.
    loss = 0.0
    for start in range(0, len(data.label_ids), batch_rows):
        end = min(start + batch_rows, len(data.label_ids))
        loss += float(
            log_loss(
                data.label_ids[start:end],
                _probabilities(model, data, start, end),
                labels=cast(_FittedClasses, model).classes_,
                normalize=False,
            )
        )
    return loss / len(data.label_ids)


def evaluate(model: SGDClassifier, data: Data, batch_rows: int) -> Evaluation:
    """Bound probability scratch to one batch; include every declared family in macro metrics."""
    count, classes = len(data.label_ids), cast(_FittedClasses, model).classes_
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    loss = 0.0
    top3 = top5 = 0
    for start in range(0, count, batch_rows):
        end = min(start + batch_rows, count)
        labels, probs = data.label_ids[start:end], _probabilities(model, data, start, end)
        loss += float(log_loss(labels, probs, labels=classes, normalize=False))
        # Stable ties prefer the lower frozen label ID, as in the MLP.
        ranked = np.argsort(-probs, axis=1, kind="stable")
        np.add.at(confusion, (labels, classes[ranked[:, 0]]), 1)
        top3 += int(np.any(classes[ranked[:, :3]] == labels[:, None], axis=1).sum())
        top5 += int(np.any(classes[ranked[:, :5]] == labels[:, None], axis=1).sum())
    measured = classification_metrics(confusion)
    precision, recall, f1, support = measured.precision, measured.recall, measured.f1, measured.support
    return Evaluation(
        count,
        loss / count,
        measured.accuracy,
        top3 / count,
        top5 / count,
        measured.macro_f1,
        confusion,
        precision,
        recall,
        f1,
        support,
    )


def _evaluation_report(evaluation: Evaluation) -> dict[str, object]:
    return {
        "val_cross_entropy": evaluation.ce,
        "val_accuracy": evaluation.accuracy,
        "val_top3_accuracy": evaluation.top3_acc,
        "val_top5_accuracy": evaluation.top5_acc,
        "val_macro_f1": evaluation.macro_f1,
    }


def fit(
    train: Data, validation: Data, penalty: Literal["l2"] | None, cfg: FitCfg, name: str
) -> tuple[SGDClassifier, dict[str, object]]:
    """Inputs are already scaled; retain an independent snapshot of the lowest validation CE."""
    model = SGDClassifier(
        loss="log_loss",
        penalty=cast(Any, penalty),  # sklearn accepts None; its inferred signature omits it.
        alpha=cfg.alpha,
        random_state=cfg.seed,
        n_jobs=cfg.workers,
        shuffle=False,  # One global permutation owns row order; no second per-batch shuffle.
    )
    classes = np.arange(len(train.classes))
    rng = np.random.default_rng(cfg.seed)
    best: SGDClassifier | None = None
    best_eval: Evaluation | None = None
    best_epoch = stale = 0
    stop_loss = math.inf
    history: list[dict[str, object]] = []
    started = time.perf_counter()
    for epoch in range(1, cfg.max_epochs + 1):
        epoch_started = time.perf_counter()
        order = rng.permutation(len(train.label_ids))
        for start in range(0, len(order), cfg.batch_rows):
            ids = order[start : start + cfg.batch_rows]
            model.partial_fit(
                train.matrix[ids], train.label_ids[ids], classes=classes if epoch == 1 and start == 0 else None
            )
        train_ce = _cross_entropy(model, train, cfg.batch_rows)
        measured = evaluate(model, validation, cfg.batch_rows)
        duration = time.perf_counter() - epoch_started
        history.append(
            {"epoch": epoch, "train_cross_entropy": train_ce, **_evaluation_report(measured), "duration_s": duration}
        )
        print(
            f"{name} — Epoch {epoch}: train CE={train_ce:.5f}; val CE={measured.ce:.5f}; "
            f"accuracy={measured.accuracy:.4f}; top-3={measured.top3_acc:.4f}; "
            f"top-5={measured.top5_acc:.4f}; macro-F1={measured.macro_f1:.4f}; {duration:.2f}s",
            file=sys.stderr,
            flush=True,
        )
        if best_eval is None or measured.ce < best_eval.ce:
            best, best_eval, best_epoch = copy.deepcopy(model), measured, epoch
        if measured.ce < stop_loss - cfg.min_delta:
            stop_loss, stale = measured.ce, 0
        else:
            stale += 1
        if stale >= cfg.patience:
            break
    assert best is not None and best_eval is not None  # Positive max_epochs established by the caller.
    return best, {
        "epochs": len(history),
        "best_epoch": best_epoch,
        "criterion": "unweighted",
        **_evaluation_report(best_eval),
        **{
            field: getattr(best_eval, field).tolist() for field in ("precision", "recall", "f1", "support", "confusion")
        },
        "history": history,
        "duration_s": time.perf_counter() - started,
    }


def _save(path: Path, checkpoint: dict[str, object], *, replace: bool) -> None:
    """Publish one complete checkpoint atomically; never truncate an existing destination."""
    if not replace and path.with_suffix(path.suffix + ".report.json").exists():
        raise FileExistsError("checkpoint report already exists; choose a new output or use --replace")
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        pending = Path(stream.name)
    try:
        joblib.dump(checkpoint, pending, protocol=5)
        if replace:
            os.replace(pending, path)
        else:
            os.link(pending, path)  # Exclusive publication also handles races after preflight.
    finally:
        pending.unlink(missing_ok=True)
    publish_model_report(path, {key: val for key, val in checkpoint.items() if key != "predictor"}, replace=replace)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and validate two one-vs-rest SGD logistic models")
    parser.add_argument(
        "--instance", type=Path, required=True, help="directory containing features.zst, split.json, vocab.zst"
    )
    parser.add_argument(
        "--labels", type=Path, help="optional policy assertion; otherwise use the frozen archive policy"
    )
    parser.add_argument("--output", type=Path, help="checkpoint directory; defaults to the instance directory")
    parser.add_argument("--replace", action="store_true", help="allow replacing existing checkpoints")
    parser.add_argument("--iters", "--max-epochs", dest="max_epochs", type=int, default=DEFAULT_FIT_CFG.max_epochs)
    for flag in ("batch_rows", "patience", "seed", "workers", "max_bytes"):
        parser.add_argument(f"--{flag.replace('_', '-')}", type=int, default=getattr(DEFAULT_FIT_CFG, flag))
    for flag in ("alpha", "min_delta"):
        parser.add_argument(f"--{flag.replace('_', '-')}", type=float, default=getattr(DEFAULT_FIT_CFG, flag))
    parser.add_argument("--scaling", choices=("maxabs", "none"), default=DEFAULT_FIT_CFG.scaling)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    cfg = FitCfg(**{name: getattr(args, name) for name in FitCfg.__dataclass_fields__})
    _check_cfg(cfg)
    started = time.perf_counter()
    root, output = args.instance, args.output or args.instance
    sources = (root / "features.zst", root / "split.json", root / "vocab.zst")
    if args.labels is not None:
        sources += (args.labels,)
    stamps = [(path.stat().st_size, path.stat().st_mtime_ns) for path in sources]
    paths = {name: output / f"logit-{name}.pkl" for name in VARIANTS}
    for path in paths.values():
        report_path = path.with_suffix(path.suffix + ".report.json")
        if path.exists() and any(path.samefile(src) for src in sources):
            raise ValueError("checkpoint would overwrite a source artifact")
        if path.exists() and not args.replace:
            raise FileExistsError(f"checkpoint exists: {path}; use --replace or another --output directory")
        if report_path.exists() and not args.replace:
            raise FileExistsError(
                f"checkpoint report exists: {report_path}; use --replace or another --output directory"
            )
    split = read_split(sources[1])
    vocab = read_vocab(sources[2])
    prepared = prepare_feature_source(sources[0])
    header = prepared.header
    if header.vocab != vocab:
        raise ValueError("vocab.zst differs from the vocabulary frozen in features.zst")
    del header, vocab  # The loaded matrices do not need a retained copy of either vocabulary.
    with sources[2].open("rb") as stream:
        vocab_id = hashlib.file_digest(stream, "sha256").hexdigest()
    policy = read_label_policy(args.labels) if args.labels is not None else None
    print("Loading cached training and validation vectors...", file=sys.stderr, flush=True)
    loading_cfg = TrainingCfg(dtype="float32")
    loaded = load_learning_data(sources[0], split, policy, cfg=loading_cfg, max_bytes=cfg.max_bytes, prepared=prepared)
    space = FeatureSpace(loaded.source.layout.width, loaded.train.classes, vocab_id)
    train, validation = (
        Data(batch.matrix, batch.label_ids, batch.theorems, space) for batch in (loaded.train, loaded.validation)
    )
    if min(len(train.label_ids), len(validation.label_ids)) < 1 or len(train.classes) < 2:
        raise ValueError("nonempty training and validation subsets and at least two declared classes are required")
    if (
        train.classes != validation.classes
        or cast(tuple[int, int], train.matrix.shape)[1] != cast(tuple[int, int], validation.matrix.shape)[1]
    ):
        raise ValueError("training and validation feature spaces differ")
    if set(train.theorems) & set(validation.theorems):
        raise ValueError("training and validation share theorems")
    scaler = _scale_owned(train, validation, cfg)
    shared = {
        "vocab_id": vocab_id,
        "width": cast(tuple[int, int], train.matrix.shape)[1],
        "classes": train.classes,
        "split_id": split_id(split),
        "train": _split_info(train),
        "validation": _split_info(validation),
        "cfg": asdict(cfg),
        "sklearn_version": sklearn.__version__,
        "method": "one-vs-rest SGD logistic regression",
    }
    if args.verbose:
        print(json.dumps(shared), file=sys.stderr, flush=True)
    fits = {name: fit(train, validation, penalty, cfg, name) for name, penalty in VARIANTS.items()}
    if any((path.stat().st_size, path.stat().st_mtime_ns) != stamp for path, stamp in zip(sources, stamps)):
        raise ValueError("a source artifact changed during training; no checkpoints published")
    output.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, object]] = {}
    for name, (model, report) in fits.items():
        predictor = Pipeline([("scaling", scaler if scaler is not None else "passthrough"), ("model", model)])
        _save(paths[name], {**shared, **report, "predictor": predictor}, replace=args.replace)
        reports[name] = {
            "checkpoint": str(paths[name]),
            **{key: val for key, val in report.items() if key != "history"},
        }
    print(json.dumps({**shared, "models": reports, "duration_s": time.perf_counter() - started}))
    return 0
