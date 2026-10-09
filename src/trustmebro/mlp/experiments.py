"""Ordered MLP experiments over one immutable feature instance; no grid or test-set selection."""

import argparse
import csv
import gc
import json
import os
import platform
import sqlite3
import sys
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import IO, Any, Literal, cast

import msgspec
import numpy as np
import scipy
import sklearn
import torch

from trustmebro.artifacts import canonical_request
from trustmebro.learning import DEFAULT_MAX_BYTES, TrainingBatch, TrainingCfg, load_learning_data, load_training_matrix
from trustmebro.preprocessing.archives import prepare_feature_source, read_vocab
from trustmebro.preprocessing.layout import count_columns
from trustmebro.preprocessing.records import read_split

from .mlp import (
    DEFAULT_FIT_CFG,
    Data,
    Epoch,
    FeatureSpace,
    FitCfg,
    FitResult,
    ModelCfg,
    PreparedData,
    Scaling,
    checkpoint_summary,
    epoch_history,
    evaluate,
    fit_prepared,
    load_checkpoint,
    prepare_training,
    save_checkpoint,
    validate_cfg,
)


class Settings(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    hidden: tuple[int, ...] | msgspec.UnsetType = msgspec.UNSET
    dropout: float | msgspec.UnsetType = msgspec.UNSET
    batch_rows: int | msgspec.UnsetType = msgspec.UNSET
    epochs: int | msgspec.UnsetType = msgspec.UNSET
    patience: int | msgspec.UnsetType = msgspec.UNSET
    min_delta: float | msgspec.UnsetType = msgspec.UNSET
    lr: float | msgspec.UnsetType = msgspec.UNSET
    weight_decay: float | msgspec.UnsetType = msgspec.UNSET
    seed: int | msgspec.UnsetType = msgspec.UNSET
    device: str | msgspec.UnsetType = msgspec.UNSET
    scaling: Scaling | msgspec.UnsetType = msgspec.UNSET
    class_weight: Literal["none", "balanced", "inverse_sqrt"] | msgspec.UnsetType = msgspec.UNSET
    label_smoothing: float | msgspec.UnsetType = msgspec.UNSET


class Run(Settings):
    name: str


class Plan(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    operation: Literal["train"] = "train"
    instance: str
    output: str
    runs: tuple[Run, ...]
    defaults: Settings = msgspec.field(default_factory=Settings)
    max_bytes: int = DEFAULT_MAX_BYTES


@dataclass(frozen=True, slots=True)
class Experiment:
    name: str
    cfg: FitCfg
    dir: Path


SUMMARY_FIELDS = (
    "name",
    "status",
    "hidden",
    "dropout",
    "lr",
    "weight_decay",
    "batch_rows",
    "scaling",
    "class_weight",
    "label_smoothing",
    "seed",
    "epochs",
    "best_epoch",
    "val_cross_entropy",
    "val_accuracy",
    "val_top3_accuracy",
    "val_top5_accuracy",
    "val_weighted_cross_entropy",
    "val_macro_f1",
    "weighted_best_epoch",
    "weighted_val_cross_entropy",
    "weighted_val_weighted_cross_entropy",
    "weighted_val_accuracy",
    "weighted_val_top3_accuracy",
    "weighted_val_top5_accuracy",
    "weighted_val_macro_f1",
    "fit_duration_s",
    "duration_s",
    "error",
)


def _cfg(defaults: Settings, run: Run) -> FitCfg:
    resolved = asdict(DEFAULT_FIT_CFG)
    resolved.update(resolved.pop("model"))
    resolved["epochs"] = resolved.pop("max_epochs")
    for settings in (defaults, run):
        resolved.update({key: val for key, val in msgspec.structs.asdict(settings).items() if val is not msgspec.UNSET})
    resolved.pop("name")
    model = ModelCfg(tuple(resolved.pop("hidden")), resolved.pop("dropout"))
    resolved["max_epochs"] = resolved.pop("epochs")
    cfg = FitCfg(model=model, **resolved)
    # Reject the whole plan before starting any fit, not halfway through a queue.
    validate_cfg(cfg)
    return cfg


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _write(path: Path, emit: Callable[[IO[str]], None]) -> None:
    """Publish each report atomically; completed runs survive later failures/interruption."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            emit(stream)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _json(path: Path, data: dict) -> None:
    _write(path, lambda stream: json.dump(data, stream, indent=2, allow_nan=False))


def _summary(output: Path, runs: tuple[Experiment, ...]) -> None:
    def emit(stream: IO[str]) -> None:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for run in runs:
            path = run.dir / "results.json"
            if not path.exists():
                continue
            data = json.loads(path.read_text())
            cfg = asdict(run.cfg)
            cfg.update(cfg.pop("model"))
            cfg["hidden"] = "x".join(map(str, cfg["hidden"]))
            row = cfg | data | {"name": run.name}
            row.update({f"weighted_{key}": val for key, val in (data.get("weighted_checkpoint") or {}).items()})
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})

    _write(output / "summary.csv", emit)


def _results(result: FitResult, duration_s: float, checkpoint: Path) -> dict:
    weighted = None
    if result.weighted_best is not None:
        path = weighted_checkpoint(checkpoint)
        weighted = checkpoint_summary(result, "weighted") | {"checkpoint_sha256": _hash(path)}
    return {
        "status": "complete",
        "checkpoint_sha256": _hash(checkpoint),
        "vocab_id": result.predictor.space.vocab_id,
        "width": result.predictor.space.width,
        "classes": result.predictor.space.classes,
        "train": result.train_split,
        "validation": result.val_split,
        "epochs": len(result.history),
        **checkpoint_summary(result),
        "class_weights": None if result.predictor.class_weights is None else result.predictor.class_weights.tolist(),
        "weighted_checkpoint": weighted,
        "history": epoch_history(result),
        "duration_s": duration_s,
    }


def run_plan(path: Path) -> None:
    """Single-writer output; paths in plans are relative to the invoking working directory."""
    plan = msgspec.json.decode(path.read_bytes(), type=Plan)
    if not plan.runs or plan.max_bytes < 1:
        raise ValueError("at least one run and a positive max_bytes limit are required")
    names = [run.name for run in plan.runs]
    if len(set(names)) != len(names) or any(
        not name
        or name in (".", "..")
        or any(not (char.isascii() and (char.isalnum() or char in "_-")) for char in name)
        for name in names
    ):
        raise ValueError("run names must be unique and contain only ASCII letters, digits, underscores or hyphens")
    instance, output = Path(plan.instance).resolve(), Path(plan.output).resolve()
    inputs = {name: instance / name for name in ("features.zst", "split.json", "vocab.zst")}
    runs = tuple(Experiment(run.name, _cfg(plan.defaults, run), output / run.name) for run in plan.runs)
    # Prevent report publication over inputs, even when the user chooses an overlapping output tree.
    destinations = {output / "summary.csv"}
    for run in runs:
        destinations.update(run.dir / name for name in ("config.json", "results.json", "checkpoint.pt"))
        if run.cfg.class_weight != "none":
            destinations.add(weighted_checkpoint(run.dir / "checkpoint.pt"))
    protected = (*inputs.values(), path.resolve())
    if any(
        dst.resolve() == src.resolve() or dst.resolve() in src.resolve().parents
        for dst in destinations
        for src in protected
    ):
        raise ValueError("experiment outputs must not overwrite or enclose input artifacts")
    print("Fingerprinting immutable inputs...", file=sys.stderr, flush=True)
    stamps = {name: (src.stat().st_size, src.stat().st_mtime_ns) for name, src in inputs.items()}
    identity: dict[str, Any] = {
        "inputs": {name: {"path": str(src), "sha256": _hash(src)} for name, src in inputs.items()},
        "code": {name: _hash(Path(__file__).with_name(name)) for name in ("experiments.py", "mlp.py")},
        "versions": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    pending: list[Experiment] = []
    configs: dict[str, dict] = {}
    for run in runs:
        config = cast(
            dict[str, Any],
            canonical_request(
                {"name": run.name, "cfg": asdict(run.cfg), "max_bytes": plan.max_bytes, "identity": identity}
            ),
        )
        configs[run.name] = config
        config_path, result_path = run.dir / "config.json", run.dir / "results.json"
        saved = json.loads(result_path.read_text()) if result_path.exists() else {}
        complete = saved.get("status") == "complete"
        code_changed = False
        if config_path.exists():
            recorded = json.loads(config_path.read_text())
            # A completed result is an immutable historical fit, not a restart of current code.
            expected = config
            if complete:
                recorded_code = recorded["identity"]["code"]
                expected = config | {"identity": identity | {"code": recorded_code}}
                code_changed = recorded_code != identity["code"]
            if recorded != expected:
                raise ValueError(f"{run.name}: settings or input/code identity changed; choose a new output directory")
        elif run.dir.exists() and any(run.dir.iterdir()):
            raise ValueError(f"{run.name}: nonempty run directory has no matching config.json")
        if complete:
            checkpoint = run.dir / "checkpoint.pt"
            hashes = [(checkpoint, saved["checkpoint_sha256"])]
            if run.cfg.class_weight != "none":
                hashes.append((weighted_checkpoint(checkpoint), saved["weighted_checkpoint"]["checkpoint_sha256"]))
            if any(not path.is_file() or _hash(path) != ident for path, ident in hashes):
                raise ValueError(f"{run.name}: completed checkpoint is missing or changed")
            if code_changed:
                print(
                    f"Warning: {run.name} was trained with different code; retaining its original results/provenance.",
                    file=sys.stderr,
                    flush=True,
                )
            print(f"Skip completed run: {run.name}", file=sys.stderr, flush=True)
        else:
            pending.append(run)
    _summary(output, runs)
    if not pending:
        return
    print("Loading cached training and validation vectors once...", file=sys.stderr, flush=True)
    data = load_training_data(
        inputs["features.zst"],
        inputs["split.json"],
        vocab_id=identity["inputs"]["vocab.zst"]["sha256"],
        max_bytes=plan.max_bytes,
        log_counts=any(run.cfg.scaling.startswith("log_") for run in pending),
    )
    if stamps != {name: (src.stat().st_size, src.stat().st_mtime_ns) for name, src in inputs.items()}:
        raise ValueError("input artifacts changed during fingerprinting/loading")
    prepared: dict[str, PreparedData] = {}
    for idx, run in enumerate(pending, 1):
        print(f"Run {idx}/{len(pending)}: {run.name}", file=sys.stderr, flush=True)
        _json(run.dir / "config.json", configs[run.name])
        _json(run.dir / "results.json", {"status": "running"})
        started = perf_counter()
        try:
            if run.cfg.scaling not in prepared:
                prepared[run.cfg.scaling] = prepare_training(data.train, data.validation, scaling=run.cfg.scaling)
            fit_started = perf_counter()
            result = fit_prepared(prepared[run.cfg.scaling], run.cfg, on_epoch=_report)
            fit_duration = perf_counter() - fit_started
            checkpoint = run.dir / "checkpoint.pt"
            save_checkpoint(checkpoint, result, replace=True)
            if result.weighted_best is not None:
                save_checkpoint(weighted_checkpoint(checkpoint), result, replace=True, crit="weighted")
            report = _results(result, perf_counter() - started, checkpoint)
            report.update(fit_duration_s=fit_duration, duration_s=perf_counter() - started)
            del result  # Do not retain a previous full-sized network while allocating the next one.
            _json(run.dir / "results.json", report)
        except (Exception, KeyboardInterrupt) as error:
            _json(
                run.dir / "results.json",
                {
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "duration_s": perf_counter() - started,
                },
            )
            _summary(output, runs)
            raise
        _summary(output, runs)
        gc.collect()  # Once per fit, never in the minibatch loop.
        print(
            f"Finished {run.name}: val CE={report['val_cross_entropy']:.5f}; "
            f"accuracy={report['val_accuracy']:.4f}; macro-F1={report['val_macro_f1']:.4f}; "
            f"top-3={report['val_top3_accuracy']:.4f}; top-5={report['val_top5_accuracy']:.4f}; "
            f"{report['duration_s']:.2f}s",
            file=sys.stderr,
            flush=True,
        )


class EvaluationPlan(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    operation: Literal["evaluate"]
    features: str
    vocab: str
    checkpoint: str
    report: str
    split: str
    test_db: str
    output: str
    device: str = DEFAULT_FIT_CFG.device
    batch_rows: int = DEFAULT_FIT_CFG.batch_rows


def run_evaluation(plan: EvaluationPlan) -> None:

    output = Path(plan.output)
    if output.exists():
        raise FileExistsError("evaluation output exists; choose a new path")
    sources = tuple(
        Path(getattr(plan, name)).resolve()
        for name in ("features", "vocab", "checkpoint", "report", "split", "test_db")
    )
    if any(output.resolve() == src or output.resolve() in src.parents for src in sources):
        raise ValueError("evaluation output must not overwrite or enclose input artifacts")
    result = evaluate_archive(*sources, device=plan.device, batch_rows=plan.batch_rows)
    _json(output, result)
    print(json.dumps({key: val for key, val in result.items() if key.startswith("test")}))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run an explicit ordered list of MLP configurations")
    parser.add_argument(
        "--config", type=Path, required=True, help="JSON plan; paths are relative to the working directory"
    )
    args = parser.parse_args(argv)
    data = args.config.read_bytes()
    operation = msgspec.json.decode(data).get("operation", "train")
    if operation == "train":
        run_plan(args.config)
    elif operation == "evaluate":
        run_evaluation(msgspec.json.decode(data, type=EvaluationPlan))
    else:
        parser.error("operation must be train or evaluate")
    return 0


@dataclass(frozen=True, slots=True)
class TrainingData:
    train: Data
    validation: Data


def load_training_data(
    features: Path, split: Path, *, vocab_id: str, max_bytes: int = DEFAULT_MAX_BYTES, log_counts: bool = False
) -> TrainingData:
    """Select both logical partitions from one completed, immutable archive."""
    manifest = read_split(split)
    stamp = features.stat().st_size, features.stat().st_mtime_ns
    loaded = load_learning_data(features, manifest, cfg=TrainingCfg(dtype="float32"), max_bytes=max_bytes)
    count_cols = count_columns(loaded.source.layout) if log_counts else None

    def data(batch: TrainingBatch) -> Data:
        space = FeatureSpace(cast(tuple[int, int], batch.matrix.shape)[1], batch.classes, vocab_id)
        return Data(batch.matrix, batch.label_ids, batch.theorems, space)

    train, val = data(loaded.train), data(loaded.validation)
    if stamp != (features.stat().st_size, features.stat().st_mtime_ns):
        raise ValueError("feature archive changed while loading training and validation")
    if count_cols is not None:
        if count_cols.shape != (train.space.width,):
            raise ValueError("feature layout differs from the loaded matrix width")
        train = Data(train.matrix, train.label_ids, train.theorems, train.space, count_cols)
    return TrainingData(train, val)


def _report(epoch: Epoch) -> None:
    val = epoch.validation
    weighted = "" if val.weighted_ce is None else f"weighted val CE={val.weighted_ce:.5f}; "
    train_name = "train CE" if val.weighted_ce is None else "weighted train CE"
    print(
        f"Epoch {epoch.epoch}: {train_name}={epoch.train_ce:.5f}; val CE={val.ce:.5f}; "
        f"{weighted}accuracy={val.accuracy:.4f}; top-3={val.top3_acc:.4f}; top-5={val.top5_acc:.4f}; "
        f"macro-F1={val.macro_f1:.4f}; {epoch.dur_sec:.2f}s",
        file=sys.stderr,
        flush=True,
    )


def weighted_checkpoint(path: Path) -> Path:
    return path.with_name(f"{path.stem}.weighted{path.suffix}")


def _names_digest(names: set[str]) -> str:
    return sha256(json.dumps(sorted(names), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def evaluate_archive(
    features: Path,
    vocab: Path,
    checkpoint: Path,
    report: Path,
    split: Path,
    test_db: Path,
    *,
    device: str,
    batch_rows: int,
) -> dict:
    """Verify provenance before inference; never fit a vocabulary, scaler, or model."""
    if batch_rows < 1:
        raise ValueError("batch_rows must be positive")
    started = perf_counter()
    saved = json.loads(report.read_text())
    checkpoint_hash = _hash(checkpoint)
    vocab_hash = _hash(vocab)
    if saved["checkpoint_sha256"] != checkpoint_hash or saved["vocab_id"] != vocab_hash:
        raise ValueError("checkpoint or vocabulary differs from the selected run's report")
    if saved["criterion"] != "unweighted":
        raise ValueError("expected the checkpoint selected by unweighted validation cross-entropy")
    manifest = read_split(split)
    for role in ("train", "validation"):
        names = set(getattr(manifest, role))
        if saved[role]["theorem_ids_sha256"] != _names_digest(names):
            raise ValueError("development manifest differs from the checkpoint's theorem population")
    prepared = prepare_feature_source(features)
    header = prepared.header
    if header.vocab != read_vocab(vocab) or Path(header.src).resolve() != test_db.resolve():
        raise ValueError("test features do not use the frozen vocabulary and requested source database")
    with sqlite3.connect(f"{test_db.resolve().as_uri()}?mode=ro&immutable=1", uri=True) as db:
        expected = dict(db.execute("SELECT name, trn_count FROM theorems"))
    if set(expected).intersection((*manifest.train, *manifest.validation)):
        raise ValueError("test theorems overlap the training or validation partition")
    # Conversion without --labels preserves tactic kinds; apply the same frozen
    # development policy, never infer a label mapping from the test population.
    batch = load_training_matrix(
        features, manifest.policy, cfg=TrainingCfg(dtype="float32"), max_bytes=2 * 2**30, prepared=prepared
    )
    names = set(batch.theorems)
    if names != set(expected) or len(batch.label_ids) != sum(expected.values()):
        raise ValueError("test archive omits theorems or transitions from the complete test partition")
    space = FeatureSpace(cast(tuple[int, int], batch.matrix.shape)[1], batch.classes, vocab_hash)
    predictor = load_checkpoint(checkpoint, device=device, expected_space=space)
    measured = evaluate(predictor, Data(batch.matrix, batch.label_ids, batch.theorems, space), batch_rows=batch_rows)
    return {
        "criterion": "unweighted_validation_cross_entropy",
        "best_epoch": saved["best_epoch"],
        "checkpoint_sha256": checkpoint_hash,
        "vocab_id": vocab_hash,
        "width": space.width,
        "classes": space.classes,
        "test": {"rows": measured.rows, "theorems": len(names), "theorem_ids_sha256": _names_digest(names)},
        "test_cross_entropy": measured.ce,
        "test_error": 1 - measured.accuracy,
        "test_accuracy": measured.accuracy,
        "test_top3_accuracy": measured.top3_acc,
        "test_top5_accuracy": measured.top5_acc,
        "test_macro_f1": measured.macro_f1,
        "precision": measured.precision.tolist(),
        "recall": measured.recall.tolist(),
        "f1": measured.f1.tolist(),
        "support": measured.support.tolist(),
        "confusion": measured.confusion.tolist(),
        "inputs": {str(path): _hash(path) for path in (features, split, test_db, report)},
        "device": device,
        "batch_rows": batch_rows,
        "duration_s": perf_counter() - started,
    }
