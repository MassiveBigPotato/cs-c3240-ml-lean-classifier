"""Small saved training reports → comparable validation figures; no inference or retraining.

This is deliberately independent of corpus scanning and density rendering.
Only trusted logistic checkpoints may be read: joblib deserialization can execute code.
MLP reports are plain JSON; their model weights are not loaded.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from hashlib import file_digest
from itertools import pairwise
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Literal

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, PercentFormatter

# Input records and plot configuration.

type Model = Literal["logistic", "mlp"]
type SubsetIdent = tuple[int, int, str]
type RunIdent = tuple[str, int, tuple[str, ...], SubsetIdent, SubsetIdent]
METRICS = {
    "val_cross_entropy": "Cross-entropy",
    "val_accuracy": "Accuracy",
    "val_top3_accuracy": "Top-3 accuracy",
    "val_top5_accuracy": "Top-5 accuracy",
    "val_macro_f1": "Macro-F1",
}
PLOTS = ("overall", "learning", "families", "confusion")
DEFAULT_COMPARE = ("logistic:default-regularization", "mlp:unweighted-dropout", "mlp:inverse-sqrt-dropout")
CHOSEN_MODEL = "mlp:inverse-sqrt-dropout"
COMPARE_COLOURS = ("#0072B2", "#E69F00", "#7A5195")
WEIGHT_NOTE = "Weighted: class weights proportional to 1/√(training class frequency)."
MLP_STYLES = {
    "baseline": ("Unweighted baseline", "#666666"),
    "deeper-unweighted": ("Two hidden layers", "#009E73"),
    "unweighted-dropout": ("Dropout", COMPARE_COLOURS[1]),
    "inverse-sqrt": ("Weighted", "#CC79A7"),
    "inverse-sqrt-dropout": ("Weighted + dropout", COMPARE_COLOURS[2]),
    "inverse-sqrt-smoothing": ("Weighted + smoothing", "#AA3377"),
    "unweighted-dropout-smoothing": ("Dropout + smoothing", "#D55E00"),
    "unweighted-smoothing": ("Smoothing", "#56B4E9"),
}


@dataclass(frozen=True, slots=True)
class Run:
    name: str
    model: Model
    src: Path
    identity: RunIdent  # vocabulary, width, classes, and both subset membership/count records
    classes: tuple[str, ...]
    best_epoch: int
    metrics: Mapping[str, float]
    history: tuple[Mapping[str, Any], ...]
    support: np.ndarray
    f1: np.ndarray
    confusion: np.ndarray

    @property
    def ident(self) -> str:
        return f"{self.model}:{self.name}"

    @property
    def label(self) -> str:
        if self.model == "logistic":
            return {"default-regularization": "L2 logistic", "no-regularization": "Unregularized logistic"}.get(
                self.name, f"Logistic: {self.name.replace('-', ' ')}"
            )
        return f"MLP: {MLP_STYLES.get(self.name, (self.name.replace('-', ' '),))[0]}"


@dataclass(frozen=True, slots=True)
class PlotCfg:
    formats: tuple[str, ...] = ("png", "pdf")
    dpi: int = 140
    theme: Literal["light", "dark"] = "light"
    chosen: str = CHOSEN_MODEL


DEFAULT_PLOT_CFG = PlotCfg()


# Source loading and experimental contracts.


def _counts(data: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    vals = np.asarray(data)
    if vals.shape != shape or vals.dtype.kind not in "iu" or np.any(vals < 0) or np.any(vals > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must contain nonnegative integer counts with shape {shape}")
    return vals.astype(np.int64, copy=False)


def _run(data: Mapping[str, Any], name: str, model: Model, src: Path) -> Run:
    if data.get("status", "complete") != "complete":
        raise ValueError("run is not complete")
    if data["criterion"] != "unweighted":
        raise ValueError("these figures compare checkpoints selected by unweighted validation cross-entropy")
    classes = tuple(data["classes"])
    if not classes or len(set(classes)) != len(classes) or not all(isinstance(label, str) for label in classes):
        raise ValueError("class labels must be unique strings")
    count = len(classes)
    support = _counts(data["support"], (count,), "support")
    confusion = _counts(data["confusion"], (count, count), "confusion")
    if not np.array_equal(support, confusion.sum(axis=1)) or support.sum() != data["validation"]["rows"]:
        raise ValueError("confusion rows and validation support disagree")
    if support.sum() == 0:
        raise ValueError("validation population is empty")
    f1 = np.asarray(data["f1"], dtype=np.float64)
    denom = support + confusion.sum(axis=0)
    expected_f1 = np.divide(2 * np.diag(confusion), denom, out=np.zeros(count), where=denom > 0)
    metrics = {key: float(data[key]) for key in METRICS}
    if not all(np.isfinite(val) and val >= 0 for val in metrics.values()):
        raise ValueError("evaluation metrics must be finite and nonnegative")
    if not 0 <= metrics["val_accuracy"] <= metrics["val_top3_accuracy"] <= metrics["val_top5_accuracy"] <= 1:
        raise ValueError("accuracy and top-k accuracies are inconsistent")
    if f1.shape != (count,) or not np.allclose(f1, expected_f1, atol=1e-10, rtol=1e-10):
        raise ValueError("class F1 disagrees with confusion counts")
    if not np.isclose(metrics["val_macro_f1"], f1.mean()) or not np.isclose(
        metrics["val_accuracy"], np.trace(confusion) / support.sum()
    ):
        raise ValueError("aggregate accuracy or macro-F1 disagrees with confusion counts")
    history = tuple(data.get("history", ()))
    if history and [row["epoch"] for row in history] != list(range(1, len(history) + 1)):
        raise ValueError("consecutive epoch history is required for learning curves")
    best_epoch = data["best_epoch"]
    if (
        not isinstance(best_epoch, int)
        or not 1 <= best_epoch <= data["epochs"]
        or (history and data["epochs"] != len(history))
    ):
        raise ValueError("selected epoch or history length is invalid")
    for row in history:
        if not all(np.isfinite(row[key]) and row[key] >= 0 for key in METRICS):
            raise ValueError("epoch metrics must be finite and nonnegative")
    if history and not all(np.isclose(metrics[key], history[best_epoch - 1][key]) for key in METRICS):
        raise ValueError("selected metrics do not describe the selected epoch")
    if history and not np.isclose(metrics["val_cross_entropy"], min(row["val_cross_entropy"] for row in history)):
        raise ValueError("selected checkpoint does not minimize validation cross-entropy")
    train, validation = (
        (data[subset]["rows"], data[subset]["theorems"], data[subset]["theorem_ids_sha256"])
        for subset in ("train", "validation")
    )
    identity = (data["vocab_id"], data["width"], classes, train, validation)
    return Run(name, model, src, identity, classes, best_epoch, metrics, history, support, f1, confusion)


def load_runs(mlp: Path, logistic: Path) -> tuple[Run, ...]:
    paths: list[tuple[Path, Model]] = [(path, "logistic") for path in sorted(logistic.glob("logit-*.pkl"))]
    paths.extend((path, "mlp") for path in sorted(mlp.glob("*/results.json")))
    if not paths or not any(model == "mlp" for _, model in paths) or not any(model == "logistic" for _, model in paths):
        raise ValueError("provide a logistic checkpoint directory and an MLP experiment-results directory")
    runs: list[Run] = []
    for path, model in paths:
        if model == "logistic":
            report = path.with_suffix(path.suffix + ".report.json")
            if report.exists():
                data = json.loads(report.read_text())
                with path.open("rb") as stream:
                    if data.get("checkpoint_sha256") != file_digest(stream, "sha256").hexdigest():
                        raise ValueError(f"{path}: report does not describe this checkpoint")
            else:
                data = joblib.load(path)
            if isinstance(data, dict):
                data.pop("predictor", None)
            name = path.stem.removeprefix("logit-")
        else:
            data = json.loads(path.read_text())
            name = path.parent.name
        if not isinstance(data, dict):
            raise TypeError(f"{path}: expected checkpoint/report metadata, not a bare estimator")
        try:
            runs.append(_run(data, name, model, path))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"{path}: missing or invalid plotting information: {error}") from error
    if len({run.ident for run in runs}) != len(runs):
        raise ValueError("duplicate experiment identifiers")
    for run in runs[1:]:
        if run.identity != runs[0].identity or not np.array_equal(run.support, runs[0].support):
            raise ValueError(f"{run.ident}: vocabulary, labels, or train/validation population differs")
    return tuple(runs)


# Figure construction. All metrics are validation metrics, not test results.


def _run_colour(run: Run) -> str:
    if run.model == "logistic":
        return COMPARE_COLOURS[0] if run.name == "default-regularization" else "#56B4E9"
    return MLP_STYLES.get(run.name, (run.label, "#666666"))[1]


def _display_runs(runs: Sequence[Run]) -> tuple[Run, ...]:
    """Omit smoothing only from figures."""
    mlp_order = {name: idx for idx, name in enumerate(MLP_STYLES)}
    return tuple(
        sorted(
            (run for run in runs if "smoothing" not in run.name.split("-")),
            key=lambda run: (run.model, mlp_order.get(run.name, len(mlp_order)), run.name),
        )
    )


def _metric_limits(metric: str, vals: Sequence[float]) -> tuple[float, float]:
    # Deliberate display domains, expanded to rounded bounds rather than clipping unexpected inputs.
    if metric == "val_cross_entropy":
        return 0, max(3.0, float(np.ceil(max(vals))))
    if metric == "val_macro_f1":
        return 0, max(0.4, float(np.ceil(max(vals) * 10) / 10))
    return 0, 1


def _comparison_bars(runs: Sequence[Run], keys: Sequence[str], title: str, chosen: str) -> Figure:
    fig, axes = plt.subplots(1, 3, figsize=(10.5, max(6, len(runs) * 0.5 + 1.5)), sharey=True)
    rows = np.arange(len(runs))
    colours = [_run_colour(run) for run in runs]
    for ax, key in zip(axes, keys):
        vals = [run.metrics[key] for run in runs]
        ax.barh(rows, vals, height=1, color=colours)
        ax.set_xlim(*_metric_limits(key, vals))
        ax.set_xlabel(f"{METRICS[key]}{' (nats)' if key == 'val_cross_entropy' else ''}")
        is_accuracy = "accuracy" in key
        if is_accuracy:
            ax.xaxis.set_major_formatter(PercentFormatter(1))
        for row, val in zip(rows, vals):
            label = f"{val:.1%}" if is_accuracy else f"{val:.3f}"
            ax.annotate(label, (val, row), xytext=(4, 0), textcoords="offset points", va="center", fontsize=12)
        ax.set_xlim(right=ax.get_xlim()[1] * 1.15)
    labels = [run.label + (" ★" if run.ident == chosen else "") for run in runs]
    axes[0].set_yticks(rows, labels)
    axes[0].invert_yaxis()
    for ax in axes:
        for idx in range(1, len(runs)):
            if runs[idx].model != runs[idx - 1].model:
                ax.axhline(idx - 0.5, color="#888888", linewidth=1)
        ax.set_axisbelow(True)
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title)
    fig.text(0.5, 0.02, f"★ Chosen configuration. Lower CE / higher scores are better.\n{WEIGHT_NOTE}", ha="center")
    fig.tight_layout(rect=(0, 0.12, 1, 0.95), w_pad=1.3)
    return fig


def overall(runs: Sequence[Run], *, chosen: str = CHOSEN_MODEL) -> Figure:
    return _comparison_bars(
        _display_runs(runs),
        ("val_cross_entropy", "val_accuracy", "val_macro_f1"),
        "Validation performance at saved checkpoints",
        chosen,
    )


def learning(runs: Sequence[Run], metric: str, *, chosen: str = CHOSEN_MODEL) -> Figure:
    fig, ax = plt.subplots(figsize=(10.5, 6))
    all_vals: list[float] = []
    for run in runs:
        colour = _run_colour(run)
        epochs = [row["epoch"] for row in run.history]
        vals = [row[metric] for row in run.history]
        all_vals.extend(vals)
        is_chosen = run.ident == chosen
        label = run.label.removeprefix("MLP: ") + (" ★" if is_chosen else "")
        ax.plot(epochs, vals, color=colour, linewidth=2.8 if is_chosen else 1.6, label=label)
        ax.scatter(run.best_epoch, run.metrics[metric], marker="D", s=45, c=colour, zorder=3)
    max_epoch = max(len(run.history) for run in runs)
    ax.set(xlabel="Epoch", ylabel=f"Validation {METRICS[metric]}", xlim=(0, np.ceil(max_epoch / 5) * 5))
    lo, hi = _metric_limits(metric, all_vals)
    if metric == "val_cross_entropy":
        lo = min(1.5, float(np.floor(min(all_vals) * 2) / 2))
    ax.set_ylim(lo, hi)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{runs[0].model.upper() if runs[0].model == 'mlp' else 'Logistic regression'}: validation {METRICS[metric]}"
    )
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
    note = "Diamonds: minimum unweighted validation-CE checkpoints. ★ Chosen configuration."
    if any(run.name.startswith("inverse-sqrt") for run in runs):
        note += f"\n{WEIGHT_NOTE}"
    fig.text(0.5, 0.015, note, ha="center")
    fig.subplots_adjust(left=0.10, right=0.67, bottom=0.20, top=0.89)
    return fig


def family_f1(runs: Sequence[Run]) -> Figure:
    base = runs[0]
    order = np.argsort(-base.support, kind="stable")
    rows = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(8, max(6, len(order) * 0.33 + 2)), sharey=True, width_ratios=(8, 1))
    for offset, run, colour in zip((-0.31, 0, 0.31), runs, COMPARE_COLOURS):
        axes[0].barh(rows + offset, run.f1[order], height=0.30, color=colour, label=run.label)
    axes[0].set(xlabel="Family F1 (higher is better)", xlim=(0, 1))
    axes[0].set_yticks(rows, [base.classes[idx].replace("_", " ") for idx in order])
    axes[0].invert_yaxis()
    axes[0].set_axisbelow(True)
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].spines[["top", "right"]].set_visible(False)
    for row, count in enumerate(base.support[order]):
        axes[1].text(0.5, row, f"{count:,}", ha="center", va="center")
    axes[1].set_title("Support")
    axes[1].set_axis_off()
    fig.suptitle("Per-family validation F1")
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.025), ncol=1, frameon=False)
    fig.text(0.5, 0.01, WEIGHT_NOTE, ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.13, 1, 0.95))
    return fig


def relative_f1(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    return np.divide(100 * (dst - src), src, out=np.full_like(src, np.nan, dtype=np.float64), where=src > 0)


def family_changes(runs: Sequence[Run]) -> Figure:
    base = runs[0]
    order = np.argsort(-base.support, kind="stable")
    rows = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, max(6, len(order) * 0.21 + 1.5)), sharey=True)
    for ax, (src, dst), colour in zip(axes, pairwise(runs), COMPARE_COLOURS[1:]):
        changes = relative_f1(src.f1, dst.f1)[order]
        valid = np.isfinite(changes)
        ax.scatter(changes[valid], rows[valid], c=colour, s=30, zorder=3)
        ax.axvline(0, color="#888888", linewidth=1)
        ax.set_xscale("symlog", linthresh=100, linscale=0.5)
        lo = min(-100, float(np.nanmin(changes, initial=0)))
        hi = max(100, float(np.nanmax(changes, initial=0)))
        ax.set_xlim(lo * 1.25, hi * 1.35)
        ticks = [-100, 0, 100, *(10**power for power in range(3, int(np.ceil(np.log10(hi * 1.35))) + 1))]
        ax.xaxis.set_major_locator(FixedLocator([tick for tick in ticks if lo * 1.25 <= tick <= hi * 1.35]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:+,.0f}%" if val else "0%"))
        ax.set(
            title=f"{dst.label.removeprefix('MLP: ')}\nvs {src.label.removeprefix('MLP: ')}",
            xlabel="Relative F1 change",
        )
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(1.04, 1.015, "F1 before → after", transform=ax.transAxes, fontsize=12)
        for row, idx in enumerate(order):
            text = f"{src.f1[idx]:.3f} → {dst.f1[idx]:.3f}"
            ax.text(1.04, row, text, transform=ax.get_yaxis_transform(), va="center", fontsize=12)
            if src.f1[idx] == 0:
                ax.text(
                    0.03, row, "undefined (baseline 0)", transform=ax.get_yaxis_transform(), va="center", fontsize=9
                )
    axes[0].set_yticks(rows, [base.classes[idx].replace("_", " ") for idx in order])
    axes[0].invert_yaxis()
    fig.suptitle("Per-family relative improvements and regressions")
    fig.text(
        0.5,
        0.015,
        f"100 × (after − before) / before. Linear within ±100%; logarithmic beyond.\n{WEIGHT_NOTE}",
        ha="center",
    )
    fig.subplots_adjust(left=0.25, right=0.84, wspace=0.85, bottom=0.14, top=0.83)
    return fig


def confusion(run: Run) -> Figure:
    count = len(run.classes)
    normalized = np.divide(
        run.confusion,
        run.support[:, None],
        out=np.zeros_like(run.confusion, dtype=np.float64),
        where=run.support[:, None] > 0,
    )
    masked = np.ma.masked_where(np.broadcast_to(run.support[:, None] == 0, normalized.shape), normalized)
    fig, ax = plt.subplots(figsize=(10.5, 9))
    cmap = LinearSegmentedColormap.from_list("white_to_blue", ("#ffffff", "#6baed6", "#08306b")).with_extremes(
        bad="#dddddd"
    )
    artist = ax.imshow(masked, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_xticks(range(count), [label.replace("_", " ") for label in run.classes], rotation=90)
    ax.set_yticks(range(count), [f"{label.replace('_', ' ')} (n={n:,})" for label, n in zip(run.classes, run.support)])
    ax.set_xticks(np.arange(count + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(count + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="#cccccc", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set(
        xlabel="Predicted tactic family",
        ylabel="True tactic family",
        title=f"{run.label} — validation confusion, epoch {run.best_epoch}",
    )
    # Annotate substantial cells only
    for row, col in np.argwhere(normalized >= 0.05):
        val = normalized[row, col]
        brightness = np.dot(cmap(val)[:3], (0.2126, 0.7152, 0.0722))
        ax.text(
            col,
            row,
            f"{val * 100:.0f}",
            ha="center",
            va="center",
            fontsize=9,
            color="black" if brightness > 0.5 else "white",
        )
    fig.colorbar(artist, ax=ax, shrink=0.8, label="Fraction within the true family", format=PercentFormatter(1))
    fig.text(
        0.5,
        0.01,
        "Darker = larger within-family proportion. Cells ≥5% show percentages; grey = no examples.",
        ha="center",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


# Rendering orchestration and CLI


def figures(
    runs: Sequence[Run], compared: Sequence[Run], selected: Sequence[str], cfg: PlotCfg
) -> Iterator[tuple[str, Figure]]:
    runs = _display_runs(runs)
    if "overall" in selected:
        yield "overall-performance", overall(runs, chosen=cfg.chosen)
        yield (
            "top-k-performance",
            _comparison_bars(
                runs,
                ("val_accuracy", "val_top3_accuracy", "val_top5_accuracy"),
                "Validation top-k accuracy",
                cfg.chosen,
            ),
        )
    if "learning" in selected:
        for model in ("logistic", "mlp"):
            group = [run for run in runs if run.model == model]
            for metric, name in (("val_cross_entropy", "cross-entropy"), ("val_macro_f1", "macro-f1")):
                yield f"learning-{name}-{model}-01", learning(group, metric, chosen=cfg.chosen)
    if "families" in selected:
        yield "family-f1", family_f1(compared)
        yield "family-improvements", family_changes(compared)
    if "confusion" in selected:
        for run in compared:
            yield f"confusion-{run.model}-{run.name}", confusion(run)


def render(
    runs: Sequence[Run],
    compared: Sequence[Run],
    output: Path,
    selected: Sequence[str],
    cfg: PlotCfg,
    *,
    replace: bool = False,
) -> None:
    if output.exists() and any(output.iterdir()) and not replace:
        raise FileExistsError(f"output directory is nonempty: {output}; use --replace or choose a new directory")
    output.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with TemporaryDirectory(prefix=".results-", dir=output) as tmp_dir:
        pending = Path(tmp_dir)
        style = "dark_background" if cfg.theme == "dark" else "default"
        with plt.style.context(style), mpl.rc_context({"font.size": 12, "pdf.fonttype": 42}):
            for name, fig in figures(runs, compared, selected, cfg):
                try:
                    for format in cfg.formats:
                        filename = f"{name}.{format}"
                        fig.savefig(pending / filename, dpi=cfg.dpi)
                        written.append(filename)
                finally:
                    plt.close(fig)
                print(f"Rendered {name}", flush=True)
        with (pending / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("model", "name", "best_epoch", *METRICS))
            writer.writeheader()
            writer.writerows(
                {"model": run.model, "name": run.name, "best_epoch": run.best_epoch, **run.metrics} for run in runs
            )
        if "families" in selected:
            with (pending / "family-improvements.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(
                    ("family", "support", "src_model", "dst_model", "src_f1", "dst_f1", "relative_change_percent")
                )
                for src, dst in pairwise(compared):
                    for idx, change in enumerate(relative_f1(src.f1, dst.f1)):
                        writer.writerow(
                            (
                                src.classes[idx],
                                src.support[idx],
                                src.ident,
                                dst.ident,
                                src.f1[idx],
                                dst.f1[idx],
                                change if np.isfinite(change) else "",
                            )
                        )
            written.append("family-improvements.csv")
        manifest = {
            "population": "validation",
            "selection": "minimum unweighted validation cross-entropy",
            "chosen_configuration": cfg.chosen,
            "plotted_runs": [run.ident for run in _display_runs(runs)],
            "omitted_from_figures": [run.ident for run in runs if "smoothing" in run.name.split("-")],
            "relative_f1": "100*(dst_f1-src_f1)/src_f1; zero baselines undefined; not percentage-point differences",
            "vocab_id": runs[0].identity[0],
            "width": runs[0].identity[1],
            "classes": runs[0].classes,
            "train": dict(zip(("rows", "theorems", "theorem_ids_sha256"), runs[0].identity[3])),
            "validation": dict(zip(("rows", "theorems", "theorem_ids_sha256"), runs[0].identity[4])),
            "compared": [run.ident for run in compared],
            "sources": {run.ident: str(run.src) for run in runs},
            "files": written + ["metrics.csv"],
        }
        (pending / "figures.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for path in pending.iterdir():
            if replace:
                os.replace(path, output / path.name)
            else:
                os.link(path, output / path.name)  # Do not overwrite a concurrently created artifact.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp", type=Path, required=True, help="experiment directory containing */results.json")
    parser.add_argument(
        "--logistic", type=Path, required=True, help="directory containing trusted logit-*.pkl checkpoints"
    )
    parser.add_argument("--output", type=Path, required=True, help="new or empty figure directory")
    parser.add_argument(
        "--replace", action="store_true", help="replace generated files in an existing output directory"
    )
    parser.add_argument(
        "--compare", nargs=3, default=DEFAULT_COMPARE, help="three model:name IDs for detailed comparisons, in order"
    )
    parser.add_argument("--plots", nargs="+", choices=PLOTS, default=PLOTS)
    parser.add_argument(
        "--chosen", default=DEFAULT_PLOT_CFG.chosen, help="model:name ID to highlight as the final configuration"
    )
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf"), default=DEFAULT_PLOT_CFG.formats)
    parser.add_argument("--dpi", type=int, default=DEFAULT_PLOT_CFG.dpi)
    parser.add_argument("--theme", choices=("light", "dark"), default=DEFAULT_PLOT_CFG.theme)
    args = parser.parse_args(argv)
    if args.dpi < 40:
        parser.error("dpi must be at least 40")
    runs = load_runs(args.mlp, args.logistic)
    visible = _display_runs(runs)
    missing = [run.ident for run in visible if not run.history]
    if "learning" in args.plots and missing:
        parser.error(
            f"learning curves require epoch histories for {', '.join(missing)}; omit learning from --plots or rerun training"
        )
    index = {run.ident: run for run in visible}
    if len(set(args.compare)) != 3 or any(ident not in index for ident in args.compare):
        parser.error(f"choose three distinct --compare IDs from: {', '.join(index)}")
    if args.chosen not in index:
        parser.error(f"choose --chosen from: {', '.join(index)}")
    compared = tuple(index[ident] for ident in args.compare)
    plt.switch_backend("Agg")
    cfg = PlotCfg(tuple(dict.fromkeys(args.formats)), args.dpi, args.theme, args.chosen)
    render(runs, compared, args.output, args.plots, cfg, replace=args.replace)
    print(
        f"Complete: {len(runs)} runs; all information required for the selected figures present; figures in {args.output}"
    )
    return 0
