"""Shared discovery, split-specific fitting, and reusable development vectors.

Candidates are label-neutral observations of the development pool. Every fitted
coverage/support/association/name product belongs to one training population.
Completed stages are reused only with identical inputs and output stamps; an
interrupted stage may replace its own unpublished/partial products on resume.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import MISSING, dataclass, fields
from functools import partial
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import monotonic
from typing import Any, cast

import msgspec
import numpy as np

from trustmebro.artifacts import canonical_request, encode_msgpack
from trustmebro.preprocessing.coverage import (
    DEFAULT_IDX_MEM_MIB,
    CoverageIdx,
    PackedShapes,
    build_coverage_idx,
    coverage_shape,
    require_coverage,
    select_coverage,
)
from trustmebro.preprocessing.layout import compile_vocab, entry_dims, feature_blocks, fixed_dimensions, stat_width
from trustmebro.preprocessing.records import (
    DEFAULT_ATTR_POLICY,
    DEFAULT_COVER_POLICY,
    NODE_NAMES,
    SUPERVISED_SCREENING,
    AttrPolicy,
    CoverageCands,
    DimBudget,
    Entry,
    LabelPolicy,
    Representation,
    SupervisedPolicy,
    Supervision,
    Vocab,
    read_label_policy,
)
from trustmebro.preprocessing.scan import convert_corpus, scan_cands
from trustmebro.runtime import Phase

from .archives import publish, read_coverage_candidates, read_selection, read_vocab
from .partition import logical_splits
from .records import DEFAULT_VALIDATION_CFG, MinSupport, SplitCfg, read_split, split_id
from .scan import _selection_results
from .supervised import select_attributes, select_supervised


@dataclass(frozen=True, slots=True)
class VocabBuildCfg:
    dims: int  # complete vector, not an enrichment-only allowance
    depths: tuple[int, ...] = (1, 2, 3, 4, 5)
    workers: int = 1
    hyp_slots: int = DEFAULT_ATTR_POLICY.hyp_slots
    name_share: float = 0.2
    min_support: int = DEFAULT_ATTR_POLICY.min_support  # labeled theorem support for optional shapes/names
    idx_mem_mib: int = DEFAULT_IDX_MEM_MIB
    score_mem_mib: int = 1024
    name_mem_mib: int = DEFAULT_ATTR_POLICY.memory_mib


def prepare_coverage(
    src: Path,
    *,
    depths: tuple[int, ...] | None = None,
    idx_mem_mib: int = DEFAULT_IDX_MEM_MIB,
    theorems: tuple[str, ...] | None = None,
) -> CoverageIdx:
    """Prepare coverage from a completed candidate archive, not the active DB.

    No solver is invoked. The retained-index guard is not a peak-RSS limit;
    full-corpus execution requires an agreed resource budget.
    """
    stamp = src.stat()
    selection = read_selection(src)
    depths = tuple(sorted(set(selection.depths if depths is None else depths)))
    if not depths or not set(depths).issubset(selection.depths):
        raise ValueError("coverage depths must be present in the candidate archive")
    with (
        Phase("Prepare expression coverage") as progress,
        closing(read_coverage_candidates(src, names=theorems)) as observations,
    ):

        def observed() -> Iterator[CoverageCands]:
            for count, theorem in enumerate(observations, start=1):
                progress.details = f"{count:,} theorems"
                yield theorem

        idx = build_coverage_idx(observed(), depths, idx_mem_mib=idx_mem_mib)
    if (src.stat().st_size, src.stat().st_mtime_ns) != (stamp.st_size, stamp.st_mtime_ns):
        raise ValueError("candidate archive changed during coverage preparation")
    return idx


def _fit_repr(cands: Path, vocab: Vocab, label_policy: LabelPolicy, policy: AttrPolicy, workers: int) -> Vocab:
    """Fit names once; callers own source guards and final publication."""
    stamp = cands.stat()
    selection = read_selection(cands)
    if vocab.representation is not None:
        raise ValueError("supply a structural vocabulary, not an already fitted representation")
    if not set(vocab.depths).issubset(selection.depths):
        raise ValueError("candidate archive lacks vocabulary discovery depths")
    if vocab.selection is None or msgspec.structs.replace(
        selection, depths=vocab.selection.depths, theorems=vocab.selection.theorems, limit=None, seed=None
    ) != msgspec.structs.replace(vocab.selection, limit=None, seed=None):
        raise ValueError("vocabulary and representation candidates must describe the same corpus selection")
    selection = vocab.selection
    layout = compile_vocab(vocab)
    mapper_total = None if selection.theorems is None else len(selection.theorems)
    mapper = partial(
        _selection_results, candidates=cands, names=selection.theorems, workers=workers, total=mapper_total
    )
    heads, names = select_attributes(mapper, layout, label_policy, policy, stage=Phase)
    label_policy_json = json.dumps(msgspec.to_builtins(label_policy), sort_keys=True, separators=(",", ":")).encode()
    repr = Representation(
        policy, heads, names, selection, str(cands.resolve()), stamp.st_size, stamp.st_mtime_ns, label_policy_json
    )
    return msgspec.structs.replace(vocab, representation=repr)


def _coverage_entries(shapes: PackedShapes, cols: np.ndarray) -> tuple[Entry, ...]:
    selected = (coverage_shape(shapes, int(col)) for col in cols)
    return tuple(Entry(shape.ident, shape.edges) for shape in selected)


def _complete_dim_costs(shapes: PackedShapes, shortlist: np.ndarray) -> np.ndarray:
    # Decode only the bounded shortlist for actual fixed-attribute costs.
    return np.fromiter(
        (entry_dims(coverage_shape(shapes, int(col)).edges, attributes=True) for col in shortlist), dtype=np.int64
    )


def check_vocab_cfg(cfg: VocabBuildCfg) -> None:
    if not np.isfinite(cfg.name_share) or not 0 <= cfg.name_share <= 1:
        raise ValueError("name share must lie between zero and one")
    if min(cfg.workers, cfg.min_support, cfg.idx_mem_mib, cfg.score_mem_mib, cfg.name_mem_mib) < 1:
        raise ValueError("workers, support and retained-memory guards must be positive")
    if cfg.hyp_slots < 0 or not cfg.depths or any(depth < 1 for depth in cfg.depths):
        raise ValueError("hypothesis slots must be nonnegative and discovery depths positive")
    stats_width = stat_width(len(NODE_NAMES), cfg.hyp_slots)
    if cfg.dims < stats_width:
        raise ValueError(
            f"total dimension budget {cfg.dims:,} cannot fit {stats_width:,} statistics/slot columns alone"
        )


def fit_vocab(
    cands: Path,
    labels: Path,
    output: Path,
    cfg: VocabBuildCfg,
    *,
    theorems: tuple[str, ...] | None = None,
    replace: bool = False,
) -> dict[str, object]:
    """Cached candidates -> training-only cover/scores/names -> frozen layout.

    Reuse the prepared in-memory coverage index across selection passes. The total cap includes fixed channels; it is not a memory/RSS limit.
    The selected heuristic baseline is preserved, not claimed globally minimal.
    """
    check_vocab_cfg(cfg)
    stats_width = stat_width(len(NODE_NAMES), cfg.hyp_slots)
    sources = (cands, labels)
    targets = (output,)
    # Reject every collision/existing output before running any expensive stage.
    for idx, path in enumerate(targets):
        for source in (*sources, *targets[:idx]):
            if path.resolve() == source.resolve() or (path.exists() and source.exists() and path.samefile(source)):
                raise ValueError("vocabulary build outputs must be distinct and cannot replace input files")
        if path.exists() and (not replace or not path.is_file()):
            raise FileExistsError(f"build output exists: {path}; use --replace for existing files")
    label_policy = read_label_policy(labels)
    stamps = [(path.stat().st_size, path.stat().st_mtime_ns) for path in sources]
    started = monotonic()
    idx = prepare_coverage(cands, depths=cfg.depths, idx_mem_mib=cfg.idx_mem_mib, theorems=theorems)
    selection = msgspec.structs.replace(
        read_selection(cands), depths=cfg.depths, theorems=idx.names, limit=None, seed=None
    )
    theorem_count = len(idx.names)

    with Phase("Select coverage baseline") as progress:
        selected = select_coverage(idx)
        baseline_entries = _coverage_entries(idx.shapes, selected.cols)
        baseline_width = fixed_dimensions(baseline_entries, cfg.hyp_slots)
        if baseline_width > cfg.dims:
            raise ValueError(
                f"total dimension budget {cfg.dims:,} cannot fit the selected coverage baseline "
                f"({baseline_width:,} columns, including {stats_width:,} statistics/slot columns); "
                "increase --dims; this heuristic cover is not a proof of the minimum feasible width"
            )
        progress.details = f"{len(baseline_entries):,} shapes; {baseline_width:,}/{cfg.dims:,} dimensions"
    remaining = cfg.dims - baseline_width
    shape_allow = min(remaining, int(remaining * (1 - cfg.name_share)))
    cols, supervision = selected.cols, None
    vocab_entries = baseline_entries
    added_dims = 0
    if shape_allow:
        shape_policy = SupervisedPolicy(dims=shape_allow, min_support=cfg.min_support, memory_mib=cfg.score_mem_mib)
        mapper = partial(
            _selection_results, candidates=cands, names=selection.theorems, workers=cfg.workers, total=len(idx.names)
        )
        dim_costs = partial(_complete_dim_costs, idx.shapes)
        enriched = select_supervised(
            mapper, idx, selected.cols, label_policy, shape_policy, depths=cfg.depths, dim_costs=dim_costs, stage=Phase
        )
        cols = np.concatenate((cols, enriched.cols))
        vocab_entries += _coverage_entries(idx.shapes, enriched.cols)
        added_dims = enriched.dims
        stamp = cands.stat()
        supervision = Supervision(
            shape_policy,
            json.dumps(msgspec.to_builtins(label_policy), sort_keys=True, separators=(",", ":")).encode(),
            str(cands.resolve()),
            stamp.st_size,
            stamp.st_mtime_ns,
            label_policy.labels,
            tuple(map(int, enriched.evidence.totals)),
            enriched.evidence.labeled_theorems,
            enriched.shortlist_shapes,
            len(enriched.cols),
            added_dims,
            dimension_cost="complete",
            available_assoc=tuple(map(float, enriched.evidence.scores.max(axis=0, initial=0))),
            selected_assoc=enriched.selected_assoc,
            label_strength=enriched.label_strength,
        )
        del enriched
    require_coverage(idx, cols)
    vocab = Vocab(
        selection.depths,
        vocab_entries,
        selection=selection,
        coverage=DEFAULT_COVER_POLICY,
        supervision=supervision,
        budget=DimBudget(cfg.dims, baseline_width, cfg.name_share),
    )
    # Release the corpus root/shape index before name evidence/scoring is built.
    del idx, selected, baseline_entries
    attrs = AttrPolicy(
        hyp_slots=cfg.hyp_slots,
        name_dims=remaining - added_dims,
        min_support=cfg.min_support,
        memory_mib=cfg.name_mem_mib,
    )
    vocab = _fit_repr(cands, vocab, label_policy, attrs, cfg.workers)
    layout = compile_vocab(vocab)  # independently enforce the complete cap
    if [(path.stat().st_size, path.stat().st_mtime_ns) for path in sources] != stamps:
        raise ValueError("database or label policy changed during vocabulary construction")
    raw_bytes = publish(output, (encode_msgpack(vocab),), sources=sources, replace=replace)
    return {
        "theorems": theorem_count,
        "shapes": len(vocab.entries),
        "dimension_budget": cfg.dims,
        "baseline_dimensions": baseline_width,
        "shape_enrichment_dimensions": added_dims,
        "shape_associations": {
            label: {"available_r2": available, "selected_r2": retained, "balanced_strength": strength}
            for label, available, retained, strength in zip(
                supervision.labels,
                supervision.available_assoc,
                supervision.selected_assoc,
                supervision.label_strength,
                strict=True,
            )
        }
        if supervision is not None
        else {},
        "name_dimensions": layout.width - baseline_width - added_dims,
        "dimensions": layout.width,
        "unused_dimensions": cfg.dims - layout.width,
        "blocks": {name: hi - lo for name, (lo, hi) in feature_blocks(layout).items()},
        "candidates": str(cands),
        "vocab": str(output),
        "stored_bytes": output.stat().st_size,
        "uncompressed_stream_bytes": raw_bytes,
        "dur_sec": monotonic() - started,
    }


@dataclass(frozen=True, slots=True)
class PrepCfg:
    vocab: VocabBuildCfg
    split: SplitCfg = DEFAULT_VALIDATION_CFG
    folds: int = 1
    pair_limit: int = 0


def _stamp(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {"path": str(path.resolve()), "bytes": stat.st_size, "modified_ns": stat.st_mtime_ns}


def _write_json(path: Path, data: object) -> None:
    """Publish small metadata atomically; compressed bulk products use archives."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=".preparation-", delete=False) as stream:
        pending = Path(stream.name)
        try:
            stream.write(msgspec.json.encode(data))
        except BaseException:
            pending.unlink(missing_ok=True)
            raise
    try:
        pending.replace(path)
    finally:
        pending.unlink(missing_ok=True)


def _stage(
    record: Path, outputs: tuple[Path, ...], srcs: tuple[Path, ...], fn: Callable[[], object], title: str
) -> dict:
    inputs = [_stamp(path) for path in srcs]
    if record.exists():
        prev = json.loads(record.read_text())
        if prev["inputs"] != inputs or any(not path.is_file() for path in outputs):
            raise ValueError(f"{title}: completed stage inputs or outputs changed; use a new output directory")
        # Old completed instances may also record the now-unused coverage dump.
        # Validate every recorded artifact, but require only current outputs.
        recorded_outputs = prev["outputs"]
        recorded_paths = tuple(Path(item["path"]) for item in recorded_outputs)
        if not {path.resolve() for path in outputs}.issubset(recorded_paths) or recorded_outputs != [
            _stamp(path) for path in recorded_paths
        ]:
            raise ValueError(f"{title}: completed artifacts changed; use a new output directory")
        print(f"{title}: reuse completed stage", file=sys.stderr)
        return {**prev, "reused": True}
    started = monotonic()
    print(f"{title}...", file=sys.stderr)
    # Nested computation owns progress; do not compete with its live line.
    result = fn()
    if inputs != [_stamp(path) for path in srcs]:
        raise ValueError(f"{title}: source changed during stage")
    report = {
        "inputs": inputs,
        "outputs": [_stamp(path) for path in outputs],
        "duration_sec": monotonic() - started,
        "result": msgspec.to_builtins(result),
    }
    _write_json(record, report)
    print(f"{title}: finished in {report['duration_sec']:.2f}s", file=sys.stderr)
    return {**report, "reused": False}


def prepare_features(db: Path, labels: Path, output: Path, cfg: PrepCfg, *, cands: Path | None = None) -> dict:
    """One holdout or K fitted instances, without copying the source database.

    Reuse requires unchanged run configuration and sources. Choose another output
    directory for another experiment, supplying the same candidates to avoid
    discovery. Existing unrelated files are never adopted or overwritten.
    """
    if cfg.folds < 1 or cfg.vocab.workers < 1 or not 0 <= cfg.pair_limit <= 2048:
        raise ValueError("folds/workers must be positive and co-occurrence prefix must lie in 0..2048")
    check_vocab_cfg(cfg.vocab)
    policy = read_label_policy(labels)
    request = cast(
        dict[str, Any],
        canonical_request(
            {
                "db": _stamp(db),
                "labels": _stamp(labels),
                "cfg": cfg,
                "shape_selection": SUPERVISED_SCREENING,
                "candidates": None if cands is None else _stamp(cands),
            }
        ),
    )
    run = output / "run.json"
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output is not a directory: {output}")
    if run.exists():
        if json.loads(run.read_text()) != request:
            raise ValueError("preparation inputs/settings changed; use a new output directory")
    elif output.exists() and any(output.iterdir()):
        raise FileExistsError("output contains unrelated artifacts; choose an empty/new directory")
    else:
        _write_json(run, request)
    started = monotonic()
    reports: dict[str, object] = {}
    summaries: list[dict[str, object]] = []
    if cands is None:
        cands = output / "candidates.zst"
        reports["candidates"] = _stage(
            output / "candidates.done.json",
            (cands,),
            (db,),
            lambda: scan_cands(db, cands, depths=cfg.vocab.depths, workers=cfg.vocab.workers, replace=True),
            "Extract shared candidates",
        )
    selection = read_selection(cands)
    if (
        Path(selection.db).resolve() != db.resolve()
        or (selection.size, selection.modified_ns) != (db.stat().st_size, db.stat().st_mtime_ns)
        or selection.limit is not None
        or selection.theorems is not None
        or not set(cfg.vocab.depths).issubset(selection.depths)
    ):
        raise ValueError("candidates must describe the complete, unchanged development database and requested depths")
    manifests = tuple(output / f"instance-{idx + 1:02d}" / "split.json" for idx in range(cfg.folds))

    def assign() -> dict[str, object]:
        with Phase("Choose logical train/validation roles") as progress:
            splits = logical_splits(db, policy, cfg.split, folds=cfg.folds)
            for path, split in zip(manifests, splits, strict=True):
                _write_json(path, split)
            progress.details = f"{len(splits)} instances; theorems remain grouped"
        return {"instances": len(splits)}

    reports["splits"] = _stage(output / "splits.done.json", manifests, (db, labels, cands), assign, "Assign roles")
    for path in manifests:
        split = read_split(path)
        root = path.parent
        vocab = root / "vocab.zst"
        prefix = f"Instance {split.instance}/{cfg.folds}"
        instance_report: dict[str, dict[str, Any]] = {
            "fitting": _stage(
                root / "vocabulary.done.json",
                (vocab,),
                (cands, labels, path),
                partial(fit_vocab, cands, labels, vocab, cfg.vocab, theorems=split.train, replace=True),
                f"{prefix}: fit training vocabulary",
            ),
            "conversion": _stage(
                root / "features.done.json",
                (root / "features.zst", root / "feature-stats.zst"),
                (cands, vocab, labels, path),
                partial(
                    convert_corpus,
                    cands,
                    vocab,
                    root / "features.zst",
                    root / "feature-stats.zst",
                    cands=True,
                    workers=cfg.vocab.workers,
                    labels=labels,
                    pair_limit=cfg.pair_limit,
                    split_id=split_id(split),
                    expected_theorems=split.train + split.validation,
                    replace=True,
                ),
                f"{prefix}: convert development pool",
            ),
        }
        reports[root.name] = instance_report
        conversion = cast(dict[str, Any], instance_report["conversion"]["result"])
        summaries.append(
            {
                "instance": split.instance,
                "train": msgspec.to_builtins(split.train_stats),
                "validation": msgspec.to_builtins(split.validation_stats),
                "dimensions": conversion["dimensions"],
                "nnz": conversion["nnz"],
                "mean_active_per_observed_state": conversion["nnz"] / max(conversion["states"], 1),
            }
        )
    if _stamp(db) != request["db"] or _stamp(labels) != request["labels"]:
        raise ValueError("development database or label policy changed during preparation")
    report = {
        "instances": cfg.folds,
        "output": str(output.resolve()),
        "duration_sec": monotonic() - started,
        "instance_summaries": summaries,
        "stages": reports,
    }
    _write_json(output / "summary.json", report)
    return report


def convert_frozen(
    db: Path,
    vocab: Path,
    output: Path,
    *,
    workers: int = 1,
    labels: Path | None = None,
    pair_limit: int = 0,
    replace: bool = False,
) -> dict:
    """Resume a compatible conversion without fitting or assigning theorem roles."""
    if read_vocab(vocab).representation is None:
        raise ValueError("frozen conversion requires a vocabulary containing its frozen label policy")
    output.mkdir(parents=True, exist_ok=True)
    request = {
        "db": _stamp(db),
        "vocab": _stamp(vocab),
        "labels": None if labels is None else _stamp(labels),
        "pair_limit": pair_limit,
    }
    marker = output / "conversion.request.json"
    if (
        not marker.exists()
        and not replace
        and any((output / name).exists() for name in ("features.zst", "feature-stats.zst"))
    ):
        raise FileExistsError("frozen conversion outputs exist without a matching request; choose a new output")
    if marker.exists() and json.loads(marker.read_text()) != request and not replace:
        raise ValueError("frozen conversion inputs changed; choose a new output or --replace")
    if replace:
        # The old completion marker must not authorize reuse after a changed request.
        (output / "features.done.json").unlink(missing_ok=True)
    _write_json(marker, request)
    report = _stage(
        output / "features.done.json",
        (output / "features.zst", output / "feature-stats.zst"),
        (db, vocab) if labels is None else (db, vocab, labels),
        partial(
            convert_corpus,
            db,
            vocab,
            output / "features.zst",
            output / "feature-stats.zst",
            workers=workers,
            labels=labels,
            pair_limit=pair_limit,
            replace=True,
        ),
        "Convert frozen features",
    )
    return report["result"]


def main(argv: list[str] | None = None) -> int:
    # dims has no default; inspect the optional fields without inventing a budget.
    defaults = {field.name: field.default for field in fields(VocabBuildCfg) if field.default is not MISSING}
    parser = argparse.ArgumentParser(description="Prepare split-specific vocabularies and cached learning features.")
    parser.add_argument("--vocab", type=Path, help="convert the database using this frozen vocabulary; never fit")
    parser.add_argument("--replace", action="store_true", help="replace frozen conversion outputs")
    parser.add_argument(
        "--db", type=Path, required=True, help="development database for fitting, or held-out database with --vocab"
    )
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="new directory, or unchanged run to resume")
    parser.add_argument("--candidates", type=Path, help="reuse complete development-pool candidates")
    parser.add_argument("--dims", type=int, help="total feature dimension cap per instance")
    parser.add_argument("--workers", type=int, default=defaults["workers"])
    parser.add_argument("--depths", type=int, nargs="+", default=defaults["depths"])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--folds", type=int, help="K disjoint grouped validation folds; K must be >= 2")
    modes.add_argument(
        "--validation-frac", type=float, help=f"holdout theorem fraction; default {DEFAULT_VALIDATION_CFG.test_frac}"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--train-min-trns", type=int, default=DEFAULT_VALIDATION_CFG.train_min.trns)
    parser.add_argument("--train-min-theorems", type=int, default=DEFAULT_VALIDATION_CFG.train_min.theorems)
    parser.add_argument("--validation-min-trns", type=int, default=DEFAULT_VALIDATION_CFG.test_min.trns)
    parser.add_argument("--validation-min-theorems", type=int, default=DEFAULT_VALIDATION_CFG.test_min.theorems)
    parser.add_argument("--solver-seconds", type=float, default=DEFAULT_VALIDATION_CFG.solver_sec)
    parser.add_argument("--hyp-slots", type=int, default=defaults["hyp_slots"])
    parser.add_argument("--name-share", type=float, default=defaults["name_share"])
    parser.add_argument("--min-support", type=int, default=defaults["min_support"])
    parser.add_argument("--index-memory-mib", type=int, default=defaults["idx_mem_mib"])
    parser.add_argument("--score-memory-mib", type=int, default=defaults["score_mem_mib"])
    parser.add_argument("--name-memory-mib", type=int, default=defaults["name_mem_mib"])
    parser.add_argument("--cooccurrence-shapes", type=int, default=0)
    supplied = sys.argv[1:] if argv is None else argv
    cfg = parser.parse_args(supplied)
    if cfg.vocab is not None:
        shared = {"--db", "--vocab", "--labels", "--output", "--workers", "--replace", "--cooccurrence-shapes"}
        explicit = {arg.split("=", 1)[0] for arg in supplied if arg.startswith("--")}
        incompatible = explicit - shared
        if incompatible:
            parser.error("fitting-only options with --vocab: " + ", ".join(sorted(incompatible)))
        report = convert_frozen(
            cfg.db,
            cfg.vocab,
            cfg.output,
            workers=cfg.workers,
            labels=cfg.labels,
            pair_limit=cfg.cooccurrence_shapes,
            replace=cfg.replace,
        )
    else:
        if cfg.labels is None or cfg.dims is None:
            parser.error("fitting requires --labels and --dims")
        if cfg.replace:
            parser.error("--replace applies only to frozen conversion; preparation resumes compatible stages")
        if cfg.folds is not None and cfg.folds < 2:
            parser.error("--folds must be at least two; omit it for a holdout")
        folds = cfg.folds or 1
        vocab = VocabBuildCfg(
            cfg.dims,
            tuple(sorted(set(cfg.depths))),
            cfg.workers,
            cfg.hyp_slots,
            cfg.name_share,
            cfg.min_support,
            cfg.index_memory_mib,
            cfg.score_memory_mib,
            cfg.name_memory_mib,
        )
        split = SplitCfg(
            1 / folds
            if folds > 1
            else (DEFAULT_VALIDATION_CFG.test_frac if cfg.validation_frac is None else cfg.validation_frac),
            cfg.seed,
            MinSupport(cfg.train_min_trns, cfg.train_min_theorems),
            MinSupport(cfg.validation_min_trns, cfg.validation_min_theorems),
            cfg.solver_seconds,
        )
        report = prepare_features(
            cfg.db, cfg.labels, cfg.output, PrepCfg(vocab, split, folds, cfg.cooccurrence_shapes), cands=cfg.candidates
        )
    print(json.dumps({key: val for key, val in report.items() if key != "stages"}))
    return 0
