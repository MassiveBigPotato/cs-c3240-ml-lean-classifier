# trustmebro

Predict the next Lean tactic family from a proof state using logistic regression and a sparse-input PyTorch MLP.

## Setup and extraction

Use [Pixi](https://pixi.sh/), Lean/Lake with the pinned toolchain, and an NVIDIA driver supporting CUDA 12.9 for GPU training.
Run commands from the repository root.
Pixi manages Python and the native graph/plotting dependencies.

```sh
lake build trustmebro-extract-state
pixi run extract-dataset --db data/mathlib.db --workers 8 --timing
```

The pipeline discovers files, extracts a sample, trains embedded compression dictionaries, extracts the rest, then retrains and recompresses.
Rerun with unchanged settings to resume.
Allow several GB per Lean worker and additional memory for dictionary training.
Exported graphs group applications/binders and rewrite registered coercion/operator heads; they are not reconstructible typed Lean terms.
Changing those rules requires a fresh database and downstream artifacts.

For manual selection:

```sh
pixi run find-files --output data/candidate-files.txt
pixi run proof-states --files-from data/candidate-files.txt --db data/manual.db --workers 8 --timing
```

Lines beginning with `#` are ignored, and completed files are skipped on resume.
Databases embed their dictionaries and record per-file statistics in a CSV log.

## Splits and features

Keep whole theorems together and reserve the outer test partition for final evaluation.
Supply an exact tactic-kind label policy with `kinds`, `unmapped`, and, for fallback labeling, `other_label`.

```sh
pixi run partition-corpus --db data/mathlib.db --labels labels.json --output data/split --test-frac 0.1 --seed 1
pixi run prepare-features --db data/split/train.db --labels labels.json --output data/learning --dims 1000000 --workers 10
```

Preparation reuses candidates, assigns train/validation roles, fits on training theorems only, and converts each learning instance.
Compatible completed stages are reused.
Each `instance-01` directory contains `vocab.zst`, `features.zst`, `feature-stats.zst`, and `split.json`.
Use `--folds` for grouped validation folds or `--validation-frac` for a holdout.

Convert held-out data without fitting or changing columns:

```sh
pixi run prepare-features --db data/split/test.db --vocab data/learning/instance-01/vocab.zst --output data/test-features --workers 10
```

Frozen conversion obtains labels from the vocabulary and writes `features.zst` and `feature-stats.zst`.
An explicit label policy must match; fitting-only flags are rejected.
The loaders in `trustmebro.learning` read cached sparse batches or complete CSR matrices without constructing graph features.

## Models and results

```sh
pixi run train-logistic --instance data/learning/instance-01 --verbose
pixi run train-mlp --config src/trustmebro/mlp/example.json
pixi run plot-results --mlp data/experiments/mlp-followup --logistic data/experiments/logreg --output data/results
```

MLP plans specify an `instance`, `output`, shared `defaults`, and one or more named `runs`; see the [MLP guide](src/trustmebro/mlp/README.md).
Both models fit scaling on training data and select saved checkpoints by validation cross-entropy.
Reports include accuracy, top-3/top-5 accuracy, macro-F1, per-family metrics, confusion matrices, and learning histories.
New checkpoints also have a lightweight `.report.json` sidecar; existing checkpoints remain usable without one.
Explicit MLP evaluation-only JSON plans load a saved checkpoint and frozen test features without fitting or selecting another checkpoint.
Only load trusted checkpoints; Joblib files can execute code.
Result plots use saved reports, never retrain or infer, and can be selected with `--plots`.

## Corpus graphs

Use a completed, immutable source database; simultaneous extraction or recompression and analysis is unsupported.

```sh
pixi run graphs --db data/mathlib.db --stats data/analysis --output data/graphs --workers 4
```

The command discovers current statistics and computes only missing prerequisites.
Complete statistics can be rendered without the source database.
Use `--replace` to regenerate prerequisites of selected families, `--graphs` to select families, and `--dot-diameter` to adjust dots.
Ordinary help lists the families.
Use `--inventory` with a frozen vocabulary for selected-shape plots and `--feature-stats` for empirical activation/coverage.
Regenerate numerical analysis blocks and embedding products together after this refactor; databases and learning artifacts do not need regeneration.
Embedding diagnostic exports share `structural-observations.npz`; mode-specific files retain transformed inputs and coordinates.
UMAP is exploratory, not evidence of semantic clusters.

## Development

```sh
pixi run python tests/run.py
pixi run ruff check src tests
pixi run ruff format --check src tests
```
