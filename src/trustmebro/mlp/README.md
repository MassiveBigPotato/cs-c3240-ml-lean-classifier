# PyTorch MLP

Run an explicit JSON experiment plan with `pixi run --locked train-mlp --config src/trustmebro/mlp/example.json`.
Paths are relative to the invoking directory.

Plans contain `instance`, `output`, optional `defaults` and `max_bytes`, and a nonempty `runs` list.
Each run has a unique `name` and optional overrides for `hidden`, `dropout`, `batch_rows`, `epochs`, `patience`, `min_delta`, `lr`, `weight_decay`, `seed`, `device`, `scaling`, `class_weight`, and `label_smoothing`.
Settings omitted from both defaults and a run inherit the model configuration.
Scaling choices are `none`, `maxabs`, `standard`, `log_maxabs`, and `log_standard`; class weights are `none`, `balanced`, or `inverse_sqrt`.
Input instances contain `features.zst`, `split.json`, and `vocab.zst`.

Inputs load once, and runs sharing a scaling policy reuse training-fitted preprocessing.
Each run writes `config.json`, `results.json`, and `checkpoint.pt`; weighted runs also save `checkpoint.weighted.pt`.
The primary checkpoint minimizes unweighted validation cross-entropy, not macro-F1 or the training objective.
Results report metrics at the selected epoch, histories, split/vocabulary identity, and durations; `summary.csv` compares completed runs.
Rerun a compatible plan to reuse completed runs and retry incomplete ones.
Use one writer per output directory and never modify inputs during a run.
`max_bytes` bounds each partition's numeric buffers, not total memory or model weights.

## Evaluation only

Set `operation` to `evaluate` explicitly and supply these fields in a JSON file:

```json
{
  "operation": "evaluate",
  "features": "data/test-features/features.zst",
  "vocab": "data/learning/instance-01/vocab.zst",
  "checkpoint": "data/experiments/mlp-followup/inverse-sqrt-dropout/checkpoint.pt",
  "report": "data/experiments/mlp-followup/inverse-sqrt-dropout/results.json",
  "split": "data/learning/instance-01/split.json",
  "test_db": "data/split/test.db",
  "output": "data/test-results.json",
  "device": "cuda",
  "batch_rows": 8192
}
```

Run it with the same `train-mlp --config` entry point.
This checks checkpoint/vocabulary identity, class order, frozen scaling, complete test rows, and theorem isolation before inference.
It neither fits nor selects anything, and refuses an existing output.
Training never evaluates test data automatically.
