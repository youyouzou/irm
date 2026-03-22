# IRM DG v2 Workflow

## What is added
- `resolved_config.json` is saved at training start in each `output_dir`.
- `run_summary.json` is saved at training end in each `output_dir`.
- Global index file is appended to `experiments/experiment_index.csv`.
- `CSIClassifier` now supports `model_variant`:
  - `baseline` (default, backward compatible)
  - `msstem` (multi-scale temporal stem)

## Required summary fields
Each `run_summary.json` contains:
- `algorithm`
- `model_variant`
- `seed`
- `best_epoch`
- `selected_val_acc`
- `val_acc`
- `val_worst_env_acc`
- `test_acc_selected`
- `test_acc_overall`
- `test_acc_worst_env`
- `checkpoint_paths`
- `timestamp`

## Run the paired ERM/IRM 12-group plan
Use the driver script (paired mode is default):

```bash
python -m scripts.run_irm_dgv2_plan --irm_config train/config.irm_dg.json --erm_config train/config.erm_dg.json --output_root experiments/irm_dgv2 --device cuda
```

Dry-run to inspect commands:

```bash
python -m scripts.run_irm_dgv2_plan --dry_run
```

Outputs:
- `experiments/irm_dgv2/plan_results.json`
- `experiments/irm_dgv2/plan_results.csv`
- `experiments/irm_dgv2/paired_compare.csv`
- `experiments/irm_dgv2/final_report.json`

Optional (not recommended): run IRM only.

```bash
python -m scripts.run_irm_dgv2_plan --skip_erm
```

## Evaluate checkpoint (baseline or msstem)
`eval_checkpoint.py` auto-detects `model_variant` from checkpoint args:

```bash
python scripts/eval_checkpoint.py --checkpoint experiments_gpu5/irm_dg_tuned/best_model.pt
```
