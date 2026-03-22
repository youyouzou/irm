import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


STAGE_A_GRID: List[Dict[str, Any]] = [
    {"lr": 3.5e-4, "weight_decay": 1e-4, "irm_lambda": 1.5, "penalty_anneal_epochs": 10, "penalty_ramp_epochs": 20},
    {"lr": 3.5e-4, "weight_decay": 1e-4, "irm_lambda": 2.0, "penalty_anneal_epochs": 10, "penalty_ramp_epochs": 20},
    {"lr": 3.0e-4, "weight_decay": 1e-4, "irm_lambda": 2.5, "penalty_anneal_epochs": 10, "penalty_ramp_epochs": 20},
    {"lr": 3.0e-4, "weight_decay": 1e-4, "irm_lambda": 3.0, "penalty_anneal_epochs": 10, "penalty_ramp_epochs": 20},
    {"lr": 3.5e-4, "weight_decay": 1e-4, "irm_lambda": 2.0, "penalty_anneal_epochs": 5, "penalty_ramp_epochs": 15},
    {"lr": 3.0e-4, "weight_decay": 1e-4, "irm_lambda": 2.5, "penalty_anneal_epochs": 5, "penalty_ramp_epochs": 15},
    {"lr": 3.5e-4, "weight_decay": 3e-4, "irm_lambda": 2.0, "penalty_anneal_epochs": 10, "penalty_ramp_epochs": 30},
    {"lr": 3.0e-4, "weight_decay": 3e-4, "irm_lambda": 2.5, "penalty_anneal_epochs": 10, "penalty_ramp_epochs": 30},
]

COMMON_OVERRIDES: Dict[str, Any] = {
    "batch_size": 16,
    "epochs": 140,
    "dropout": 0.45,
    "label_smoothing": 0.0,
    "val_selection_metric": "hybrid",
    "aug_prob": 0.55,
    "amp_scale_low": 0.85,
    "amp_scale_high": 1.15,
    "noise_std_ratio": 0.01,
    "time_mask_ratio": 0.05,
    "subcarrier_mask_ratio": 0.08,
    "model_variant": "baseline",
    "seed": 42,
}

IRM_ONLY_KEYS = {"irm_lambda", "penalty_anneal_epochs", "penalty_ramp_epochs", "val_selection_metric"}


@dataclass
class RunResult:
    algorithm: str
    stage: str
    pair_id: str
    run_name: str
    output_dir: str
    overrides: Dict[str, Any]
    summary: Dict[str, Any]


def _to_cli_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _run_command(cmd: List[str], dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def _load_summary(output_dir: str) -> Dict[str, Any]:
    summary_path = os.path.join(output_dir, "run_summary.json")
    if not os.path.isfile(summary_path):
        raise FileNotFoundError(f"run_summary.json not found: {summary_path}")
    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _filter_overrides_for_algorithm(overrides: Dict[str, Any], algorithm: str) -> Dict[str, Any]:
    if algorithm == "irm":
        return dict(overrides)
    if algorithm == "erm":
        return {k: v for k, v in overrides.items() if k not in IRM_ONLY_KEYS}
    raise ValueError(f"Unknown algorithm: {algorithm}")


def _train_one(
    *,
    algorithm: str,
    config_path: str,
    output_root: str,
    stage: str,
    pair_id: str,
    run_name: str,
    device: str,
    dry_run: bool,
    overrides: Dict[str, Any],
) -> RunResult:
    output_dir = os.path.join(output_root, stage, run_name)
    os.makedirs(output_dir, exist_ok=True)
    module = "train.train_irm" if algorithm == "irm" else "train.train_erm"
    cmd = [
        sys.executable,
        "-m",
        module,
        "--config",
        config_path,
        "--output_dir",
        output_dir,
        "--device",
        device,
    ]
    for k, v in overrides.items():
        cmd.extend([f"--{k}", _to_cli_value(v)])
    _run_command(cmd, dry_run=dry_run)
    summary = {} if dry_run else _load_summary(output_dir)
    return RunResult(
        algorithm=algorithm,
        stage=stage,
        pair_id=pair_id,
        run_name=run_name,
        output_dir=output_dir,
        overrides=overrides,
        summary=summary,
    )


def _train_paired(
    *,
    irm_config: str,
    erm_config: str,
    output_root: str,
    stage: str,
    pair_id: str,
    device: str,
    dry_run: bool,
    merged_overrides: Dict[str, Any],
    run_erm: bool,
) -> Tuple[RunResult, Optional[RunResult]]:
    irm_overrides = _filter_overrides_for_algorithm(merged_overrides, "irm")
    irm_result = _train_one(
        algorithm="irm",
        config_path=irm_config,
        output_root=output_root,
        stage=stage,
        pair_id=pair_id,
        run_name=f"{pair_id}_irm",
        device=device,
        dry_run=dry_run,
        overrides=irm_overrides,
    )

    erm_result: Optional[RunResult] = None
    if run_erm:
        erm_overrides = _filter_overrides_for_algorithm(merged_overrides, "erm")
        erm_result = _train_one(
            algorithm="erm",
            config_path=erm_config,
            output_root=output_root,
            stage=stage,
            pair_id=pair_id,
            run_name=f"{pair_id}_erm",
            device=device,
            dry_run=dry_run,
            overrides=erm_overrides,
        )
    return irm_result, erm_result


def _rank_key(summary: Dict[str, Any]) -> tuple:
    return (
        float(summary.get("selected_val_acc", -1.0)),
        float(summary.get("val_worst_env_acc", -1.0)),
    )


def _paired_rows(irm_runs: List[RunResult], erm_runs: List[RunResult]) -> List[Dict[str, Any]]:
    erm_by_key = {(r.stage, r.pair_id): r for r in erm_runs}
    rows: List[Dict[str, Any]] = []
    for irm in irm_runs:
        key = (irm.stage, irm.pair_id)
        erm = erm_by_key.get(key)
        irm_test = irm.summary.get("test_acc_selected")
        erm_test = erm.summary.get("test_acc_selected") if erm is not None else None
        gap = None
        if irm_test is not None and erm_test is not None:
            gap = float(irm_test) - float(erm_test)
        rows.append(
            {
                "stage": irm.stage,
                "pair_id": irm.pair_id,
                "irm_run_name": irm.run_name,
                "erm_run_name": erm.run_name if erm else None,
                "irm_selected_val_acc": irm.summary.get("selected_val_acc"),
                "erm_selected_val_acc": erm.summary.get("selected_val_acc") if erm else None,
                "irm_test_acc_selected": irm_test,
                "erm_test_acc_selected": erm_test,
                "test_acc_gap_irm_minus_erm": gap,
                "model_variant": irm.summary.get("model_variant"),
                "seed": irm.summary.get("seed"),
            }
        )
    return rows


def _paired_summary_stats(paired_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in paired_rows if r["test_acc_gap_irm_minus_erm"] is not None]
    gaps = [float(r["test_acc_gap_irm_minus_erm"]) for r in valid]
    irm_better = sum(1 for g in gaps if g > 0)
    erm_better = sum(1 for g in gaps if g < 0)
    ties = sum(1 for g in gaps if g == 0)
    return {
        "num_pairs": len(valid),
        "irm_better_count": irm_better,
        "erm_better_count": erm_better,
        "tie_count": ties,
        "mean_test_acc_gap_irm_minus_erm": statistics.mean(gaps) if gaps else None,
        "std_test_acc_gap_irm_minus_erm": statistics.stdev(gaps) if len(gaps) > 1 else (0.0 if gaps else None),
    }


def _save_results(
    output_root: str,
    all_runs: List[RunResult],
    paired_rows: List[Dict[str, Any]],
    final_report: Dict[str, Any],
) -> None:
    os.makedirs(output_root, exist_ok=True)
    json_path = os.path.join(output_root, "plan_results.json")
    csv_path = os.path.join(output_root, "plan_results.csv")
    paired_csv_path = os.path.join(output_root, "paired_compare.csv")
    report_path = os.path.join(output_root, "final_report.json")

    data = []
    for r in all_runs:
        data.append(
            {
                "algorithm": r.algorithm,
                "stage": r.stage,
                "pair_id": r.pair_id,
                "run_name": r.run_name,
                "output_dir": r.output_dir,
                "overrides": r.overrides,
                "summary": r.summary,
            }
        )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    headers = [
        "algorithm",
        "stage",
        "pair_id",
        "run_name",
        "output_dir",
        "selected_val_acc",
        "val_acc",
        "val_worst_env_acc",
        "test_acc_selected",
        "test_acc_overall",
        "test_acc_worst_env",
        "model_variant",
        "seed",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in all_runs:
            s = r.summary
            writer.writerow(
                {
                    "algorithm": r.algorithm,
                    "stage": r.stage,
                    "pair_id": r.pair_id,
                    "run_name": r.run_name,
                    "output_dir": r.output_dir,
                    "selected_val_acc": s.get("selected_val_acc"),
                    "val_acc": s.get("val_acc"),
                    "val_worst_env_acc": s.get("val_worst_env_acc"),
                    "test_acc_selected": s.get("test_acc_selected"),
                    "test_acc_overall": s.get("test_acc_overall"),
                    "test_acc_worst_env": s.get("test_acc_worst_env"),
                    "model_variant": s.get("model_variant"),
                    "seed": s.get("seed"),
                }
            )

    if paired_rows:
        paired_headers = list(paired_rows[0].keys())
        with open(paired_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=paired_headers)
            writer.writeheader()
            writer.writerows(paired_rows)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run paired ERM/IRM DG v2 plan: same conditions, directly comparable."
    )
    parser.add_argument("--irm_config", type=str, default="train/config.irm_dg.json")
    parser.add_argument("--erm_config", type=str, default="train/config.erm_dg.json")
    parser.add_argument("--output_root", type=str, default="experiments/irm_dgv2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true", help="Print commands only, do not execute.")
    parser.add_argument("--skip_erm", action="store_true", help="Run only IRM (not recommended).")
    args = parser.parse_args()

    run_erm = not args.skip_erm
    all_runs: List[RunResult] = []
    irm_runs: List[RunResult] = []
    erm_runs: List[RunResult] = []

    # Stage A: 8 baseline pairs
    stage_a_irm: List[RunResult] = []
    for idx, grid_item in enumerate(STAGE_A_GRID, start=1):
        pair_id = f"a{idx}"
        merged = dict(COMMON_OVERRIDES)
        merged.update(grid_item)
        irm_result, erm_result = _train_paired(
            irm_config=args.irm_config,
            erm_config=args.erm_config,
            output_root=args.output_root,
            stage="stage_a",
            pair_id=pair_id,
            device=args.device,
            dry_run=args.dry_run,
            merged_overrides=merged,
            run_erm=run_erm,
        )
        stage_a_irm.append(irm_result)
        irm_runs.append(irm_result)
        all_runs.append(irm_result)
        if erm_result is not None:
            erm_runs.append(erm_result)
            all_runs.append(erm_result)

    if args.dry_run:
        print("Dry-run finished. No training executed.")
        return

    ranked_a = sorted(stage_a_irm, key=lambda r: _rank_key(r.summary), reverse=True)
    top2_a = ranked_a[:2]

    # Stage B: IRM top2 -> paired rerun with msstem
    stage_b_irm: List[RunResult] = []
    for idx, parent in enumerate(top2_a, start=1):
        pair_id = f"b{idx}"
        merged = dict(parent.overrides)
        merged["model_variant"] = "msstem"
        irm_result, erm_result = _train_paired(
            irm_config=args.irm_config,
            erm_config=args.erm_config,
            output_root=args.output_root,
            stage="stage_b",
            pair_id=pair_id,
            device=args.device,
            dry_run=False,
            merged_overrides=merged,
            run_erm=run_erm,
        )
        stage_b_irm.append(irm_result)
        irm_runs.append(irm_result)
        all_runs.append(irm_result)
        if erm_result is not None:
            erm_runs.append(erm_result)
            all_runs.append(erm_result)

    best_b = sorted(stage_b_irm, key=lambda r: _rank_key(r.summary), reverse=True)[0]

    # Stage C: best stage B config, paired seed robustness
    stage_c_irm: List[RunResult] = []
    stage_c_erm: List[RunResult] = []
    for seed in (7, 99):
        pair_id = f"c_seed{seed}"
        merged = dict(best_b.overrides)
        merged["seed"] = seed
        irm_result, erm_result = _train_paired(
            irm_config=args.irm_config,
            erm_config=args.erm_config,
            output_root=args.output_root,
            stage="stage_c",
            pair_id=pair_id,
            device=args.device,
            dry_run=False,
            merged_overrides=merged,
            run_erm=run_erm,
        )
        stage_c_irm.append(irm_result)
        irm_runs.append(irm_result)
        all_runs.append(irm_result)
        if erm_result is not None:
            stage_c_erm.append(erm_result)
            erm_runs.append(erm_result)
            all_runs.append(erm_result)

    irm_group = [best_b] + stage_c_irm
    irm_scores = [float(r.summary["test_acc_selected"]) for r in irm_group]
    erm_group_scores: List[float] = []
    if run_erm:
        stage_b_erm = [r for r in erm_runs if r.stage == "stage_b" and r.pair_id == best_b.pair_id]
        erm_group_runs = stage_b_erm + stage_c_erm
        erm_group_scores = [float(r.summary["test_acc_selected"]) for r in erm_group_runs]

    paired_rows = _paired_rows(irm_runs, erm_runs) if run_erm else []
    paired_stats = _paired_summary_stats(paired_rows) if run_erm else {}

    final_report = {
        "selection_policy": {
            "stage_a_to_b": "rank by IRM selected_val_acc, tie-break by IRM val_worst_env_acc",
            "stage_c": "best Stage B IRM config, seeds=[7, 99]",
            "pairing": "ERM and IRM run under matched settings for direct comparison",
        },
        "best_irm_group": [
            {
                "run_name": r.run_name,
                "stage": r.stage,
                "pair_id": r.pair_id,
                "output_dir": r.output_dir,
                "seed": r.summary.get("seed"),
                "test_acc_selected": r.summary.get("test_acc_selected"),
                "selected_val_acc": r.summary.get("selected_val_acc"),
                "model_variant": r.summary.get("model_variant"),
            }
            for r in irm_group
        ],
        "irm_group_metrics": {
            "test_acc_selected_mean": statistics.mean(irm_scores),
            "test_acc_selected_std": statistics.stdev(irm_scores) if len(irm_scores) > 1 else 0.0,
        },
        "erm_group_metrics": {
            "test_acc_selected_mean": statistics.mean(erm_group_scores) if erm_group_scores else None,
            "test_acc_selected_std": (
                statistics.stdev(erm_group_scores) if len(erm_group_scores) > 1 else (0.0 if erm_group_scores else None)
            ),
        },
        "paired_summary": paired_stats,
    }

    _save_results(args.output_root, all_runs, paired_rows, final_report)
    print(f"Paired plan completed. Results saved under: {args.output_root}")


if __name__ == "__main__":
    main()
