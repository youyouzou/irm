import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Dict, List


BASE_COMMON: Dict[str, Any] = {
    "batch_size": 16,
    "epochs": 140,
    "dropout": 0.45,
    "label_smoothing": 0.0,
    "aug_prob": 0.55,
    "amp_scale_low": 0.85,
    "amp_scale_high": 1.15,
    "noise_std_ratio": 0.01,
    "time_mask_ratio": 0.05,
    "subcarrier_mask_ratio": 0.08,
    "model_variant": "baseline",
    "seed": 42,
}

A2_IRM_BASE: Dict[str, Any] = {
    "lr": 3.5e-4,
    "weight_decay": 1e-4,
    "irm_lambda": 2.0,
    "penalty_anneal_epochs": 10,
    "penalty_ramp_epochs": 20,
    "val_selection_metric": "hybrid",
}

REFINE_GRID: List[Dict[str, Any]] = [
    {"id": "r1_base", "overrides": {}},
    {"id": "r2_lambda18", "overrides": {"irm_lambda": 1.8}},
    {"id": "r3_lambda22", "overrides": {"irm_lambda": 2.2}},
    {"id": "r4_ramp15", "overrides": {"penalty_ramp_epochs": 15}},
    {"id": "r5_ramp30", "overrides": {"penalty_ramp_epochs": 30}},
    {"id": "r6_wd3e4", "overrides": {"weight_decay": 3e-4}},
    {"id": "r7_lr3e4", "overrides": {"lr": 3e-4}},
    {
        "id": "r8_aug_plus",
        "overrides": {
            "aug_prob": 0.60,
            "noise_std_ratio": 0.012,
            "time_mask_ratio": 0.06,
            "subcarrier_mask_ratio": 0.10,
        },
    },
]

IRM_ONLY_KEYS = {"irm_lambda", "penalty_anneal_epochs", "penalty_ramp_epochs", "val_selection_metric"}


@dataclass
class RunResult:
    algorithm: str
    run_id: str
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


def _train_one(
    *,
    algorithm: str,
    config_path: str,
    output_dir: str,
    device: str,
    dry_run: bool,
    overrides: Dict[str, Any],
    run_id: str,
) -> RunResult:
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
        run_id=run_id,
        output_dir=output_dir,
        overrides=overrides,
        summary=summary,
    )


def _filter_for_erm(overrides: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in overrides.items() if k not in IRM_ONLY_KEYS}


def _save_outputs(output_root: str, runs: List[RunResult], paired_rows: List[Dict[str, Any]], final_report: Dict[str, Any]) -> None:
    os.makedirs(output_root, exist_ok=True)

    plan_csv = os.path.join(output_root, "refine_results.csv")
    paired_csv = os.path.join(output_root, "refine_paired_compare.csv")
    report_json = os.path.join(output_root, "refine_final_report.json")
    full_json = os.path.join(output_root, "refine_results.json")

    with open(plan_csv, "w", newline="", encoding="utf-8") as f:
        headers = [
            "algorithm",
            "run_id",
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
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in runs:
            s = r.summary
            writer.writerow(
                {
                    "algorithm": r.algorithm,
                    "run_id": r.run_id,
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
        with open(paired_csv, "w", newline="", encoding="utf-8") as f:
            headers = list(paired_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(paired_rows)

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    full_payload = [
        {
            "algorithm": r.algorithm,
            "run_id": r.run_id,
            "output_dir": r.output_dir,
            "overrides": r.overrides,
            "summary": r.summary,
        }
        for r in runs
    ]
    with open(full_json, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="A2-centered paired ERM/IRM refinement plan.")
    parser.add_argument("--irm_config", type=str, default="train/config.irm_dg.json")
    parser.add_argument("--erm_config", type=str, default="train/config.erm_dg.json")
    parser.add_argument("--output_root", type=str, default="experiments/irm_a2_refine")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    runs: List[RunResult] = []
    paired_rows: List[Dict[str, Any]] = []

    for item in REFINE_GRID:
        run_id = item["id"]
        merged = dict(BASE_COMMON)
        merged.update(A2_IRM_BASE)
        merged.update(item["overrides"])

        irm_overrides = dict(merged)
        erm_overrides = _filter_for_erm(merged)

        irm_dir = os.path.join(args.output_root, run_id, "irm")
        erm_dir = os.path.join(args.output_root, run_id, "erm")

        irm_run = _train_one(
            algorithm="irm",
            config_path=args.irm_config,
            output_dir=irm_dir,
            device=args.device,
            dry_run=args.dry_run,
            overrides=irm_overrides,
            run_id=run_id,
        )
        erm_run = _train_one(
            algorithm="erm",
            config_path=args.erm_config,
            output_dir=erm_dir,
            device=args.device,
            dry_run=args.dry_run,
            overrides=erm_overrides,
            run_id=run_id,
        )

        runs.extend([irm_run, erm_run])

        if not args.dry_run:
            irm_test = float(irm_run.summary.get("test_acc_selected", 0.0))
            erm_test = float(erm_run.summary.get("test_acc_selected", 0.0))
            paired_rows.append(
                {
                    "run_id": run_id,
                    "irm_run_dir": irm_dir,
                    "erm_run_dir": erm_dir,
                    "irm_selected_val_acc": irm_run.summary.get("selected_val_acc"),
                    "erm_selected_val_acc": erm_run.summary.get("selected_val_acc"),
                    "irm_test_acc_selected": irm_test,
                    "erm_test_acc_selected": erm_test,
                    "test_acc_gap_irm_minus_erm": irm_test - erm_test,
                    "model_variant": irm_run.summary.get("model_variant"),
                    "seed": irm_run.summary.get("seed"),
                }
            )

    if args.dry_run:
        print("Dry-run finished. No training executed.")
        return

    irm_tests = [float(r.summary["test_acc_selected"]) for r in runs if r.algorithm == "irm"]
    erm_tests = [float(r.summary["test_acc_selected"]) for r in runs if r.algorithm == "erm"]
    gaps = [float(r["test_acc_gap_irm_minus_erm"]) for r in paired_rows]

    best_irm = sorted(
        [r for r in runs if r.algorithm == "irm"],
        key=lambda x: float(x.summary.get("test_acc_selected", -1.0)),
        reverse=True,
    )[0]

    final_report = {
        "baseline": "A2",
        "selection_policy": "rank by IRM test_acc_selected under paired ERM comparison",
        "best_irm_run": {
            "run_id": best_irm.run_id,
            "output_dir": best_irm.output_dir,
            "test_acc_selected": best_irm.summary.get("test_acc_selected"),
            "selected_val_acc": best_irm.summary.get("selected_val_acc"),
            "seed": best_irm.summary.get("seed"),
            "model_variant": best_irm.summary.get("model_variant"),
        },
        "irm_metrics": {
            "mean_test_acc_selected": statistics.mean(irm_tests),
            "std_test_acc_selected": statistics.stdev(irm_tests) if len(irm_tests) > 1 else 0.0,
        },
        "erm_metrics": {
            "mean_test_acc_selected": statistics.mean(erm_tests),
            "std_test_acc_selected": statistics.stdev(erm_tests) if len(erm_tests) > 1 else 0.0,
        },
        "paired_summary": {
            "num_pairs": len(gaps),
            "irm_better_count": sum(1 for g in gaps if g > 0),
            "erm_better_count": sum(1 for g in gaps if g < 0),
            "tie_count": sum(1 for g in gaps if g == 0),
            "mean_test_acc_gap_irm_minus_erm": statistics.mean(gaps),
            "std_test_acc_gap_irm_minus_erm": statistics.stdev(gaps) if len(gaps) > 1 else 0.0,
        },
    }

    _save_outputs(args.output_root, runs, paired_rows, final_report)
    print(f"A2 refinement completed. Results saved under: {args.output_root}")


if __name__ == "__main__":
    main()
