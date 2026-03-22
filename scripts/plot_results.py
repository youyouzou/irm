import argparse
import csv
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def load_logs(path: Path) -> Dict[str, List[float]]:
    logs: Dict[str, List[float]] = {}
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                if k not in logs:
                    logs[k] = []
                # epoch 是整数，其他是浮点数
                if k == "epoch":
                    logs[k].append(int(v))
                else:
                    logs[k].append(float(v))
    return logs


def load_run_summary(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def plot_erm(log_path: Path, output_dir: Path) -> None:
    logs = load_logs(log_path)
    epochs = logs["epoch"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) train/val acc
    plt.figure()
    plt.plot(epochs, logs["train_acc"], label="train_acc")
    plt.plot(epochs, logs["val_acc"], label="val_acc")
    if "test_acc" in logs:
        plt.plot(epochs, logs["test_acc"], label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("ERM Accuracy Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "erm_accuracy.png", dpi=200)
    plt.close()

    # 2) train loss
    plt.figure()
    plt.plot(epochs, logs["train_loss"], label="train_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("ERM Training Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "erm_train_loss.png", dpi=200)
    plt.close()


def plot_irm(log_path: Path, output_dir: Path) -> None:
    logs = load_logs(log_path)
    epochs = logs["epoch"]

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) val acc
    plt.figure()
    plt.plot(epochs, logs["val_acc"], label="val_acc")
    if "test_acc" in logs:
        plt.plot(epochs, logs["test_acc"], label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("IRM Accuracy Curves")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "irm_accuracy.png", dpi=200)
    plt.close()

    # 2) loss & penalty
    plt.figure()
    plt.plot(epochs, logs["erm_loss"], label="erm_loss")
    plt.plot(epochs, logs["penalty"], label="penalty")
    plt.xlabel("Epoch")
    plt.ylabel("Loss / Penalty")
    plt.title("IRM Loss and Penalty")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "irm_loss_penalty.png", dpi=200)
    plt.close()


def plot_compare_erm_irm(
    erm_log_path: Path,
    irm_log_path: Path,
    output_dir: Path,
) -> None:
    erm_logs = load_logs(erm_log_path)
    irm_logs = load_logs(irm_log_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    # 优先比较 test_acc，如果没有则比较 val_acc
    metric = None
    if "test_acc" in erm_logs and "test_acc" in irm_logs:
        metric = "test_acc"
        ylabel = "Test Accuracy"
        title = "ERM vs IRM Test Accuracy"
    elif "val_acc" in erm_logs and "val_acc" in irm_logs:
        metric = "val_acc"
        ylabel = "Validation Accuracy"
        title = "ERM vs IRM Validation Accuracy"

    if metric is None:
        print("No common metric (test_acc or val_acc) found for ERM vs IRM comparison.")
        return

    plt.figure()
    plt.plot(erm_logs["epoch"], erm_logs[metric], label=f"ERM {metric}")
    plt.plot(irm_logs["epoch"], irm_logs[metric], label=f"IRM {metric}")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_dir / "erm_vs_irm_compare.png", dpi=200)
    plt.close()


def summarize_logs(erm_log_path: Path, irm_log_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    lines = []

    if erm_log_path.is_file():
        erm = load_logs(erm_log_path)
        erm_summary = load_run_summary(erm_log_path.parent / "run_summary.json")
        if "train_acc" in erm and erm["train_acc"]:
            best_train = max(erm["train_acc"])
            best_train_epoch = erm["epoch"][erm["train_acc"].index(best_train)]
            lines.append(f"ERM best_train_acc: {best_train:.6f} @ epoch {best_train_epoch}")
        if "val_acc" in erm and erm["val_acc"]:
            best_val = max(erm["val_acc"])
            best_val_epoch = erm["epoch"][erm["val_acc"].index(best_val)]
            lines.append(f"ERM best_val_acc:   {best_val:.6f} @ epoch {best_val_epoch}")
        if "selected_val_acc" in erm and erm["selected_val_acc"]:
            best_sel = max(erm["selected_val_acc"])
            best_sel_epoch = erm["epoch"][erm["selected_val_acc"].index(best_sel)]
            lines.append(f"ERM best_selected:  {best_sel:.6f} @ epoch {best_sel_epoch}")
        if erm_summary:
            lines.append(
                "ERM test(selected/overall/worst): "
                f"{erm_summary.get('test_acc_selected')} / "
                f"{erm_summary.get('test_acc_overall')} / "
                f"{erm_summary.get('test_acc_worst_env')}"
            )

    if irm_log_path.is_file():
        irm = load_logs(irm_log_path)
        irm_summary = load_run_summary(irm_log_path.parent / "run_summary.json")
        if "train_acc" in irm and irm["train_acc"]:
            best_train = max(irm["train_acc"])
            best_train_epoch = irm["epoch"][irm["train_acc"].index(best_train)]
            lines.append(f"IRM best_train_acc: {best_train:.6f} @ epoch {best_train_epoch}")
        if "val_acc" in irm and irm["val_acc"]:
            best_val = max(irm["val_acc"])
            best_val_epoch = irm["epoch"][irm["val_acc"].index(best_val)]
            lines.append(f"IRM best_val_acc:   {best_val:.6f} @ epoch {best_val_epoch}")
        if "selected_val_acc" in irm and irm["selected_val_acc"]:
            best_sel = max(irm["selected_val_acc"])
            best_sel_epoch = irm["epoch"][irm["selected_val_acc"].index(best_sel)]
            lines.append(f"IRM best_selected:  {best_sel:.6f} @ epoch {best_sel_epoch}")
        if "val_hybrid_acc" in irm and irm["val_hybrid_acc"]:
            best_hybrid = max(irm["val_hybrid_acc"])
            best_hybrid_epoch = irm["epoch"][irm["val_hybrid_acc"].index(best_hybrid)]
            lines.append(f"IRM best_hybrid:    {best_hybrid:.6f} @ epoch {best_hybrid_epoch}")
        if "val_worst_env_acc" in irm and irm["val_worst_env_acc"]:
            best_worst = max(irm["val_worst_env_acc"])
            best_worst_epoch = irm["epoch"][irm["val_worst_env_acc"].index(best_worst)]
            lines.append(f"IRM best_worst_env: {best_worst:.6f} @ epoch {best_worst_epoch}")
        if irm_summary:
            lines.append(
                "IRM test(selected/overall/worst): "
                f"{irm_summary.get('test_acc_selected')} / "
                f"{irm_summary.get('test_acc_overall')} / "
                f"{irm_summary.get('test_acc_worst_env')}"
            )

    (output_dir / "summary.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ERM/IRM training curves.")
    parser.add_argument(
        "--erm_logs",
        type=str,
        default="experiments/erm/logs.csv",
        help="Path to ERM logs.csv",
    )
    parser.add_argument(
        "--irm_logs",
        type=str,
        default="experiments/irm/logs.csv",
        help="Path to IRM logs.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/plots",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    erm_log_path = Path(args.erm_logs)
    irm_log_path = Path(args.irm_logs)
    output_dir = Path(args.output_dir)

    if erm_log_path.is_file():
        plot_erm(erm_log_path, output_dir)
    else:
        print(f"ERM log file not found: {erm_log_path}")

    if irm_log_path.is_file():
        plot_irm(irm_log_path, output_dir)
    else:
        print(f"IRM log file not found: {irm_log_path}")

    if erm_log_path.is_file() and irm_log_path.is_file():
        plot_compare_erm_irm(erm_log_path, irm_log_path, output_dir)
    summarize_logs(erm_log_path, irm_log_path, output_dir)


if __name__ == "__main__":
    main()
