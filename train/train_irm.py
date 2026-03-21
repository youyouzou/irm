import argparse

import torch

from train.config_utils import load_config_file
from train.trainer import run_irm


def str2bool(value: str) -> bool:
    v = value.lower()
    if v in {"1", "true", "yes", "y"}:
        return True
    if v in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IRM training using unified trainer.")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML config.")
    parser.add_argument("--train_npz", type=str, default="data/processed/train.npz")
    parser.add_argument("--val_npz", type=str, default="data/processed/val.npz")
    parser.add_argument("--test_npz", type=str, default="data/processed/test.npz")
    parser.add_argument("--num_classes", type=int, required=True)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--irm_lambda", type=float, default=2.0, help="IRM penalty coefficient.")
    parser.add_argument("--penalty_anneal_epochs", type=int, default=15)
    parser.add_argument("--penalty_ramp_epochs", type=int, default=30)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--early_stop_patience", type=int, default=30)
    parser.add_argument(
        "--val_selection_metric",
        type=str,
        default="overall",
        choices=["overall", "worst_env", "hybrid"],
    )

    parser.add_argument("--input_norm", type=str, default="sample_zscore", choices=["none", "sample_zscore"])
    parser.add_argument("--use_augment", type=str2bool, default=True)
    parser.add_argument("--aug_prob", type=float, default=0.5)
    parser.add_argument("--amp_scale_low", type=float, default=0.85)
    parser.add_argument("--amp_scale_high", type=float, default=1.15)
    parser.add_argument("--noise_std_ratio", type=float, default=0.015)
    parser.add_argument("--time_mask_ratio", type=float, default=0.08)
    parser.add_argument("--subcarrier_mask_ratio", type=float, default=0.12)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="experiments/irm")
    return parser


def parse_args():
    parser = build_parser()
    early_args, _ = parser.parse_known_args()
    if early_args.config:
        parser.set_defaults(**load_config_file(early_args.config))
    return parser.parse_args()


def main():
    args = parse_args()
    run_irm(args)


if __name__ == "__main__":
    main()
