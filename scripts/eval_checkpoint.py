import argparse
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets.wifi_dataset import WiFiCSIDataset, build_csi_transform
from models.cnn import CSIClassifier
from train.trainer import evaluate, evaluate_by_env


def _build_args_from_checkpoint(
    ckpt_args: Dict[str, Any],
    cli_args: argparse.Namespace,
    device: str,
) -> SimpleNamespace:
    def get(key: str, default: Any = None) -> Any:
        if hasattr(cli_args, key) and getattr(cli_args, key) is not None:
            return getattr(cli_args, key)
        if key in ckpt_args:
            return ckpt_args[key]
        return default

    return SimpleNamespace(
        test_npz=get("test_npz", "data/processed/test.npz"),
        batch_size=get("batch_size", 64),
        # Default to single-process loading for Windows-safe standalone eval.
        num_workers=get("num_workers", 0) if cli_args.num_workers is not None else 0,
        num_classes=get("num_classes"),
        dropout=get("dropout", 0.3),
        model_variant=get("model_variant", "baseline"),
        input_norm=get("input_norm", "sample_zscore"),
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved ERM/IRM checkpoint on the test set."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint, e.g. experiments_gpu7/irm_dg_tuned/best_model.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help='Device to use: "cuda" or "cpu".',
    )
    parser.add_argument(
        "--test_npz",
        type=str,
        default=None,
        help="Optional: override test NPZ path saved in checkpoint.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Optional: override batch size for evaluation.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Optional: override num_workers for DataLoader.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=None,
        help="Optional: override num_classes.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Optional: override dropout.",
    )

    cli_args = parser.parse_args()
    device = cli_args.device

    ckpt = torch.load(cli_args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        raise ValueError("Checkpoint 'args' field is expected to be a dict.")

    args = _build_args_from_checkpoint(ckpt_args, cli_args, device)
    if args.num_classes is None:
        raise ValueError(
            "num_classes is not found in checkpoint args and not provided via CLI."
        )

    eval_transform = build_csi_transform(
        normalize=args.input_norm,
        augment=False,
    )

    print(f"Using device: {device}")
    print(f"Checkpoint: {cli_args.checkpoint}")
    print(f"Test NPZ: {args.test_npz}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Num classes: {args.num_classes}")
    print(f"Dropout: {args.dropout}")
    print(f"Model variant: {args.model_variant}")
    print(f"Input norm: {args.input_norm}")

    test_ds = WiFiCSIDataset(args.test_npz, transform=None)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
    )

    model = CSIClassifier(
        num_classes=args.num_classes,
        dropout=args.dropout,
        model_variant=args.model_variant,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    test_acc = evaluate(model, test_loader, device, transform=eval_transform)
    overall_acc, worst_env_acc, per_env_acc = evaluate_by_env(
        model, test_loader, device, transform=eval_transform
    )

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test overall accuracy: {overall_acc:.4f}")
    print(f"Test worst-env accuracy: {worst_env_acc:.4f}")
    print("Per-env accuracy:")
    for env_id, env_acc in per_env_acc.items():
        print(f"  env {env_id}: {env_acc:.4f}")


if __name__ == "__main__":
    main()
