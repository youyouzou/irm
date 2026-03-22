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

from datasets.wifi_dataset import WiFiCSIDataset
from models.cnn import CSIClassifier
from train.trainer import evaluate


def _build_args_from_checkpoint(
    ckpt_args: Dict[str, Any],
    cli_args: argparse.Namespace,
    device: str,
) -> SimpleNamespace:
    """
    根据 checkpoint 中保存的 args 和命令行参数，构造一个用于测试的简化 args。

    优先级：命令行 > checkpoint 默认。
    """
    def get(key: str, default: Any = None) -> Any:
        if hasattr(cli_args, key) and getattr(cli_args, key) is not None:
            return getattr(cli_args, key)
        if key in ckpt_args:
            return ckpt_args[key]
        return default

    return SimpleNamespace(
        test_npz=get("test_npz", "data/processed/test.npz"),
        batch_size=get("batch_size", 64),
        num_workers=get("num_workers", 0),
        num_classes=get("num_classes"),
        dropout=get("dropout", 0.3),
        model_variant=get("model_variant", "baseline"),
        device=device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ERM model on test set.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/erm/best_model.pt",
        help="Path to ERM checkpoint (best_model.pt).",
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
        help="Optional: override num_classes (usually keep as in checkpoint).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Optional: override dropout (usually keep as in checkpoint).",
    )
    parser.add_argument(
        "--model_variant",
        type=str,
        default=None,
        choices=["baseline", "msstem"],
        help="Optional: override model variant.",
    )

    cli_args = parser.parse_args()

    device = cli_args.device
    print(f"Using device: {device}")

    # 1) 加载 checkpoint
    ckpt = torch.load(cli_args.checkpoint, map_location=device)
    ckpt_args = ckpt.get("args", {})
    if not isinstance(ckpt_args, dict):
        raise ValueError("Checkpoint 'args' field is expected to be a dict.")

    # 2) 构造测试所需的 args（组合 checkpoint 与命令行）
    args = _build_args_from_checkpoint(ckpt_args, cli_args, device)

    if args.num_classes is None:
        raise ValueError(
            "num_classes is not found in checkpoint args and not provided via CLI."
        )

    print(f"Test NPZ: {args.test_npz}")
    print(f"Batch size: {args.batch_size}")
    print(f"Num workers: {args.num_workers}")
    print(f"Num classes: {args.num_classes}")
    print(f"Dropout: {args.dropout}")
    print(f"Model variant: {args.model_variant}")

    # 3) 构建测试集 DataLoader
    test_ds = WiFiCSIDataset(args.test_npz)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    # 4) 构建模型并加载权重
    model = CSIClassifier(
        num_classes=args.num_classes,
        dropout=args.dropout,
        model_variant=args.model_variant,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    # 5) 在测试集上评估
    test_acc = evaluate(model, test_loader, device)
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()

