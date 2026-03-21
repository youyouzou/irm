import copy
import csv
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from datasets.wifi_dataset import WiFiCSIDataset, build_csi_transform, build_dataloaders_for_envs
from losses.irm_loss import aggregate_irm_loss
from models.cnn import CSIClassifier


@dataclass
class TrainerArtifacts:
    log_path: str
    best_model_path: str


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        ema_state = self.ema.state_dict()
        model_state = model.state_dict()
        for k, v in ema_state.items():
            if k not in model_state:
                continue
            src = model_state[k].detach()
            if torch.is_floating_point(v):
                v.mul_(self.decay).add_(src, alpha=1.0 - self.decay)
            else:
                v.copy_(src)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _maybe_to_device(x: torch.Tensor, device: str) -> torch.Tensor:
    if device.startswith("cuda"):
        return x.to(device, non_blocking=True)
    return x.to(device)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y, _env in loader:
            x = _maybe_to_device(x, device)
            y = _maybe_to_device(y, device)
            if transform is not None:
                x = transform(x)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def evaluate_by_env(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[float, float, Dict[int, float]]:
    model.eval()
    correct_by_env: Dict[int, int] = {}
    total_by_env: Dict[int, int] = {}
    with torch.no_grad():
        for x, y, env in loader:
            env_list = [int(e) for e in env.tolist()]
            x = _maybe_to_device(x, device)
            y = _maybe_to_device(y, device)
            if transform is not None:
                x = transform(x)
            logits = model(x)
            preds = logits.argmax(dim=1)
            matched = preds.eq(y).detach().cpu().tolist()
            for ok, e in zip(matched, env_list):
                correct_by_env[e] = correct_by_env.get(e, 0) + int(ok)
                total_by_env[e] = total_by_env.get(e, 0) + 1
    per_env_acc = {
        e: correct_by_env[e] / max(total_by_env[e], 1)
        for e in sorted(total_by_env.keys())
    }
    total_correct = sum(correct_by_env.values())
    total_count = sum(total_by_env.values())
    overall_acc = total_correct / max(total_count, 1)
    worst_env_acc = min(per_env_acc.values()) if per_env_acc else 0.0
    return overall_acc, worst_env_acc, per_env_acc


def infinite_loader(loader: DataLoader) -> Iterator:
    while True:
        for batch in loader:
            yield batch


def _prepare_artifacts(output_dir: str) -> TrainerArtifacts:
    os.makedirs(output_dir, exist_ok=True)
    return TrainerArtifacts(
        log_path=os.path.join(output_dir, "logs.csv"),
        best_model_path=os.path.join(output_dir, "best_model.pt"),
    )


def _metadata_path_for(model_path: str) -> str:
    stem, _ = os.path.splitext(model_path)
    return f"{stem}.json"


def _write_best_model_metadata(model_path: str, args, **metrics) -> None:
    payload = {
        "model_path": model_path,
        "hyperparameters": dict(vars(args)),
    }
    payload.update(metrics)
    with open(_metadata_path_for(model_path), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _build_transforms(args) -> Tuple:
    input_norm = getattr(args, "input_norm", "sample_zscore")
    use_augment = getattr(args, "use_augment", True)
    aug_prob = getattr(args, "aug_prob", 0.5)
    amp_scale_low = getattr(args, "amp_scale_low", 0.9)
    amp_scale_high = getattr(args, "amp_scale_high", 1.1)
    noise_std_ratio = getattr(args, "noise_std_ratio", 0.01)
    time_mask_ratio = getattr(args, "time_mask_ratio", 0.0)
    subcarrier_mask_ratio = getattr(args, "subcarrier_mask_ratio", 0.0)

    train_transform = build_csi_transform(
        normalize=input_norm,
        augment=use_augment,
        aug_prob=aug_prob,
        amp_scale_low=amp_scale_low,
        amp_scale_high=amp_scale_high,
        noise_std_ratio=noise_std_ratio,
        time_mask_ratio=time_mask_ratio,
        subcarrier_mask_ratio=subcarrier_mask_ratio,
    )
    eval_transform = build_csi_transform(
        normalize=input_norm,
        augment=False,
    )
    return train_transform, eval_transform


def _pin_memory_for(device: str) -> bool:
    return device.startswith("cuda")


def _compute_irm_weight(epoch: int, irm_lambda: float, anneal_epochs: int, ramp_epochs: int) -> float:
    if irm_lambda <= 1.0:
        return irm_lambda
    if epoch <= anneal_epochs:
        return 1.0
    if ramp_epochs <= 0:
        return irm_lambda
    progress = min(1.0, (epoch - anneal_epochs) / float(ramp_epochs))
    return 1.0 + progress * (irm_lambda - 1.0)


def _select_val_score(metric: str, overall: float, worst: float) -> float:
    if metric == "worst_env":
        return worst
    if metric == "overall":
        return overall
    if metric == "hybrid":
        return 0.5 * (overall + worst)
    raise ValueError(f"Unknown val_selection_metric: {metric}")


def _build_erm_loaders(args) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # NOTE: transforms are applied on GPU in the training loop to reduce CPU work.
    train_ds = WiFiCSIDataset(args.train_npz, transform=None)
    val_ds = WiFiCSIDataset(args.val_npz, transform=None)
    test_ds = WiFiCSIDataset(args.test_npz, transform=None)

    pin_memory = _pin_memory_for(args.device)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def _build_irm_loaders(args) -> Tuple[Dict[int, DataLoader], DataLoader, DataLoader]:
    # NOTE: transforms are applied on GPU in the training loop to reduce CPU work.
    train_ds = WiFiCSIDataset(args.train_npz, transform=None)
    val_ds = WiFiCSIDataset(args.val_npz, transform=None)
    test_ds = WiFiCSIDataset(args.test_npz, transform=None)

    env_loaders = build_dataloaders_for_envs(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    pin_memory = _pin_memory_for(args.device)
    # build_dataloaders_for_envs doesn't set pin_memory; we rebuild env loaders here if needed
    if pin_memory:
        env_loaders = {
            e: DataLoader(
                loader.dataset,
                batch_size=loader.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                drop_last=True,
                pin_memory=True,
            )
            for e, loader in env_loaders.items()
        }

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    return env_loaders, val_loader, test_loader


def run_erm(args) -> None:
    set_seed(args.seed)
    artifacts = _prepare_artifacts(args.output_dir)
    device = args.device
    print(f"Using device: {device}")

    train_loader, val_loader, test_loader = _build_erm_loaders(args)
    train_transform, eval_transform = _build_transforms(args)

    model = CSIClassifier(num_classes=args.num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)
    ema = ModelEMA(model, args.ema_decay) if args.ema_decay > 0 else None

    best_selected_val = 0.0
    best_train_acc = 0.0
    best_train_epoch = 0
    epochs_without_improve = 0

    with open(artifacts.log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            "epoch",
            "train_loss",
            "train_acc",
            "val_acc",
            "ema_val_acc",
            "selected_val_acc",
            "lr",
        ])

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, correct_train, total_train = 0.0, 0, 0

        for x, y, _env in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x = _maybe_to_device(x, device)
            y = _maybe_to_device(y, device)
            if train_transform is not None:
                x = train_transform(x)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            running_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct_train += (preds == y).sum().item()
            total_train += y.numel()

        train_loss = running_loss / max(total_train, 1)
        train_acc = correct_train / max(total_train, 1)
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_epoch = epoch
        val_acc = evaluate(model, val_loader, device, transform=eval_transform)
        ema_val_acc = evaluate(ema.ema, val_loader, device, transform=eval_transform) if ema is not None else val_acc
        selected_val_acc = max(val_acc, ema_val_acc)

        scheduler.step(selected_val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch}] train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_acc={val_acc:.4f}, ema_val_acc={ema_val_acc:.4f}, lr={current_lr:.6g}"
        )

        with open(artifacts.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [epoch, train_loss, train_acc, val_acc, ema_val_acc, selected_val_acc, current_lr]
            )

        if selected_val_acc > best_selected_val:
            best_selected_val = selected_val_acc
            epochs_without_improve = 0
            best_state = ema.ema.state_dict() if (ema is not None and ema_val_acc >= val_acc) else model.state_dict()
            checkpoint = {
                "model_state": best_state,
                "epoch": epoch,
                "val_acc": val_acc,
                "ema_val_acc": ema_val_acc,
                "selected_val_acc": selected_val_acc,
                "best_train_acc": best_train_acc,
                "best_train_epoch": best_train_epoch,
                "args": vars(args),
                "trainer_mode": "erm",
            }
            torch.save(
                checkpoint,
                artifacts.best_model_path,
            )
            _write_best_model_metadata(
                artifacts.best_model_path,
                args,
                trainer_mode="erm",
                best_epoch=epoch,
                val_acc=val_acc,
                ema_val_acc=ema_val_acc,
                selected_val_acc=selected_val_acc,
                best_train_acc=best_train_acc,
                best_train_epoch=best_train_epoch,
                current_lr=current_lr,
            )
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best selected_val_acc={best_selected_val:.4f}")
            break

    print(
        f"ERM training finished. Best selected_val_acc={best_selected_val:.4f}, "
        f"best_train_acc={best_train_acc:.4f} (epoch {best_train_epoch}). Evaluating on test set..."
    )
    best_ckpt = torch.load(artifacts.best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_acc = evaluate(model, test_loader, device, transform=eval_transform)
    print(f"Test accuracy: {test_acc:.4f}")


def run_irm(args) -> None:
    set_seed(args.seed)
    artifacts = _prepare_artifacts(args.output_dir)
    device = args.device
    print(f"Using device: {device}")

    env_loaders, val_loader, test_loader = _build_irm_loaders(args)
    train_transform, eval_transform = _build_transforms(args)
    env_ids = sorted(env_loaders.keys())
    env_iters: Dict[int, Iterator] = {e: infinite_loader(loader) for e, loader in env_loaders.items()}
    env_batch_steps = {e: len(loader) for e, loader in env_loaders.items()}
    print(f"Train envs: {env_ids}")
    print(f"Batches per env: {env_batch_steps}")

    model = CSIClassifier(num_classes=args.num_classes, dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    ema = ModelEMA(model, args.ema_decay) if getattr(args, "ema_decay", 0.0) > 0 else None

    val_selection_metric = getattr(args, "val_selection_metric", "hybrid")
    penalty_ramp_epochs = getattr(args, "penalty_ramp_epochs", 20)
    best_selected_val = 0.0
    best_overall_val = 0.0
    best_worst_env_val = 0.0
    best_train_acc = 0.0
    best_train_epoch = 0
    epochs_without_improve = 0

    best_overall_path = os.path.join(args.output_dir, "best_overall_model.pt")
    best_worst_env_path = os.path.join(args.output_dir, "best_worst_env_model.pt")

    with open(artifacts.log_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            [
                "epoch",
                "total_loss",
                "erm_loss",
                "penalty",
                "train_acc",
                "val_acc",
                "val_worst_env_acc",
                "ema_val_acc",
                "ema_val_worst_env_acc",
                "selected_val_acc",
                "val_hybrid_acc",
                "ema_val_hybrid_acc",
                "irm_weight",
                "lr",
            ]
        )

    steps_per_epoch = min(env_batch_steps.values())

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_total_loss = 0.0
        running_erm_loss = 0.0
        running_penalty = 0.0
        train_correct = 0
        train_total = 0

        irm_weight = _compute_irm_weight(
            epoch=epoch,
            irm_lambda=args.irm_lambda,
            anneal_epochs=args.penalty_anneal_epochs,
            ramp_epochs=penalty_ramp_epochs,
        )

        for _ in tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            per_env_logits = []
            per_env_targets = []

            for e in env_ids:
                x, y, _env = next(env_iters[e])
                x = _maybe_to_device(x, device)
                y = _maybe_to_device(y, device)
                if train_transform is not None:
                    x = train_transform(x)
                logits = model(x)
                per_env_logits.append(logits)
                per_env_targets.append(y)
                preds = logits.argmax(dim=1)
                train_correct += (preds == y).sum().item()
                train_total += y.numel()

            optimizer.zero_grad()
            total_loss, erm_loss, penalty = aggregate_irm_loss(
                per_env_logits,
                per_env_targets,
                criterion,
                penalty_weight=irm_weight,
            )
            total_loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()
            if ema is not None:
                ema.update(model)

            batch_size_total = sum(t.size(0) for t in per_env_targets)
            running_total_loss += total_loss.item() * batch_size_total
            running_erm_loss += erm_loss.item() * batch_size_total
            running_penalty += penalty.item() * batch_size_total

        denom = max(steps_per_epoch * args.batch_size * len(env_ids), 1)
        avg_total_loss = running_total_loss / denom
        avg_erm_loss = running_erm_loss / denom
        avg_penalty = running_penalty / denom
        train_acc = train_correct / max(train_total, 1)
        if train_acc > best_train_acc:
            best_train_acc = train_acc
            best_train_epoch = epoch

        val_acc, val_worst_env_acc, per_env_acc = evaluate_by_env(model, val_loader, device, transform=eval_transform)
        val_hybrid_acc = 0.5 * (val_acc + val_worst_env_acc)

        ema_val_acc = val_acc
        ema_val_worst_env_acc = val_worst_env_acc
        ema_val_hybrid_acc = val_hybrid_acc
        ema_per_env_acc = per_env_acc
        if ema is not None:
            ema_val_acc, ema_val_worst_env_acc, ema_per_env_acc = evaluate_by_env(
                ema.ema, val_loader, device, transform=eval_transform
            )
            ema_val_hybrid_acc = 0.5 * (ema_val_acc + ema_val_worst_env_acc)

        model_selected_val = _select_val_score(val_selection_metric, val_acc, val_worst_env_acc)
        ema_selected_val = _select_val_score(val_selection_metric, ema_val_acc, ema_val_worst_env_acc)
        use_ema_for_selected = ema is not None and ema_selected_val >= model_selected_val
        selected_val_acc = ema_selected_val if use_ema_for_selected else model_selected_val
        scheduler.step(selected_val_acc)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"[Epoch {epoch}] total_loss={avg_total_loss:.4f}, erm_loss={avg_erm_loss:.4f}, "
            f"penalty={avg_penalty:.4f}, train_acc={train_acc:.4f}, val_acc={val_acc:.4f}, "
            f"ema_val_acc={ema_val_acc:.4f}, val_worst_env_acc={val_worst_env_acc:.4f}, "
            f"ema_val_worst_env_acc={ema_val_worst_env_acc:.4f}, val_hybrid_acc={val_hybrid_acc:.4f}, "
            f"ema_val_hybrid_acc={ema_val_hybrid_acc:.4f}, select={selected_val_acc:.4f}, "
            f"irm_weight={irm_weight:.2f}, lr={current_lr:.6g}"
        )
        if per_env_acc:
            env_msg = ", ".join([f"env{e}:{a:.4f}" for e, a in per_env_acc.items()])
            print(f"           val_per_env: {env_msg}")
        if ema is not None and ema_per_env_acc:
            ema_env_msg = ", ".join([f"env{e}:{a:.4f}" for e, a in ema_per_env_acc.items()])
            print(f"           ema_val_per_env: {ema_env_msg}")

        with open(artifacts.log_path, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(
                [
                    epoch,
                    avg_total_loss,
                    avg_erm_loss,
                    avg_penalty,
                    train_acc,
                    val_acc,
                    val_worst_env_acc,
                    ema_val_acc,
                    ema_val_worst_env_acc,
                    selected_val_acc,
                    val_hybrid_acc,
                    ema_val_hybrid_acc,
                    irm_weight,
                    current_lr,
                ]
            )

        overall_candidate_acc = ema_val_acc if (ema is not None and ema_val_acc >= val_acc) else val_acc
        overall_candidate_worst = (
            ema_val_worst_env_acc if (ema is not None and ema_val_acc >= val_acc) else val_worst_env_acc
        )
        overall_candidate_hybrid = (
            ema_val_hybrid_acc if (ema is not None and ema_val_acc >= val_acc) else val_hybrid_acc
        )
        overall_candidate_state = (
            ema.ema.state_dict() if (ema is not None and ema_val_acc >= val_acc) else model.state_dict()
        )

        worst_candidate_worst = (
            ema_val_worst_env_acc if (ema is not None and ema_val_worst_env_acc >= val_worst_env_acc) else val_worst_env_acc
        )
        worst_candidate_acc = ema_val_acc if (ema is not None and ema_val_worst_env_acc >= val_worst_env_acc) else val_acc
        worst_candidate_hybrid = (
            ema_val_hybrid_acc if (ema is not None and ema_val_worst_env_acc >= val_worst_env_acc) else val_hybrid_acc
        )
        worst_candidate_state = (
            ema.ema.state_dict()
            if (ema is not None and ema_val_worst_env_acc >= val_worst_env_acc)
            else model.state_dict()
        )

        selected_candidate_acc = ema_val_acc if use_ema_for_selected else val_acc
        selected_candidate_worst = ema_val_worst_env_acc if use_ema_for_selected else val_worst_env_acc
        selected_candidate_hybrid = ema_val_hybrid_acc if use_ema_for_selected else val_hybrid_acc
        selected_candidate_state = ema.ema.state_dict() if use_ema_for_selected else model.state_dict()

        if overall_candidate_acc > best_overall_val:
            best_overall_val = overall_candidate_acc
            checkpoint = {
                "model_state": overall_candidate_state,
                "epoch": epoch,
                "val_acc": overall_candidate_acc,
                "val_worst_env_acc": overall_candidate_worst,
                "selected_val_acc": overall_candidate_acc,
                "selected_val_metric": "overall",
                "best_train_acc": best_train_acc,
                "best_train_epoch": best_train_epoch,
                "args": vars(args),
                "env_ids": env_ids,
                "irm_lambda": args.irm_lambda,
                "penalty_anneal_epochs": args.penalty_anneal_epochs,
                "penalty_ramp_epochs": penalty_ramp_epochs,
                "trainer_mode": "irm",
            }
            torch.save(
                checkpoint,
                best_overall_path,
            )
            _write_best_model_metadata(
                best_overall_path,
                args,
                trainer_mode="irm",
                best_epoch=epoch,
                selected_val_metric="overall",
                val_acc=overall_candidate_acc,
                val_worst_env_acc=overall_candidate_worst,
                val_hybrid_acc=overall_candidate_hybrid,
                selected_val_acc=overall_candidate_acc,
                best_train_acc=best_train_acc,
                best_train_epoch=best_train_epoch,
                current_lr=current_lr,
                irm_weight=irm_weight,
            )

        if worst_candidate_worst > best_worst_env_val:
            best_worst_env_val = worst_candidate_worst
            checkpoint = {
                "model_state": worst_candidate_state,
                "epoch": epoch,
                "val_acc": worst_candidate_acc,
                "val_worst_env_acc": worst_candidate_worst,
                "selected_val_acc": worst_candidate_worst,
                "selected_val_metric": "worst_env",
                "best_train_acc": best_train_acc,
                "best_train_epoch": best_train_epoch,
                "args": vars(args),
                "env_ids": env_ids,
                "irm_lambda": args.irm_lambda,
                "penalty_anneal_epochs": args.penalty_anneal_epochs,
                "penalty_ramp_epochs": penalty_ramp_epochs,
                "trainer_mode": "irm",
            }
            torch.save(
                checkpoint,
                best_worst_env_path,
            )
            _write_best_model_metadata(
                best_worst_env_path,
                args,
                trainer_mode="irm",
                best_epoch=epoch,
                selected_val_metric="worst_env",
                val_acc=worst_candidate_acc,
                val_worst_env_acc=worst_candidate_worst,
                val_hybrid_acc=worst_candidate_hybrid,
                selected_val_acc=worst_candidate_worst,
                best_train_acc=best_train_acc,
                best_train_epoch=best_train_epoch,
                current_lr=current_lr,
                irm_weight=irm_weight,
            )

        if selected_val_acc > best_selected_val:
            best_selected_val = selected_val_acc
            epochs_without_improve = 0
            checkpoint = {
                "model_state": selected_candidate_state,
                "epoch": epoch,
                "val_acc": selected_candidate_acc,
                "val_worst_env_acc": selected_candidate_worst,
                "selected_val_acc": selected_val_acc,
                "selected_val_metric": val_selection_metric,
                "best_train_acc": best_train_acc,
                "best_train_epoch": best_train_epoch,
                "args": vars(args),
                "env_ids": env_ids,
                "irm_lambda": args.irm_lambda,
                "penalty_anneal_epochs": args.penalty_anneal_epochs,
                "penalty_ramp_epochs": penalty_ramp_epochs,
                "trainer_mode": "irm",
            }
            torch.save(
                checkpoint,
                artifacts.best_model_path,
            )
            _write_best_model_metadata(
                artifacts.best_model_path,
                args,
                trainer_mode="irm",
                best_epoch=epoch,
                selected_val_metric=val_selection_metric,
                val_acc=selected_candidate_acc,
                val_worst_env_acc=selected_candidate_worst,
                val_hybrid_acc=selected_candidate_hybrid,
                selected_val_acc=selected_val_acc,
                best_train_acc=best_train_acc,
                best_train_epoch=best_train_epoch,
                current_lr=current_lr,
                irm_weight=irm_weight,
            )
        else:
            epochs_without_improve += 1

        if epochs_without_improve >= args.early_stop_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"Best selected_val_acc={best_selected_val:.4f} ({val_selection_metric})."
            )
            break

    print(
        f"IRM training finished. Best selected_val_acc={best_selected_val:.4f} ({val_selection_metric}), "
        f"best_train_acc={best_train_acc:.4f} (epoch {best_train_epoch}). Evaluating on test set..."
    )
    best_ckpt = torch.load(artifacts.best_model_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state"])
    test_acc_selected = evaluate(model, test_loader, device, transform=eval_transform)
    print(f"Test accuracy (best-selected model, metric={val_selection_metric}): {test_acc_selected:.4f}")

    if os.path.isfile(best_overall_path):
        ckpt_overall = torch.load(best_overall_path, map_location=device)
        model.load_state_dict(ckpt_overall["model_state"])
        test_acc_overall = evaluate(model, test_loader, device, transform=eval_transform)
        print(f"Test accuracy (best-overall-val model): {test_acc_overall:.4f}")

    if os.path.isfile(best_worst_env_path):
        ckpt_worst = torch.load(best_worst_env_path, map_location=device)
        model.load_state_dict(ckpt_worst["model_state"])
        test_acc_worst = evaluate(model, test_loader, device, transform=eval_transform)
        print(f"Test accuracy (best-worst-env-val model): {test_acc_worst:.4f}")
