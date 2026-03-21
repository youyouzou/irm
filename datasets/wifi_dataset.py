import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


class ComposeTransforms:
    """Apply transforms in sequence."""

    def __init__(self, transforms: List[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x


class SampleWiseZScore:
    """Normalize each sample to zero-mean and unit-std."""

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean()
        std = x.std(unbiased=False).clamp_min(self.eps)
        return (x - mean) / std


class RandomAmplitudeScale:
    """Randomly scale signal amplitude."""

    def __init__(self, low: float = 0.9, high: float = 1.1, p: float = 0.5) -> None:
        self.low = low
        self.high = high
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand((), device=x.device).item() > self.p:
            return x
        scale = torch.empty((), device=x.device).uniform_(self.low, self.high).item()
        return x * scale


class RandomGaussianNoise:
    """Inject Gaussian noise relative to sample std."""

    def __init__(self, std_ratio: float = 0.01, p: float = 0.5) -> None:
        self.std_ratio = std_ratio
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.std_ratio <= 0 or torch.rand((), device=x.device).item() > self.p:
            return x
        sigma = x.std(unbiased=False).clamp_min(1e-6) * self.std_ratio
        return x + torch.randn_like(x) * sigma


class RandomTimeMask:
    """Mask a random continuous time range."""

    def __init__(self, max_ratio: float = 0.0, p: float = 0.5) -> None:
        self.max_ratio = max_ratio
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_ratio <= 0 or torch.rand((), device=x.device).item() > self.p:
            return x
        out = x.clone()
        if out.dim() == 2:
            t = out.size(0)
            max_len = max(1, int(t * self.max_ratio))
            length = int(torch.randint(1, max_len + 1, (1,), device=x.device).item())
            start = int(torch.randint(0, t - length + 1, (1,), device=x.device).item())
            out[start : start + length, :] = 0.0
            return out
        if out.dim() == 3:
            t = out.size(1)
            max_len = max(1, int(t * self.max_ratio))
            length = int(torch.randint(1, max_len + 1, (1,), device=x.device).item())
            start = int(torch.randint(0, t - length + 1, (1,), device=x.device).item())
            out[:, start : start + length, :] = 0.0
            return out
        return x


class RandomSubcarrierMask:
    """Mask a random continuous subcarrier range."""

    def __init__(self, max_ratio: float = 0.0, p: float = 0.5) -> None:
        self.max_ratio = max_ratio
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_ratio <= 0 or torch.rand((), device=x.device).item() > self.p:
            return x
        out = x.clone()
        if out.dim() == 2:
            s = out.size(1)
            max_len = max(1, int(s * self.max_ratio))
            length = int(torch.randint(1, max_len + 1, (1,), device=x.device).item())
            start = int(torch.randint(0, s - length + 1, (1,), device=x.device).item())
            out[:, start : start + length] = 0.0
            return out
        if out.dim() == 3:
            s = out.size(2)
            max_len = max(1, int(s * self.max_ratio))
            length = int(torch.randint(1, max_len + 1, (1,), device=x.device).item())
            start = int(torch.randint(0, s - length + 1, (1,), device=x.device).item())
            out[:, :, start : start + length] = 0.0
            return out
        return x


def build_csi_transform(
    normalize: str = "sample_zscore",
    augment: bool = False,
    aug_prob: float = 0.5,
    amp_scale_low: float = 0.9,
    amp_scale_high: float = 1.1,
    noise_std_ratio: float = 0.01,
    time_mask_ratio: float = 0.0,
    subcarrier_mask_ratio: float = 0.0,
) -> Optional[Callable[[torch.Tensor], torch.Tensor]]:
    transforms: List[Callable[[torch.Tensor], torch.Tensor]] = []

    if normalize == "sample_zscore":
        transforms.append(SampleWiseZScore())
    elif normalize != "none":
        raise ValueError(f"Unsupported normalize mode: {normalize}")

    if augment:
        transforms.extend(
            [
                RandomAmplitudeScale(low=amp_scale_low, high=amp_scale_high, p=aug_prob),
                RandomGaussianNoise(std_ratio=noise_std_ratio, p=aug_prob),
                RandomTimeMask(max_ratio=time_mask_ratio, p=aug_prob),
                RandomSubcarrierMask(max_ratio=subcarrier_mask_ratio, p=aug_prob),
            ]
        )

    if not transforms:
        return None
    return ComposeTransforms(transforms)


class WiFiCSIDataset(Dataset):
    """
    通用 WiFi/CSI 数据集：
    - 假设使用一个 .npz 文件，里面至少包含:
      - 'x': 形状 [N, C, H, W] 或 [N, H, W] 的特征
      - 'y': 形状 [N] 的类别标签 (int)
      - 'env': 形状 [N] 的环境编号 (int 或可转为 int)

    你可以用自己的预处理脚本把原始数据转成这种格式。
    """

    def __init__(
        self,
        npz_path: str,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        data = np.load(npz_path)
        self.x = data["x"]  # numpy array
        self.y = data["y"].astype(np.int64)
        self.env = data["env"].astype(np.int64)
        self.transform = transform

        assert len(self.x) == len(self.y) == len(self.env), "x/y/env must have same length"

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        x = self.x[idx]
        y = int(self.y[idx])
        env = int(self.env[idx])

        # 转成 tensor（尽量避免不必要的拷贝）
        x_tensor = torch.from_numpy(x)
        if x_tensor.dtype != torch.float32:
            x_tensor = x_tensor.to(dtype=torch.float32)

        if self.transform is not None:
            x_tensor = self.transform(x_tensor)

        return x_tensor, y, env


def build_env_subsets(dataset: WiFiCSIDataset) -> Dict[int, Subset]:
    """
    根据 env 字段，把一个 WiFiCSIDataset 划分为多个环境子集。
    返回 {env_id: Subset}。
    """
    env_ids = dataset.env
    env_to_indices: Dict[int, list] = {}
    for idx, e in enumerate(env_ids):
        e_int = int(e)
        if e_int not in env_to_indices:
            env_to_indices[e_int] = []
        env_to_indices[e_int].append(idx)

    env_subsets: Dict[int, Subset] = {}
    for e, indices in env_to_indices.items():
        env_subsets[e] = Subset(dataset, indices)
    return env_subsets


def build_dataloaders_for_envs(
    dataset: WiFiCSIDataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Dict[int, DataLoader]:
    """
    为每个环境创建一个独立的 DataLoader，用于 IRM 训练。
    """
    env_subsets = build_env_subsets(dataset)
    env_loaders: Dict[int, DataLoader] = {}
    for e, subset in env_subsets.items():
        env_loaders[e] = DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=True,  # IRM 中各环境 batch 尺寸一致更方便
        )
    return env_loaders

