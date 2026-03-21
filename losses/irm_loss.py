from typing import Tuple, List

import torch
import torch.nn as nn
from torch import autograd


def irm_penalty(logits: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    """
    IRM penalty 的常用实现：
    - 引入一个标量 w，对 logits 做缩放；
    - 在该环境上计算 loss 对 w 的梯度，并取其平方和。
    参考 IRM 论文附录和常见代码实现。
    """
    device = logits.device
    w = torch.tensor(1.0, device=device, requires_grad=True)

    loss = criterion(logits * w, y)
    grad = autograd.grad(loss, [w], create_graph=True)[0]
    return torch.sum(grad**2)


def aggregate_irm_loss(
    per_env_logits: List[torch.Tensor],
    per_env_targets: List[torch.Tensor],
    criterion: nn.Module,
    penalty_weight: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    给多个环境的 logits/targets，计算：
    - 平均 ERM loss
    - 平均 IRM penalty
    - 总损失 total_loss = erm_loss + penalty_weight * penalty
    """
    assert len(per_env_logits) == len(per_env_targets) > 0

    erm_losses = []
    penalties = []
    for logits, targets in zip(per_env_logits, per_env_targets):
        erm_losses.append(criterion(logits, targets))
        penalties.append(irm_penalty(logits, targets, criterion))

    erm_loss = torch.stack(erm_losses).mean()
    penalty = torch.stack(penalties).mean()
    total_loss = erm_loss + penalty_weight * penalty
    return total_loss, erm_loss, penalty

