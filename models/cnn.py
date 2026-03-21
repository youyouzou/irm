import math
import torch
import torch.nn as nn


def _pick_groups(channels: int, max_groups: int = 8) -> int:
    """为 GroupNorm 选择“组数”，保证能整除通道数，且不超过 max_groups。"""
    for g in range(min(max_groups, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1  # 理论上不会走到这里，保底返回 1 组（等价于 LayerNorm）


class SEBlock(nn.Module):
    """Squeeze-and-Excitation（通道注意力）模块。"""

    def __init__(self, channels: int, reduction: int = 8):
        """
        channels: 当前特征图的通道数
        reduction: 通道压缩比例，越大表示中间层维度越小
        """
        super().__init__()
        # 压缩后的隐藏维度，至少为 4，避免过小
        hidden = max(channels // reduction, 4)

        # 自适应全局平均池化到 1x1，用来做通道汇聚
        self.pool = nn.AdaptiveAvgPool2d(1)

        # 两层全连接实现“先压缩再扩展”，最后用 Sigmoid 得到 [0,1] 的通道权重
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        b, c, _, _ = x.shape
        # 全局平均池化 -> [B, C, 1, 1] -> [B, C]
        s = self.pool(x).view(b, c)
        # 通过全连接生成每个通道的权重，再 reshape 回 [B, C, 1, 1]
        s = self.fc(s).view(b, c, 1, 1)
        # 通道上加权：每个通道乘上相应的权重
        return x * s


class ResidualBlock(nn.Module):
    """带 SE 和 GroupNorm 的残差块，用于构建 CNN 的深层特征提取部分。"""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        """
        in_ch: 输入通道数
        out_ch: 输出通道数
        stride: 第一层卷积的步长（>1 时会下采样）
        dropout: 残差内部使用的 Dropout2d 比例
        """
        super().__init__()
        # 为 GroupNorm 选择“组数”，保证可以整除通道数
        g1 = _pick_groups(out_ch)
        g2 = _pick_groups(out_ch)

        # 第一个 3x3 卷积：可以通过 stride 实现空间下采样
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.norm1 = nn.GroupNorm(g1, out_ch)
        self.act1 = nn.ReLU(inplace=True)

        # 第二个 3x3 卷积：保持特征图大小不变（stride=1）
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g2, out_ch)

        # 通道注意力模块
        self.se = SEBlock(out_ch)

        # 残差分支上的 Dropout2d，用于正则化
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        # 如果通道数或分辨率发生变化，需要对 shortcut 也做对应变换
        if stride != 1 or in_ch != out_ch:
            # 1x1 卷积 + GroupNorm，把输入映射到和主分支同样的形状
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_pick_groups(out_ch), out_ch),
            )
        else:
            # 否则直接用恒等映射
            self.shortcut = nn.Identity()

        # 残差加和后的激活函数
        self.out_act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 计算 shortcut 分支（可能包含 1x1 卷积和下采样）
        identity = self.shortcut(x)

        # 主分支：Conv -> GN -> ReLU
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)

        # Conv -> GN -> SE -> Dropout
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.se(out)
        out = self.drop(out)

        # 残差相加
        out = out + identity
        # 再通过 ReLU 激活
        return self.out_act(out)


class CSIClassifier(nn.Module):
    """
    用于 WiFi CSI 数据的分类模型。
    输入：形状 [B, 2000, 30] 或 [B, 1, 2000, 30]
    输出：形状 [B, num_classes] 的 logits。
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        """
        num_classes: 分类类别数（你的任务中是 7）
        dropout: 全连接层和残差块内部的 dropout 比例
        """
        super().__init__()

        # stem：把单通道 CSI 特征映射到 32 通道的特征图
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(_pick_groups(32), 32),
            nn.ReLU(inplace=True),
        )

        # feature：堆叠多个 ResidualBlock，逐步提取高层特征并下采样
        self.feature = nn.Sequential(
            # 保持分辨率不变，增加非线性和表达能力
            ResidualBlock(32, 32, stride=1, dropout=dropout * 0.5),
            # stride=2 降采样，通道从 32 -> 64
            ResidualBlock(32, 64, stride=2, dropout=dropout * 0.5),
            # 再次 stride=2 降采样，通道从 64 -> 128
            ResidualBlock(64, 128, stride=2, dropout=dropout),
            # 不再降采样，继续堆叠 128 通道残差块
            ResidualBlock(128, 128, stride=1, dropout=dropout),
            # 全局平均池化到 1x1，得到每个通道一个特征
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # classifier：把 128 维特征映射到 num_classes
        self.classifier = nn.Sequential(
            nn.Flatten(),                # [B, 128, 1, 1] -> [B, 128]
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self) -> None:
        """对卷积层和线性层做 Kaiming 初始化，偏置初始化为 0。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:
          - 若为 [B, 2000, 30]（3 维），自动在前面 unsqueeze 一维当作通道；
          - 若为 [B, C, H, W]（4 维），要求 C=1。
        """
        # 如果是 [B, H, W]，自动加一个通道维度 -> [B, 1, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(1)
        elif x.dim() != 4:
            raise ValueError(f"Expected input with 3 or 4 dims, got shape: {tuple(x.shape)}")

        # 卷积 + 残差块特征提取
        x = self.stem(x)
        x = self.feature(x)
        # 全连接分类
        return self.classifier(x)
        
