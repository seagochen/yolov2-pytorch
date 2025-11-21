"""
Common layers for YOLOv2
使用最新的PyTorch API和最佳实践
"""

import torch
import torch.nn as nn
from typing import Optional


class ConvBNAct(nn.Module):
    """
    卷积 + BatchNorm + 激活函数

    使用最新PyTorch API的标准组合层
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: Optional[int] = None,
        groups: int = 1,
        activation: str = 'leaky_relu'
    ):
        super().__init__()

        # 自动计算padding以保持尺寸（当stride=1时）
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False  # BN层会处理bias
        )

        # 使用track_running_stats=True确保BN正确工作
        self.bn = nn.BatchNorm2d(
            out_channels,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True
        )

        # 激活函数
        if activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.1, inplace=True)
        elif activation == 'relu':
            self.act = nn.ReLU(inplace=True)
        elif activation == 'silu':  # Swish/SiLU
            self.act = nn.SiLU(inplace=True)
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SpaceToDepth(nn.Module):
    """
    Space-to-Depth转换（Passthrough layer）

    将高分辨率特征图重组为低分辨率但高通道的特征图
    PyTorch没有内置，需要手动实现
    """
    def __init__(self, block_size: int = 2):
        super().__init__()
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C*block_size^2, H//block_size, W//block_size)
        """
        B, C, H, W = x.shape
        bs = self.block_size

        # 重塑张量
        # (B, C, H, W) -> (B, C, H//bs, bs, W//bs, bs)
        x = x.view(B, C, H // bs, bs, W // bs, bs)

        # 调整维度顺序并合并
        # (B, C, H//bs, bs, W//bs, bs) -> (B, C, bs, bs, H//bs, W//bs)
        # -> (B, C*bs*bs, H//bs, W//bs)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * bs * bs, H // bs, W // bs)

        return x


class Residual(nn.Module):
    """
    残差连接块（可选）
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: Optional[int] = None,
        activation: str = 'leaky_relu'
    ):
        super().__init__()

        if hidden_channels is None:
            hidden_channels = in_channels // 2

        self.conv1 = ConvBNAct(in_channels, hidden_channels, 1, activation=activation)
        self.conv2 = ConvBNAct(hidden_channels, in_channels, 3, activation=activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv2(self.conv1(x))


def autopad(kernel_size: int, padding: Optional[int] = None) -> int:
    """
    自动计算padding以保持空间尺寸
    """
    if padding is None:
        padding = kernel_size // 2
    return padding


if __name__ == '__main__':
    # 测试层
    print("Testing layers...")

    # ConvBNAct
    x = torch.randn(2, 64, 32, 32)
    conv = ConvBNAct(64, 128, 3, 2)
    y = conv(x)
    print(f"ConvBNAct: {x.shape} -> {y.shape}")

    # SpaceToDepth
    x = torch.randn(2, 512, 40, 40)
    s2d = SpaceToDepth(block_size=2)
    y = s2d(x)
    print(f"SpaceToDepth: {x.shape} -> {y.shape}")

    # Residual
    x = torch.randn(2, 256, 20, 20)
    res = Residual(256)
    y = res(x)
    print(f"Residual: {x.shape} -> {y.shape}")

    print("✓ All tests passed!")
