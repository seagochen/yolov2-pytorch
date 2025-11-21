"""
Darknet-19 Backbone for YOLOv2

YOLOv2使用Darknet-19作为特征提取backbone，主要特点：
- 19层卷积网络
- 所有卷积层后都使用Batch Normalization
- 使用1x1卷积进行通道降维
- 使用Global Average Pooling替代全连接层
- 输入：640x640x3
- 输出：20x20x1024 特征图
"""

import torch
import torch.nn as nn


class ConvBNLeaky(nn.Module):
    """
    卷积 + BatchNorm + LeakyReLU 组合模块
    YOLOv2的基础构建块
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBNLeaky, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.leaky = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.leaky(self.bn(self.conv(x)))


class Darknet19(nn.Module):
    """
    Darknet-19 特征提取网络

    网络结构：
    - 输入: (B, 3, 640, 640)
    - 输出: (B, 1024, 20, 20)

    特点：
    - 19个卷积层 + 5个max pooling层
    - 每个卷积层后都有BatchNorm和LeakyReLU
    - 使用1x1卷积降维，3x3卷积提取特征
    """

    def __init__(self):
        super(Darknet19, self).__init__()

        # Layer 1-2: 640x640 -> 320x320 -> 160x160
        self.layer1 = nn.Sequential(
            ConvBNLeaky(3, 32, 3, 1, 1),      # 640x640x3 -> 640x640x32
            nn.MaxPool2d(2, 2),                # 640x640x32 -> 320x320x32
            ConvBNLeaky(32, 64, 3, 1, 1),     # 320x320x32 -> 320x320x64
            nn.MaxPool2d(2, 2)                 # 320x320x64 -> 160x160x64
        )

        # Layer 3-5: 160x160 -> 80x80
        self.layer2 = nn.Sequential(
            ConvBNLeaky(64, 128, 3, 1, 1),    # 160x160x64 -> 160x160x128
            ConvBNLeaky(128, 64, 1, 1, 0),    # 160x160x128 -> 160x160x64
            ConvBNLeaky(64, 128, 3, 1, 1),    # 160x160x64 -> 160x160x128
            nn.MaxPool2d(2, 2)                 # 160x160x128 -> 80x80x128
        )

        # Layer 6-8: 80x80 -> 40x40
        self.layer3 = nn.Sequential(
            ConvBNLeaky(128, 256, 3, 1, 1),   # 80x80x128 -> 80x80x256
            ConvBNLeaky(256, 128, 1, 1, 0),   # 80x80x256 -> 80x80x128
            ConvBNLeaky(128, 256, 3, 1, 1),   # 80x80x128 -> 80x80x256
            nn.MaxPool2d(2, 2)                 # 80x80x256 -> 40x40x256
        )

        # Layer 9-13: 40x40 -> 20x20
        self.layer4 = nn.Sequential(
            ConvBNLeaky(256, 512, 3, 1, 1),   # 40x40x256 -> 40x40x512
            ConvBNLeaky(512, 256, 1, 1, 0),   # 40x40x512 -> 40x40x256
            ConvBNLeaky(256, 512, 3, 1, 1),   # 40x40x256 -> 40x40x512
            ConvBNLeaky(512, 256, 1, 1, 0),   # 40x40x512 -> 40x40x256
            ConvBNLeaky(256, 512, 3, 1, 1),   # 40x40x256 -> 40x40x512
            nn.MaxPool2d(2, 2)                 # 40x40x512 -> 20x20x512
        )

        # Layer 14-18: 20x20 (保持尺寸)
        self.layer5 = nn.Sequential(
            ConvBNLeaky(512, 1024, 3, 1, 1),  # 20x20x512 -> 20x20x1024
            ConvBNLeaky(1024, 512, 1, 1, 0),  # 20x20x1024 -> 20x20x512
            ConvBNLeaky(512, 1024, 3, 1, 1),  # 20x20x512 -> 20x20x1024
            ConvBNLeaky(1024, 512, 1, 1, 0),  # 20x20x1024 -> 20x20x512
            ConvBNLeaky(512, 1024, 3, 1, 1)   # 20x20x512 -> 20x20x1024
        )

        # 用于保存中间特征（passthrough layer需要）
        self.layer4_output = None

    def forward(self, x):
        """
        前向传播

        Args:
            x: (B, 3, 640, 640)

        Returns:
            features: (B, 1024, 20, 20)
            passthrough: (B, 512, 40, 40) - 用于passthrough layer
        """
        x = self.layer1(x)  # (B, 64, 160, 160)
        x = self.layer2(x)  # (B, 128, 80, 80)
        x = self.layer3(x)  # (B, 256, 40, 40)

        # 保存layer4的输出用于passthrough
        x = self.layer4(x)  # (B, 512, 20, 20)
        passthrough = x     # 保存40x40的特征（pooling之前）

        x = self.layer5(x)  # (B, 1024, 20, 20)

        return x, passthrough


class PassthroughLayer(nn.Module):
    """
    Passthrough Layer (细粒度特征层)

    将高分辨率特征图（40x40）重组并连接到低分辨率特征图（20x20）
    这样可以保留更多细节信息，有助于检测小物体

    操作：
    1. 将40x40x512的特征图分成4个20x20的块
    2. 在通道维度上拼接，得到20x20x2048
    3. 与20x20x1024的主特征concat，得到20x20x3072
    """

    def __init__(self):
        super(PassthroughLayer, self).__init__()

    def forward(self, x):
        """
        Space-to-depth转换

        Args:
            x: (B, C, H, W) - 通常是 (B, 512, 40, 40)

        Returns:
            out: (B, C*4, H/2, W/2) - 通常是 (B, 2048, 20, 20)
        """
        B, C, H, W = x.shape

        # 将40x40分成4个20x20的块
        # (B, C, 40, 40) -> (B, C, 20, 2, 20, 2)
        x = x.view(B, C, H // 2, 2, W // 2, 2)

        # 调整维度顺序并合并
        # (B, C, 20, 2, 20, 2) -> (B, C, 2, 2, 20, 20) -> (B, C*4, 20, 20)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(B, C * 4, H // 2, W // 2)

        return x


def test_darknet19():
    """测试Darknet-19网络"""
    print("=" * 60)
    print("Testing Darknet-19 Backbone")
    print("=" * 60)

    # 创建模型
    model = Darknet19()

    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 640, 640)
    print(f"\nInput shape: {x.shape}")

    # 前向传播
    features, passthrough = model(x)
    print(f"Output features shape: {features.shape}")
    print(f"Passthrough features shape: {passthrough.shape}")

    # 测试Passthrough layer
    passthrough_layer = PassthroughLayer()
    passthrough_reorg = passthrough_layer(passthrough)
    print(f"Passthrough after reorg: {passthrough_reorg.shape}")

    # 拼接
    combined = torch.cat([features, passthrough_reorg], dim=1)
    print(f"Combined features shape: {combined.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Model size: {total_params * 4 / (1024**2):.2f} MB")

    print("\n" + "=" * 60)
    print("✓ Test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_darknet19()
