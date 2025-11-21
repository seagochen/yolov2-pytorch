"""
YOLOv1 Network Architecture - Version 102
升级版支持:
- 640x640 输入尺寸
- 20x20 网格划分 (更精细，支持多尺度物体检测)
- 每个格子2个边界框
- 80个COCO类别

网络结构改进:
- 调整卷积层使输出特征图为20x20
- 保持与原始YOLOv1类似的架构
"""

import torch


class YoloV1NetworkV102(torch.nn.Module):
    """
    YOLOv1 网络架构 - 升级版

    参数:
        grids_size: 网格尺寸，默认(20, 20)
        confidences: 置信度数量，固定为1 (每个格子一个置信度)
        bounding_boxes: 每个格子的边界框数量，默认2
        object_categories: 物体类别数，默认80 (COCO)

    输入:
        (B, 3, 640, 640) - 批次大小B，3通道RGB，640x640图像

    输出:
        (B, C, G) 其中:
            C = confidences + bboxes*4 + categories
            G = grid_h * grid_w
    """

    def __init__(self, grids_size=(20, 20), confidences=1, bounding_boxes=2, object_categories=80):
        super().__init__()

        self.grids_size = grids_size
        self.confidences = confidences
        self.bounding_boxes = bounding_boxes
        self.object_categories = object_categories

        # 计算最终输出特征数
        out_features = (confidences + bounding_boxes * 4 + object_categories) * grids_size[0] * grids_size[1]

        # ============================================================
        # 卷积层设计 (640x640 -> 20x20)
        # ============================================================

        # Layer 1: 640x640 -> 320x320 -> 160x160
        # Input: (B, 3, 640, 640)
        self.conv_1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Output: (B, 64, 160, 160)

        # Layer 2: 160x160 -> 80x80
        # Input: (B, 64, 160, 160)
        self.conv_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(192),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Output: (B, 192, 80, 80)

        # Layer 3: 80x80 -> 40x40
        # Input: (B, 192, 80, 80)
        self.conv_3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Output: (B, 512, 40, 40)

        # Layer 4: 40x40 -> 20x20
        # Input: (B, 512, 40, 40)
        self.conv_4 = torch.nn.Sequential(
            # 4个 1x1 和 3x3 卷积交替
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
            # 最后的卷积
            torch.nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )  # Output: (B, 1024, 20, 20)

        # Layer 5: 20x20 -> 20x20 (保持尺寸不变)
        # Input: (B, 1024, 20, 20)
        self.conv_5 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
            torch.nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1)
        )  # Output: (B, 1024, 20, 20)

        # Layer 6: 20x20 -> 20x20 (保持尺寸不变)
        # Input: (B, 1024, 20, 20)
        self.conv_6 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.1)
        )  # Output: (B, 1024, 20, 20)

        # ============================================================
        # 全连接层
        # ============================================================

        # Layer 7: Flatten + FC
        # Input: (B, 1024 * 20 * 20)
        self.fc_7 = torch.nn.Sequential(
            torch.nn.Linear(in_features=20 * 20 * 1024, out_features=4096),
            torch.nn.Dropout(0.5),
            torch.nn.LeakyReLU(0.1)
        )  # Output: (B, 4096)

        # Layer 8: 输出层
        # Input: (B, 4096)
        self.fc_8 = torch.nn.Sequential(
            torch.nn.Linear(in_features=4096, out_features=out_features),
            torch.nn.Sigmoid()  # 使用Sigmoid激活，确保输出在[0,1]范围
        )  # Output: (B, out_features)

    def forward(self, data):
        """
        前向传播

        参数:
            data: (B, 3, 640, 640) 输入图像

        返回:
            output: (B, C, G) YOLO预测结果
        """
        # 获取输入维度
        B, C, H, W = data.shape

        # 检查输入尺寸
        assert C == 3, f'输入通道数必须为3，当前为{C}'
        assert H == 640, f'输入高度必须为640，当前为{H}'
        assert W == 640, f'输入宽度必须为640，当前为{W}'

        # 卷积层前向传播
        data = self.conv_1(data)  # (B, 64, 160, 160)
        data = self.conv_2(data)  # (B, 192, 80, 80)
        data = self.conv_3(data)  # (B, 512, 40, 40)
        data = self.conv_4(data)  # (B, 1024, 20, 20)
        data = self.conv_5(data)  # (B, 1024, 20, 20)
        data = self.conv_6(data)  # (B, 1024, 20, 20)

        # 展平并通过全连接层
        data = data.reshape(B, -1)  # (B, 1024*20*20)
        data = self.fc_7(data)      # (B, 4096)
        data = self.fc_8(data)      # (B, out_features)

        # 重塑为YOLO输出格式 (B, C, G)
        grid_cells = self.grids_size[0] * self.grids_size[1]
        data = data.reshape(B, -1, grid_cells)

        return data

    def get_output_info(self):
        """获取输出信息"""
        grid_cells = self.grids_size[0] * self.grids_size[1]
        features_per_cell = self.confidences + self.bounding_boxes * 4 + self.object_categories

        info = {
            'grids_size': self.grids_size,
            'grid_cells': grid_cells,
            'confidences': self.confidences,
            'bounding_boxes': self.bounding_boxes,
            'object_categories': self.object_categories,
            'features_per_cell': features_per_cell,
            'total_features': features_per_cell * grid_cells
        }
        return info


def test_network():
    """测试网络"""
    print("=" * 60)
    print("Testing YOLOv1 Network V102")
    print("=" * 60)

    batch_size = 4
    grids_size = (20, 20)
    confidences = 1
    bounding_boxes = 2
    object_categories = 80

    # 创建模型
    model = YoloV1NetworkV102(
        grids_size=grids_size,
        confidences=confidences,
        bounding_boxes=bounding_boxes,
        object_categories=object_categories
    )

    # 打印模型信息
    info = model.get_output_info()
    print("\n模型配置:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # 测试前向传播
    print("\n测试前向传播:")
    data = torch.randn(batch_size, 3, 640, 640)
    print(f"  输入尺寸: {data.shape}")

    output = model(data)
    print(f"  输出尺寸: {output.shape}")
    print(f"  输出范围: [{output.min():.4f}, {output.max():.4f}]")

    # 验证输出形状
    expected_C = confidences + bounding_boxes * 4 + object_categories
    expected_G = grids_size[0] * grids_size[1]
    assert output.shape == (batch_size, expected_C, expected_G), \
        f"输出形状错误: 期望{(batch_size, expected_C, expected_G)}, 实际{output.shape}"

    print("\n✓ 测试通过!")

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    print(f"  模型大小: {total_params * 4 / (1024**2):.2f} MB")

    print("=" * 60)


if __name__ == "__main__":
    test_network()
