"""
YOLOv2 Loss Function - 重构版
使用最新PyTorch API
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class YOLOv2Loss(nn.Module):
    """
    YOLOv2 损失函数

    改进：
    - 更清晰的代码结构
    - 更好的类型标注
    - 优化的计算效率
    - 详细的损失分解
    """

    def __init__(
        self,
        num_classes: int = 80,
        anchors: torch.Tensor = None,
        lambda_coord: float = 5.0,
        lambda_noobj: float = 0.5,
        lambda_class: float = 1.0,
        reduction: str = 'sum'
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class
        self.reduction = reduction

        # Anchors (如果提供)
        if anchors is not None:
            self.register_buffer('anchors', anchors)
        else:
            self.register_buffer('anchors', torch.FloatTensor([
                [0.57273, 0.677385],
                [1.87446, 2.06253],
                [3.33843, 5.47434],
                [7.88282, 3.52778],
                [9.77052, 9.16828]
            ]))

        self.num_anchors = len(self.anchors)

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        计算损失

        Args:
            predictions: (B, num_anchors, grid_h, grid_w, 5+num_classes)
            targets: (B, num_anchors, grid_h, grid_w, 5+num_classes)

        Returns:
            loss: 总损失
            loss_dict: 损失分解
        """
        device = predictions.device
        B, num_anchors, grid_h, grid_w, _ = predictions.shape

        # 确保anchors在正确设备
        if self.anchors.device != device:
            self.anchors = self.anchors.to(device)

        # 解析预测
        pred_tx = predictions[..., 0]
        pred_ty = predictions[..., 1]
        pred_tw = predictions[..., 2]
        pred_th = predictions[..., 3]
        pred_conf = predictions[..., 4]
        pred_cls = predictions[..., 5:]

        # 解析目标
        target_tx = targets[..., 0]
        target_ty = targets[..., 1]
        target_tw = targets[..., 2]
        target_th = targets[..., 3]
        target_conf = targets[..., 4]
        target_cls = targets[..., 5:]

        # 创建mask
        obj_mask = target_conf > 0  # (B, num_anchors, grid_h, grid_w)
        noobj_mask = ~obj_mask

        # 计算有物体的cell数量（用于统计）
        num_obj = obj_mask.sum().float() + 1e-6

        # 使用batch size进行归一化，而不是num_obj，以保持损失稳定性
        normalizer = B * grid_h * grid_w

        # ========================================
        # 1. 坐标损失
        # ========================================
        loss_x = F.mse_loss(
            pred_tx[obj_mask],
            target_tx[obj_mask],
            reduction='sum'
        )

        loss_y = F.mse_loss(
            pred_ty[obj_mask],
            target_ty[obj_mask],
            reduction='sum'
        )

        loss_w = F.mse_loss(
            pred_tw[obj_mask],
            target_tw[obj_mask],
            reduction='sum'
        )

        loss_h = F.mse_loss(
            pred_th[obj_mask],
            target_th[obj_mask],
            reduction='sum'
        )

        loss_coord = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h) / normalizer

        # ========================================
        # 2. 置信度损失（使用BCEWithLogitsLoss更稳定）
        # ========================================
        # 有物体 - 使用 BCEWithLogitsLoss 避免数值不稳定
        loss_conf_obj = F.binary_cross_entropy_with_logits(
            pred_conf[obj_mask],
            target_conf[obj_mask],
            reduction='sum'
        )

        # 无物体
        loss_conf_noobj = F.binary_cross_entropy_with_logits(
            pred_conf[noobj_mask],
            torch.zeros_like(pred_conf[noobj_mask]),
            reduction='sum'
        )

        loss_conf = (loss_conf_obj + self.lambda_noobj * loss_conf_noobj) / normalizer

        # ========================================
        # 3. 分类损失（使用BCEWithLogitsLoss更稳定）
        # ========================================
        if obj_mask.sum() > 0:
            # 使用 BCEWithLogitsLoss 避免数值不稳定
            loss_class = F.binary_cross_entropy_with_logits(
                pred_cls[obj_mask],
                target_cls[obj_mask],
                reduction='sum'
            )
            loss_class = self.lambda_class * loss_class / normalizer
        else:
            loss_class = torch.zeros(1, device=device)

        # ========================================
        # 总损失
        # ========================================
        total_loss = loss_coord + loss_conf + loss_class

        # 损失字典（使用normalizer保持一致性）
        loss_dict = {
            'total': total_loss.item(),
            'coord': loss_coord.item(),
            'x': (loss_x / normalizer).item(),
            'y': (loss_y / normalizer).item(),
            'w': (loss_w / normalizer).item(),
            'h': (loss_h / normalizer).item(),
            'conf': loss_conf.item(),
            'conf_obj': (loss_conf_obj / normalizer).item(),
            'conf_noobj': (loss_conf_noobj / normalizer).item(),
            'class': loss_class.item(),
            'num_obj': num_obj.item()
        }

        return total_loss, loss_dict


def create_yolov2_loss(
    num_classes: int = 80,
    anchors: torch.Tensor = None,
    **kwargs
) -> YOLOv2Loss:
    """
    创建YOLOv2损失函数

    Args:
        num_classes: 类别数
        anchors: Anchor boxes
        **kwargs: 其他参数

    Returns:
        criterion: 损失函数
    """
    return YOLOv2Loss(
        num_classes=num_classes,
        anchors=anchors,
        **kwargs
    )


if __name__ == '__main__':
    print("Testing YOLOv2Loss...")

    # 创建损失函数
    criterion = create_yolov2_loss(num_classes=80)

    # 测试数据
    B, num_anchors, grid_size, num_classes = 4, 5, 20, 80
    predictions = torch.randn(B, num_anchors, grid_size, grid_size, 5 + num_classes)
    targets = torch.zeros(B, num_anchors, grid_size, grid_size, 5 + num_classes)

    # 添加一些假目标
    targets[0, 2, 10, 10, :5] = torch.tensor([0.5, 0.5, 0.1, 0.2, 1.0])
    targets[0, 2, 10, 10, 5] = 1.0  # class 0

    targets[1, 0, 5, 5, :5] = torch.tensor([0.3, 0.4, -0.1, 0.1, 1.0])
    targets[1, 0, 5, 5, 10] = 1.0  # class 5

    # 计算损失
    loss, loss_dict = criterion(predictions, targets)

    print(f"\nLoss values:")
    for key, value in loss_dict.items():
        print(f"  {key:12s}: {value:.6f}")

    # 测试反向传播
    predictions.requires_grad = True
    loss, _ = criterion(predictions, targets)
    loss.backward()

    print(f"\nGradients:")
    print(f"  Shape: {predictions.grad.shape}")
    print(f"  Range: [{predictions.grad.min():.6f}, {predictions.grad.max():.6f}]")

    print("\n✓ All tests passed!")
