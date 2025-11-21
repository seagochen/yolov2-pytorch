"""
YOLOv2 Loss Function

YOLOv2损失函数包含三个部分：
1. 坐标损失 (coordinate loss): tx, ty, tw, th
2. 置信度损失 (confidence loss): objectness score
3. 分类损失 (classification loss): class probabilities

关键公式：
- bx = sigmoid(tx) + cx
- by = sigmoid(ty) + cy
- bw = pw * exp(tw)
- bh = ph * exp(th)

其中cx, cy是grid cell的左上角坐标，pw, ph是anchor的宽高
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class YOLOv2Loss(nn.Module):
    """
    YOLOv2损失函数

    参数:
        num_classes: 类别数量
        anchors: anchor boxes, shape (num_anchors, 2)
        lambda_coord: 坐标损失权重（默认5.0）
        lambda_noobj: 无物体置信度损失权重（默认0.5）
        lambda_class: 分类损失权重（默认1.0）
    """

    def __init__(self, num_classes=80, anchors=None,
                 lambda_coord=5.0, lambda_noobj=0.5, lambda_class=1.0):
        super(YOLOv2Loss, self).__init__()

        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.lambda_class = lambda_class

        if anchors is None:
            # 默认anchors
            self.anchors = torch.FloatTensor([
                [0.57273, 0.677385],
                [1.87446, 2.06253],
                [3.33843, 5.47434],
                [7.88282, 3.52778],
                [9.77052, 9.16828]
            ])
        else:
            self.anchors = anchors

        self.num_anchors = len(self.anchors)

        # 用于计算BCE loss
        self.bce_loss = nn.BCELoss(reduction='sum')
        self.mse_loss = nn.MSELoss(reduction='sum')

    def forward(self, predictions, targets):
        """
        计算YOLOv2损失

        Args:
            predictions: 模型输出
                shape: (B, num_anchors, grid_h, grid_w, 5+num_classes)
            targets: 真实标签
                shape: (B, num_anchors, grid_h, grid_w, 5+num_classes)

        Returns:
            total_loss: 总损失
            loss_dict: 各项损失的字典
        """
        device = predictions.device
        B, num_anchors, grid_h, grid_w, _ = predictions.shape

        # 将anchors移到正确的设备
        if self.anchors.device != device:
            self.anchors = self.anchors.to(device)

        # ========================================================
        # 解析预测和目标
        # ========================================================
        # 预测值
        pred_tx = predictions[..., 0]  # (B, num_anchors, grid_h, grid_w)
        pred_ty = predictions[..., 1]
        pred_tw = predictions[..., 2]
        pred_th = predictions[..., 3]
        pred_conf_raw = predictions[..., 4]  # 原始置信度
        pred_class_raw = predictions[..., 5:]  # (B, num_anchors, grid_h, grid_w, num_classes)

        # 目标值
        target_tx = targets[..., 0]
        target_ty = targets[..., 1]
        target_tw = targets[..., 2]
        target_th = targets[..., 3]
        target_conf = targets[..., 4]
        target_class = targets[..., 5:]

        # ========================================================
        # 创建mask: 区分有物体和无物体的grid cell
        # ========================================================
        obj_mask = target_conf > 0  # (B, num_anchors, grid_h, grid_w)
        noobj_mask = ~obj_mask

        num_obj = obj_mask.sum().float() + 1e-6  # 避免除零

        # ========================================================
        # 1. 坐标损失 (只对有物体的grid计算)
        # ========================================================
        # YOLOv2直接回归tx, ty, tw, th
        # 使用MSE loss

        loss_x = self.mse_loss(
            pred_tx[obj_mask],
            target_tx[obj_mask]
        ) / num_obj

        loss_y = self.mse_loss(
            pred_ty[obj_mask],
            target_ty[obj_mask]
        ) / num_obj

        loss_w = self.mse_loss(
            pred_tw[obj_mask],
            target_tw[obj_mask]
        ) / num_obj

        loss_h = self.mse_loss(
            pred_th[obj_mask],
            target_th[obj_mask]
        ) / num_obj

        loss_coord = self.lambda_coord * (loss_x + loss_y + loss_w + loss_h)

        # ========================================================
        # 2. 置信度损失
        # ========================================================
        # 对置信度应用sigmoid
        pred_conf = torch.sigmoid(pred_conf_raw)

        # 有物体的置信度损失
        loss_conf_obj = self.bce_loss(
            pred_conf[obj_mask],
            target_conf[obj_mask]
        ) / num_obj

        # 无物体的置信度损失
        loss_conf_noobj = self.bce_loss(
            pred_conf[noobj_mask],
            torch.zeros_like(pred_conf[noobj_mask])
        ) / num_obj

        loss_conf = loss_conf_obj + self.lambda_noobj * loss_conf_noobj

        # ========================================================
        # 3. 分类损失 (只对有物体的grid计算)
        # ========================================================
        # 使用softmax + cross entropy
        if obj_mask.sum() > 0:
            # 展平有物体的预测和目标
            pred_class_obj = pred_class_raw[obj_mask]  # (N, num_classes)
            target_class_obj = target_class[obj_mask]  # (N, num_classes)

            # 使用BCE loss (因为target是one-hot)
            loss_class = self.bce_loss(
                torch.sigmoid(pred_class_obj),
                target_class_obj
            ) / num_obj
        else:
            loss_class = torch.tensor(0.0, device=device)

        loss_class = self.lambda_class * loss_class

        # ========================================================
        # 总损失
        # ========================================================
        total_loss = loss_coord + loss_conf + loss_class

        # 返回损失字典
        loss_dict = {
            'total': total_loss.item(),
            'coord': loss_coord.item(),
            'x': loss_x.item(),
            'y': loss_y.item(),
            'w': loss_w.item(),
            'h': loss_h.item(),
            'conf': loss_conf.item(),
            'conf_obj': loss_conf_obj.item(),
            'conf_noobj': loss_conf_noobj.item(),
            'class': loss_class.item()
        }

        return total_loss, loss_dict


def test_yolov2_loss():
    """测试YOLOv2损失函数"""
    print("=" * 60)
    print("Testing YOLOv2 Loss Function")
    print("=" * 60)

    # 参数
    batch_size = 4
    num_anchors = 5
    grid_size = 20
    num_classes = 80

    # 创建损失函数
    loss_fn = YOLOv2Loss(num_classes=num_classes)

    # 随机预测和目标
    predictions = torch.randn(batch_size, num_anchors, grid_size, grid_size, 5 + num_classes)
    targets = torch.zeros(batch_size, num_anchors, grid_size, grid_size, 5 + num_classes)

    # 添加一些假的目标
    targets[0, 2, 10, 10, 0] = 0.5  # tx
    targets[0, 2, 10, 10, 1] = 0.5  # ty
    targets[0, 2, 10, 10, 2] = 0.1  # tw
    targets[0, 2, 10, 10, 3] = 0.2  # th
    targets[0, 2, 10, 10, 4] = 1.0  # confidence
    targets[0, 2, 10, 10, 5] = 1.0  # class 0

    targets[1, 0, 5, 5, 0] = 0.3
    targets[1, 0, 5, 5, 1] = 0.4
    targets[1, 0, 5, 5, 2] = -0.1
    targets[1, 0, 5, 5, 3] = 0.1
    targets[1, 0, 5, 5, 4] = 1.0
    targets[1, 0, 5, 5, 10] = 1.0  # class 10

    # 计算损失
    total_loss, loss_dict = loss_fn(predictions, targets)

    print(f"\nInput shapes:")
    print(f"  Predictions: {predictions.shape}")
    print(f"  Targets: {targets.shape}")

    print(f"\nLoss values:")
    for key, value in loss_dict.items():
        print(f"  {key:12s}: {value:.6f}")

    print(f"\nTotal loss: {total_loss.item():.6f}")

    # 测试反向传播
    predictions.requires_grad = True
    total_loss, _ = loss_fn(predictions, targets)
    total_loss.backward()

    print(f"\nGradient check:")
    print(f"  Predictions grad shape: {predictions.grad.shape}")
    print(f"  Gradient range: [{predictions.grad.min():.6f}, {predictions.grad.max():.6f}]")

    print("\n" + "=" * 60)
    print("✓ Test passed!")
    print("=" * 60)


if __name__ == "__main__":
    test_yolov2_loss()
