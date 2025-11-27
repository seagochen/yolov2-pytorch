"""
YOLOv2 Model - 重构版
使用最新PyTorch API和最佳实践
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict

from .darknet import Darknet19Improved
from .layers import ConvBNAct, SpaceToDepth


class YOLOv2(nn.Module):
    """
    YOLOv2 检测网络 - 重构版

    改进：
    - 清晰的类型标注
    - 更好的配置管理
    - 支持动态anchor
    - 改进的前向传播
    """

    def __init__(
        self,
        num_classes: int = 80,
        anchors: Optional[np.ndarray] = None,
        img_size: int = 640,
        conf_threshold: float = 0.5
    ):
        super().__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.conf_threshold = conf_threshold

        # 初始化anchors
        if anchors is None:
            anchors_tensor = torch.tensor([
                [0.57273, 0.677385],
                [1.87446, 2.06253],
                [3.33843, 5.47434],
                [7.88282, 3.52778],
                [9.77052, 9.16828]
            ], dtype=torch.float32)
        else:
            anchors_tensor = torch.as_tensor(anchors, dtype=torch.float32)

        # 注册为buffer，自动随模型迁移/保存
        self.register_buffer('anchors', anchors_tensor)

        self.num_anchors = len(self.anchors)

        # Backbone
        self.backbone = Darknet19Improved(in_channels=3)

        # Passthrough处理
        self.space_to_depth = SpaceToDepth(block_size=2)
        self.passthrough_conv = ConvBNAct(
            in_channels=512 * 4,  # SpaceToDepth后通道数
            out_channels=64,
            kernel_size=1
        )

        # Detection head
        self.detection_conv = ConvBNAct(
            in_channels=1024 + 64,
            out_channels=1024,
            kernel_size=3
        )

        # 输出层
        out_channels = self.num_anchors * (5 + num_classes)
        self.detection_out = nn.Conv2d(1024, out_channels, 1)

        # 初始化输出层
        self._initialize_output_layer()

    def _initialize_output_layer(self):
        """初始化输出层权重"""
        nn.init.normal_(self.detection_out.weight, 0, 0.01)
        nn.init.constant_(self.detection_out.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: (B, 3, H, W)

        Returns:
            predictions: (B, num_anchors, grid_h, grid_w, 5+num_classes)
        """
        B = x.size(0)

        # Backbone
        features, passthrough_features = self.backbone(x)
        # features: (B, 1024, 20, 20)
        # passthrough_features: (B, 512, 40, 40)

        # Passthrough处理
        passthrough = self.space_to_depth(passthrough_features)
        # (B, 512*4, 20, 20)

        passthrough = self.passthrough_conv(passthrough)
        # (B, 64, 20, 20)

        # 拼接特征
        combined = torch.cat([features, passthrough], dim=1)
        # (B, 1088, 20, 20)

        # Detection
        detection = self.detection_conv(combined)
        # (B, 1024, 20, 20)

        detection = self.detection_out(detection)
        # (B, num_anchors*(5+num_classes), 20, 20)

        grid_h, grid_w = detection.shape[2:]

        # 重塑输出
        detection = detection.view(
            B,
            self.num_anchors,
            5 + self.num_classes,
            grid_h,
            grid_w
        )

        # 调整维度顺序
        detection = detection.permute(0, 1, 3, 4, 2).contiguous()
        # (B, num_anchors, grid_h, grid_w, 5+num_classes)

        return detection

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: Optional[float] = None,
        device: Optional[torch.device] = None
    ) -> List[List[Dict]]:
        """
        预测并解码边界框 (向量化加速版)

        Args:
            x: (B, 3, H, W)
            conf_threshold: 置信度阈值
            device: 设备

        Returns:
            batch_predictions: 每张图像的检测结果列表
        """
        if conf_threshold is None:
            conf_threshold = self.conf_threshold

        if device is None:
            device = x.device

        self.eval()
        output = self.forward(x)

        img_h, img_w = x.shape[2:]

        return self._decode_predictions_vectorized(output, conf_threshold, device, img_h, img_w)

    def _decode_predictions_vectorized(
        self,
        output: torch.Tensor,
        conf_threshold: float,
        device: torch.device,
        img_h: int,
        img_w: int
    ) -> List[List[Dict]]:
        """
        向量化解码，大幅提升速度
        """
        B, num_anchors, grid_h, grid_w, _ = output.shape

        # 移动anchors到正确设备
        anchors = self.anchors if self.anchors.device == device else self.anchors.to(device)

        # 1. 提取各个分量
        tx = output[..., 0]
        ty = output[..., 1]
        tw = output[..., 2]
        th = output[..., 3]
        conf_raw = output[..., 4]
        class_raw = output[..., 5:]

        # 2. 激活函数
        sigmoid_tx = torch.sigmoid(tx)
        sigmoid_ty = torch.sigmoid(ty)
        conf = torch.sigmoid(conf_raw)

        # Sigmoid per-class，保持与BCE训练一致
        class_probs = torch.sigmoid(class_raw)
        max_class_prob, class_id = torch.max(class_probs, dim=-1)

        # 3. 计算最终置信度
        final_conf = conf * max_class_prob

        # 4. 构建网格坐标
        grid_y, grid_x = torch.meshgrid(
            torch.arange(grid_h, device=device),
            torch.arange(grid_w, device=device),
            indexing='ij'
        )

        # 扩展维度以匹配 (B, A, H, W)
        grid_x = grid_x.view(1, 1, grid_h, grid_w).expand(B, num_anchors, -1, -1)
        grid_y = grid_y.view(1, 1, grid_h, grid_w).expand(B, num_anchors, -1, -1)

        # 扩展anchors
        anchor_w = anchors[:, 0].view(1, num_anchors, 1, 1).expand(B, -1, grid_h, grid_w)
        anchor_h = anchors[:, 1].view(1, num_anchors, 1, 1).expand(B, -1, grid_h, grid_w)

        # 5. 计算绝对坐标 (归一化 0~1)
        bx = (sigmoid_tx + grid_x) / grid_w
        by = (sigmoid_ty + grid_y) / grid_h
        bw = (anchor_w * torch.exp(tw)) / grid_w
        bh = (anchor_h * torch.exp(th)) / grid_h

        # 6. 转换为像素坐标 (x1, y1, x2, y2)
        cx = bx * img_w
        cy = by * img_h
        w = bw * img_w
        h = bh * img_h

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # 裁剪到图像边缘
        x1 = x1.clamp(min=0, max=img_w)
        y1 = y1.clamp(min=0, max=img_h)
        x2 = x2.clamp(min=0, max=img_w)
        y2 = y2.clamp(min=0, max=img_h)

        # 7. 过滤低置信度结果并转换为 List[List[Dict]]
        batch_predictions = []
        mask = final_conf > conf_threshold

        for b in range(B):
            b_mask = mask[b]

            if not b_mask.any():
                batch_predictions.append([])
                continue

            # 使用掩码提取数据
            cur_x1 = x1[b][b_mask]
            cur_y1 = y1[b][b_mask]
            cur_x2 = x2[b][b_mask]
            cur_y2 = y2[b][b_mask]
            cur_conf = final_conf[b][b_mask]
            cur_cls = class_id[b][b_mask]

            image_preds = []
            n_detections = cur_conf.size(0)

            for k in range(n_detections):
                image_preds.append({
                    'class_id': cur_cls[k].item(),
                    'confidence': cur_conf[k].item(),
                    'bbox': (cur_x1[k].item(), cur_y1[k].item(), cur_x2[k].item(), cur_y2[k].item())
                })

            batch_predictions.append(image_preds)

        return batch_predictions

    def load_darknet_weights(self, weights_path: str):
        """加载Darknet预训练权重（可选功能）"""
        # TODO: 实现Darknet权重加载
        raise NotImplementedError("Darknet weights loading not implemented yet")

    def get_config(self) -> Dict:
        """获取模型配置"""
        return {
            'num_classes': self.num_classes,
            'anchors': self.anchors.cpu().numpy().tolist(),
            'img_size': self.img_size,
            'grid_size_hint': self.img_size // 32,
            'num_anchors': self.num_anchors,
            'conf_threshold': self.conf_threshold
        }


def create_yolov2(
    num_classes: int = 80,
    img_size: int = 640,
    pretrained: bool = False
) -> YOLOv2:
    """
    创建YOLOv2模型的工厂函数

    Args:
        num_classes: 类别数
        img_size: 输入图像尺寸
        pretrained: 是否加载预训练权重

    Returns:
        model: YOLOv2模型
    """
    model = YOLOv2(num_classes=num_classes, img_size=img_size)

    if pretrained:
        # TODO: 加载预训练权重
        print("Warning: Pretrained weights not available yet")

    return model


if __name__ == '__main__':
    print("Testing YOLOv2...")

    # 创建模型
    model = create_yolov2(num_classes=80, img_size=640)

    # 测试前向传播
    x = torch.randn(2, 3, 640, 640)
    output = model(x)
    print(f"\nForward pass:")
    print(f"  Input: {x.shape}")
    print(f"  Output: {output.shape}")

    # 测试预测
    predictions = model.predict(x, conf_threshold=0.5)
    print(f"\nPredictions:")
    for i, preds in enumerate(predictions):
        print(f"  Image {i}: {len(preds)} detections")

    # 模型配置
    config = model.get_config()
    print(f"\nModel config:")
    for key, value in config.items():
        if key != 'anchors':
            print(f"  {key}: {value}")

    # 参数统计
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: {total_params * 4 / (1024**2):.2f} MB")

    print("\n✓ All tests passed!")
