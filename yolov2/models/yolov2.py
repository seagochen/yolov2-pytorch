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
        self.grid_size = img_size // 32  # 默认20 for 640x640
        self.conf_threshold = conf_threshold

        # 初始化anchors
        if anchors is None:
            self.anchors = torch.tensor([
                [0.57273, 0.677385],
                [1.87446, 2.06253],
                [3.33843, 5.47434],
                [7.88282, 3.52778],
                [9.77052, 9.16828]
            ], dtype=torch.float32)
        else:
            self.anchors = torch.from_numpy(anchors).float()

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

        # 重塑输出
        detection = detection.view(
            B,
            self.num_anchors,
            5 + self.num_classes,
            self.grid_size,
            self.grid_size
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
        预测并解码边界框

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

        # 确保模型在评估模式
        self.eval()

        # 前向传播
        output = self.forward(x)  # (B, num_anchors, grid_h, grid_w, 5+num_classes)

        # 解码
        predictions = self._decode_predictions(
            output,
            conf_threshold=conf_threshold,
            device=device
        )

        return predictions

    def _decode_predictions(
        self,
        output: torch.Tensor,
        conf_threshold: float,
        device: torch.device
    ) -> List[List[Dict]]:
        """
        解码YOLO输出为边界框

        Args:
            output: (B, num_anchors, grid_h, grid_w, 5+num_classes)
            conf_threshold: 置信度阈值
            device: 设备

        Returns:
            batch_predictions: 预测列表
        """
        B, num_anchors, grid_h, grid_w, _ = output.shape

        # 移动anchors到正确设备
        if self.anchors.device != device:
            self.anchors = self.anchors.to(device)

        batch_predictions = []

        for b in range(B):
            image_predictions = []

            for i in range(grid_h):
                for j in range(grid_w):
                    for a in range(num_anchors):
                        # 提取预测
                        pred = output[b, a, i, j, :]  # (5+num_classes,)

                        # 解析
                        tx, ty, tw, th, conf_raw = pred[:5]
                        class_raw = pred[5:]

                        # 应用sigmoid到置信度
                        conf = torch.sigmoid(conf_raw)

                        if conf < conf_threshold:
                            continue

                        # 类别预测
                        class_probs = torch.softmax(class_raw, dim=0)
                        class_id = torch.argmax(class_probs).item()
                        class_prob = class_probs[class_id].item()

                        # 最终置信度
                        final_conf = conf.item() * class_prob

                        if final_conf < conf_threshold:
                            continue

                        # 解码边界框（YOLOv2公式）
                        anchor_w, anchor_h = self.anchors[a]

                        # bx = sigmoid(tx) + cx
                        # by = sigmoid(ty) + cy
                        # bw = pw * exp(tw)
                        # bh = ph * exp(th)

                        bx = (torch.sigmoid(tx).item() + j) / grid_w
                        by = (torch.sigmoid(ty).item() + i) / grid_h

                        bw = anchor_w * torch.exp(tw).item() / grid_w
                        bh = anchor_h * torch.exp(th).item() / grid_h

                        # 转换为像素坐标 (x1, y1, x2, y2)
                        x1 = (bx - bw / 2) * self.img_size
                        y1 = (by - bh / 2) * self.img_size
                        x2 = (bx + bw / 2) * self.img_size
                        y2 = (by + bh / 2) * self.img_size

                        # 裁剪到图像范围
                        x1 = max(0, min(self.img_size, x1))
                        y1 = max(0, min(self.img_size, y1))
                        x2 = max(0, min(self.img_size, x2))
                        y2 = max(0, min(self.img_size, y2))

                        # 添加检测结果
                        image_predictions.append({
                            'class_id': class_id,
                            'confidence': final_conf,
                            'bbox': (x1, y1, x2, y2)
                        })

            batch_predictions.append(image_predictions)

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
            'grid_size': self.grid_size,
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
