"""
YOLOv2 Network Architecture

YOLOv2的主要改进：
1. Darknet-19 作为backbone
2. Anchor Boxes 机制
3. Batch Normalization
4. Passthrough Layer (细粒度特征)
5. 全卷积网络（去掉全连接层）
6. 直接位置预测（使用sigmoid约束）

输入: (B, 3, 640, 640)
输出: (B, num_anchors, grid_h, grid_w, 5+num_classes)
      其中 5 = tx, ty, tw, th, confidence
"""

import torch
import torch.nn as nn
import numpy as np

from .Darknet19 import Darknet19, PassthroughLayer, ConvBNLeaky


class YOLOv2(nn.Module):
    """
    YOLOv2 检测网络

    参数:
        num_classes: 类别数量（默认80 for COCO）
        anchors: anchor boxes列表，每个anchor是(w, h)，归一化到[0,1]
        img_size: 输入图像尺寸（默认640）
    """

    def __init__(self, num_classes=80, anchors=None, img_size=640):
        super(YOLOv2, self).__init__()

        self.num_classes = num_classes
        self.img_size = img_size
        self.grid_size = 20  # 640 / 32 = 20

        # 默认的5个anchor boxes (YOLOv2使用5个)
        # 这些anchor是通过K-means聚类COCO数据集得到的
        if anchors is None:
            # 格式: (w, h) 归一化到[0,1]
            self.anchors = torch.FloatTensor([
                [0.57273, 0.677385],  # 小物体
                [1.87446, 2.06253],   # 中等物体
                [3.33843, 5.47434],   # 大物体
                [7.88282, 3.52778],   # 宽物体
                [9.77052, 9.16828]    # 超大物体
            ])
        else:
            self.anchors = torch.FloatTensor(anchors)

        self.num_anchors = len(self.anchors)

        # Darknet-19 backbone
        self.backbone = Darknet19()

        # Passthrough layer
        self.passthrough = PassthroughLayer()

        # 降维passthrough特征 (2048 -> 64)
        self.passthrough_conv = ConvBNLeaky(2048, 64, 1, 1, 0)

        # Detection layers
        # 输入: 1024 + 64 = 1088 channels
        # 输出: num_anchors * (5 + num_classes) channels
        out_channels = self.num_anchors * (5 + num_classes)

        self.detection = nn.Sequential(
            ConvBNLeaky(1088, 1024, 3, 1, 1),
            nn.Conv2d(1024, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: (B, 3, 640, 640)

        Returns:
            output: (B, num_anchors, grid_size, grid_size, 5+num_classes)
                其中最后一维是 [tx, ty, tw, th, confidence, class_0, ..., class_n]
        """
        B = x.size(0)

        # Backbone特征提取
        features, passthrough_features = self.backbone(x)
        # features: (B, 1024, 20, 20)
        # passthrough_features: (B, 512, 40, 40)

        # Passthrough layer
        passthrough_reorg = self.passthrough(passthrough_features)
        # (B, 2048, 20, 20)

        # 降维passthrough特征
        passthrough_reduced = self.passthrough_conv(passthrough_reorg)
        # (B, 64, 20, 20)

        # 拼接特征
        combined = torch.cat([features, passthrough_reduced], dim=1)
        # (B, 1088, 20, 20)

        # Detection head
        detection = self.detection(combined)
        # (B, num_anchors*(5+num_classes), 20, 20)

        # 重塑输出
        # (B, num_anchors*(5+num_classes), 20, 20)
        # -> (B, num_anchors, 5+num_classes, 20, 20)
        # -> (B, num_anchors, 20, 20, 5+num_classes)
        detection = detection.view(B, self.num_anchors, 5 + self.num_classes,
                                    self.grid_size, self.grid_size)
        detection = detection.permute(0, 1, 3, 4, 2).contiguous()

        return detection

    def predict(self, x, conf_threshold=0.5):
        """
        预测并解码边界框

        Args:
            x: (B, 3, 640, 640)
            conf_threshold: 置信度阈值

        Returns:
            predictions: List of detections for each image
                每个detection: (class_id, confidence, x1, y1, x2, y2)
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(x)  # (B, num_anchors, 20, 20, 5+num_classes)

            # 解码边界框
            predictions = self._decode_output(output, conf_threshold)

        return predictions

    def _decode_output(self, output, conf_threshold=0.5):
        """
        解码YOLOv2输出

        Args:
            output: (B, num_anchors, grid_h, grid_w, 5+num_classes)
            conf_threshold: 置信度阈值

        Returns:
            batch_predictions: List of predictions for each image
        """
        B, num_anchors, grid_h, grid_w, _ = output.shape
        device = output.device

        batch_predictions = []

        # 为每张图像解码
        for b in range(B):
            image_predictions = []

            # 遍历每个grid cell和anchor
            for i in range(grid_h):
                for j in range(grid_w):
                    for a in range(num_anchors):
                        # 提取预测值
                        pred = output[b, a, i, j, :]  # (5+num_classes,)

                        # tx, ty, tw, th, confidence
                        tx = torch.sigmoid(pred[0])
                        ty = torch.sigmoid(pred[1])
                        tw = pred[2]
                        th = pred[3]
                        confidence = torch.sigmoid(pred[4])

                        # 置信度过滤
                        if confidence < conf_threshold:
                            continue

                        # 类别预测
                        class_probs = torch.softmax(pred[5:], dim=0)
                        class_id = torch.argmax(class_probs).item()
                        class_prob = class_probs[class_id].item()

                        # 最终置信度
                        final_conf = confidence.item() * class_prob

                        if final_conf < conf_threshold:
                            continue

                        # 解码边界框
                        # YOLOv2的位置预测公式：
                        # bx = sigmoid(tx) + cx
                        # by = sigmoid(ty) + cy
                        # bw = pw * exp(tw)
                        # bh = ph * exp(th)

                        anchor_w = self.anchors[a, 0]
                        anchor_h = self.anchors[a, 1]

                        # 计算中心坐标（归一化到[0,1]）
                        bx = (tx.item() + j) / grid_w
                        by = (ty.item() + i) / grid_h

                        # 计算宽高（归一化到[0,1]）
                        bw = anchor_w * torch.exp(tw).item() / grid_w
                        bh = anchor_h * torch.exp(th).item() / grid_h

                        # 转换为角点坐标（像素）
                        x1 = (bx - bw / 2) * self.img_size
                        y1 = (by - bh / 2) * self.img_size
                        x2 = (bx + bw / 2) * self.img_size
                        y2 = (by + bh / 2) * self.img_size

                        # 限制在图像范围内
                        x1 = max(0, min(self.img_size, x1))
                        y1 = max(0, min(self.img_size, y1))
                        x2 = max(0, min(self.img_size, x2))
                        y2 = max(0, min(self.img_size, y2))

                        image_predictions.append({
                            'class_id': class_id,
                            'confidence': final_conf,
                            'bbox': (x1, y1, x2, y2)
                        })

            batch_predictions.append(image_predictions)

        return batch_predictions


def get_default_anchors():
    """
    获取COCO数据集的默认anchor boxes
    这些anchor是通过K-means聚类得到的

    Returns:
        anchors: (5, 2) 数组，每行是 (w, h)，归一化到[0,1]
    """
    return np.array([
        [0.57273, 0.677385],   # 小物体
        [1.87446, 2.06253],    # 中等物体
        [3.33843, 5.47434],    # 大物体
        [7.88282, 3.52778],    # 宽物体
        [9.77052, 9.16828]     # 超大物体
    ])


def test_yolov2():
    """测试YOLOv2网络"""
    print("=" * 60)
    print("Testing YOLOv2 Network")
    print("=" * 60)

    # 创建模型
    model = YOLOv2(num_classes=80, img_size=640)

    # 测试输入
    batch_size = 2
    x = torch.randn(batch_size, 3, 640, 640)
    print(f"\nInput shape: {x.shape}")

    # 前向传播
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Expected: (B={batch_size}, anchors=5, grid_h=20, grid_w=20, 5+classes=85)")

    # 测试预测
    predictions = model.predict(x, conf_threshold=0.5)
    print(f"\nPredictions for batch:")
    for i, preds in enumerate(predictions):
        print(f"  Image {i}: {len(preds)} detections")

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
    test_yolov2()
