"""
YOLOv2 Inference Script

支持:
- 单张图像检测
- 批量图像检测
- 数据集验证
- NMS后处理
"""

import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from YoloVer2.model.YOLOv2 import YOLOv2


def nms(detections, iou_threshold=0.5):
    """
    非极大值抑制 (NMS)

    Args:
        detections: List of dict, 每个dict包含 'class_id', 'confidence', 'bbox'
        iou_threshold: IoU阈值

    Returns:
        filtered_detections: NMS后的检测结果
    """
    if len(detections) == 0:
        return []

    # 按置信度排序
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

    filtered = []

    while len(detections) > 0:
        # 取置信度最高的
        best = detections.pop(0)
        filtered.append(best)

        # 计算与剩余框的IoU
        best_box = best['bbox']

        remaining = []
        for det in detections:
            det_box = det['bbox']

            # 计算IoU
            iou = compute_iou(best_box, det_box)

            # 如果IoU小于阈值，保留
            if iou < iou_threshold or det['class_id'] != best['class_id']:
                remaining.append(det)

        detections = remaining

    return filtered


def compute_iou(box1, box2):
    """
    计算两个框的IoU

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        iou: IoU值
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # 计算交集
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)

    # 计算并集
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / (union_area + 1e-6)
    return iou


def draw_detections(image, detections, class_names, thickness=2):
    """
    在图像上绘制检测结果

    Args:
        image: 图像 (H, W, 3) numpy array
        detections: 检测结果 List of dict
        class_names: 类别名称列表
        thickness: 线条粗细

    Returns:
        image: 绘制后的图像
    """
    image = image.copy()
    h, w = image.shape[:2]

    # 颜色
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128),
        (128, 128, 0), (128, 0, 128), (0, 128, 128)
    ]

    for det in detections:
        class_id = det['class_id']
        confidence = det['confidence']
        x1, y1, x2, y2 = det['bbox']

        # 转换为整数
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        # 选择颜色
        color = colors[class_id % len(colors)]

        # 绘制边界框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

        # 绘制标签
        class_name = class_names[class_id] if class_id < len(class_names) else f'class_{class_id}'
        label = f'{class_name}: {confidence:.2f}'

        # 计算标签背景大小
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )

        # 绘制标签背景
        cv2.rectangle(
            image,
            (x1, y1 - label_h - baseline - 5),
            (x1 + label_w, y1),
            color,
            -1
        )

        # 绘制标签文字
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

    return image


def run_inference_on_image(model, image_path, device, class_names,
                            conf_threshold=0.5, nms_threshold=0.5,
                            output_dir=None, show=False):
    """
    对单张图像进行推理

    Args:
        model: YOLOv2模型
        image_path: 图像路径
        device: 设备
        class_names: 类别名称
        conf_threshold: 置信度阈值
        nms_threshold: NMS阈值
        output_dir: 输出目录
        show: 是否显示结果
    """
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f'Failed to load image: {image_path}')
        return

    # 预处理
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, (640, 640))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)  # (1, 3, 640, 640)

    # 推理
    model.eval()
    batch_detections = model.predict(tensor, conf_threshold=conf_threshold)
    detections = batch_detections[0]

    # NMS
    detections = nms(detections, iou_threshold=nms_threshold)

    print(f'\nImage: {image_path.name}')
    print(f'Detected {len(detections)} objects (after NMS)')

    # 绘制结果
    result_image = draw_detections(resized, detections, class_names)

    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'det_{image_path.name}')
        cv2.imwrite(output_path, result_image)
        print(f'Saved to: {output_path}')

    # 显示结果
    if show:
        cv2.imshow('YOLOv2 Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


def main():
    parser = argparse.ArgumentParser(description='YOLOv2 Inference')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--data', type=str, default='data/coco.yaml',
                        help='COCO数据集YAML配置文件')
    parser.add_argument('--source', type=str, default='',
                        help='图像路径或目录')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--nms-threshold', type=float, default=0.5,
                        help='NMS IoU阈值')
    parser.add_argument('--output-dir', type=str, default='runs/detect_v2',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--show', action='store_true',
                        help='显示结果')

    args = parser.parse_args()

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # =====================================================
    # 加载模型
    # =====================================================
    print(f'\nLoading model from: {args.weights}')

    checkpoint = torch.load(args.weights, map_location=device)

    # 获取anchors
    anchors = checkpoint.get('anchors', None)
    if anchors is None:
        from YoloVer2.model.YOLOv2 import get_default_anchors
        anchors = get_default_anchors()
        print('Using default anchors')
    else:
        print('Using anchors from checkpoint')

    # 创建模型
    model = YOLOv2(
        num_classes=80,
        anchors=anchors,
        img_size=640
    )

    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print('Model loaded successfully')

    # =====================================================
    # 加载类别名称
    # =====================================================
    if os.path.exists(args.data):
        import yaml
        with open(args.data, 'r') as f:
            config = yaml.safe_load(f)
        class_names = config.get('names', [f'class_{i}' for i in range(80)])
    else:
        class_names = [f'class_{i}' for i in range(80)]

    # =====================================================
    # 推理
    # =====================================================
    if args.source:
        source_path = Path(args.source)

        if source_path.is_file():
            # 单张图像
            run_inference_on_image(
                model, source_path, device, class_names,
                conf_threshold=args.conf_threshold,
                nms_threshold=args.nms_threshold,
                output_dir=args.output_dir,
                show=args.show
            )
        elif source_path.is_dir():
            # 目录中的所有图像
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                image_files.extend(list(source_path.glob(ext)))
                image_files.extend(list(source_path.glob(ext.upper())))

            print(f'\nFound {len(image_files)} images in {source_path}')

            for img_path in tqdm(image_files, desc='Processing'):
                run_inference_on_image(
                    model, img_path, device, class_names,
                    conf_threshold=args.conf_threshold,
                    nms_threshold=args.nms_threshold,
                    output_dir=args.output_dir,
                    show=False
                )
        else:
            print(f'Invalid source: {args.source}')
    else:
        print('Please specify --source for inference')

    print('\nInference completed!')


if __name__ == '__main__':
    main()
