"""
YOLOv1 V102 推理脚本
支持COCO数据集的目标检测和可视化

功能:
- 加载训练好的模型
- 对图像进行目标检测
- 可视化检测结果
- 支持单张图像或批量图像处理
"""

import os
import argparse
import torch
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from YoloVer102.model.YoloNetworkV102 import YoloV1NetworkV102
from Generic.dataset.COCO.COCODataset import COCODataset


def parse_yolo_output(predictions, conf_threshold=0.5, grid_size=(20, 20), img_size=640):
    """
    解析YOLO输出

    参数:
        predictions: 模型输出 (B, C, G)
        conf_threshold: 置信度阈值
        grid_size: 网格尺寸
        img_size: 图像尺寸

    返回:
        detections: List of List of (class_id, confidence, bbox)
            bbox = (x1, y1, x2, y2) 绝对坐标
    """
    B, C, G = predictions.shape
    grid_h, grid_w = grid_size
    grid_cell_w = img_size / grid_w
    grid_cell_h = img_size / grid_h

    batch_detections = []

    for b in range(B):
        image_detections = []

        for g in range(G):
            # 获取网格索引
            grid_y = g // grid_w
            grid_x = g % grid_w

            # 解析预测
            pred = predictions[b, :, g]
            confidence = pred[0].item()

            # 置信度过滤
            if confidence < conf_threshold:
                continue

            # 解析两个边界框
            bboxes = [
                pred[1:5].cpu().numpy(),  # bbox1
                pred[5:9].cpu().numpy()   # bbox2
            ]

            # 解析类别
            class_probs = pred[9:].cpu().numpy()
            class_id = np.argmax(class_probs)
            class_prob = class_probs[class_id]

            # 选择第一个边界框 (简化处理)
            # TODO: 可以改进为选择IoU更高或置信度更高的框
            bbox = bboxes[0]

            # 转换坐标
            cx_rel, cy_rel, w, h = bbox

            # 计算绝对中心坐标
            grid_left = grid_x * grid_cell_w
            grid_top = grid_y * grid_cell_h

            cx_abs = grid_left + cx_rel * grid_cell_w
            cy_abs = grid_top + cy_rel * grid_cell_h

            # 计算边界框
            w_abs = w * img_size
            h_abs = h * img_size

            x1 = cx_abs - w_abs / 2
            y1 = cy_abs - h_abs / 2
            x2 = cx_abs + w_abs / 2
            y2 = cy_abs + h_abs / 2

            # 限制在图像范围内
            x1 = max(0, min(img_size, x1))
            y1 = max(0, min(img_size, y1))
            x2 = max(0, min(img_size, x2))
            y2 = max(0, min(img_size, y2))

            # 添加检测结果
            image_detections.append({
                'class_id': int(class_id),
                'confidence': float(confidence * class_prob),
                'bbox': (x1, y1, x2, y2)
            })

        batch_detections.append(image_detections)

    return batch_detections


def draw_detections(image, detections, class_names, thickness=2):
    """
    在图像上绘制检测结果

    参数:
        image: 图像 (H, W, 3) numpy array
        detections: 检测结果 List of dict
        class_names: 类别名称列表
        thickness: 线条粗细

    返回:
        image: 绘制后的图像
    """
    image = image.copy()
    h, w = image.shape[:2]

    # 颜色 (BGR)
    colors = [
        (255, 0, 0), (0, 255, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
        (128, 0, 0), (0, 128, 0), (0, 0, 128)
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


def run_inference_on_image(model, image_path, device, class_names, conf_threshold=0.5,
                            output_dir=None, show=True):
    """
    对单张图像进行推理

    参数:
        model: YOLO模型
        image_path: 图像路径
        device: 设备
        class_names: 类别名称
        conf_threshold: 置信度阈值
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
    with torch.no_grad():
        predictions = model(tensor)

    # 解析输出
    batch_detections = parse_yolo_output(
        predictions,
        conf_threshold=conf_threshold,
        grid_size=(20, 20),
        img_size=640
    )
    detections = batch_detections[0]

    print(f'\nImage: {image_path.name}')
    print(f'Detected {len(detections)} objects')

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
        cv2.imshow('YOLOv1 Detection', result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return detections


def run_inference_on_dataset(model, dataset, device, class_names, conf_threshold=0.5,
                              output_dir=None, num_images=10):
    """
    对数据集中的图像进行推理

    参数:
        model: YOLO模型
        dataset: COCODataset
        device: 设备
        class_names: 类别名称
        conf_threshold: 置信度阈值
        output_dir: 输出目录
        num_images: 处理的图像数量
    """
    model.eval()

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    num_images = min(num_images, len(dataset))

    for idx in tqdm(range(num_images), desc='Processing'):
        # 获取数据
        image, label = dataset[idx]
        img_path = dataset.img_files[idx]

        # 推理
        tensor = image.unsqueeze(0).to(device)  # (1, 3, 640, 640)
        with torch.no_grad():
            predictions = model(tensor)

        # 解析输出
        batch_detections = parse_yolo_output(
            predictions,
            conf_threshold=conf_threshold,
            grid_size=(20, 20),
            img_size=640
        )
        detections = batch_detections[0]

        # 转换为numpy图像用于绘制
        image_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 绘制结果
        result_image = draw_detections(image_bgr, detections, class_names)

        # 保存结果
        if output_dir:
            output_path = os.path.join(output_dir, f'det_{idx:04d}_{img_path.name}')
            cv2.imwrite(output_path, result_image)

        print(f'[{idx+1}/{num_images}] {img_path.name}: {len(detections)} objects')


def main():
    parser = argparse.ArgumentParser(description='YOLOv1 V102 Inference')
    parser.add_argument('--weights', type=str, required=True,
                        help='模型权重路径')
    parser.add_argument('--data', type=str, default='data/coco.yaml',
                        help='COCO数据集YAML配置文件')
    parser.add_argument('--source', type=str, default='',
                        help='图像路径或目录 (如果为空则使用数据集)')
    parser.add_argument('--conf-threshold', type=float, default=0.5,
                        help='置信度阈值')
    parser.add_argument('--output-dir', type=str, default='runs/detect',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--num-images', type=int, default=10,
                        help='处理的图像数量 (数据集模式)')
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
    model = YoloV1NetworkV102(
        grids_size=(20, 20),
        confidences=1,
        bounding_boxes=2,
        object_categories=80
    )

    checkpoint = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

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
                    output_dir=args.output_dir,
                    show=False
                )
        else:
            print(f'Invalid source: {args.source}')
    else:
        # 使用数据集
        print(f'\nLoading dataset from: {args.data}')
        dataset = COCODataset(
            yaml_path=args.data,
            split='val',
            img_size=640,
            grids_size=(20, 20),
            bounding_boxes=2,
            augment=False
        )

        run_inference_on_dataset(
            model, dataset, device, class_names,
            conf_threshold=args.conf_threshold,
            output_dir=args.output_dir,
            num_images=args.num_images
        )

    print('\nInference completed!')


if __name__ == '__main__':
    main()
