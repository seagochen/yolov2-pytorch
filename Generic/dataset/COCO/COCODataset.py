"""
COCO Dataset Loader for YOLOv1
支持Ultralytics格式的YAML配置和TXT标注文件

YAML格式示例:
    path: /path/to/dataset
    train: images/train
    val: images/val
    test: images/test

    nc: 80  # number of classes
    names: ['person', 'bicycle', 'car', ...]

TXT标注格式 (每行一个对象):
    class_id center_x center_y width height
    其中坐标都是归一化到[0,1]的值
"""

import os
import yaml
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class COCODataset(Dataset):
    """
    COCO格式数据集加载器

    参数:
        yaml_path: YAML配置文件路径
        split: 'train', 'val', 或 'test'
        img_size: 目标图像尺寸 (默认640x640)
        grids_size: YOLO网格尺寸 (默认20x20)
        bounding_boxes: 每个格子的边界框数量 (默认2)
        augment: 是否进行数据增强
    """

    def __init__(self, yaml_path, split='train', img_size=640,
                 grids_size=(20, 20), bounding_boxes=2, augment=False):
        super(COCODataset, self).__init__()

        self.img_size = img_size
        self.grids_size = grids_size
        self.grid_h, self.grid_w = grids_size
        self.bounding_boxes = bounding_boxes
        self.augment = augment
        self.split = split

        # 读取YAML配置
        with open(yaml_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # 获取数据集路径
        self.dataset_root = Path(self.config.get('path', '')).resolve()
        if not self.dataset_root.exists():
            self.dataset_root = Path(yaml_path).parent

        # 获取类别信息
        self.num_classes = self.config['nc']
        self.class_names = self.config['names']

        # 获取图像路径
        img_dir = self.config.get(split, f'images/{split}')
        self.img_dir = self.dataset_root / img_dir

        # 获取标签路径 (通常是images替换为labels)
        self.label_dir = self.dataset_root / img_dir.replace('images', 'labels')

        # 扫描所有图像文件
        self.img_files = self._load_image_files()

        # 计算网格单元大小
        self.grid_cell_w = self.img_size / self.grid_w
        self.grid_cell_h = self.img_size / self.grid_h

        print(f"[COCODataset] Loaded {len(self.img_files)} images from {self.img_dir}")
        print(f"[COCODataset] Classes: {self.num_classes}, Grid: {grids_size}, BBoxes: {bounding_boxes}")

    def _load_image_files(self):
        """扫描图像文件"""
        img_files = []
        if not self.img_dir.exists():
            print(f"Warning: Image directory not found: {self.img_dir}")
            return img_files

        # 支持的图像格式
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

        for ext in valid_extensions:
            img_files.extend(list(self.img_dir.glob(f'*{ext}')))
            img_files.extend(list(self.img_dir.glob(f'*{ext.upper()}')))

        img_files = sorted(img_files)
        return img_files

    def _load_label(self, label_path):
        """
        加载TXT标注文件
        返回: List of (class_id, cx, cy, w, h) 归一化坐标
        """
        if not label_path.exists():
            return []

        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])

                # 验证坐标范围
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    continue

                labels.append((class_id, cx, cy, w, h))

        return labels

    def _encode_labels_to_grid(self, labels):
        """
        将标注转换为YOLO网格格式

        输出格式: (C, G) 其中
            C = confidences + bboxes*4 + classes = 1 + 2*4 + num_classes
            G = grid_h * grid_w

        每个格子包含:
            [conf, bbox1_cx_rel, bbox1_cy_rel, bbox1_w, bbox1_h,
                   bbox2_cx_rel, bbox2_cy_rel, bbox2_w, bbox2_h,
             class_0, class_1, ..., class_n]
        """
        # 计算每个格子的特征数
        features_per_cell = 1 + self.bounding_boxes * 4 + self.num_classes

        # 初始化网格 (C, G)
        grid_cells = self.grid_h * self.grid_w
        grid_labels = torch.zeros(features_per_cell, grid_cells)

        for class_id, cx, cy, w, h in labels:
            # 计算对象中心所在的网格
            grid_x = int(cx * self.grid_w)
            grid_y = int(cy * self.grid_h)
            grid_x = min(grid_x, self.grid_w - 1)
            grid_y = min(grid_y, self.grid_h - 1)

            grid_idx = grid_y * self.grid_w + grid_x

            # 计算相对于格子的中心坐标
            grid_left = grid_x * self.grid_cell_w
            grid_top = grid_y * self.grid_cell_h

            cx_abs = cx * self.img_size
            cy_abs = cy * self.img_size

            cx_rel = (cx_abs - grid_left) / self.grid_cell_w
            cy_rel = (cy_abs - grid_top) / self.grid_cell_h

            # 限制相对坐标在[0, 1]范围内
            cx_rel = max(0.0, min(1.0, cx_rel))
            cy_rel = max(0.0, min(1.0, cy_rel))

            # 检查该格子是否已有对象
            if grid_labels[0, grid_idx] == 0:
                # 第一个边界框
                grid_labels[0, grid_idx] = 1.0  # confidence
                grid_labels[1, grid_idx] = cx_rel
                grid_labels[2, grid_idx] = cy_rel
                grid_labels[3, grid_idx] = w
                grid_labels[4, grid_idx] = h
            elif self.bounding_boxes > 1 and grid_labels[5, grid_idx] == 0:
                # 第二个边界框 (如果支持)
                # 注意：这里不设置confidence，因为YOLO v1只对每个格子设置一个confidence
                # 但我们保留两个bbox的能力
                grid_labels[5, grid_idx] = cx_rel
                grid_labels[6, grid_idx] = cy_rel
                grid_labels[7, grid_idx] = w
                grid_labels[8, grid_idx] = h

            # 设置类别 (one-hot编码)
            class_start_idx = 1 + self.bounding_boxes * 4
            if 0 <= class_id < self.num_classes:
                grid_labels[class_start_idx + class_id, grid_idx] = 1.0

        return grid_labels

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        返回:
            image: (3, img_size, img_size) 归一化到[0, 1]
            labels: (C, G) YOLO网格格式标签
        """
        # 加载图像
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            # 返回空白图像和标签
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            labels = []
        else:
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 加载标签
            label_path = self.label_dir / (img_path.stem + '.txt')
            labels = self._load_label(label_path)

        # 调整图像大小到目标尺寸
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 数据增强 (可选)
        if self.augment and self.split == 'train':
            img = self._augment_image(img)

        # 转换为tensor并归一化
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 编码标签到网格
        grid_labels = self._encode_labels_to_grid(labels)

        return img, grid_labels

    def _augment_image(self, img):
        """
        简单的数据增强
        - 随机亮度调整
        - 随机对比度调整
        """
        # 随机亮度 (±30)
        if np.random.rand() > 0.5:
            brightness = np.random.randint(-30, 30)
            img = np.clip(img.astype(np.int32) + brightness, 0, 255).astype(np.uint8)

        # 随机对比度
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)

        return img

    def get_class_name(self, class_id):
        """获取类别名称"""
        if 0 <= class_id < len(self.class_names):
            return self.class_names[class_id]
        return f"class_{class_id}"


def create_coco_yaml_template(save_path='data/coco.yaml'):
    """
    创建COCO数据集YAML配置模板
    """
    template = """# COCO Dataset Configuration for YOLOv1
# 数据集根目录
path: /path/to/coco/dataset

# 数据集划分
train: images/train2017
val: images/val2017
test: images/test2017

# 类别数量
nc: 80

# 类别名称
names:
  - person
  - bicycle
  - car
  - motorcycle
  - airplane
  - bus
  - train
  - truck
  - boat
  - traffic light
  - fire hydrant
  - stop sign
  - parking meter
  - bench
  - bird
  - cat
  - dog
  - horse
  - sheep
  - cow
  - elephant
  - bear
  - zebra
  - giraffe
  - backpack
  - umbrella
  - handbag
  - tie
  - suitcase
  - frisbee
  - skis
  - snowboard
  - sports ball
  - kite
  - baseball bat
  - baseball glove
  - skateboard
  - surfboard
  - tennis racket
  - bottle
  - wine glass
  - cup
  - fork
  - knife
  - spoon
  - bowl
  - banana
  - apple
  - sandwich
  - orange
  - broccoli
  - carrot
  - hot dog
  - pizza
  - donut
  - cake
  - chair
  - couch
  - potted plant
  - bed
  - dining table
  - toilet
  - tv
  - laptop
  - mouse
  - remote
  - keyboard
  - cell phone
  - microwave
  - oven
  - toaster
  - sink
  - refrigerator
  - book
  - clock
  - vase
  - scissors
  - teddy bear
  - hair drier
  - toothbrush
"""

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(template)

    print(f"YAML template created at: {save_path}")


if __name__ == "__main__":
    # 测试代码
    print("Creating COCO YAML template...")
    create_coco_yaml_template('data/coco.yaml')

    print("\nExample usage:")
    print("  dataset = COCODataset('data/coco.yaml', split='train')")
    print("  img, label = dataset[0]")
    print("  print(img.shape, label.shape)")
