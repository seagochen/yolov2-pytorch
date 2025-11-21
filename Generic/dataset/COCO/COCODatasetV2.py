"""
COCO Dataset Loader for YOLOv2
支持Anchor Boxes机制的数据加载器

与YOLOv1的主要区别：
- 使用anchor boxes
- 标签格式: (num_anchors, grid_h, grid_w, 5+num_classes)
- 需要计算与anchor的IoU来分配ground truth
"""

import os
import yaml
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path


class COCODatasetV2(Dataset):
    """
    COCO格式数据集加载器 - YOLOv2版本

    参数:
        yaml_path: YAML配置文件路径
        split: 'train', 'val', 或 'test'
        img_size: 目标图像尺寸 (默认640x640)
        grid_size: YOLO网格尺寸 (默认20)
        anchors: anchor boxes列表 [(w, h), ...], 归一化到[0,1]
        augment: 是否进行数据增强
    """

    def __init__(self, yaml_path, split='train', img_size=640,
                 grid_size=20, anchors=None, augment=False):
        super(COCODatasetV2, self).__init__()

        self.img_size = img_size
        self.grid_size = grid_size
        self.augment = augment
        self.split = split

        # 默认anchors (YOLOv2的5个anchor)
        if anchors is None:
            self.anchors = np.array([
                [0.57273, 0.677385],
                [1.87446, 2.06253],
                [3.33843, 5.47434],
                [7.88282, 3.52778],
                [9.77052, 9.16828]
            ])
        else:
            self.anchors = np.array(anchors)

        self.num_anchors = len(self.anchors)

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

        # 获取标签路径
        self.label_dir = self.dataset_root / img_dir.replace('images', 'labels')

        # 扫描所有图像文件
        self.img_files = self._load_image_files()

        # 计算网格单元大小
        self.grid_cell_size = self.img_size / self.grid_size

        print(f"[COCODatasetV2] Loaded {len(self.img_files)} images from {self.img_dir}")
        print(f"[COCODatasetV2] Classes: {self.num_classes}, Grid: {grid_size}x{grid_size}, Anchors: {self.num_anchors}")

    def _load_image_files(self):
        """扫描图像文件"""
        img_files = []
        if not self.img_dir.exists():
            print(f"Warning: Image directory not found: {self.img_dir}")
            return img_files

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

    def _compute_iou(self, box1, box2):
        """
        计算两个框的IoU（只考虑宽高，中心对齐）
        用于anchor匹配

        Args:
            box1: (w1, h1)
            box2: (w2, h2)

        Returns:
            iou: IoU值
        """
        w1, h1 = box1
        w2, h2 = box2

        # 中心对齐，计算交集
        inter_w = min(w1, w2)
        inter_h = min(h1, h2)
        inter_area = inter_w * inter_h

        # 并集
        union_area = w1 * h1 + w2 * h2 - inter_area

        iou = inter_area / (union_area + 1e-6)
        return iou

    def _encode_labels_to_yolov2(self, labels):
        """
        将标注转换为YOLOv2格式

        输出格式: (num_anchors, grid_size, grid_size, 5+num_classes)
        每个位置包含:
            [tx, ty, tw, th, confidence, class_0, ..., class_n]

        Args:
            labels: List of (class_id, cx, cy, w, h) 归一化坐标

        Returns:
            target: (num_anchors, grid_size, grid_size, 5+num_classes)
        """
        target = np.zeros((self.num_anchors, self.grid_size, self.grid_size,
                           5 + self.num_classes), dtype=np.float32)

        for class_id, cx, cy, w, h in labels:
            # 计算对象中心所在的网格
            grid_x = int(cx * self.grid_size)
            grid_y = int(cy * self.grid_size)
            grid_x = min(grid_x, self.grid_size - 1)
            grid_y = min(grid_y, self.grid_size - 1)

            # 找到与当前框IoU最大的anchor
            best_anchor_idx = 0
            best_iou = 0

            for anchor_idx in range(self.num_anchors):
                anchor_w, anchor_h = self.anchors[anchor_idx]
                # 将anchor尺寸转换为与grid相同的单位
                anchor_w_grid = anchor_w
                anchor_h_grid = anchor_h

                iou = self._compute_iou((w, h), (anchor_w_grid, anchor_h_grid))

                if iou > best_iou:
                    best_iou = iou
                    best_anchor_idx = anchor_idx

            # 计算目标值
            # tx, ty: 相对于grid cell左上角的偏移（需要sigmoid(tx) + cx）
            # 因此 tx = cx * grid_size - grid_x
            tx = cx * self.grid_size - grid_x
            ty = cy * self.grid_size - grid_y

            # tw, th: 相对于anchor的缩放（需要anchor * exp(tw)）
            # 因此 tw = log(w / anchor_w)
            anchor_w, anchor_h = self.anchors[best_anchor_idx]
            tw = np.log(w / (anchor_w + 1e-6) + 1e-6)
            th = np.log(h / (anchor_h + 1e-6) + 1e-6)

            # 填充target
            target[best_anchor_idx, grid_y, grid_x, 0] = tx
            target[best_anchor_idx, grid_y, grid_x, 1] = ty
            target[best_anchor_idx, grid_y, grid_x, 2] = tw
            target[best_anchor_idx, grid_y, grid_x, 3] = th
            target[best_anchor_idx, grid_y, grid_x, 4] = 1.0  # confidence

            # one-hot class
            if 0 <= class_id < self.num_classes:
                target[best_anchor_idx, grid_y, grid_x, 5 + class_id] = 1.0

        return target

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        返回:
            image: (3, img_size, img_size) 归一化到[0, 1]
            target: (num_anchors, grid_size, grid_size, 5+num_classes)
        """
        # 加载图像
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Warning: Failed to load image {img_path}")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
            labels = []
        else:
            # BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 加载标签
            label_path = self.label_dir / (img_path.stem + '.txt')
            labels = self._load_label(label_path)

        # 调整图像大小
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 数据增强
        if self.augment and self.split == 'train':
            img = self._augment_image(img)

        # 转换为tensor并归一化
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # 编码标签
        target = self._encode_labels_to_yolov2(labels)
        target = torch.from_numpy(target).float()

        return img, target

    def _augment_image(self, img):
        """简单的数据增强"""
        # 随机亮度
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


if __name__ == "__main__":
    # 测试代码
    print("COCODatasetV2 for YOLOv2")
    print("=" * 60)

    # 创建示例dataset
    dataset = COCODatasetV2(
        'data/coco.yaml',
        split='train',
        img_size=640,
        grid_size=20
    )

    print(f"\nDataset size: {len(dataset)}")

    if len(dataset) > 0:
        # 测试加载一个样本
        img, target = dataset[0]
        print(f"\nSample 0:")
        print(f"  Image shape: {img.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Target range: [{target.min():.4f}, {target.max():.4f}]")

        # 统计有多少个目标
        num_objects = (target[:, :, :, 4] > 0).sum()
        print(f"  Number of objects: {num_objects}")

    print("\n" + "=" * 60)
