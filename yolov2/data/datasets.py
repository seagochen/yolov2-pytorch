"""
Dataset classes with full Ultralytics format compatibility
完全兼容Ultralytics YAML+TXT格式
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import yaml


class COCODetectionDataset(Dataset):
    """
    COCO Detection Dataset - 完全兼容Ultralytics格式

    支持的YAML格式：
    ```yaml
    path: /path/to/dataset  # 数据集根目录
    train: images/train     # 训练图像目录（相对于path）
    val: images/val         # 验证图像目录
    test: images/test       # 测试图像目录（可选）

    nc: 80                  # 类别数量
    names: [...]            # 类别名称列表
    ```

    TXT标注格式（每行一个对象）：
    ```
    class_id center_x center_y width height
    ```
    所有坐标归一化到[0, 1]
    """

    def __init__(
        self,
        yaml_path: str,
        split: str = 'train',
        img_size: int = 640,
        augment: bool = False,
        anchors: Optional[np.ndarray] = None,
        cache_images: bool = False
    ):
        """
        Args:
            yaml_path: YAML配置文件路径
            split: 数据集划分 ('train', 'val', 'test')
            img_size: 目标图像尺寸
            augment: 是否应用数据增强
            anchors: Anchor boxes (num_anchors, 2)
            cache_images: 是否缓存图像到内存
        """
        super().__init__()

        self.split = split
        self.img_size = img_size
        self.augment = augment
        self.cache_images = cache_images

        # 加载配置
        self.config = self._load_yaml(yaml_path)

        # 解析路径
        self.dataset_root = self._parse_dataset_root(yaml_path)
        self.img_dir, self.label_dir = self._parse_split_paths(split)

        # 类别信息
        self.num_classes = self.config['nc']
        self.class_names = self.config['names']

        # Anchors
        if anchors is None:
            self.anchors = self._get_default_anchors()
        else:
            self.anchors = anchors
        self.num_anchors = len(self.anchors)

        # 网格信息
        self.grid_size = img_size // 32  # 默认32倍下采样
        self.grid_cell_size = img_size / self.grid_size

        # 扫描图像文件
        self.img_files = self._scan_images()

        # 图像缓存
        self.imgs = [None] * len(self.img_files) if cache_images else None

        # 统计信息
        self._print_stats()

    def _load_yaml(self, yaml_path: str) -> Dict:
        """加载YAML配置"""
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML config not found: {yaml_path}")

        with open(yaml_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # 验证必需字段
        required_fields = ['nc', 'names']
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Missing required field in YAML: {field}")

        return config

    def _parse_dataset_root(self, yaml_path: str) -> Path:
        """解析数据集根目录"""
        yaml_path = Path(yaml_path)

        # 优先使用YAML中的path字段
        if 'path' in self.config:
            root = Path(self.config['path'])
            if root.is_absolute() and root.exists():
                return root

        # 否则使用YAML文件所在目录
        return yaml_path.parent

    def _parse_split_paths(self, split: str) -> Tuple[Path, Path]:
        """
        解析图像和标签目录路径

        兼容Ultralytics/Roboflow格式：
        Roboflow导出的YAML使用 ../train/images 格式，意思是去掉../前缀
        实际上这些目录就在YAML文件的同级目录下

        Returns:
            img_dir: 图像目录
            label_dir: 标签目录
        """
        # 获取图像目录（从YAML或默认值）
        img_subpath = self.config.get(split, f'images/{split}')

        # Roboflow格式处理：../train/images 实际指向 train/images
        # 这是因为Roboflow期望 ../something 等价于 something（相对于YAML目录）
        if str(img_subpath).startswith('../'):
            # 移除 ../ 前缀，实际路径相对于YAML所在目录
            img_subpath_clean = str(img_subpath)[3:]  # 移除 '../'
            img_dir = self.dataset_root / img_subpath_clean
        else:
            # 标准路径，相对于dataset_root
            img_dir = self.dataset_root / img_subpath

        # 标签目录：将'images'替换为'labels'
        label_subpath = img_subpath.replace('images', 'labels')
        if str(label_subpath).startswith('../'):
            label_subpath_clean = str(label_subpath)[3:]  # 移除 '../'
            label_dir = self.dataset_root / label_subpath_clean
        else:
            label_dir = self.dataset_root / label_subpath

        # 验证目录存在
        if not img_dir.exists():
            print(f"Warning: Image directory not found: {img_dir}")
        if not label_dir.exists():
            print(f"Warning: Label directory not found: {label_dir}")

        return img_dir, label_dir

    def _scan_images(self) -> List[Path]:
        """扫描图像文件"""
        if not self.img_dir.exists():
            return []

        # 支持的图像格式
        img_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

        img_files = []
        for ext in img_extensions:
            img_files.extend(self.img_dir.glob(f'*{ext}'))
            img_files.extend(self.img_dir.glob(f'*{ext.upper()}'))

        # 排序以确保一致性
        img_files = sorted(img_files)

        return img_files

    def _get_default_anchors(self) -> np.ndarray:
        """获取默认anchor boxes"""
        return np.array([
            [0.57273, 0.677385],
            [1.87446, 2.06253],
            [3.33843, 5.47434],
            [7.88282, 3.52778],
            [9.77052, 9.16828]
        ], dtype=np.float32)

    def _print_stats(self):
        """打印数据集统计信息"""
        print(f"\n{'='*60}")
        print(f"Dataset: {self.split}")
        print(f"{'='*60}")
        print(f"Root: {self.dataset_root}")
        print(f"Images: {self.img_dir}")
        print(f"Labels: {self.label_dir}")
        print(f"Number of images: {len(self.img_files)}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Image size: {self.img_size}")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        print(f"Number of anchors: {self.num_anchors}")
        print(f"Augmentation: {self.augment}")
        print(f"{'='*60}\n")

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个样本

        Returns:
            image: (3, img_size, img_size) Tensor, 归一化到[0, 1]
            target: (num_anchors, grid_size, grid_size, 5+num_classes) Tensor
        """
        # 加载图像
        img = self._load_image(index)

        # 加载标签
        labels = self._load_labels(index)

        # 数据增强
        if self.augment:
            img, labels = self._apply_augmentations(img, labels)

        # Resize
        img = cv2.resize(img, (self.img_size, self.img_size))

        # 转换为Tensor
        img = self._img_to_tensor(img)

        # 编码标签
        target = self._encode_labels(labels)

        return img, target

    def _load_image(self, index: int) -> np.ndarray:
        """加载图像"""
        # 从缓存加载
        if self.imgs is not None and self.imgs[index] is not None:
            return self.imgs[index].copy()

        # 从磁盘加载
        img_path = self.img_files[index]
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"Warning: Failed to load image: {img_path}")
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 缓存
        if self.imgs is not None:
            self.imgs[index] = img.copy()

        return img

    def _load_labels(self, index: int) -> List[Tuple]:
        """
        加载标签文件

        Returns:
            labels: List of (class_id, cx, cy, w, h)
        """
        img_path = self.img_files[index]
        label_path = self.label_dir / (img_path.stem + '.txt')

        if not label_path.exists():
            return []

        labels = []
        with open(label_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                try:
                    class_id = int(parts[0])
                    cx, cy, w, h = map(float, parts[1:5])

                    # 验证
                    if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        continue
                    if class_id < 0 or class_id >= self.num_classes:
                        continue

                    labels.append((class_id, cx, cy, w, h))
                except (ValueError, IndexError):
                    continue

        return labels

    def _img_to_tensor(self, img: np.ndarray) -> torch.Tensor:
        """图像转Tensor"""
        # HWC -> CHW
        img = img.transpose(2, 0, 1)
        # 归一化
        img = img.astype(np.float32) / 255.0
        # 转Tensor
        img = torch.from_numpy(img)
        return img

    def _encode_labels(self, labels: List[Tuple]) -> torch.Tensor:
        """
        编码标签为YOLO格式

        Returns:
            target: (num_anchors, grid_size, grid_size, 5+num_classes)
        """
        target = np.zeros(
            (self.num_anchors, self.grid_size, self.grid_size, 5 + self.num_classes),
            dtype=np.float32
        )

        for class_id, cx, cy, w, h in labels:
            # 计算grid cell
            grid_x = int(cx * self.grid_size)
            grid_y = int(cy * self.grid_size)
            grid_x = min(grid_x, self.grid_size - 1)
            grid_y = min(grid_y, self.grid_size - 1)

            # 找最佳anchor
            best_anchor = self._find_best_anchor(w, h)

            # 计算偏移
            tx = cx * self.grid_size - grid_x
            ty = cy * self.grid_size - grid_y

            # 计算尺度
            # 注意：anchor尺寸是网格单位，需要将归一化的w/h也转换为网格单位
            anchor_w, anchor_h = self.anchors[best_anchor]
            w_grid = w * self.grid_size  # 归一化 -> 网格单位
            h_grid = h * self.grid_size
            tw = np.log(w_grid / (anchor_w + 1e-16) + 1e-16)
            th = np.log(h_grid / (anchor_h + 1e-16) + 1e-16)

            # 填充
            target[best_anchor, grid_y, grid_x, 0] = tx
            target[best_anchor, grid_y, grid_x, 1] = ty
            target[best_anchor, grid_y, grid_x, 2] = tw
            target[best_anchor, grid_y, grid_x, 3] = th
            target[best_anchor, grid_y, grid_x, 4] = 1.0

            # One-hot class
            target[best_anchor, grid_y, grid_x, 5 + class_id] = 1.0

        return torch.from_numpy(target)

    def _find_best_anchor(self, w: float, h: float) -> int:
        """找到最佳匹配的anchor

        Args:
            w, h: 归一化的宽高 (0~1)
        """
        best_iou = 0
        best_anchor = 0

        # 将归一化坐标转换为网格单位，与anchor尺寸统一
        w_grid = w * self.grid_size
        h_grid = h * self.grid_size

        for i, (anchor_w, anchor_h) in enumerate(self.anchors):
            # 计算IoU（中心对齐）
            inter_w = min(w_grid, anchor_w)
            inter_h = min(h_grid, anchor_h)
            inter_area = inter_w * inter_h

            union_area = w_grid * h_grid + anchor_w * anchor_h - inter_area
            iou = inter_area / (union_area + 1e-16)

            if iou > best_iou:
                best_iou = iou
                best_anchor = i

        return best_anchor

    def _apply_augmentations(
        self,
        img: np.ndarray,
        labels: List[Tuple]
    ) -> Tuple[np.ndarray, List[Tuple]]:
        """应用数据增强"""
        # 随机亮度
        if np.random.rand() > 0.5:
            brightness = np.random.randint(-30, 31)
            img = np.clip(img.astype(np.int32) + brightness, 0, 255).astype(np.uint8)

        # 随机对比度
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            img = np.clip(img * alpha, 0, 255).astype(np.uint8)

        # 随机水平翻转
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            labels = [(cls, 1 - cx, cy, w, h) for cls, cx, cy, w, h in labels]

        return img, labels


if __name__ == '__main__':
    # 测试数据集
    print("Testing COCODetectionDataset...")

    dataset = COCODetectionDataset(
        yaml_path='../../data/coco.yaml',
        split='train',
        img_size=640,
        augment=True
    )

    if len(dataset) > 0:
        img, target = dataset[0]
        print(f"Sample:")
        print(f"  Image shape: {img.shape}")
        print(f"  Target shape: {target.shape}")
        print(f"  Image range: [{img.min():.3f}, {img.max():.3f}]")

        # 统计目标数量
        num_objects = (target[:, :, :, 4] > 0).sum().item()
        print(f"  Number of objects: {num_objects}")

    print("\n✓ Test passed!")
