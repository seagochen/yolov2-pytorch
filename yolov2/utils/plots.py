"""
Plotting utilities for training visualization
绘制训练曲线、混淆矩阵、PR曲线等
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 非GUI后端
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
import seaborn as sns


class TrainingPlotter:
    """
    训练过程可视化器

    类似Ultralytics的风格，生成训练曲线等图表
    """

    def __init__(self, save_dir: Path):
        """
        Args:
            save_dir: 保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # 存储历史数据
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'precision': [],
            'recall': [],
            'mAP@0.5': [],
            'mAP@0.5:0.95': [],
            'f1': []
        }

    def update(self, epoch: int, metrics: Dict[str, float]):
        """
        更新指标

        Args:
            epoch: 当前epoch
            metrics: 指标字典
        """
        self.history['epoch'].append(epoch)

        for key in self.history.keys():
            if key != 'epoch' and key in metrics:
                self.history[key].append(metrics[key])

    def plot_training_curves(self):
        """绘制训练曲线"""
        if not self.history['epoch']:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Training Metrics', fontsize=16, fontweight='bold')

        epochs = self.history['epoch']

        # 1. Loss曲线
        ax = axes[0, 0]
        if self.history['train_loss']:
            ax.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=2)
        if self.history['val_loss']:
            ax.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curve')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Precision & Recall
        ax = axes[0, 1]
        if self.history['precision']:
            ax.plot(epochs, self.history['precision'], 'g-', label='Precision', linewidth=2)
        if self.history['recall']:
            ax.plot(epochs, self.history['recall'], 'b-', label='Recall', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.set_title('Precision & Recall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 3. mAP
        ax = axes[1, 0]
        if self.history['mAP@0.5']:
            ax.plot(epochs, self.history['mAP@0.5'], 'purple', label='mAP@0.5', linewidth=2)
        if self.history['mAP@0.5:0.95']:
            ax.plot(epochs, self.history['mAP@0.5:0.95'], 'orange', label='mAP@0.5:0.95', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('mAP')
        ax.set_title('Mean Average Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        # 4. F1 Score
        ax = axes[1, 1]
        if self.history['f1']:
            ax.plot(epochs, self.history['f1'], 'red', label='F1 Score', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

        plt.tight_layout()
        save_path = self.save_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f'Training curves saved to {save_path}')

    def plot_confusion_matrix(self, cm_matrix: np.ndarray, class_names: List[str]):
        """
        绘制混淆矩阵

        Args:
            cm_matrix: 混淆矩阵
            class_names: 类别名称
        """
        # 只显示前20个类别（如果类别太多）
        nc = min(20, len(class_names))
        cm = cm_matrix[:nc, :nc]
        names = class_names[:nc]

        fig, ax = plt.subplots(figsize=(12, 10))

        # 归一化
        cm_norm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-16)

        # 绘制热图
        sns.heatmap(
            cm_norm,
            annot=True,
            fmt='.2f',
            cmap='Blues',
            xticklabels=names,
            yticklabels=names,
            square=True,
            cbar_kws={'label': 'Normalized Count'},
            ax=ax
        )

        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('True', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')

        plt.tight_layout()
        save_path = self.save_dir / 'confusion_matrix.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f'Confusion matrix saved to {save_path}')

    def plot_pr_curve(self, precision: np.ndarray, recall: np.ndarray, ap: float):
        """
        绘制PR曲线

        Args:
            precision: Precision数组
            recall: Recall数组
            ap: Average Precision
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        ax.plot(recall, precision, 'b-', linewidth=2, label=f'AP = {ap:.3f}')
        ax.fill_between(recall, precision, alpha=0.2, color='blue')

        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

        plt.tight_layout()
        save_path = self.save_dir / 'PR_curve.png'
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()

        print(f'PR curve saved to {save_path}')

    def save_metrics_csv(self):
        """保存指标到CSV文件"""
        import pandas as pd

        df = pd.DataFrame(self.history)
        save_path = self.save_dir / 'metrics.csv'
        df.to_csv(save_path, index=False)
        print(f'Metrics saved to {save_path}')


def plot_detection_samples(
    images: List[np.ndarray],
    predictions: List[List[Dict]],
    targets: List[np.ndarray],
    class_names: List[str],
    save_dir: Path,
    max_images: int = 16
):
    """
    绘制检测样本（类似YOLOv8的val_batch图）

    Args:
        images: 图像列表
        predictions: 预测结果列表
        targets: 目标列表
        class_names: 类别名称
        save_dir: 保存目录
        max_images: 最大显示图像数
    """
    n_images = min(len(images), max_images)
    n_cols = 4
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx in range(n_images):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        # 获取图像
        img = images[idx].copy()
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)

        # 转换为RGB（如果需要）
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[0] == 3:  # CHW格式
            img = img.transpose(1, 2, 0)

        # 确保数组是连续的（OpenCV要求）
        img = np.ascontiguousarray(img)

        # 绘制GT（绿色）
        if idx < len(targets) and len(targets[idx]) > 0:
            for gt in targets[idx]:
                cls_id = int(gt[0])
                x1, y1, x2, y2 = gt[1:5]
                h, w = img.shape[:2]

                # 转换归一化坐标到像素坐标
                if x2 <= 1.0:  # 归一化坐标
                    cx, cy, bw, bh = x1, y1, x2, y2
                    x1 = int((cx - bw / 2) * w)
                    y1 = int((cy - bh / 2) * h)
                    x2 = int((cx + bw / 2) * w)
                    y2 = int((cy + bh / 2) * h)
                else:
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = class_names[cls_id] if cls_id < len(class_names) else f'cls{cls_id}'
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.5, (0, 255, 0), 2)

        # 绘制预测（红色）
        if idx < len(predictions):
            for pred in predictions[idx]:
                cls_id = pred['class_id']
                conf = pred['confidence']
                x1, y1, x2, y2 = map(int, pred['bbox'])

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                label = f"{class_names[cls_id] if cls_id < len(class_names) else f'cls{cls_id}'} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                           0.4, (255, 0, 0), 1)

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Image {idx}', fontsize=10)

    # 隐藏多余的子图
    for idx in range(n_images, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')

    plt.tight_layout()
    save_path = save_dir / 'val_batch_predictions.jpg'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f'Detection samples saved to {save_path}')


def plot_labels_distribution(targets: List[np.ndarray], class_names: List[str], save_dir: Path):
    """
    绘制标签分布

    Args:
        targets: 目标列表
        class_names: 类别名称
        save_dir: 保存目录
    """
    # 统计每个类别的数量
    class_counts = {}
    box_sizes = []

    for target in targets:
        if len(target) == 0:
            continue

        for obj in target:
            cls_id = int(obj[0])
            class_counts[cls_id] = class_counts.get(cls_id, 0) + 1

            # 计算框的尺寸
            if len(obj) >= 5:
                w, h = obj[3], obj[4]
                box_sizes.append(w * h)

    # 绘制类别分布
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 类别分布柱状图
    ax = axes[0]
    if class_counts:
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        labels = [class_names[c] if c < len(class_names) else f'cls{c}' for c in classes]

        # 只显示前20个类别
        if len(classes) > 20:
            classes = classes[:20]
            counts = counts[:20]
            labels = labels[:20]

        ax.bar(range(len(classes)), counts, color='steelblue')
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_title('Class Distribution')
        ax.grid(True, alpha=0.3, axis='y')

    # 框尺寸分布
    ax = axes[1]
    if box_sizes:
        ax.hist(box_sizes, bins=50, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Box Area (normalized)')
        ax.set_ylabel('Count')
        ax.set_title('Bounding Box Size Distribution')
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = save_dir / 'labels_distribution.png'
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()

    print(f'Labels distribution saved to {save_path}')


if __name__ == '__main__':
    print("Testing plotting utilities...")

    # 创建临时目录
    save_dir = Path('test_plots')
    save_dir.mkdir(exist_ok=True)

    # 测试训练曲线
    plotter = TrainingPlotter(save_dir)

    for epoch in range(1, 11):
        metrics = {
            'train_loss': 0.5 - epoch * 0.03,
            'val_loss': 0.6 - epoch * 0.025,
            'precision': epoch * 0.08,
            'recall': epoch * 0.07,
            'mAP@0.5': epoch * 0.06,
            'mAP@0.5:0.95': epoch * 0.04,
            'f1': epoch * 0.075
        }
        plotter.update(epoch, metrics)

    plotter.plot_training_curves()

    # 测试混淆矩阵
    cm = np.random.rand(10, 10) * 100
    class_names = [f'class_{i}' for i in range(10)]
    plotter.plot_confusion_matrix(cm, class_names)

    # 测试PR曲线
    recall = np.linspace(0, 1, 100)
    precision = 1 - recall * 0.5
    plotter.plot_pr_curve(precision, recall, 0.75)

    print("\n✓ All tests passed!")
    print(f"Plots saved to {save_dir}")
