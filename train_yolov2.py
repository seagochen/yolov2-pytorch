"""
YOLOv2 Training Script

支持的特性：
- Darknet-19 backbone
- Anchor boxes mechanism
- Batch Normalization
- Passthrough layer
- Multi-scale training (可选)
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from YoloVer2.model.YOLOv2 import YOLOv2, get_default_anchors
from Generic.dataset.COCO.COCODatasetV2 import COCODatasetV2
from Generic.loss.YOLOv2Loss import YOLOv2Loss


def train_one_epoch(model, device, train_loader, optimizer, criterion, epoch):
    """
    训练一个epoch

    Args:
        model: YOLOv2模型
        device: 设备
        train_loader: 训练数据加载器
        optimizer: 优化器
        criterion: 损失函数
        epoch: 当前epoch

    Returns:
        avg_loss_dict: 平均损失字典
    """
    model.train()

    # 累积损失
    epoch_loss = 0.0
    epoch_loss_dict = {}

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')

    for batch_idx, (images, targets) in enumerate(pbar):
        # 移动数据到设备
        images = images.to(device)
        targets = targets.to(device)

        # 前向传播
        optimizer.zero_grad()
        predictions = model(images)

        # 计算损失
        loss, loss_dict = criterion(predictions, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 累积损失
        epoch_loss += loss.item()
        for key, value in loss_dict.items():
            if key not in epoch_loss_dict:
                epoch_loss_dict[key] = 0.0
            epoch_loss_dict[key] += value

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'coord': f'{loss_dict["coord"]:.4f}',
            'conf': f'{loss_dict["conf"]:.4f}',
            'class': f'{loss_dict["class"]:.4f}'
        })

    # 计算平均损失
    num_batches = len(train_loader)
    avg_loss_dict = {k: v / num_batches for k, v in epoch_loss_dict.items()}

    print(f'\nEpoch {epoch} - Train Loss: {avg_loss_dict["total"]:.4f} '
          f'(Coord: {avg_loss_dict["coord"]:.4f}, '
          f'Conf: {avg_loss_dict["conf"]:.4f}, '
          f'Class: {avg_loss_dict["class"]:.4f})')

    return avg_loss_dict


def validate(model, device, val_loader, criterion):
    """
    验证模型

    Args:
        model: YOLOv2模型
        device: 设备
        val_loader: 验证数据加载器
        criterion: 损失函数

    Returns:
        avg_loss_dict: 平均损失字典
    """
    model.eval()

    epoch_loss = 0.0
    epoch_loss_dict = {}

    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc='Validation'):
            # 移动数据到设备
            images = images.to(device)
            targets = targets.to(device)

            # 前向传播
            predictions = model(images)

            # 计算损失
            loss, loss_dict = criterion(predictions, targets)

            # 累积损失
            epoch_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in epoch_loss_dict:
                    epoch_loss_dict[key] = 0.0
                epoch_loss_dict[key] += value

    # 计算平均损失
    num_batches = len(val_loader)
    avg_loss_dict = {k: v / num_batches for k, v in epoch_loss_dict.items()}

    print(f'Validation - Loss: {avg_loss_dict["total"]:.4f} '
          f'(Coord: {avg_loss_dict["coord"]:.4f}, '
          f'Conf: {avg_loss_dict["conf"]:.4f}, '
          f'Class: {avg_loss_dict["class"]:.4f})')

    return avg_loss_dict


def main():
    parser = argparse.ArgumentParser(description='YOLOv2 Training')
    parser.add_argument('--data', type=str, default='data/coco.yaml',
                        help='COCO数据集YAML配置文件路径')
    parser.add_argument('--epochs', type=int, default=100,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='权重衰减')
    parser.add_argument('--lambda-coord', type=float, default=5.0,
                        help='坐标损失权重')
    parser.add_argument('--lambda-noobj', type=float, default=0.5,
                        help='无物体置信度损失权重')
    parser.add_argument('--lambda-class', type=float, default=1.0,
                        help='分类损失权重')
    parser.add_argument('--save-dir', type=str, default='YoloVer2/weights',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的模型路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备 (cuda/cpu)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='数据加载线程数')

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # =====================================================
    # 创建anchor boxes
    # =====================================================
    anchors = get_default_anchors()
    print(f'\nUsing anchors:')
    for i, (w, h) in enumerate(anchors):
        print(f'  Anchor {i}: w={w:.4f}, h={h:.4f}')

    # =====================================================
    # 创建数据集和数据加载器
    # =====================================================
    print(f'\nLoading dataset from: {args.data}')

    train_dataset = COCODatasetV2(
        yaml_path=args.data,
        split='train',
        img_size=640,
        grid_size=20,
        anchors=anchors,
        augment=True
    )

    val_dataset = COCODatasetV2(
        yaml_path=args.data,
        split='val',
        img_size=640,
        grid_size=20,
        anchors=anchors,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f'Train dataset: {len(train_dataset)} images')
    print(f'Val dataset: {len(val_dataset)} images')

    # =====================================================
    # 创建模型
    # =====================================================
    print('\nCreating YOLOv2 model...')
    model = YOLOv2(
        num_classes=80,
        anchors=anchors,
        img_size=640
    )
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Model size: {total_params * 4 / (1024**2):.2f} MB')

    # =====================================================
    # 创建损失函数
    # =====================================================
    criterion = YOLOv2Loss(
        num_classes=80,
        anchors=torch.FloatTensor(anchors),
        lambda_coord=args.lambda_coord,
        lambda_noobj=args.lambda_noobj,
        lambda_class=args.lambda_class
    )

    # =====================================================
    # 创建优化器
    # =====================================================
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度器
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 90],
        gamma=0.1
    )

    # =====================================================
    # 恢复训练 (可选)
    # =====================================================
    start_epoch = 1
    best_val_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        print(f'\nResuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f'Resumed from epoch {start_epoch}')

    # =====================================================
    # 训练循环
    # =====================================================
    print('\n' + '=' * 60)
    print('Starting training...')
    print('=' * 60)

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'\n--- Epoch {epoch}/{args.epochs} ---')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # 训练
        train_loss_dict = train_one_epoch(
            model, device, train_loader, optimizer, criterion, epoch
        )

        # 验证
        val_loss_dict = validate(
            model, device, val_loader, criterion
        )

        # 更新学习率
        scheduler.step()

        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss_dict': train_loss_dict,
            'val_loss_dict': val_loss_dict,
            'best_val_loss': best_val_loss,
            'anchors': anchors,
        }

        # 保存最新模型
        latest_path = os.path.join(args.save_dir, 'yolov2_latest.pth')
        torch.save(checkpoint, latest_path)

        # 保存最佳模型
        val_loss = val_loss_dict['total']
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, 'yolov2_best.pth')
            torch.save(checkpoint, best_path)
            print(f'★ New best model saved! (val_loss: {val_loss:.4f})')

        # 定期保存
        if epoch % 10 == 0:
            epoch_path = os.path.join(args.save_dir, f'yolov2_epoch_{epoch}.pth')
            torch.save(checkpoint, epoch_path)

    print('\n' + '=' * 60)
    print('Training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
