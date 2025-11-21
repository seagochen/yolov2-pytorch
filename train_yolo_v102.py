"""
YOLOv1 V102 训练脚本
支持COCO数据集训练

特性:
- 支持YAML配置文件
- 支持Ultralytics格式的txt标注
- 640x640输入尺寸
- 20x20网格
- 每个格子2个边界框
- 80个COCO类别
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from YoloVer102.model.YoloNetworkV102 import YoloV1NetworkV102
from Generic.dataset.COCO.COCODataset import COCODataset


def train_one_epoch(model, device, train_loader, optimizer, epoch, lambda_coord=5.0, lambda_noobj=0.5):
    """
    训练一个epoch

    参数:
        model: YOLO模型
        device: 设备 (cuda/cpu)
        train_loader: 训练数据加载器
        optimizer: 优化器
        epoch: 当前epoch
        lambda_coord: 坐标损失权重
        lambda_noobj: 无物体置信度损失权重
    """
    model.train()
    total_loss = 0.0
    total_coord_loss = 0.0
    total_conf_loss = 0.0
    total_class_loss = 0.0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for batch_idx, (images, labels) in enumerate(pbar):
        # 移动数据到设备
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        optimizer.zero_grad()
        predictions = model(images)

        # 计算损失
        loss, coord_loss, conf_loss, class_loss = compute_yolo_loss(
            predictions, labels,
            lambda_coord=lambda_coord,
            lambda_noobj=lambda_noobj
        )

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        total_coord_loss += coord_loss.item()
        total_conf_loss += conf_loss.item()
        total_class_loss += class_loss.item()

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'coord': f'{coord_loss.item():.4f}',
            'conf': f'{conf_loss.item():.4f}',
            'class': f'{class_loss.item():.4f}'
        })

    # 计算平均损失
    num_batches = len(train_loader)
    avg_loss = total_loss / num_batches
    avg_coord = total_coord_loss / num_batches
    avg_conf = total_conf_loss / num_batches
    avg_class = total_class_loss / num_batches

    print(f'\nEpoch {epoch} - Avg Loss: {avg_loss:.4f} '
          f'(Coord: {avg_coord:.4f}, Conf: {avg_conf:.4f}, Class: {avg_class:.4f})')

    return avg_loss


def validate(model, device, val_loader, lambda_coord=5.0, lambda_noobj=0.5):
    """
    验证模型

    参数:
        model: YOLO模型
        device: 设备
        val_loader: 验证数据加载器
        lambda_coord: 坐标损失权重
        lambda_noobj: 无物体置信度损失权重
    """
    model.eval()
    total_loss = 0.0
    total_coord_loss = 0.0
    total_conf_loss = 0.0
    total_class_loss = 0.0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            # 移动数据到设备
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            predictions = model(images)

            # 计算损失
            loss, coord_loss, conf_loss, class_loss = compute_yolo_loss(
                predictions, labels,
                lambda_coord=lambda_coord,
                lambda_noobj=lambda_noobj
            )

            total_loss += loss.item()
            total_coord_loss += coord_loss.item()
            total_conf_loss += conf_loss.item()
            total_class_loss += class_loss.item()

    # 计算平均损失
    num_batches = len(val_loader)
    avg_loss = total_loss / num_batches
    avg_coord = total_coord_loss / num_batches
    avg_conf = total_conf_loss / num_batches
    avg_class = total_class_loss / num_batches

    print(f'Validation - Avg Loss: {avg_loss:.4f} '
          f'(Coord: {avg_coord:.4f}, Conf: {avg_conf:.4f}, Class: {avg_class:.4f})')

    return avg_loss


def compute_yolo_loss(predictions, labels, lambda_coord=5.0, lambda_noobj=0.5):
    """
    计算YOLO损失函数

    参数:
        predictions: 模型预测 (B, C, G)
        labels: 真实标签 (B, C, G)
        lambda_coord: 坐标损失权重
        lambda_noobj: 无物体置信度损失权重

    返回:
        total_loss: 总损失
        coord_loss: 坐标损失
        conf_loss: 置信度损失
        class_loss: 分类损失
    """
    B, C, G = predictions.shape

    # 解析标签
    # 格式: [conf, bbox1(4), bbox2(4), classes(80)]
    label_conf = labels[:, 0:1, :]  # (B, 1, G)
    label_bbox1 = labels[:, 1:5, :]  # (B, 4, G)
    label_bbox2 = labels[:, 5:9, :]  # (B, 4, G)
    label_class = labels[:, 9:, :]   # (B, 80, G)

    # 解析预测
    pred_conf = predictions[:, 0:1, :]  # (B, 1, G)
    pred_bbox1 = predictions[:, 1:5, :]  # (B, 4, G)
    pred_bbox2 = predictions[:, 5:9, :]  # (B, 4, G)
    pred_class = predictions[:, 9:, :]   # (B, 80, G)

    # =====================================================
    # 1. 坐标损失 (只计算有物体的格子)
    # =====================================================
    # 物体存在的mask
    obj_mask = label_conf > 0  # (B, 1, G)

    # 计算两个bbox的IoU，选择IoU更高的那个
    # 这里简化处理：对于有标注的格子，bbox1用于主要物体，bbox2用于次要物体
    # 如果只有一个物体，bbox2不参与损失计算

    # Bbox1 损失 (所有有物体的格子)
    coord_loss_bbox1 = torch.nn.functional.mse_loss(
        pred_bbox1 * obj_mask,
        label_bbox1 * obj_mask,
        reduction='sum'
    )

    # Bbox2 损失 (只有标注了第二个bbox的格子)
    bbox2_mask = (label_bbox2.sum(dim=1, keepdim=True) > 0) & obj_mask  # (B, 1, G)
    coord_loss_bbox2 = torch.nn.functional.mse_loss(
        pred_bbox2 * bbox2_mask,
        label_bbox2 * bbox2_mask,
        reduction='sum'
    )

    coord_loss = lambda_coord * (coord_loss_bbox1 + coord_loss_bbox2)

    # =====================================================
    # 2. 置信度损失
    # =====================================================
    # 有物体的置信度损失
    conf_loss_obj = torch.nn.functional.mse_loss(
        pred_conf * obj_mask,
        label_conf * obj_mask,
        reduction='sum'
    )

    # 无物体的置信度损失
    noobj_mask = ~obj_mask
    conf_loss_noobj = torch.nn.functional.mse_loss(
        pred_conf * noobj_mask,
        label_conf * noobj_mask,
        reduction='sum'
    )

    conf_loss = conf_loss_obj + lambda_noobj * conf_loss_noobj

    # =====================================================
    # 3. 分类损失 (只计算有物体的格子)
    # =====================================================
    class_loss = torch.nn.functional.mse_loss(
        pred_class * obj_mask,
        label_class * obj_mask,
        reduction='sum'
    )

    # =====================================================
    # 总损失
    # =====================================================
    # 归一化 (除以有物体的格子数量)
    num_obj = obj_mask.sum() + 1e-6  # 避免除零
    total_loss = (coord_loss + conf_loss + class_loss) / num_obj

    return total_loss, coord_loss / num_obj, conf_loss / num_obj, class_loss / num_obj


def main():
    parser = argparse.ArgumentParser(description='YOLOv1 V102 Training')
    parser.add_argument('--data', type=str, default='data/coco.yaml',
                        help='COCO数据集YAML配置文件路径')
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='权重衰减')
    parser.add_argument('--lambda-coord', type=float, default=5.0,
                        help='坐标损失权重')
    parser.add_argument('--lambda-noobj', type=float, default=0.5,
                        help='无物体置信度损失权重')
    parser.add_argument('--save-dir', type=str, default='YoloVer102/weights',
                        help='模型保存目录')
    parser.add_argument('--resume', type=str, default='',
                        help='恢复训练的模型路径')
    parser.add_argument('--device', type=str, default='cuda',
                        help='训练设备 (cuda/cpu)')

    args = parser.parse_args()

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # =====================================================
    # 创建数据集和数据加载器
    # =====================================================
    print(f'\nLoading dataset from: {args.data}')

    train_dataset = COCODataset(
        yaml_path=args.data,
        split='train',
        img_size=640,
        grids_size=(20, 20),
        bounding_boxes=2,
        augment=True
    )

    val_dataset = COCODataset(
        yaml_path=args.data,
        split='val',
        img_size=640,
        grids_size=(20, 20),
        bounding_boxes=2,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f'Train dataset: {len(train_dataset)} images')
    print(f'Val dataset: {len(val_dataset)} images')

    # =====================================================
    # 创建模型
    # =====================================================
    print('\nCreating model...')
    model = YoloV1NetworkV102(
        grids_size=(20, 20),
        confidences=1,
        bounding_boxes=2,
        object_categories=80
    )
    model = model.to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

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
        milestones=[30, 40],
        gamma=0.1
    )

    # =====================================================
    # 恢复训练 (可选)
    # =====================================================
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f'\nResuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resumed from epoch {start_epoch}')

    # =====================================================
    # 训练循环
    # =====================================================
    print('\n' + '=' * 60)
    print('Starting training...')
    print('=' * 60)

    best_val_loss = float('inf')

    for epoch in range(start_epoch, args.epochs + 1):
        print(f'\n--- Epoch {epoch}/{args.epochs} ---')
        print(f'Learning rate: {optimizer.param_groups[0]["lr"]:.6f}')

        # 训练
        train_loss = train_one_epoch(
            model, device, train_loader, optimizer, epoch,
            lambda_coord=args.lambda_coord,
            lambda_noobj=args.lambda_noobj
        )

        # 验证
        val_loss = validate(
            model, device, val_loader,
            lambda_coord=args.lambda_coord,
            lambda_noobj=args.lambda_noobj
        )

        # 更新学习率
        scheduler.step()

        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }

        # 保存最新模型
        latest_path = os.path.join(args.save_dir, 'yolo_v102_latest.pth')
        torch.save(checkpoint, latest_path)
        print(f'Saved checkpoint to {latest_path}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, 'yolo_v102_best.pth')
            torch.save(checkpoint, best_path)
            print(f'★ New best model saved to {best_path} (val_loss: {val_loss:.4f})')

    print('\n' + '=' * 60)
    print('Training completed!')
    print(f'Best validation loss: {best_val_loss:.4f}')
    print('=' * 60)


if __name__ == '__main__':
    main()
