"""
YOLOv2 Training Script - 重构版
清晰、模块化的训练流程
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from yolov2.models import create_yolov2
from yolov2.data import COCODetectionDataset
from yolov2.utils import create_yolov2_loss, init_seeds, check_img_size, colorstr, increment_path


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv2 Training')

    # 数据相关
    parser.add_argument('--data', type=str, default='data/coco.yaml',
                        help='Dataset config YAML path')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Input image size')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--workers', type=int, default=8,
                        help='Number of dataloader workers')
    parser.add_argument('--cache', action='store_true',
                        help='Cache images for faster training')

    # 训练相关
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Warmup epochs')

    # 损失权重
    parser.add_argument('--lambda-coord', type=float, default=5.0,
                        help='Coordinate loss weight')
    parser.add_argument('--lambda-noobj', type=float, default=0.5,
                        help='No-object confidence loss weight')
    parser.add_argument('--lambda-class', type=float, default=1.0,
                        help='Classification loss weight')

    # 其他
    parser.add_argument('--project', type=str, default='runs/train',
                        help='Save directory')
    parser.add_argument('--name', type=str, default='exp',
                        help='Experiment name')
    parser.add_argument('--resume', type=str, default='',
                        help='Resume from checkpoint')
    parser.add_argument('--device', type=str, default='0',
                        help='CUDA device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    parser.add_argument('--exist-ok', action='store_true',
                        help='Allow existing project/name')

    return parser.parse_args()


def setup_device(device_str: str):
    """设置训练设备"""
    if device_str == 'cpu':
        return torch.device('cpu')

    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = device_str
        return torch.device('cuda:0')

    print(colorstr('yellow', 'WARNING: CUDA not available, using CPU'))
    return torch.device('cpu')


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """训练一个epoch"""
    model.train()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    total_loss = 0
    loss_components = {}

    for batch_idx, (images, targets) in enumerate(pbar):
        # 移动到设备
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

        # 统计
        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v

        # 更新进度条
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'coord': f'{loss_dict["coord"]:.4f}',
            'conf': f'{loss_dict["conf"]:.4f}',
            'cls': f'{loss_dict["class"]:.4f}'
        })

    # 计算平均
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    return avg_loss, avg_components


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()

    total_loss = 0
    loss_components = {}

    for images, targets in tqdm(dataloader, desc='Validation'):
        images = images.to(device)
        targets = targets.to(device)

        predictions = model(images)
        loss, loss_dict = criterion(predictions, targets)

        total_loss += loss.item()
        for k, v in loss_dict.items():
            loss_components[k] = loss_components.get(k, 0) + v

    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    return avg_loss, avg_components


def main():
    """主函数"""
    args = parse_args()

    # 初始化随机种子
    init_seeds(args.seed)

    # 设置保存目录
    save_dir = Path(increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    weights_dir = save_dir / 'weights'
    weights_dir.mkdir(exist_ok=True)

    # 打印配置
    print(colorstr('bright_blue', 'bold', '\nYOLOv2 Training'))
    print('=' * 60)
    print(f'Device: {args.device}')
    print(f'Dataset: {args.data}')
    print(f'Image size: {args.img_size}')
    print(f'Batch size: {args.batch_size}')
    print(f'Epochs: {args.epochs}')
    print(f'Save directory: {save_dir}')
    print('=' * 60)

    # 设置设备
    device = setup_device(args.device)

    # 验证img_size
    img_size = check_img_size(args.img_size, stride=32)

    # 创建数据集
    print(colorstr('bright_green', '\nLoading datasets...'))
    train_dataset = COCODetectionDataset(
        yaml_path=args.data,
        split='train',
        img_size=img_size,
        augment=True,
        cache_images=args.cache
    )

    val_dataset = COCODetectionDataset(
        yaml_path=args.data,
        split='val',
        img_size=img_size,
        augment=False,
        cache_images=False
    )

    # 创建dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    # 创建模型
    print(colorstr('bright_green', '\nCreating model...'))
    model = create_yolov2(
        num_classes=train_dataset.num_classes,
        img_size=img_size
    )
    model = model.to(device)

    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Parameters: {total_params:,} ({total_params * 4 / 1e6:.2f} MB)')

    # 创建损失函数
    criterion = create_yolov2_loss(
        num_classes=train_dataset.num_classes,
        anchors=torch.from_numpy(train_dataset.anchors),
        lambda_coord=args.lambda_coord,
        lambda_noobj=args.lambda_noobj,
        lambda_class=args.lambda_class
    )

    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 学习率调度
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 90],
        gamma=0.1
    )

    # 恢复训练
    start_epoch = 0
    best_loss = float('inf')

    if args.resume and os.path.exists(args.resume):
        print(colorstr('bright_cyan', f'\nResuming from {args.resume}'))
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_loss = ckpt.get('best_loss', float('inf'))
        print(f'Resumed from epoch {start_epoch}')

    # 训练循环
    print(colorstr('bright_green', 'bold', '\nStarting training...'))
    print('=' * 60)

    for epoch in range(start_epoch, args.epochs):
        # 训练
        train_loss, train_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # 验证
        val_loss, val_dict = validate(
            model, val_loader, criterion, device
        )

        # 更新学习率
        scheduler.step()

        # 打印
        print(f'\nEpoch {epoch}:')
        print(f'  Train Loss: {train_loss:.4f}')
        print(f'  Val Loss: {val_loss:.4f}')
        print(f'  LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 保存检查点
        ckpt = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_loss': best_loss
        }

        # 保存最新
        torch.save(ckpt, weights_dir / 'last.pt')

        # 保存最佳
        if val_loss < best_loss:
            best_loss = val_loss
            ckpt['best_loss'] = best_loss
            torch.save(ckpt, weights_dir / 'best.pt')
            print(colorstr('bright_green', f'  ★ New best model! Loss: {val_loss:.4f}'))

    print('\n' + '=' * 60)
    print(colorstr('bright_green', 'bold', 'Training completed!'))
    print(f'Best loss: {best_loss:.4f}')
    print(f'Weights saved to: {weights_dir}')
    print('=' * 60)


if __name__ == '__main__':
    main()
