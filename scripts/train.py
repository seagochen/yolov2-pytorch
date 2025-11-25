"""
YOLOv2 Training Script - 完整版
包含训练、验证、评估指标计算和可视化
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
import numpy as np

from yolov2.models import create_yolov2
from yolov2.data import COCODetectionDataset
from yolov2.utils import (
    create_yolov2_loss,
    init_seeds,
    check_img_size,
    colorstr,
    increment_path,
    nms,
    # 微调组件
    ReduceLROnPlateau,
    EarlyStopping,
    ModelEMA,
    GradientAccumulator,
    LabelSmoothingBCE
)
from yolov2.utils.metrics import ConfusionMatrix, DetectionMetrics
from yolov2.utils.plots import (
    TrainingPlotter,
    plot_detection_samples,
    plot_labels_distribution
)


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
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Initial learning rate (降低默认值以提高稳定性)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    parser.add_argument('--warmup-epochs', type=int, default=3,
                        help='Warmup epochs')
    parser.add_argument('--grad-clip', type=float, default=10.0,
                        help='Gradient clipping max norm (0 to disable)')

    # 损失权重
    parser.add_argument('--lambda-coord', type=float, default=5.0,
                        help='Coordinate loss weight')
    parser.add_argument('--lambda-noobj', type=float, default=0.5,
                        help='No-object confidence loss weight')
    parser.add_argument('--lambda-class', type=float, default=1.0,
                        help='Classification loss weight')

    # 微调相关
    parser.add_argument('--patience', type=int, default=5,
                        help='Patience for ReduceLROnPlateau (epochs without improvement)')
    parser.add_argument('--lr-factor', type=float, default=0.1,
                        help='Factor to reduce learning rate')
    parser.add_argument('--min-lr', type=float, default=1e-7,
                        help='Minimum learning rate')
    parser.add_argument('--max-lr-reductions', type=int, default=3,
                        help='Max LR reductions before early stopping')
    parser.add_argument('--early-stopping', action='store_true',
                        help='Enable early stopping')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--amp', action='store_true',
                        help='Use Automatic Mixed Precision training')
    parser.add_argument('--label-smoothing', type=float, default=0.0,
                        help='Label smoothing factor (0.0 to disable)')
    parser.add_argument('--ema', action='store_true',
                        help='Use Exponential Moving Average for model weights')
    parser.add_argument('--ema-decay', type=float, default=0.9999,
                        help='EMA decay factor')

    # 评估相关
    parser.add_argument('--eval-interval', type=int, default=5,
                        help='Compute full metrics every N epochs (0=only last epoch)')
    parser.add_argument('--conf-thres', type=float, default=0.001,
                        help='Confidence threshold for evaluation')
    parser.add_argument('--iou-thres', type=float, default=0.6,
                        help='IoU threshold for NMS')
    parser.add_argument('--plot-samples', type=int, default=16,
                        help='Number of detection samples to plot')

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


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
    epoch,
    grad_clip=0.0,
    scaler=None,
    accumulator=None,
    ema=None
):
    """训练一个epoch

    Args:
        grad_clip: 梯度裁剪阈值，0表示不裁剪
        scaler: GradScaler，用于混合精度训练
        accumulator: GradientAccumulator，用于梯度累积
        ema: ModelEMA，用于指数移动平均
    """
    model.train()

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    total_loss = 0
    loss_components = {}

    # 是否使用混合精度
    use_amp = scaler is not None

    for batch_idx, (images, targets) in enumerate(pbar):
        # 移动到设备
        images = images.to(device)
        targets = targets.to(device)

        # 混合精度前向传播
        with torch.cuda.amp.autocast(enabled=use_amp):
            predictions = model(images)
            loss, loss_dict = criterion(predictions, targets)

        # 反向传播（支持梯度累积）
        if accumulator is not None:
            # 梯度累积模式
            if use_amp:
                scaler.scale(loss / accumulator.accumulation_steps).backward()
            else:
                (loss / accumulator.accumulation_steps).backward()

            # 检查是否应该更新参数
            if accumulator.should_step(batch_idx):
                if use_amp:
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                optimizer.zero_grad()

                # 更新EMA
                if ema is not None:
                    ema.update(model)
        else:
            # 标准模式
            optimizer.zero_grad()
            if use_amp:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            # 更新EMA
            if ema is not None:
                ema.update(model)

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
def validate(
    model,
    dataloader,
    criterion,
    device,
    conf_thres=0.001,
    iou_thres=0.6,
    nc=80,
    compute_metrics=True,
    save_dir=None,
    plot_samples=16
):
    """
    验证并计算评估指标

    Returns:
        metrics: 包含loss和detection metrics的字典
        images: 用于可视化的图像
        predictions: 预测结果
        targets: 真实标签
    """
    model.eval()

    total_loss = 0
    loss_components = {}

    # 收集预测和目标
    all_predictions = []
    all_targets = []
    sample_images = []
    sample_preds = []
    sample_targets = []

    # 混淆矩阵和检测指标
    confusion_matrix = ConfusionMatrix(nc=nc, conf=conf_thres, iou_thres=iou_thres)
    detection_metrics = DetectionMetrics(nc=nc)

    pbar = tqdm(dataloader, desc='Validation')

    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            targets_device = targets.to(device)

            # 前向传播（计算损失）
            predictions_raw = model(images)
            loss, loss_dict = criterion(predictions_raw, targets_device)

            total_loss += loss.item()
            for k, v in loss_dict.items():
                loss_components[k] = loss_components.get(k, 0) + v

            # 解码预测（用于评估）- 只在需要时计算
            if compute_metrics:
                batch_predictions = model.predict(images, conf_threshold=conf_thres, device=device)

                # 转换targets格式 - 从YOLO target tensor解码为 [class_id, x1, y1, x2, y2]
                batch_targets = []
                img_size = model.img_size
                anchors = model.anchors.cpu().numpy()

                for i in range(targets.size(0)):
                    target = targets[i]  # (num_anchors, grid_h, grid_w, 5+nc)
                    num_anchors, grid_h, grid_w = target.size(0), target.size(1), target.size(2)

                    # 提取有物体的位置
                    obj_mask = target[:, :, :, 4] > 0
                    target_list = []

                    for a in range(num_anchors):
                        for cy in range(grid_h):
                            for cx in range(grid_w):
                                if obj_mask[a, cy, cx]:
                                    # 提取类别
                                    class_probs = target[a, cy, cx, 5:]
                                    class_id = torch.argmax(class_probs).item()

                                    # 提取并解码bbox (tx, ty, tw, th)
                                    tx = target[a, cy, cx, 0].item()
                                    ty = target[a, cy, cx, 1].item()
                                    tw = target[a, cy, cx, 2].item()
                                    th = target[a, cy, cx, 3].item()

                                    # YOLOv2解码公式
                                    # bx = sigmoid(tx) + cx, by = sigmoid(ty) + cy
                                    # bw = pw * exp(tw), bh = ph * exp(th)
                                    anchor_w, anchor_h = anchors[a]

                                    bx = (1 / (1 + np.exp(-tx)) + cx) / grid_w
                                    by = (1 / (1 + np.exp(-ty)) + cy) / grid_h
                                    bw = anchor_w * np.exp(tw) / grid_w
                                    bh = anchor_h * np.exp(th) / grid_h

                                    # 转换为像素坐标 (x1, y1, x2, y2)
                                    x1 = (bx - bw / 2) * img_size
                                    y1 = (by - bh / 2) * img_size
                                    x2 = (bx + bw / 2) * img_size
                                    y2 = (by + bh / 2) * img_size

                                    # 裁剪到图像范围
                                    x1 = max(0, min(img_size, x1))
                                    y1 = max(0, min(img_size, y1))
                                    x2 = max(0, min(img_size, x2))
                                    y2 = max(0, min(img_size, y2))

                                    target_list.append([class_id, x1, y1, x2, y2])

                    batch_targets.append(np.array(target_list) if target_list else np.zeros((0, 5)))

                all_predictions.extend(batch_predictions)
                all_targets.extend(batch_targets)

                # 应用NMS
                for i, preds in enumerate(batch_predictions):
                    batch_predictions[i] = nms(preds, iou_threshold=iou_thres)

                # 更新检测指标
                detection_metrics.update(batch_predictions, batch_targets)

                # 收集样本用于可视化（仅在需要保存时）
                if batch_idx == 0 and save_dir:
                    n_samples = min(plot_samples, images.size(0))
                    for i in range(n_samples):
                        img = images[i].cpu().numpy()
                        sample_images.append(img)
                        sample_preds.append(batch_predictions[i])
                        sample_targets.append(batch_targets[i])

    # 计算平均损失
    num_batches = len(dataloader)
    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in loss_components.items()}

    # 计算检测指标
    metrics = {'val_loss': avg_loss}
    metrics.update(avg_components)

    if compute_metrics and all_predictions:
        det_metrics = detection_metrics.compute_metrics()
        metrics.update(det_metrics)

    # 绘制检测样本
    if save_dir and sample_images:
        plot_detection_samples(
            sample_images,
            sample_preds,
            sample_targets,
            dataloader.dataset.class_names,
            save_dir,
            max_images=plot_samples
        )

    return metrics, sample_images, sample_preds, sample_targets


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
    print(colorstr('bright_blue', 'bold', '\n' + '='*60))
    print(colorstr('bright_blue', 'bold', 'YOLOv2 Training with Evaluation'))
    print('='*60)
    print(f'Device: {args.device}')
    print(f'Dataset: {args.data}')
    print(f'Image size: {args.img_size}')
    print(f'Batch size: {args.batch_size}')
    if args.accumulation_steps > 1:
        effective_batch = args.batch_size * args.accumulation_steps
        print(f'Effective batch size: {effective_batch} (accumulation: {args.accumulation_steps})')
    print(f'Epochs: {args.epochs}')
    print(f'Save directory: {save_dir}')
    print(colorstr('bright_yellow', '--- Fine-tuning Features ---'))
    print(f'ReduceLROnPlateau: patience={args.patience}, factor={args.lr_factor}')
    print(f'Early Stopping: {"enabled" if args.early_stopping else "disabled"} (max_lr_reductions={args.max_lr_reductions})')
    print(f'Mixed Precision (AMP): {"enabled" if args.amp else "disabled"}')
    print(f'Model EMA: {"enabled" if args.ema else "disabled"} (decay={args.ema_decay})')
    print(f'Label Smoothing: {args.label_smoothing}')
    print('='*60)

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

    # 绘制标签分布
    print(colorstr('bright_cyan', '\nAnalyzing dataset...'))
    # Get raw labels instead of encoded targets
    raw_labels = []
    for i in range(min(1000, len(train_dataset))):
        labels = train_dataset._load_labels(i)
        if labels:
            # Convert list of tuples to numpy array: [(class_id, cx, cy, w, h), ...]
            raw_labels.append(np.array(labels, dtype=np.float32))
        else:
            raw_labels.append(np.array([], dtype=np.float32).reshape(0, 5))

    plot_labels_distribution(
        raw_labels,
        train_dataset.class_names,
        save_dir
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
        lambda_class=args.lambda_class,
        label_smoothing=args.label_smoothing
    )

    # 创建优化器 - 使用AdamW更稳定
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8
    )

    # 学习率调度 - 使用CosineAnnealing + ReduceLROnPlateau
    # warmup阶段使用LambdaLR，之后由ReduceLROnPlateau接管
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            # Warmup阶段：线性增长
            return (epoch + 1) / args.warmup_epochs
        else:
            # Cosine decay作为基础调度
            progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * progress))

    warmup_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ReduceLROnPlateau - 当验证损失不下降时降低学习率
    lr_plateau = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=args.lr_factor,
        patience=args.patience,
        min_lr=args.min_lr,
        verbose=True
    )

    # Early Stopping - 连续多次降低学习率后仍无改善则停止
    early_stopping = EarlyStopping(
        patience=args.patience,
        mode='min',
        check_lr_reductions=True,
        max_lr_reductions=args.max_lr_reductions,
        verbose=True
    ) if args.early_stopping else None

    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler() if args.amp and device.type == 'cuda' else None

    # 梯度累积
    accumulator = GradientAccumulator(args.accumulation_steps) if args.accumulation_steps > 1 else None

    # Model EMA
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema else None

    # 创建可视化器
    plotter = TrainingPlotter(save_dir)

    # 恢复训练
    start_epoch = 0
    best_map = 0.0
    best_val_loss = float('inf')

    # EMA平滑验证损失（用于稳定模型选择）
    ema_val_loss = None
    ema_alpha = 0.1  # EMA衰减系数，越小越平滑

    if args.resume and os.path.exists(args.resume):
        print(colorstr('bright_cyan', f'\nResuming from {args.resume}'))
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch']
        best_map = ckpt.get('best_map', 0.0)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        # 恢复微调组件状态
        if 'lr_plateau' in ckpt:
            lr_plateau.load_state_dict(ckpt['lr_plateau'])
        if early_stopping and 'early_stopping' in ckpt:
            early_stopping.load_state_dict(ckpt['early_stopping'])
        if ema and 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
        if scaler and 'scaler' in ckpt:
            scaler.load_state_dict(ckpt['scaler'])
        print(f'Resumed from epoch {start_epoch}')

    # 训练循环
    print(colorstr('bright_green', 'bold', '\nStarting training...'))
    print('='*60)

    for epoch in range(start_epoch, args.epochs):
        print(f'\n{colorstr("bright_cyan", "bold", f"Epoch {epoch}/{args.epochs-1}")}')
        print(f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        # 训练（传入新的组件）
        train_loss, train_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            grad_clip=args.grad_clip,
            scaler=scaler,
            accumulator=accumulator,
            ema=ema
        )

        # 验证：根据 eval_interval 决定是否计算完整指标
        # eval_interval=0 表示只在最后一个epoch计算，否则每N个epoch计算一次
        is_last_epoch = (epoch == args.epochs - 1)
        is_eval_epoch = (args.eval_interval > 0 and epoch % args.eval_interval == 0)
        compute_full_metrics = is_last_epoch or is_eval_epoch

        # 使用EMA模型进行验证（如果启用）
        val_model = ema.ema if ema else model
        val_metrics, _, _, _ = validate(
            val_model,
            val_loader,
            criterion,
            device,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            nc=train_dataset.num_classes,
            compute_metrics=compute_full_metrics,
            save_dir=save_dir if compute_full_metrics else None,
            plot_samples=args.plot_samples
        )

        # 更新学习率调度
        # Warmup阶段使用LambdaLR
        if epoch < args.warmup_epochs:
            warmup_scheduler.step()
        else:
            # Warmup结束后，由ReduceLROnPlateau根据验证损失调整
            current_val_loss_for_lr = val_metrics.get("val_loss", 0)
            lr_reduced = lr_plateau.step(current_val_loss_for_lr)

            # 检查早停
            if early_stopping:
                should_stop = early_stopping.step(
                    current_val_loss_for_lr,
                    lr_reduced=lr_reduced,
                    num_lr_reductions=lr_plateau.num_lr_reductions
                )
                if should_stop:
                    print(colorstr('red', 'bold', '\n⚠ Early stopping triggered!'))
                    # 早停时计算完整指标
                    print(colorstr('cyan', 'Computing final metrics before stopping...'))
                    val_metrics, _, _, _ = validate(
                        val_model,
                        val_loader,
                        criterion,
                        device,
                        conf_thres=args.conf_thres,
                        iou_thres=args.iou_thres,
                        nc=train_dataset.num_classes,
                        compute_metrics=True,
                        save_dir=save_dir,
                        plot_samples=args.plot_samples
                    )
                    # 更新plotter以包含完整指标
                    all_metrics = {'train_loss': train_loss, **val_metrics}
                    plotter.update(epoch, all_metrics)
                    plotter.save_metrics_csv()
                    # 打印最终指标
                    print(f'\n{colorstr("bright_yellow", "Final Metrics")}:')
                    print(f'  Precision: {val_metrics.get("precision", 0):.4f}')
                    print(f'  Recall: {val_metrics.get("recall", 0):.4f}')
                    print(f'  mAP@0.5: {val_metrics.get("mAP@0.5", 0):.4f}')
                    print(f'  mAP@0.5:0.95: {val_metrics.get("mAP@0.5:0.95", 0):.4f}')
                    print(f'  F1: {val_metrics.get("f1", 0):.4f}')
                    break

        # 合并指标（非早停情况）
        all_metrics = {
            'train_loss': train_loss,
            **val_metrics
        }

        # 更新可视化（非早停情况）
        plotter.update(epoch, all_metrics)

        # 实时保存CSV，防止中断丢失数据
        plotter.save_metrics_csv()

        # 打印指标
        print(f'\n{colorstr("bright_yellow", "Results")}:')
        print(f'  Train Loss: {train_loss:.4f}')
        current_val_loss = val_metrics.get("val_loss", 0)
        print(f'  Val Loss: {current_val_loss:.4f}')

        # 计算EMA平滑验证损失
        if ema_val_loss is None:
            ema_val_loss = current_val_loss
        else:
            ema_val_loss = ema_alpha * current_val_loss + (1 - ema_alpha) * ema_val_loss

        print(f'  Val Loss (EMA): {ema_val_loss:.4f}')

        if 'precision' in val_metrics:
            print(f'  Precision: {val_metrics["precision"]:.4f}')
            print(f'  Recall: {val_metrics["recall"]:.4f}')
            print(f'  mAP@0.5: {val_metrics.get("mAP@0.5", 0):.4f}')
            print(f'  mAP@0.5:0.95: {val_metrics.get("mAP@0.5:0.95", 0):.4f}')
            print(f'  F1: {val_metrics.get("f1", 0):.4f}')

        # 保存检查点（包含微调组件状态）
        ckpt = {
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics': all_metrics,
            'best_map': best_map,
            'ema_val_loss': ema_val_loss,
            'best_val_loss': best_val_loss,
            # 微调组件状态
            'lr_plateau': lr_plateau.state_dict(),
        }
        if early_stopping:
            ckpt['early_stopping'] = early_stopping.state_dict()
        if ema:
            ckpt['ema'] = ema.state_dict()
        if scaler:
            ckpt['scaler'] = scaler.state_dict()

        # 保存最新
        torch.save(ckpt, weights_dir / 'last.pt')

        # 保存最佳（基于mAP或EMA平滑后的val_loss）
        current_map = val_metrics.get('mAP@0.5', 0)
        if current_map > 0:  # 如果计算了mAP，使用mAP作为标准
            if current_map > best_map:
                best_map = current_map
                ckpt['best_map'] = best_map
                torch.save(ckpt, weights_dir / 'best.pt')
                print(colorstr('bright_green', f'  ★ New best model! mAP@0.5: {best_map:.4f}'))
        else:  # 否则使用EMA平滑后的val_loss作为标准（越小越好）
            if ema_val_loss < best_val_loss:
                best_val_loss = ema_val_loss
                ckpt['best_val_loss'] = best_val_loss
                torch.save(ckpt, weights_dir / 'best.pt')
                print(colorstr('bright_green', f'  ★ New best model! Val Loss (EMA): {ema_val_loss:.4f}'))

        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt, weights_dir / f'epoch_{epoch+1}.pt')

    # 训练结束
    print('\n' + '='*60)
    print(colorstr('bright_green', 'bold', 'Training completed!'))
    print(f'Best mAP@0.5: {best_map:.4f}')
    print(f'Results saved to: {save_dir}')
    print('='*60)

    # 生成最终可视化
    print(colorstr('bright_cyan', '\nGenerating final plots...'))
    plotter.plot_training_curves()
    plotter.save_metrics_csv()

    # 最终验证（使用最后保存的模型）
    print(colorstr('bright_cyan', '\nFinal evaluation...'))
    # 优先使用best.pt，如果不存在则使用last.pt
    best_weights = weights_dir / 'best.pt'
    last_weights = weights_dir / 'last.pt'

    if best_weights.exists():
        print(f'Loading best model from {best_weights}')
        ckpt = torch.load(best_weights, weights_only=False)
        model.load_state_dict(ckpt['model'])
        # 如果启用了EMA，也加载EMA权重
        if ema and 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
    elif last_weights.exists():
        print(f'Loading last model from {last_weights}')
        ckpt = torch.load(last_weights, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if ema and 'ema' in ckpt:
            ema.load_state_dict(ckpt['ema'])
    else:
        print('No saved weights found, using current model state')

    # 使用EMA模型进行最终评估
    final_model = ema.ema if ema else model
    final_metrics, _, _, _ = validate(
        final_model,
        val_loader,
        criterion,
        device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        nc=train_dataset.num_classes,
        compute_metrics=True,
        save_dir=save_dir,
        plot_samples=args.plot_samples
    )

    print(colorstr('bright_green', '\n✓ All done!'))
    print(f'Results: {save_dir}')
    print(f'  - Training curves: training_curves.png')
    print(f'  - Detection samples: val_batch_predictions.jpg')
    print(f'  - Labels distribution: labels_distribution.png')
    print(f'  - Metrics CSV: metrics.csv')
    print(f'  - Weights: weights/best.pt')


if __name__ == '__main__':
    main()
