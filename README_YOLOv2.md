# YOLOv2 Implementation - 完整升级版

> **从YOLOv1到YOLOv2的完整实现 - 引入Anchor Boxes和Darknet-19**

## 🎯 YOLOv2 vs YOLOv1/V102

### 核心改进对比

| 特性 | YOLOv1 V100 | YOLOv1 V102 | **YOLOv2** |
|------|-------------|-------------|------------|
| **输入尺寸** | 448×448 | 640×640 | **640×640** |
| **Backbone** | 自定义CNN | 自定义CNN | **Darknet-19** |
| **网格** | 8×8 | 20×20 | **20×20** |
| **Anchor Boxes** | ❌ 无 | ❌ 无 | **✅ 5个** |
| **检测方式** | 直接回归 | 直接回归 | **Anchor-based** |
| **Passthrough** | ❌ 无 | ❌ 无 | **✅ 有** |
| **Batch Norm** | ✅ 有 | ✅ 有 | **✅ 全部** |
| **全连接层** | ✅ 有 | ✅ 有 | **❌ 无** |
| **架构类型** | CNN+FC | CNN+FC | **全卷积** |
| **参数量** | ~100M | ~202M | **~50M** |

### YOLOv2的关键创新

#### 1. **Darknet-19 Backbone**
- 19层卷积网络
- 全部使用Batch Normalization
- 使用1×1卷积降维
- 更高效的特征提取

#### 2. **Anchor Boxes机制**
- 预定义5个anchor尺寸（通过K-means聚类获得）
- 每个grid cell可以检测多个尺度的物体
- 提高了小物体和多物体检测能力

#### 3. **Passthrough Layer**
- 类似ResNet的skip connection
- 将40×40的高分辨率特征连接到20×20
- 保留更多细节信息，增强小物体检测

#### 4. **直接位置预测**
```python
bx = sigmoid(tx) + cx
by = sigmoid(ty) + cy
bw = pw * exp(tw)
bh = ph * exp(th)
```
- 使用sigmoid约束中心坐标，确保在grid cell内
- 使用exponential缩放anchor尺寸

#### 5. **全卷积网络**
- 去掉全连接层
- 减少参数量
- 支持多尺度输入（可选）

---

## 🏗️ 架构详解

### 网络结构

```
输入: (B, 3, 640, 640)
  ↓
┌──────────────────────────────────────┐
│ Darknet-19 Backbone                  │
├──────────────────────────────────────┤
│ Conv 3×3, 32   → 640×640×32          │
│ MaxPool 2×2    → 320×320×32          │
│ Conv 3×3, 64   → 320×320×64          │
│ MaxPool 2×2    → 160×160×64          │
│                                      │
│ Conv 3×3, 128  → 160×160×128         │
│ Conv 1×1, 64   → 160×160×64          │
│ Conv 3×3, 128  → 160×160×128         │
│ MaxPool 2×2    → 80×80×128           │
│                                      │
│ Conv 3×3, 256  → 80×80×256           │
│ Conv 1×1, 128  → 80×80×128           │
│ Conv 3×3, 256  → 80×80×256           │
│ MaxPool 2×2    → 40×40×256           │
│                                      │
│ [多层1×1和3×3交替] → 40×40×512       │
│ MaxPool 2×2    → 20×20×512 ─┐        │
│                             │ (保存用于Passthrough)
│ [多层1×1和3×3交替] → 20×20×1024      │
└──────────────────────────────────────┘
  ↓
┌──────────────────────────────────────┐
│ Passthrough Layer                    │
├──────────────────────────────────────┤
│ 40×40×512 → Space-to-depth           │
│          → 20×20×2048                │
│          → Conv 1×1, 64              │
│          → 20×20×64                  │
└──────────────────────────────────────┘
  ↓
Concat: [20×20×1024, 20×20×64] → 20×20×1088
  ↓
┌──────────────────────────────────────┐
│ Detection Head                       │
├──────────────────────────────────────┤
│ Conv 3×3, 1024 → 20×20×1024          │
│ Conv 1×1, 425  → 20×20×425           │
│   (5 anchors × 85 = 425)             │
└──────────────────────────────────────┘
  ↓
Reshape: (B, 5, 20, 20, 85)

其中 85 = 5 (tx,ty,tw,th,conf) + 80 (classes)
```

### Anchor Boxes

YOLOv2使用5个预定义的anchor boxes（通过K-means聚类COCO数据集得到）：

```python
anchors = [
    [0.57273, 0.677385],   # 小物体 (36×43 pixels @ 640×640)
    [1.87446, 2.06253],    # 中等物体 (120×132 pixels)
    [3.33843, 5.47434],    # 大物体 (214×350 pixels)
    [7.88282, 3.52778],    # 宽物体 (505×226 pixels)
    [9.77052, 9.16828]     # 超大物体 (626×587 pixels)
]
```

每个anchor负责检测特定尺寸范围的物体。

---

## 🚀 快速开始

### 1️⃣ 环境安装

```bash
# 克隆仓库
git clone <repository_url>
cd YOLOv1

# 安装依赖
pip install torch torchvision opencv-python matplotlib pillow pyyaml tqdm numpy
```

### 2️⃣ 准备数据集

使用Ultralytics格式的COCO数据集：

```
coco_dataset/
├── images/
│   ├── train2017/
│   └── val2017/
└── labels/
    ├── train2017/
    └── val2017/
```

每个txt标注文件：
```
class_id center_x center_y width height
```
（所有坐标归一化到[0,1]）

### 3️⃣ 配置YAML

编辑 `data/coco.yaml`:
```yaml
path: /path/to/coco_dataset
train: images/train2017
val: images/val2017
nc: 80
names: [person, bicycle, car, ...]
```

### 4️⃣ 训练模型

#### 基础训练

```bash
python train_yolov2.py \
    --data data/coco.yaml \
    --epochs 100 \
    --batch-size 16 \
    --lr 0.001
```

#### 高级参数

```bash
python train_yolov2.py \
    --data data/coco.yaml \
    --epochs 160 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0.0005 \
    --lambda-coord 5.0 \
    --lambda-noobj 0.5 \
    --lambda-class 1.0 \
    --save-dir YoloVer2/weights \
    --device cuda \
    --num-workers 8
```

**参数说明**:
- `--epochs`: 训练轮数（YOLOv2原文使用160 epochs）
- `--batch-size`: 批次大小
- `--lr`: 初始学习率
- `--lambda-coord`: 坐标损失权重（default: 5.0）
- `--lambda-noobj`: 无物体损失权重（default: 0.5）
- `--lambda-class`: 分类损失权重（default: 1.0）

#### 恢复训练

```bash
python train_yolov2.py \
    --data data/coco.yaml \
    --resume YoloVer2/weights/yolov2_latest.pth \
    --epochs 160
```

### 5️⃣ 推理检测

#### 单张图像

```bash
python run_yolov2.py \
    --weights YoloVer2/weights/yolov2_best.pth \
    --source path/to/image.jpg \
    --conf-threshold 0.5 \
    --nms-threshold 0.5 \
    --show
```

#### 批量图像

```bash
python run_yolov2.py \
    --weights YoloVer2/weights/yolov2_best.pth \
    --source path/to/images/ \
    --conf-threshold 0.5 \
    --nms-threshold 0.5 \
    --output-dir runs/detect_v2
```

**参数说明**:
- `--conf-threshold`: 置信度阈值
- `--nms-threshold`: NMS IoU阈值
- `--show`: 显示检测结果

---

## 📊 损失函数

YOLOv2使用三部分损失的加权和：

```
Loss = λ_coord × L_coord + L_conf + λ_class × L_class
```

### 1️⃣ 坐标损失

对有物体的grid cell计算tx, ty, tw, th的MSE损失：

```python
L_coord = MSE(pred_tx, target_tx) + MSE(pred_ty, target_ty)
        + MSE(pred_tw, target_tw) + MSE(pred_th, target_th)
```

权重: `λ_coord = 5.0`

### 2️⃣ 置信度损失

```python
L_conf = BCE(pred_conf[obj], 1.0)              # 有物体
       + λ_noobj × BCE(pred_conf[noobj], 0.0)  # 无物体
```

权重: `λ_noobj = 0.5`

### 3️⃣ 分类损失

对有物体的grid cell计算类别概率的BCE损失：

```python
L_class = BCE(pred_class[obj], target_class[obj])
```

权重: `λ_class = 1.0`

---

## 📈 训练技巧

### 学习率调度

默认使用MultiStepLR:
- Epoch 0-60: lr = 0.001
- Epoch 60-90: lr = 0.0001
- Epoch 90+: lr = 0.00001

### 数据增强

当前支持：
- 随机亮度调整
- 随机对比度调整

可扩展：
- 随机翻转
- 随机裁剪
- Mosaic augmentation
- MixUp

### 多尺度训练（可选）

YOLOv2支持多尺度训练，可以修改训练脚本动态改变输入尺寸：

```python
# 每10个batch随机选择一个尺寸
if batch_idx % 10 == 0:
    img_size = random.choice([320, 352, 384, 416, 448, 480, 512, 544, 576, 608, 640])
    # 调整数据加载器...
```

---

## 🎨 高级特性

### 1. 自定义Anchor Boxes

如果你的数据集物体尺寸分布与COCO不同，可以使用K-means聚类生成自定义anchors：

```python
import numpy as np
from sklearn.cluster import KMeans

# 收集所有边界框的宽高
widths = []
heights = []

for label in all_labels:
    for obj in label:
        widths.append(obj['w'])
        heights.append(obj['h'])

# K-means聚类
boxes = np.column_stack([widths, heights])
kmeans = KMeans(n_clusters=5)
kmeans.fit(boxes)

anchors = kmeans.cluster_centers_
print("Custom anchors:", anchors)
```

### 2. 迁移学习

使用预训练的Darknet-19权重：

```python
# 加载Darknet-19 ImageNet预训练权重
backbone_weights = torch.load('darknet19_imagenet.pth')
model.backbone.load_state_dict(backbone_weights)

# 冻结backbone，只训练detection head
for param in model.backbone.parameters():
    param.requires_grad = False
```

### 3. 混合精度训练

使用torch.cuda.amp加速训练：

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for images, targets in train_loader:
    optimizer.zero_grad()

    with autocast():
        predictions = model(images)
        loss, _ = criterion(predictions, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 🔧 模型导出

### TorchScript

```python
model.eval()
example = torch.randn(1, 3, 640, 640)
traced = torch.jit.trace(model, example)
traced.save('yolov2_traced.pt')
```

### ONNX

```python
torch.onnx.export(
    model,
    example,
    'yolov2.onnx',
    input_names=['images'],
    output_names=['predictions'],
    dynamic_axes={
        'images': {0: 'batch'},
        'predictions': {0: 'batch'}
    }
)
```

---

## 📚 代码结构

```
YOLOv1/
├── YoloVer2/                        # YOLOv2实现
│   └── model/
│       ├── Darknet19.py             # Darknet-19 backbone
│       ├── YOLOv2.py                # YOLOv2主网络
│       └── __init__.py
│
├── Generic/
│   ├── dataset/COCO/
│   │   ├── COCODatasetV2.py         # YOLOv2数据加载器
│   │   └── ...
│   └── loss/
│       ├── YOLOv2Loss.py            # YOLOv2损失函数
│       └── ...
│
├── train_yolov2.py                  # 训练脚本
├── run_yolov2.py                    # 推理脚本
├── data/coco.yaml                   # 数据集配置
└── README_YOLOv2.md                 # 本文档
```

---

## 🐛 常见问题

### Q1: OOM (Out of Memory) 错误？

**解决方案**:
- 减小batch size: `--batch-size 8`
- 减小输入尺寸（需要修改代码）
- 使用梯度累积
- 使用混合精度训练

### Q2: 损失NaN？

**解决方案**:
- 检查数据标注是否正确
- 降低学习率: `--lr 0.0001`
- 检查梯度裁剪
- 确保归一化正确

### Q3: Anchor匹配问题？

**检查**:
- ✓ Anchor尺寸是否合适
- ✓ 是否需要重新聚类
- ✓ IoU计算是否正确

### Q4: 小物体检测不佳？

**优化**:
- 确认Passthrough layer工作正常
- 增加数据增强
- 调整Anchor尺寸
- 增加训练epochs

---

## 📖 参考文献

- **YOLO9000: Better, Faster, Stronger**
  Joseph Redmon, Ali Farhadi
  [arXiv:1612.08242](https://arxiv.org/abs/1612.08242)

- **YOLOv1 Paper**
  Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"
  [arXiv:1506.02640](https://arxiv.org/abs/1506.02640)

- **Darknet Framework**
  [https://pjreddie.com/darknet/](https://pjreddie.com/darknet/)

---

## 🎯 性能对比

### 理论性能

| 指标 | YOLOv1 V100 | YOLOv1 V102 | YOLOv2 |
|------|-------------|-------------|---------|
| **参数量** | ~100M | ~202M | **~50M** ✓ |
| **FLOPs** | ~30B | ~50B | **~34B** |
| **最大检测数** | 64 | 800 | **2000** |
| **Anchor支持** | ❌ | ❌ | **✅** |
| **小物体检测** | 弱 | 中 | **强** ✓ |
| **推理速度** | 快 | 中 | **快** ✓ |

### 预期mAP (COCO val)

- **YOLOv1**: ~45% mAP
- **YOLOv2**: ~68% mAP @640×640
- **YOLOv2 (原文)**: ~76% mAP @544×544 with multi-scale

---

## 🤝 版本历史

- **YOLOv2** (当前): Anchor boxes, Darknet-19, Passthrough layer
- **V102**: 640×640输入, 20×20网格, COCO支持
- **V100** (初版): 448×448输入, 8×8网格, MNIST

---

## 📝 License

MIT License

---

**祝训练顺利！🚀**

如有问题请参考代码注释或提Issue。
