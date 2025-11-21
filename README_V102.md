# YOLOv1 Version 102 - 升级版

> **从MNIST数字检测到COCO目标检测的完整升级**

## 📋 更新内容

### ✨ Version 102 新特性

1. **输入尺寸升级**: 448x448 → **640x640**
2. **网格精细化**: 8x8 (64格) → **20x20 (400格)**
3. **多边界框**: 1个边界框/格 → **2个边界框/格**
4. **类别扩展**: 10个数字类别 → **80个COCO类别**
5. **数据格式**: 自定义MNIST → **Ultralytics YAML+TXT格式**

### 🎯 核心改进

| 特性 | V100 (旧版) | V102 (新版) |
|------|-------------|-------------|
| 输入尺寸 | 448×448 | **640×640** |
| 网格尺寸 | 8×8 | **20×20** |
| 边界框/格子 | 1 | **2** |
| 类别数 | 10 | **80** |
| 数据集 | MNIST | **COCO** |
| 格子总数 | 64 | **400** |
| 检测能力 | 单尺度 | **多尺度** |

---

## 🏗️ 项目结构

```
YOLOv1/
├── Generic/                          # 通用组件
│   ├── dataset/
│   │   ├── MNIST/                    # MNIST数据集 (V100)
│   │   │   ├── MNISTDataset.py
│   │   │   └── PlotMNISTImage.py
│   │   └── COCO/                     # ✨ COCO数据集 (V102)
│   │       ├── COCODataset.py        # Ultralytics格式加载器
│   │       └── __init__.py
│   ├── grids/                        # 网格系统
│   │   ├── YoloGrids.py
│   │   └── BoundingBox.py
│   ├── loss/                         # 损失函数
│   │   ├── YoloLoss.py
│   │   └── IoU.py
│   ├── scores/                       # 评估指标
│   │   └── YoloScores.py
│   └── tools/                        # 工具函数
│       ├── Normalizer.py
│       ├── Convertor.py
│       ├── ImagePlotter.py
│       └── TorchSetOp.py
│
├── YoloVer100/                       # V100版本 (MNIST)
│   └── model/
│       └── YoloNetwork.py            # 448x448, 8x8网格
│
├── YoloVer102/                       # ✨ V102版本 (COCO)
│   ├── model/
│   │   ├── YoloNetworkV102.py        # 640x640, 20x20网格
│   │   └── __init__.py
│   └── weights/                      # 模型权重保存目录
│
├── data/                             # 数据配置
│   ├── coco.yaml                     # ✨ COCO完整数据集配置
│   └── coco_sample.yaml              # ✨ COCO示例配置
│
├── train_yolo_v100.py                # V100训练脚本 (MNIST)
├── run_yolo_v100.py                  # V100推理脚本 (MNIST)
├── train_yolo_v102.py                # ✨ V102训练脚本 (COCO)
├── run_yolo_v102.py                  # ✨ V102推理脚本 (COCO)
│
├── Requirements.txt
├── README_V102.md                    # ✨ 本文档
└── .gitignore
```

---

## 🚀 快速开始

### 1️⃣ 环境安装

```bash
# 克隆仓库
git clone <repository_url>
cd YOLOv1

# 安装依赖
pip install torch torchvision opencv-python matplotlib pillow pyyaml tqdm
```

### 2️⃣ 准备COCO数据集

#### 数据集结构

YOLOv102使用**Ultralytics格式**的数据集：

```
coco_dataset/
├── images/
│   ├── train2017/          # 训练图像
│   │   ├── 000001.jpg
│   │   ├── 000002.jpg
│   │   └── ...
│   ├── val2017/            # 验证图像
│   │   ├── 000001.jpg
│   │   └── ...
│   └── test2017/           # 测试图像 (可选)
│
└── labels/
    ├── train2017/          # 训练标注
    │   ├── 000001.txt      # 对应图像的标注文件
    │   ├── 000002.txt
    │   └── ...
    ├── val2017/            # 验证标注
    └── test2017/           # 测试标注 (可选)
```

#### 标注文件格式 (TXT)

每个图像对应一个同名的`.txt`文件，每行表示一个目标：

```
class_id center_x center_y width height
```

- **class_id**: 类别ID (0-79)
- **center_x, center_y**: 边界框中心坐标 (归一化到0-1)
- **width, height**: 边界框宽高 (归一化到0-1)

**示例** (`000001.txt`):
```
0 0.5 0.5 0.3 0.4      # person, 中心(0.5, 0.5), 宽0.3, 高0.4
2 0.2 0.3 0.15 0.2     # car, 中心(0.2, 0.3), 宽0.15, 高0.2
```

#### 配置YAML文件

编辑 `data/coco.yaml`:

```yaml
# 数据集根目录 (修改为你的路径)
path: /path/to/coco_dataset

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
  # ... (80个类别)
```

### 3️⃣ 训练模型

#### 基础训练

```bash
python train_yolo_v102.py \
    --data data/coco.yaml \
    --epochs 50 \
    --batch-size 16 \
    --lr 0.001
```

#### 高级参数

```bash
python train_yolo_v102.py \
    --data data/coco.yaml \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001 \
    --weight-decay 0.0005 \
    --lambda-coord 5.0 \
    --lambda-noobj 0.5 \
    --save-dir YoloVer102/weights \
    --device cuda
```

**参数说明**:
- `--data`: YAML配置文件路径
- `--epochs`: 训练轮数
- `--batch-size`: 批次大小
- `--lr`: 学习率
- `--weight-decay`: 权重衰减
- `--lambda-coord`: 坐标损失权重
- `--lambda-noobj`: 无物体置信度损失权重
- `--save-dir`: 模型保存目录
- `--device`: 训练设备 (cuda/cpu)

#### 恢复训练

```bash
python train_yolo_v102.py \
    --data data/coco.yaml \
    --resume YoloVer102/weights/yolo_v102_latest.pth \
    --epochs 100
```

### 4️⃣ 推理检测

#### 单张图像

```bash
python run_yolo_v102.py \
    --weights YoloVer102/weights/yolo_v102_best.pth \
    --source path/to/image.jpg \
    --conf-threshold 0.5 \
    --output-dir runs/detect \
    --show
```

#### 图像目录

```bash
python run_yolo_v102.py \
    --weights YoloVer102/weights/yolo_v102_best.pth \
    --source path/to/images/ \
    --conf-threshold 0.5 \
    --output-dir runs/detect
```

#### 数据集验证

```bash
python run_yolo_v102.py \
    --weights YoloVer102/weights/yolo_v102_best.pth \
    --data data/coco.yaml \
    --conf-threshold 0.5 \
    --output-dir runs/detect \
    --num-images 100
```

**参数说明**:
- `--weights`: 模型权重路径
- `--source`: 图像路径或目录 (不指定则使用数据集)
- `--data`: YAML配置文件
- `--conf-threshold`: 置信度阈值
- `--output-dir`: 检测结果保存目录
- `--show`: 显示检测结果
- `--num-images`: 处理的图像数量 (数据集模式)

---

## 🧮 模型架构

### 网络结构

YOLOv102采用卷积神经网络架构，将640×640图像映射到20×20的网格预测：

```
输入: (B, 3, 640, 640) RGB图像
  ↓
Conv1: 7×7, stride 2 → (B, 64, 320, 320)
  ↓ MaxPool 2×2
  → (B, 64, 160, 160)
  ↓
Conv2: 3×3 → (B, 192, 160, 160)
  ↓ MaxPool 2×2
  → (B, 192, 80, 80)
  ↓
Conv3: 多层3×3 → (B, 512, 80, 80)
  ↓ MaxPool 2×2
  → (B, 512, 40, 40)
  ↓
Conv4: 多层3×3 → (B, 1024, 40, 40)
  ↓ MaxPool 2×2
  → (B, 1024, 20, 20)
  ↓
Conv5: 多层3×3 → (B, 1024, 20, 20)
  ↓
Conv6: 3×3 → (B, 1024, 20, 20)
  ↓
Flatten + FC7 → (B, 4096)
  ↓
FC8 → (B, 35600)  # 89 × 400
  ↓
Reshape → (B, 89, 400)
```

### 输出格式

**输出张量**: `(B, 89, 400)`

- **B**: 批次大小
- **89**: 每个格子的特征数 = 1 (置信度) + 2×4 (2个边界框) + 80 (类别)
- **400**: 网格格子数 = 20×20

**每个格子包含**:
```
[confidence,                      # 1: 置信度
 bbox1_cx, bbox1_cy, bbox1_w, bbox1_h,  # 4: 第一个边界框
 bbox2_cx, bbox2_cy, bbox2_w, bbox2_h,  # 4: 第二个边界框
 class_0, class_1, ..., class_79]       # 80: 类别概率
```

### 模型参数

- **总参数量**: ~202M
- **模型大小**: ~808 MB
- **推理速度**: ~20 FPS (NVIDIA RTX 3090)

---

## 📊 损失函数

YOLOv102使用多项损失的加权和：

```
Loss = λ_coord × L_coord + L_conf + L_class
```

### 1️⃣ 坐标损失 (L_coord)

仅对有物体的格子计算：

```
L_coord = Σ [MSE(pred_bbox, true_bbox)]
```

- 权重: `λ_coord = 5.0`
- 使用MSE损失

### 2️⃣ 置信度损失 (L_conf)

```
L_conf = Σ_obj [MSE(pred_conf, 1)]
       + λ_noobj × Σ_noobj [MSE(pred_conf, 0)]
```

- 有物体格子: 目标置信度为1
- 无物体格子: 目标置信度为0，权重`λ_noobj = 0.5`

### 3️⃣ 分类损失 (L_class)

仅对有物体的格子计算：

```
L_class = Σ [MSE(pred_class, true_class)]
```

- 使用one-hot编码
- 80个类别的概率分布

---

## 🎨 数据增强

YOLOv102内置简单的数据增强：

- **随机亮度调整**: ±30
- **随机对比度调整**: 0.8-1.2×

可在`COCODataset.py`中自定义更多增强：

```python
def _augment_image(self, img):
    # 添加更多增强操作
    # - 随机翻转
    # - 随机裁剪
    # - 随机缩放
    # - 色彩抖动
    # etc.
    return img
```

---

## 🔧 进阶使用

### 自定义类别数

如果你的数据集不是80个类别，可以修改：

1. **YAML配置**:
```yaml
nc: 20  # 自定义类别数
names:
  - class_0
  - class_1
  # ...
```

2. **训练脚本**:
```python
model = YoloV1NetworkV102(
    grids_size=(20, 20),
    confidences=1,
    bounding_boxes=2,
    object_categories=20  # 修改为你的类别数
)
```

### 调整网格尺寸

支持不同的网格尺寸 (需要重新训练):

```python
# 更粗的网格 (检测大物体)
grids_size = (14, 14)

# 更细的网格 (检测小物体)
grids_size = (28, 28)
```

⚠️ **注意**: 修改网格尺寸需要调整模型架构中的全连接层。

### 可视化训练过程

可使用TensorBoard或wandb记录训练：

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/yolo_v102')
writer.add_scalar('Loss/train', train_loss, epoch)
writer.add_scalar('Loss/val', val_loss, epoch)
```

---

## 📈 性能优化建议

### 训练优化

1. **学习率调度**:
   - 使用warmup: 前几个epoch逐渐增加学习率
   - 使用cosine annealing或step decay

2. **批次大小**:
   - GPU内存充足时增大batch size
   - 使用梯度累积模拟大batch

3. **混合精度训练**:
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    predictions = model(images)
    loss = compute_loss(predictions, labels)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 推理优化

1. **TorchScript导出**:
```python
model.eval()
traced_model = torch.jit.trace(model, example_input)
traced_model.save('yolo_v102_traced.pt')
```

2. **ONNX导出**:
```python
torch.onnx.export(
    model,
    example_input,
    'yolo_v102.onnx',
    input_names=['images'],
    output_names=['predictions']
)
```

---

## 🐛 常见问题

### Q1: 训练时显存不足？

**解决方案**:
- 减小batch size: `--batch-size 8`
- 使用梯度累积
- 使用混合精度训练

### Q2: 损失不收敛？

**解决方案**:
- 检查数据集标注是否正确
- 降低学习率: `--lr 0.0001`
- 增加warmup epochs
- 检查lambda权重是否合理

### Q3: 检测效果不佳?

**解决方案**:
- 增加训练epochs
- 使用数据增强
- 调整置信度阈值
- 增加训练数据量
- 尝试不同的网格尺寸

### Q4: 标注文件格式错误？

**检查清单**:
- ✓ 坐标是否归一化到[0, 1]
- ✓ class_id是否在[0, nc-1]范围内
- ✓ 每行5个数值，空格分隔
- ✓ 文件名与图像对应

---

## 📚 参考资料

- [YOLO原始论文](https://arxiv.org/abs/1506.02640): Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection"
- [COCO数据集](https://cocodataset.org/)
- [Ultralytics格式说明](https://docs.ultralytics.com/datasets/detect/)

---

## 📝 版本历史

- **V102** (2024): COCO支持，640×640输入，20×20网格，2个边界框/格
- **V100** (初版): MNIST支持，448×448输入，8×8网格，1个边界框/格

---

## 🤝 贡献

欢迎提交Issue和Pull Request！

---

## 📄 License

MIT License

---

## 👨‍💻 作者

YOLOv1 Implementation & Upgrade

---

**祝训练顺利！🚀**
