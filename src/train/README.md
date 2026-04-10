# Training Pipeline

This directory contains the core training logic for the Vision Transformer (ViT) and DynamicViT models. The pipeline is designed as a three-stage process, optimized for **ImageNet-1k (128x128)**.

---

## Training Stages

### 1. SSL Teacher Pre-training (`train_teacher_SSL.py`)
Trains a standard Teacher ViT using **Self-Supervised Learning (SSL)** via a Masked Autoencoder (MAE) objective. The model learns to reconstruct missing patches from a heavily masked image (default 75% mask ratio).

**Key Features**:
- MAE Reconstruction Loss (MSE).
- Layer-wise Learning Rate Decay (LLRD).
- Learning rate warmup and cosine annealing.

**Example CLI**:
```bash
python -m src.train.train_teacher_SSL --epochs 400 --mask_ratio 0.75
```

---

### 2. Teacher Supervised Training with REPA (`train_teacher.py`)
Fine-tunes the SSL Teacher using supervised labels and **REPA** (Representation Bottleneck). It uses a frozen DINOv1 model to provide rich feature representations that the Teacher must emulate.

**Key Features**:
- Combined Cross-Entropy and REPA (Cosine Similarity) loss.
- Distillation from DINOv1 features.
- Supports resuming from SSL checkpoints.

**Example CLI**:
```bash
# Resume from SSL pre-training
python -m src.train.train_teacher \
  --resume-from checkpoints/imagenet/ssl_teacher/ssl_teacher_best.pth \
  --lambda_repa 2.0 \
  --epochs 120
```

---

### 3. Student DynamicViT Training (`train_student.py`)
Trains the **Student** (DynamicViT) to efficiently drop less informative patches. The student is initialized with the Teacher's backbone and trained using a complex composite loss.

**Key Features**:
- **Distillation**: Learns from the frozen Teacher's logits and features.
- **Dynamic Pruning**: Learns patch-dropping masks at specific layers (default: 4, 7, 10).
- **Composite Loss** (`dynamic_loss.py`):
  - Classification Loss (Ground Truth).
  - Distillation Loss (Teacher Logits).
  - KL Divergence (Feature alignment).
  - Sparsity/Ratio Loss (Enforcing the pruning factor $\rho$).

**Example CLI**:
```bash
# Train student with a specific pruning factor (rho)
python -m src.train.train_student \
  --teacher_checkpoint checkpoints/imagenet/teacher/teacher_checkpoint_best.pth \
  --rho 0.7 \
  --run_name my_student_run
```

---

## Outputs & Monitoring

### Checkpoints
Saved in `checkpoints/imagenet/`:
- `ssl_teacher/ssl_teacher_best.pth`
- `teacher/teacher_checkpoint_best.pth`
- `{run_name}/student_best.pth`

### Logs & Visualizations
Saved in `logs/imagenet/`:
- **Tensorboard**: Real-time tracking of losses, accuracy, and learning rates.
- **Graphs**:
  - `teacher/graphs/`: Loss and accuracy curves for the teacher.
  - `student/graphs/`: Detailed student metrics including `ratio_loss`, `distill_loss`, and `rho` evolution.

---

## Argument Reference (Common)

| Argument | Description | Default (ImageNet) |
|----------|-------------|---------------------|
| `--dataset` | `imagenet` or `cifar10` | `imagenet` |
| `--epochs` | Number of training epochs | From `configs/` |
| `--batch_size` | Training batch size | `256` |
| `--alpha` | Base learning rate | `0.0015` |
| `--device` | Hardware acceleration (`cuda`, `mps`, `cpu`) | Auto-detect |
| `--resume-from` | Path to a checkpoint to resume training | `None` |
