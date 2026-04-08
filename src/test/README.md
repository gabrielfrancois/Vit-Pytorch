# Model Evaluation, Comparison & Analysis

This directory contains scripts for evaluating, benchmarking, and visualizing the performance of your Vision Transformers (ViT) and DynamicViTs. The pipeline is optimized for **ImageNet-1k (128x128)**, focusing on accuracy, inference speed, and computational efficiency.

---

## Requirements

Ensure your environment is set up and dependencies are installed:

```bash
uv sync
# or
pip install -r requirements.txt
```

---

## Usage & Scripts

### 1. Teacher vs. Student Comparison (`compare_st-te.py`)
This is the primary evaluation script. It compares a **Teacher** (Standard ViT) and a **Student** (DynamicViT) across several metrics.

- **Metrics**: Top-1 Accuracy, Loss, and Throughput (img/sec).
- **Visuals**: Generates side-by-side Confusion Matrices and Performance Comparison bar charts.
- **DynamicViT Insights**: Produces per-layer pruning visualizations to show which patches the Student model "drops."

**Example CLI**:
```bash
# Evaluate both models with default checkpoints
python -m src.test.compare_st-te --test-teacher --test-student

# Evaluate only the Student with a custom checkpoint
python -m src.test.compare_st-te --test-student --student-checkpoint checkpoints/imagenet/student_best.pth

# Run with custom visualization settings
python -m src.test.compare_st-te --test-student --visualize --num_images 12 --device cuda
```

---

### 2. Computational Cost Analysis (`parameters_computing.py`)
Use this script to analyze the structural complexity and theoretical speed of your models. It uses `calflops` to provide precise measurements.

- **Metrics**: Total Parameters (M), FLOPs (G), and MACs (G).

**Example CLI**:
```bash
# Calculate costs for default ImageNet checkpoints
python -m src.test.parameters_computing --dataset imagenet

# Specify custom checkpoint paths
python -m src.test.parameters_computing \
  --teacher-checkpoint checkpoints/imagenet/teacher_checkpoint_best.pth \
  --student-checkpoint checkpoints/imagenet/student_best.pth
```

---

### 3. SSL Teacher Reconstruction (`test_SSLteacher.py`)
Evaluates the Self-Supervised Learning (SSL) performance of a Teacher model trained with Masked Autoencoder (MAE) objectives.

- **Metrics**: MSE, MAE, PSNR (Peak Signal-to-Noise Ratio), and Threshold Accuracy.
- **Visuals**: Generates triple-panel images: Original vs. Masked Input vs. Model Reconstruction.

**Example CLI**:
```bash
# Standard SSL evaluation
python -m src.test.test_SSLteacher --dataset imagenet

# Evaluate with a specific mask ratio and visualization count
python -m src.test.test_SSLteacher \
  --checkpoint checkpoints/imagenet/ssl_teacher/ssl_teacher_best.pth \
  --mask_ratio 0.75 \
  --num_images 10
```

---

## Outputs & Logs

All results are automatically saved to the `logs/imagenet/` directory:

- **Performance Graphs**: `logs/imagenet/evaluation_results/`
  - `compare_performance.png`: Accuracy & Throughput bars.
  - `compare_confusion_matrices.png`: Top-20 class confusion matrices.
- **Pruning Visuals**: `logs/imagenet/pruning_visualizations/`
  - `pruning_vis_X.png`: Visualizes patch dropping at each `pruning_index` [4, 7, 10].
- **SSL Visuals**: `logs/imagenet/ssl_teacher/visualizations/`
  - `ssl_reconstruction_X.png`: MAE reconstruction quality samples.

---

## Key Arguments (Common)

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset selection (focus on `imagenet`) | `imagenet` |
| `--device` | Force device (`cpu`, `cuda`, `mps`) | Auto-detect |
| `--num_images` | Number of visualization samples to generate | `8` (or `5` for SSL) |
| `--teacher-checkpoint` | Path to Teacher `.pth` | `checkpoints/imagenet/...` |
| `--student-checkpoint` | Path to Student `.pth` | `checkpoints/imagenet/...` |
