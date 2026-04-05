# test.py — Model Evaluation & Visualization

Evaluate and compare a **Teacher** ViT against a **Student** DynamicViT on CIFAR-10 or ImageNet-1k.
Produces accuracy/speed comparison charts, confusion matrices, and per-layer pruning visualizations.

---

## Requirements

Install dependencies before running:
```bash
uv sync
```
or 

```bash
pip install -r requirements.txt
```

Then:
```bash
source .venv/bin/activate
```

## Checkpoints

By default the script looks for:

| Model   | Default path                                    |
|---------|-------------------------------------------------|
| Teacher | `checkpoints/{dataset}/teacher_checkpoint_best.pth` |
| Student | `checkpoints/{dataset}/student_best.pth`        |

You can override either path with `--teacher_checkpoint` / `--student_checkpoint`.

---

## Usage

### Minimal — evaluate both models on imagenet
```bash
python -m src.test.test
```

### Evaluate on ImageNet-1k
```bash
python -m src.test.test --dataset imagenet
```

### Evaluate only the Student (skip Teacher)
```bash
python -m src.test.test --test_teacher False --test_student True
```

### Custom checkpoints
```bash
python -m src.test.test \
  --teacher_checkpoint path/to/my_teacher.pth \
  --student_checkpoint path/to/my_student.pth
```

### Override model architecture (**must match your checkpoint**)
```bash
python -m src.test.test --d_model 384 --n_heads 6 --n_layers 12
```

### Control the number of pruning visualizations
```bash
python -m src.test.test --visualize --num_images 16
```

### Force a specific device
```bash
python -m src.test.test --device cuda   # or cpu / mps
```

---

## All Arguments

| Argument               | Type    | Default          | Description |
|------------------------|---------|------------------|-------------|
| `--dataset`            | `str`   | `cifar10`        | Dataset to load. Choices: `cifar10`, `imagenet`. |
| `--test_teacher`       | flag    | `True`           | Run evaluation on the Teacher model. |
| `--test_student`       | flag    | `True`           | Run evaluation on the Student model. |
| `--teacher_checkpoint` | `str`   | `None`           | Path to Teacher `.pth` checkpoint. Falls back to default if not set or missing. |
| `--student_checkpoint` | `str`   | `None`           | Path to Student `.pth` checkpoint. Falls back to default if not set or missing. |
| `--visualize`          | flag    | `True`           | Generate per-image pruning mask visualizations for the Student. |
| `--num_images`         | `int`   | `8`              | Number of images to visualize when `--visualize` is set. |
| `--d_model`            | `int`   | from config      | Embedding dimension. Overrides the value in the config file. |
| `--n_layers`           | `int`   | from config      | Number of Transformer layers. |
| `--n_heads`            | `int`   | from config      | Number of attention heads. |
| `--batch_size`         | `int`   | from config      | Batch size for the DataLoader. |
| `--device`             | `str`   | auto-detect      | Force a specific device. Choices: `cpu`, `cuda`, `mps`. |

---

## Outputs

All outputs are written relative to the project root:
```
logs/
├── {dataset}/
│   ├── evaluation_results/
│   │   ├── compare_confusion_matrices.png   # Side-by-side confusion matrices
│   │   └── compare_performance.png          # Accuracy & throughput bar charts
│   └── pruning_visualizations/
│       ├── pruning_vis_0.png                # Original + masked image per pruning layer
│       ├── pruning_vis_1.png
│       └── ...
```

The confusion matrix plot clips to the **top 10 classes** on CIFAR-10 and **top 20** on ImageNet for readability.

---

## Example — Full Run on CIFAR-10 with Custom Paths
```bash
python -m src.test.test \
  --dataset cifar10 \
  --teacher_checkpoint checkpoints/cifar10/teacher_v2.pth \
  --student_checkpoint checkpoints/cifar10/student_v2.pth \
  --num_images 12 \
  --device cuda
```

Expected terminal output per model:
```
------------------------------
Results for Teacher:
  Accuracy:   94.31%
  Loss:       0.1823
  Throughput: 3412 img/sec
------------------------------
```

At the end of a joint run, a summary is printed:
```
 Student Speed-Up: +47.23%
 Accuracy Drop: 1.15%
```