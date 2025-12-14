import os
import time
import logging
from pathlib import Path
from typing import Tuple, List, Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from omegaconf import OmegaConf

from data.imagenet_loader import load_imagenet1k
from models.vision_transformer import VisionTransformer
from models.dynamicViT import DynamicVisionTransformer
from helper_function.print import bold, blue, orange, yellow, red, green

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("Evaluation")

# --- Load Configurations ---
cfg_base = OmegaConf.load("configs/base.yaml")
cfg_imagenet = OmegaConf.load("configs/imagenet.yaml")
cfg = OmegaConf.merge(cfg_base, cfg_imagenet)

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# --- Directories ---
checkpoint_dir = Path(cfg.outputs.checkpoint_dir)
results_dir = Path(cfg.result.result_dir)
checkpoint_dir.mkdir(parents=True, exist_ok=True)
results_dir.mkdir(parents=True, exist_ok=True)

teacher_path = Path(cfg.student.teacher_checkpoint)
student_path = Path(cfg.student.student_checkpoint)

# --- Evaluation ---
def evaluate_model(
    model: nn.Module, loader: DataLoader, device: torch.device, model_name: str = "Model"
) -> Tuple[float, float, float, List[int], List[int]]:
    """
    Evaluates a model (Teacher or Student) on the given loader.

    Returns:
        accuracy: float, top-1 accuracy in %
        avg_loss: float, average cross-entropy loss
        throughput: float, images/sec
        all_preds: List[int], predicted class indices
        all_labels: List[int], true class indices
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    start_time = time.time()
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Testing {model_name}"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            loss = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    end_time = time.time()
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    throughput = total / max(end_time - start_time, 1e-8)

    logger.info(f"{model_name} Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}, Throughput: {throughput:.2f} img/s")
    return accuracy, avg_loss, throughput, all_preds, all_labels


# --- Plotting ---
def plot_confusion_matrices(
    teacher_cm: np.ndarray, student_cm: np.ndarray, class_names: List[str], save_dir: Path
) -> None:
    """Plot side-by-side confusion matrices for Teacher and Student models."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(teacher_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Teacher Confusion Matrix"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")

    sns.heatmap(student_cm, annot=True, fmt="d", cmap="Oranges", ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title("Student Confusion Matrix"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(save_dir / "compare_confusion_matrices.png")
    plt.close()
    logger.info(f"Saved confusion matrices to {save_dir}")


def plot_performance_comparison(
    teacher_acc: float, student_acc: float,
    teacher_speed: float, student_speed: float,
    save_dir: Path
) -> None:
    """Plot bar charts for accuracy and inference speed."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    models, colors = ["Teacher", "Student"], ["#1f77b4", "#ff7f0e"]

    # Accuracy
    axes[0].bar(models, [teacher_acc, student_acc], color=colors, alpha=0.8)
    axes[0].set_ylabel("Accuracy (%)"); axes[0].set_ylim(0, 100)
    axes[0].set_title("Top-1 Accuracy Comparison")
    for i, v in enumerate([teacher_acc, student_acc]):
        axes[0].text(i, v + 1, f"{v:.2f}%", ha="center", fontweight="bold")

    # Speed
    axes[1].bar(models, [teacher_speed, student_speed], color=colors, alpha=0.8)
    axes[1].set_ylabel("Images / Second"); axes[1].set_title("Inference Speed")
    for i, v in enumerate([teacher_speed, student_speed]):
        axes[1].text(i, v + 5, f"{int(v)} img/s", ha="center", fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_dir / "compare_performance.png")
    plt.close()
    logger.info(f"Saved performance comparison to {save_dir}")


def plot_per_class_accuracy(
    teacher_cm: np.ndarray, student_cm: np.ndarray, class_names: List[str], save_dir: Path
) -> None:
    """Plot per-class accuracy comparison based on confusion matrices."""
    teacher_acc = np.divide(
        teacher_cm.diagonal(),
        teacher_cm.sum(axis=1),
        out=np.zeros_like(teacher_cm.diagonal(), dtype=float),
        where=teacher_cm.sum(axis=1) != 0
    ) * 100

    student_acc = np.divide(
        student_cm.diagonal(),
        student_cm.sum(axis=1),
        out=np.zeros_like(student_cm.diagonal(), dtype=float),
        where=student_cm.sum(axis=1) != 0
    ) * 100

    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width/2, teacher_acc, width, label="Teacher", color="#1f77b4")
    ax.bar(x + width/2, student_acc, width, label="Student", color="#ff7f0e")
    ax.set_xticks(x); ax.set_xticklabels(class_names)
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Per-Class Accuracy Comparison")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "compare_per_class_accuracy.png")
    plt.close()
    logger.info(f"Saved per-class accuracy to {save_dir}")


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("Loading test data...")
    _, test_loader, _ = load_imagenet1k(
        batch_size= cfg.dataset.batch_size,
        max_items_train = cfg.dataset.max_items_train,
        max_items_val = cfg.dataset.max_items_val,
    )

    # Initialize models
    m_cfg = cfg.model
    pruning_index = cfg.dynamicvit.get("pruning_index", 0)
    teacher = VisionTransformer(
        d_model=m_cfg.d_model,
        n_classes=m_cfg.n_classes,
        img_size=m_cfg.img_size,
        patch_size=m_cfg.patch_size,
        n_channels=m_cfg.n_channels,
        n_heads=m_cfg.n_heads,
        n_layers=m_cfg.n_layers
    ).to(device)
    student = DynamicVisionTransformer(
        d_model=m_cfg.d_model,
        n_classes=m_cfg.n_classes,
        img_size=m_cfg.img_size,
        patch_size=m_cfg.patch_size,
        n_channels=m_cfg.n_channels,
        n_heads=m_cfg.n_heads,
        n_layers=m_cfg.n_layers,
        pruning_index=pruning_index
    ).to(device)

    # Load checkpoints if they exist
    if teacher_path.exists():
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    else:
        logger.warning("Teacher checkpoint not found. Using random weights.")

    if student_path.exists():
        student.load_state_dict(torch.load(student_path, map_location=device))
    else:
        logger.warning("Student checkpoint not found. Using random weights.")

    # Evaluate
    t_acc, t_loss, t_speed, t_preds, t_labels = evaluate_model(teacher, test_loader, device, "Teacher")
    s_acc, s_loss, s_speed, s_preds, s_labels = evaluate_model(student, test_loader, device, "Student")

    # Generate plots
    labels = sorted(set(t_labels + s_labels))
    class_names = [f"class_{i}" for i in labels]
    plot_confusion_matrices(confusion_matrix(t_labels, t_preds), confusion_matrix(s_labels, s_preds), class_names, results_dir)
    plot_performance_comparison(t_acc, s_acc, t_speed, s_speed, results_dir)
    plot_per_class_accuracy(confusion_matrix(t_labels, t_preds), confusion_matrix(s_labels, s_preds), class_names, results_dir)

    # Speed bonus
    speed_diff = ((s_speed - t_speed) / t_speed) * 100
    logger.info(f"Student increased throughput by {speed_diff:.2f}%")
