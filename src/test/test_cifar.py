# This file handles the evaluation of both Teacher and Student models on the Test Set.
# It generates comparison graphs for Accuracy, Speed (Throughput), and Confusion Matrices.
# python -m test.test_cifar
import os
import time
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from typing import Sequence

# Imports
from src.models.vision_transformer import VisionTransformer
from src.models.dynamicViT import DynamicVisionTransformer
from data.load.load_data import load_CIFAR
from configs.train_cifar10 import * 
from helper_function.print import *

# Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(bold(f"Using device: {device}"))

# Directories (best by default)
checkpoint_dir = "checkpoints/cifar10/student_2th_try"
teacher_path = "checkpoints/cifar10/teacher_2th_try/teacher_checkpoint_best.pth"
student_path = "checkpoints/cifar10/student_2th_try/student_best.pth"
results_dir = "./logs/cifar10/student/graphs/Evaluation_Graphs"
os.makedirs(results_dir, exist_ok=True)


# Universal Evaluation Function
def evaluate_model(model, loader, device, model_name="Model"):
    """
    Evaluates any model (Teacher or Student) on the provided loader.
    Handles different return signatures automatically.
    Returns: accuracy, average_loss, throughput, predictions, true_labels
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    # Timing for Throughput
    start_time = time.time()
    
    with torch.no_grad():
        loop = tqdm(loader, desc=f"Testing {model_name}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Forward pass
            # Student returns 4 values, Teacher returns 2, we only need the first one (logits) for accuracy.
            outputs = model(imgs)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            
            loss = criterion(logits, labels)

            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    end_time = time.time()
    
    # Metrics
    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    total_time = end_time - start_time
    throughput = len(loader.dataset) / total_time # Images per second
    
    print(orange("-"*10+ f"\nResults for {model_name}:" + "-"*10))
    print(bold(f"  Accuracy: {accuracy:.2f}%"))
    print(bold(f"  Loss: {avg_loss:.4f}"))
    print(bold(f"  Throughput: {throughput:.2f} img/sec"))
    
    return accuracy, avg_loss, throughput, all_preds, all_labels

# Plotting Functions



def plot_confusion_matrices(
    teacher_cm: np.ndarray,
    student_cm: np.ndarray,
    class_names: Sequence[str],
    save_dir: str
) -> None:
    """
    Plot and save side-by-side confusion matrices for teacher and student models.

    This function visualizes the confusion matrices using heatmaps in order
    to qualitatively compare the prediction behavior of a static teacher
    model and a dynamically pruned student model.

    Args:
        teacher_cm (np.ndarray):
            Confusion matrix of the teacher model (shape: [num_classes, num_classes]).
        student_cm (np.ndarray):
            Confusion matrix of the student model (shape: [num_classes, num_classes]).
        class_names (Sequence[str]):
            List of class names corresponding to matrix indices.
        save_dir (str):
            Directory where the generated figure will be saved.
    """

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    # Teacher confusion matrix
    sns.heatmap(
        teacher_cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        ax=axes[0],
        xticklabels=class_names,
        yticklabels=class_names
    )
    axes[0].set_title("Teacher Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # Student confusion matrix
    sns.heatmap(
        student_cm,
        annot=True,
        fmt='d',
        cmap='Oranges',
        ax=axes[1],
        xticklabels=class_names,
        yticklabels=class_names
    )
    axes[1].set_title("Student (DynamicViT) Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_confusion_matrices.png"))
    plt.close()

    print(blue(f"Saved confusion matrices to {save_dir}"))


def plot_performance_comparison(
    teacher_acc: float,
    student_acc: float,
    teacher_speed: float,
    student_speed: float,
    save_dir: str
) -> None:
    """
    Plot and save a comparison of accuracy and inference throughput
    between teacher and student models.

    Args:
        teacher_acc (float):
            Top-1 accuracy of the teacher model (percentage).
        student_acc (float):
            Top-1 accuracy of the student model (percentage).
        teacher_speed (float):
            Inference throughput of the teacher model (images per second).
        student_speed (float):
            Inference throughput of the student model (images per second).
        save_dir (str):
            Directory where the generated figure will be saved.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    models = ['Teacher', 'Student']
    colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange

    # Accuracy comparison
    axes[0].bar(models, [teacher_acc, student_acc], color=colors, alpha=0.8)
    axes[0].set_title("Top-1 Accuracy Comparison")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)

    for i, v in enumerate([teacher_acc, student_acc]):
        axes[0].text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')

    # Throughput comparison
    axes[1].bar(models, [teacher_speed, student_speed], color=colors, alpha=0.8)
    axes[1].set_title("Inference Throughput Comparison")
    axes[1].set_ylabel("Images / Second")

    for i, v in enumerate([teacher_speed, student_speed]):
        axes[1].text(i, v + 10, f"{int(v)} img/s", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_performance.png"))
    plt.close()

    print(blue(f"Saved performance comparison to {save_dir}"))


def plot_per_class_accuracy(
    teacher_cm: np.ndarray,
    student_cm: np.ndarray,
    class_names: Sequence[str],
    save_dir: str
) -> None:
    """
    Plot and save a per-class accuracy comparison between teacher and student models.

    Per-class accuracy is computed as the ratio of correct predictions
    (diagonal of the confusion matrix) to the total number of samples
    per class.

    Args:
        teacher_cm (np.ndarray):
            Confusion matrix of the teacher model.
        student_cm (np.ndarray):
            Confusion matrix of the student model.
        class_names (Sequence[str]):
            List of class names.
        save_dir (str):
            Directory where the generated figure will be saved.
    """

    teacher_cls_acc: np.ndarray = (
        teacher_cm.diagonal() / teacher_cm.sum(axis=1) * 100
    )
    student_cls_acc: np.ndarray = (
        student_cm.diagonal() / student_cm.sum(axis=1) * 100
    )

    x = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, teacher_cls_acc, width, label='Teacher', color='#1f77b4')
    ax.bar(x + width / 2, student_cls_acc, width, label='Student', color='#ff7f0e')

    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_per_class_accuracy.png"))
    plt.close()

    print(blue(f"Saved per-class accuracy comparison to {save_dir}"))

# Main Execution
if __name__ == "__main__":
    # CIFAR-10 Class Names
    class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    # Load Data
    print(yellow("Loading Test Data..."))
    # We only need test_loader here
    data_path = "./data/load/cifar10"
    _, test_loader, _ = load_CIFAR(data_path, CIFAR=10)

    # Load Teacher & student
    print(yellow("Loading Teacher Model..."))
    teacher = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    else:
        print(red("Teacher checkpoint not found! Running with random weights (Results will be meaningless)."))
        
    print(yellow("Loading Student Model..."))
    student = DynamicVisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index=pruning_index).to(device)
    if os.path.exists(student_path):
        student.load_state_dict(torch.load(student_path, map_location=device))
    else:
        print(red("Student checkpoint not found! Running with random weights."))

    # Run Evaluation
    t_acc, t_loss, t_speed, t_preds, t_labels = evaluate_model(teacher, test_loader, device, "Teacher")
    s_acc, s_loss, s_speed, s_preds, s_labels = evaluate_model(student, test_loader, device, "Student")
    
    # Generate plot
    print(yellow("\nGenerating Graphs..."))
    t_cm = confusion_matrix(t_labels, t_preds)
    s_cm = confusion_matrix(s_labels, s_preds)
    plot_confusion_matrices(t_cm, s_cm, class_names, results_dir)
    plot_performance_comparison(t_acc, s_acc, t_speed, s_speed, results_dir)
    plot_per_class_accuracy(t_cm, s_cm, class_names, results_dir)
    
    print(green(f"Evaluation Complete. All graphs saved in {results_dir}"))

    # Compute the speed bonus
    diff_speed = ((s_speed-t_speed)/t_speed)*100
    print(bold(f"student increased by {diff_speed}% the number of images processed per second."))