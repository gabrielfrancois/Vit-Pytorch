# This Python file gathers all the functions / classes needed to handle training, validation or inference from the ViT model
# python -m training.test_imagenet

import os
import time
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from models.vision_transformer import VisionTransformer
from models.dynamicViT_imagenet import DynamicVisionTransformer
from data.imagenet_loader import load_imagenet1k
from configs.train_imagenet1k import *
from helper_function.print import *
from typing import List, Tuple, Optional, Any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(bold(f"Using device: {device}"))

# Checkpoint paths
checkpoint_dir = "checkpoints/imagenet1K"
checkpoint_path = f"{checkpoint_dir}/student_checkpoint_last.pth"
teacher_checkpoint = f"{checkpoint_dir}/teacher_checkpoint_best.pth"
student_path = f"{checkpoint_dir}/student_best.pth"
os.makedirs(checkpoint_dir, exist_ok=True)





results_dir = "testing/log/Imagenet/best/Evaluation_Graphs_Test"
pruning_vis_dir = "testing/log/Imagenet/best/Pruning_Images"
os.makedirs(results_dir, exist_ok=True)

# Initialize and load TEACHER
print(yellow("Initializing Teacher..."))
teacher = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)

if os.path.exists(teacher_checkpoint):
    print(green(f"Loading Teacher weights from {teacher_checkpoint}"))
    teacher.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
else:
    raise FileNotFoundError(red(f"Teacher checkpoint not found at {teacher_checkpoint}. Please run run_teacher.py first!"))


# Initialize student
print(yellow("Initializing Student..."))
student = DynamicVisionTransformer(
    d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index, rho_init
).to(device)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate_teacher_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_name: str = "Model"
) -> Tuple[float, float, float, List[int], List[int]]:
    """
    Evaluate a model on a dataset and compute loss, accuracy, throughput, and predictions.

    Args:
        model: PyTorch model to evaluate (teacher or student).
        loader: DataLoader providing the evaluation dataset.
        device: Device to run the evaluation on.
        model_name: Name used for printing/logging purposes.

    Returns:
        accuracy: Classification accuracy in percentage.
        avg_loss: Average cross-entropy loss.
        throughput: Images processed per second.
        all_preds: List of predicted class indices.
        all_labels: List of ground-truth class indices.
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

            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_time = time.time() - start_time
    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    throughput = len(loader.dataset) / total_time

    print(orange("-" * 10 + f"\nResults for {model_name}:" + "-" * 10))
    print(bold(f"  Accuracy: {accuracy:.2f}%"))
    print(bold(f"  Loss: {avg_loss:.4f}"))
    print(bold(f"  Throughput: {throughput:.2f} img/sec"))

    return accuracy, avg_loss, throughput, all_preds, all_labels

# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------
def plot_confusion_matrices(
    teacher_cm: np.ndarray,
    student_cm: np.ndarray,
    class_names: Optional[List[str]],
    save_dir: str,
    top_k: int = 10
) -> None:
    """
    Plot and save the top-k class confusion matrices for teacher and student models.

    Args:
        teacher_cm: Confusion matrix of the teacher model.
        student_cm: Confusion matrix of the student model.
        class_names: List of class names. If None, default names are generated.
        save_dir: Directory where the plot will be saved.
        top_k: Number of top classes to display based on frequency in teacher_cm.
    """
    # Create class names if there's not
    if class_names is None:
        num_classes = teacher_cm.shape[0]
        class_names = [f"Class {i}" for i in range(num_classes)]
   
    freq = np.sum(teacher_cm, axis=1)  
    top_indices = np.argsort(freq)[-top_k:][::-1]  # top-k indices sorted in descending order

    # Subsample the matrices and class name
    teacher_cm_top = teacher_cm[np.ix_(top_indices, top_indices)]
    student_cm_top = student_cm[np.ix_(top_indices, top_indices)]
    class_names_top = [class_names[i] for i in top_indices]

    #  Plot
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(teacher_cm_top, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names_top, yticklabels=class_names_top, ax=axes[0])
    axes[0].set_title("Teacher Confusion Matrix (Top {})".format(top_k))

    sns.heatmap(student_cm_top, annot=True, fmt="d", cmap="Oranges",
                xticklabels=class_names_top, yticklabels=class_names_top, ax=axes[1])
    axes[1].set_title("Student (DynamicViT) Confusion Matrix (Top {})".format(top_k))

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "compare_confusion_matrices_top{}.png".format(top_k)))
    plt.close()


def plot_performance_comparison(
    t_acc: float,
    s_acc: float,
    t_speed: float,
    s_speed: float,
    save_dir: str,
    device: str = "cpu",
) -> None:
    """
    Plot and save a bar chart comparing teacher and student performance.

    Args:
        t_acc: Teacher accuracy (%).
        s_acc: Student accuracy (%).
        t_speed: Teacher throughput (img/sec).
        s_speed: Student throughput (img/sec).
        save_dir: Directory where the plot will be saved.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    models = ['Teacher', 'Student']
    colors = ['#1f77b4', '#ff7f0e']  # Blue, Orange

    # Accuracy Comparison
    axes[0].bar(models, [t_acc, s_acc], color=colors, alpha=0.8)
    axes[0].set_title("Top-1 Accuracy Comparison")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    # Add text labels
    for i, v in enumerate([t_acc, s_acc]):
        axes[0].text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')

    # Speed Comparison
    axes[1].bar(models, [t_speed, s_speed], color=colors, alpha=0.8)
    axes[1].set_title("Inference Speed (Throughput)")
    axes[1].set_ylabel("Images / Second")
    # Add text labels
    for i, v in enumerate([t_speed, s_speed]):
        axes[1].text(i, v + 10, f"{int(v)} img/s", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"compare_performance_{device}.png"))
    plt.close()
    print(blue(f"Saved Performance Comparison to {save_dir}"))

def plot_per_class_accuracy(
    t_cm: np.ndarray,
    s_cm: np.ndarray,
    class_names: Optional[List[str]],
    save_dir: str
) -> None:
    """
    Plot and save a per-class accuracy comparison for teacher and student models.

    Args:
        t_cm: Teacher confusion matrix.
        s_cm: Student confusion matrix.
        class_names: List of class names. If None, default names are generated.
        save_dir: Directory where the plot will be saved.
    """

    # Create class names if it's necessary
    if class_names is None:
        num_classes = t_cm.shape[0]
        class_names = [f"Class {i}" for i in range(num_classes)]
    t_acc = t_cm.diagonal() / t_cm.sum(axis=1) * 100
    s_acc = s_cm.diagonal() / s_cm.sum(axis=1) * 100
    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width / 2, t_acc, width, label="Teacher")
    ax.bar(x + width / 2, s_acc, width, label="Student")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Accuracy (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_per_class_accuracy.png"))
    plt.close()



def evaluate_student_model(
    student: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "Validation"
) -> tuple[float, float, float, list, list]:
    """
    Evaluate the student model on a dataset and compute loss, accuracy, throughput, and predictions.

    Args:
        student: Student Vision Transformer model.
        loader: DataLoader for validation or test data.
        device: Device used for evaluation.
        desc: Description shown in the progress bar.

    Returns:
        accuracy: Classification accuracy (%).
        avg_loss: Average cross-entropy loss.
        throughput: Images processed per second.
        all_preds: List of predicted class indices.
        all_labels: List of ground-truth class indices.
    """
    student.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    start_time = time.time()

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Testing Student - {desc}"):
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = student(imgs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            # Compute loss
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)

            # Compute accuracy
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    total_time = time.time() - start_time
    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    throughput = len(loader.dataset) / total_time

    print(orange("-" * 10 + f"\nResults for Student ({desc}):" + "-" * 10))
    print(bold(f"  Accuracy: {accuracy:.2f}%"))
    print(bold(f"  Loss: {avg_loss:.4f}"))
    print(bold(f"  Throughput: {throughput:.2f} img/sec"))

    return accuracy, avg_loss, throughput, all_preds, all_labels


def rho_schedule(
    epoch: int,
    max_epoch: int,
    rho_init: float = 1.0,
    rho_final: float = 0.7,
    steepness: float = 10.0
) -> float:
    """
    Compute a smooth pruning ratio schedule using a sigmoid function.

    The schedule starts close to rho_init, transitions smoothly around
    mid-training, and converges towards rho_final.

    Args:
        epoch: Current epoch index.
        max_epoch: Total number of epochs.
        rho_init: Initial pruning ratio.
        rho_final: Final pruning ratio.
        steepness: Controls how sharp the transition is.

    Returns:
        rho: Pruning ratio for the given epoch.
    """
    x = epoch / (max_epoch - 1)
    s = 1 / (1 + np.exp(-steepness * (x - 0.5)))
    return rho_init + (rho_final - rho_init) * s


if __name__ == "__main__":
    start_time = time.time()

    class_names = None
    # Load data
    print(blue("Loading Data..."))
    train_loader, val_loader, test_loader = load_imagenet1k()

    print(yellow("Starting Student Testing..."))
    start_epoch = 0
    history = {
        'train_loss': [], 'ratio_loss': [], "distill_loss": [], "kl_loss": [],
        'train_acc': [], 'val_acc': [], 'lrs': [], 'rho': []
    }
    scaler = torch.amp.GradScaler()  # Mixed precision scaler

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        last_epoch = ckpt['epoch']
        ans = input(f"Checkpoint found at epoch {last_epoch}. Test last model or best one ? [last/best/n]")
        if ans.lower() == 'last':
            student.load_state_dict(ckpt['student_state'])
            start_epoch = last_epoch + 1
            results_dir = "testing/log/Imagenet/last/Evaluation_Graphs_Test"
            pruning_vis_dir = "testing/log/Imagenet/last/Pruning_Images"
        elif ans.lower() == 'best':
            student.load_state_dict(torch.load(student_path, map_location=device))
            start_epoch = 85
            results_dir = "testing/log/Imagenet/best/Evaluation_Graphs_Test"
            pruning_vis_dir = "testing/log/Imagenet/best/Pruning_Images"

    best_val_acc = 0.0
    rho = rho_schedule(start_epoch, epochs)

    
    # Evaluation
    t_acc, t_loss, t_speed, t_preds, t_labels = evaluate_teacher_model(
        teacher, test_loader, device, "Teacher"
    )

    # best
    

    s_acc, s_loss, s_speed, s_preds, s_labels  = evaluate_student_model(student, test_loader, device, desc="Testing Student")




    print(yellow("Generating graphs..."))
    plot_confusion_matrices(
        confusion_matrix(t_labels, t_preds),
        confusion_matrix(s_labels, s_preds),
        class_names, results_dir
    )
    plot_performance_comparison(
        t_acc, s_acc, t_speed, s_speed, results_dir, device
    )
    plot_per_class_accuracy(
        confusion_matrix(t_labels, t_preds),
        confusion_matrix(s_labels, s_preds),
        class_names, results_dir
    )

    diff_speed = ((s_speed - t_speed) / t_speed) * 100
    print(bold(f"Student speed-up: {diff_speed:.2f}%"))

    # print(yellow("Visualizing pruning..."))
    # visualize_pruning_on_images(student, test_loader, device)
    # print(green("Done."))

    # Display time taken
    seconds = time.time() - start_time
    print(cyan('Time Taken:'), cyan(time.strftime("%H:%M:%S", time.gmtime(seconds))))


