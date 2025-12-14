# This file handles the evaluation of both Teacher and Student models on the Test Set.
# It generates comparison graphs for Accuracy, Speed (Throughput), Confusion Matrices, and DynamicViT pruning visualization.
# python -m testing.run_test
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from tqdm import tqdm

from configs.train_cifar10 import *
from data.load_data import load_CIFAR
from helper_function.print import *
from models.dynamicViT import DynamicVisionTransformer
from models.vision_transformer import VisionTransformer

# Setup
device = torch.device("cpu")
print(bold(f"Using device: {device}"))

# Directories
checkpoint_dir = "checkpoints"
teacher_path = f"{checkpoint_dir}/teacher_checkpoint_best.pth"
student_path = f"{checkpoint_dir}/student_best.pth"
results_dir = "testing/log/Evaluation_Graphs_cifar"
os.makedirs(results_dir, exist_ok=True)

# --- Pruning Visualization Function ---
def visualize_pruning(images, all_masks, patch_size, stage_names=None, idx=0, save_path=None):
    """
    Visualize token pruning stage by stage for a single image in the batch.
    """
    img = images[idx].permute(1,2,0).cpu().numpy()  # H,W,C
    n_stages = len(all_masks)
    n_patches_h = img.shape[0] // patch_size[0]
    n_patches_w = img.shape[1] // patch_size[1]

    plt.figure(figsize=(4*n_stages,4))
    for i, mask in enumerate(all_masks):
        stage_mask = mask[idx,1:].cpu().numpy()  # ignore CLS token
        stage_mask = stage_mask.reshape(n_patches_h, n_patches_w)

        plt.subplot(1, n_stages, i+1)
        plt.imshow(img)
        plt.imshow(np.kron(1-stage_mask, np.ones(patch_size)), cmap='Reds', alpha=0.5)
        plt.axis('off')
        if stage_names:
            plt.title(stage_names[i])
        else:
            plt.title(f"Stage {i+1}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved pruning visualization to {save_path}")
    else:
        plt.show()

# --- Universal Evaluation Function ---
def evaluate_model(model, loader, device, model_name="Model"):
    """
    Evaluates any model (Teacher or Student) on the provided loader.
    Returns: accuracy, average_loss, throughput, predictions, true_labels, all_masks (if Student)
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_masks = None

    start_time = time.time()

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Testing {model_name}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)

            # Teacher returns logits only, Student returns tuple
            if isinstance(outputs, tuple):
                logits, _, all_masks_batch, _ = outputs
                if all_masks is None:
                    all_masks = all_masks_batch  # store masks from first batch
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

    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    total_time = end_time - start_time
    throughput = len(loader.dataset) / total_time  # images/sec

    print(orange("-"*10 + f"\nResults for {model_name}:" + "-"*10))
    print(bold(f"  Accuracy: {accuracy:.2f}%"))
    print(bold(f"  Loss: {avg_loss:.4f}"))
    print(bold(f"  Throughput: {throughput:.2f} img/sec"))

    return accuracy, avg_loss, throughput, all_preds, all_labels, all_masks

# --- Plotting Functions (unchanged) ---
def plot_confusion_matrices(teacher_cm, student_cm, class_names, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(teacher_cm, annot=True, fmt="d", cmap="Blues", ax=axes[0], xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Teacher Confusion Matrix"); axes[0].set_xlabel("Predicted"); axes[0].set_ylabel("True")
    sns.heatmap(student_cm, annot=True, fmt="d", cmap="Oranges", ax=axes[1], xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title("Student (DynamicViT) Confusion Matrix"); axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("True")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_confusion_matrices.png"))
    plt.close()
    print(blue(f"Saved Confusion Matrices to {save_dir}"))

def plot_performance_comparison(teacher_acc, student_acc, teacher_speed, student_speed, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    models = ["Teacher", "Student"]
    colors = ["#1f77b4", "#ff7f0e"]
    axes[0].bar(models, [teacher_acc, student_acc], color=colors, alpha=0.8)
    axes[0].set_title("Top-1 Accuracy Comparison"); axes[0].set_ylabel("Accuracy (%)"); axes[0].set_ylim(0,100)
    for i, v in enumerate([teacher_acc, student_acc]):
        axes[0].text(i, v+1, f"{v:.2f}%", ha="center", fontweight="bold")
    axes[1].bar(models, [teacher_speed, student_speed], color=colors, alpha=0.8)
    axes[1].set_title("Inference Speed (Throughput)"); axes[1].set_ylabel("Images / Second")
    for i, v in enumerate([teacher_speed, student_speed]):
        axes[1].text(i, v+10, f"{int(v)} img/s", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_performance.png"))
    plt.close()
    print(blue(f"Saved Performance Comparison to {save_dir}"))

def plot_per_class_accuracy(teacher_cm, student_cm, class_names, save_dir):
    teacher_cls_acc = teacher_cm.diagonal() / teacher_cm.sum(axis=1) * 100
    student_cls_acc = student_cm.diagonal() / student_cm.sum(axis=1) * 100
    x = np.arange(len(class_names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(14,6))
    ax.bar(x - width/2, teacher_cls_acc, width, label="Teacher", color="#1f77b4")
    ax.bar(x + width/2, student_cls_acc, width, label="Student", color="#ff7f0e")
    ax.set_ylabel("Accuracy (%)"); ax.set_title("Per-Class Accuracy Comparison")
    ax.set_xticks(x); ax.set_xticklabels(class_names); ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_per_class_accuracy.png"))
    plt.close()
    print(blue(f"Saved Per-Class Accuracy to {save_dir}"))

# --- Main Execution ---
if __name__ == "__main__":
    class_names = ["Plane","Car","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"]
    print(yellow("Loading Test Data..."))
    _, test_loader, _ = load_CIFAR("/home/onyxia/work/Vit-Pytorch/data", CIFAR=10)

    print(yellow("Loading Teacher Model..."))
    teacher = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
    if os.path.exists(teacher_path):
        teacher.load_state_dict(torch.load(teacher_path, map_location=device))
    else:
        print(red("Teacher checkpoint not found! Running with random weights."))

    print(yellow("Loading Student Model..."))
    student = DynamicVisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index=pruning_index).to(device)
    if os.path.exists(student_path):
        student.load_state_dict(torch.load(student_path, map_location=device))
    else:
        print(red("Student checkpoint not found! Running with random weights."))

    # --- Evaluation ---
    t_acc, t_loss, t_speed, t_preds, t_labels, _ = evaluate_model(teacher, test_loader, device, "Teacher")
    s_acc, s_loss, s_speed, s_preds, s_labels, s_masks = evaluate_model(student, test_loader, device, "Student")

    # --- Graphs ---
    t_cm = confusion_matrix(t_labels, t_preds)
    s_cm = confusion_matrix(s_labels, s_preds)
    plot_confusion_matrices(t_cm, s_cm, class_names, results_dir)
    plot_performance_comparison(t_acc, s_acc, t_speed, s_speed, results_dir)
    plot_per_class_accuracy(t_cm, s_cm, class_names, results_dir)

    # --- Visualisation du pruning ---
    sample_imgs, _ = next(iter(test_loader))
    sample_imgs = sample_imgs.to(device)
    with torch.no_grad():
        _, _, all_masks, _ = student(sample_imgs)

    visualize_pruning(
        sample_imgs,
        all_masks,
        patch_size=patch_size,
        stage_names=[f"Stage {i+1}" for i in range(len(all_masks))],
        idx=0,
        save_path=os.path.join(results_dir, "student_pruning_visualization.png")
    )

    print(green(f"Evaluation Complete. All graphs saved in {results_dir}"))
    diff_speed = ((s_speed - t_speed) / t_speed) * 100
    print(bold(f"student increased by {diff_speed:.2f}% the number of images processed per second."))
