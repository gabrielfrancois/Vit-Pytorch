# This file handles the evaluation of both Teacher and Student models on the Test Set.
# It generates comparison graphs for Accuracy, Speed (Throughput), and Confusion Matrices.
# python -m testing.run_test
import os
import time
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Imports
from models.vision_transformer import VisionTransformer
from models.dynamicViT import DynamicVisionTransformer
from data.load_data import load_CIFAR
from configs.train_cifar10 import * 
from helper_function.print import *

# Setup
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(orange(f"Using device: {device}"))

# Directories (best by default)
checkpoint_dir = "checkpoints"
teacher_path = f"{checkpoint_dir}/teacher_checkpoint_best.pth"
student_path = f"{checkpoint_dir}/student_best.pth"
results_dir = "testing/log/Evaluation_Graphs"
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
def plot_confusion_matrices(teacher_cm, student_cm, class_names, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Teacher
    sns.heatmap(teacher_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Teacher Confusion Matrix")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    
    # Student
    sns.heatmap(student_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1], xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title("Student (DynamicViT) Confusion Matrix")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_confusion_matrices.png"))
    plt.close()
    print(blue(f"Saved Confusion Matrices to {save_dir}"))

def plot_performance_comparison(teacher_acc, student_acc, teacher_speed, student_speed, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = ['Teacher', 'Student']
    colors = ['#1f77b4', '#ff7f0e'] # Blue, Orange
    
    # Accuracy Comparison
    axes[0].bar(models, [teacher_acc, student_acc], color=colors, alpha=0.8)
    axes[0].set_title("Top-1 Accuracy Comparison")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 100)
    # Add text labels
    for i, v in enumerate([teacher_acc, student_acc]):
        axes[0].text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')
        
    # Speed Comparison
    axes[1].bar(models, [teacher_speed, student_speed], color=colors, alpha=0.8)
    axes[1].set_title("Inference Speed (Throughput)")
    axes[1].set_ylabel("Images / Second")
    # Add text labels
    for i, v in enumerate([teacher_speed, student_speed]):
        axes[1].text(i, v + 10, f"{int(v)} img/s", ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_performance.png"))
    plt.close()
    print(blue(f"Saved Performance Comparison to {save_dir}"))

def plot_per_class_accuracy(teacher_cm, student_cm, class_names, save_dir):
    # Calculate per-class accuracy from CM (Diagonal / Row Sum)
    teacher_cls_acc = teacher_cm.diagonal() / teacher_cm.sum(axis=1) * 100
    student_cls_acc = student_cm.diagonal() / student_cm.sum(axis=1) * 100
    
    x = np.arange(len(class_names))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 6))
    rects1 = ax.bar(x - width/2, teacher_cls_acc, width, label='Teacher', color='#1f77b4')
    rects2 = ax.bar(x + width/2, student_cls_acc, width, label='Student', color='#ff7f0e')
    
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Per-Class Accuracy Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_per_class_accuracy.png"))
    plt.close()
    print(blue(f"Saved Per-Class Accuracy to {save_dir}"))

# Main Execution
if __name__ == "__main__":
    # CIFAR-10 Class Names
    class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
    # Load Data
    print(yellow("Loading Test Data..."))
    # We only need test_loader here
    _, test_loader, _ = load_CIFAR("/home/onyxia/work/Vit-Pytorch/data", CIFAR=10)

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