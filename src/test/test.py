import argparse
import os
import time

import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import List, Tuple, Optional

from helper_function.print import *
from src.models.vision_transformer import VisionTransformer
from src.models.dynamicViT import DynamicVisionTransformer

# ----------------------------------------- Core Evaluation Function -----------------------------------------

def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    model_name: str = "Model"
) -> Tuple[float, float, float, List[int], List[int]]:
    """
    Run a full evaluation pass over the test set and return key metrics.
    Args:
        model       (nn.Module):
        loader      (DataLoader):
        device      (torch.device):
        model_name  (str):
    Returns:
        accuracy    (float):                Top-1 accuracy in percent [0, 100].
        avg_loss    (float):                Mean CrossEntropy loss over the full dataset.
        throughput  (float):                Inference speed in images per second.
        all_preds   (List[int]):            Predicted class indices, one per sample.
        all_labels  (List[int]):            Ground-truth class indices, one per sample.
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
            # Handle both Teacher (returns tuple) and Student (returns bigger tuple)
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

    print(blue("-" * 30))
    print(blue(f"Results for {model_name}:"))
    print(bold(f"  Accuracy:   {accuracy:.2f}%"))
    print(bold(f"  Loss:       {avg_loss:.4f}"))
    print(bold(f"  Throughput: {throughput:.2f} img/sec"))
    print(blue("-" * 30))

    return accuracy, avg_loss, throughput, all_preds, all_labels

# ----------------------------------------- Plot Functions -----------------------------------------

def plot_confusion_matrices(
    teacher_cm: np.ndarray, student_cm: np.ndarray,
    class_names: Optional[List[str]], save_dir: str, top_k: int = 10
) -> None:
    """
        Save a side-by-side confusion matrix plot for Teacher and Student.
        If the number of classes exceeds top_k (e.g. ImageNet), only the top_k
        most frequent classes are shown.
        Args:
            teacher_cm  (np.ndarray):           Confusion matrix for the Teacher, shape [C, C].
            student_cm  (np.ndarray):           Confusion matrix for the Student, shape [C, C].
            class_names (Optional[List[str]]):  Human-readable label for each class, length C.
                                                If None, labels default to "Class 0", "Class 1", ...
            save_dir    (str):                  Directory where the PNG file is written.
            top_k       (int):                  Max number of classes to display. Default: 10.
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(teacher_cm.shape[0])]
    
    # If dealing with ImageNet (1000 classes), only show Top K 
    if teacher_cm.shape[0] > top_k:
        freq = np.sum(teacher_cm, axis=1)  
        top_indices = np.argsort(freq)[-top_k:][::-1]
        teacher_cm = teacher_cm[np.ix_(top_indices, top_indices)]
        student_cm = student_cm[np.ix_(top_indices, top_indices)]
        class_names = [class_names[i] for i in top_indices]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(teacher_cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"Teacher Confusion Matrix (Top {len(class_names)})")

    sns.heatmap(student_cm, annot=True, fmt="d", cmap="Oranges", xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"Student Confusion Matrix (Top {len(class_names)})")

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "compare_confusion_matrices.png"))
    plt.close()

def plot_performance_comparison(t_acc: float, s_acc: float, t_speed: float, s_speed: float, save_dir: str) -> None:
    """
        Save a two-panel bar chart comparing Teacher vs Student accuracy and throughput.
        Args:
            t_acc   (float):    Teacher top-1 accuracy in percent.
            s_acc   (float):    Student top-1 accuracy in percent.
            t_speed (float):    Teacher inference throughput in images/second.
            s_speed (float):    Student inference throughput in images/second.
            save_dir (str):     Directory where the PNG file is written.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    models = ['Teacher', 'Student']
    colors = ['#1f77b4', '#ff7f0e']

    axes[0].bar(models, [t_acc, s_acc], color=colors, alpha=0.8)
    axes[0].set_title("Top-1 Accuracy Comparison")
    axes[0].set_ylabel("Accuracy (%)")
    for i, v in enumerate([t_acc, s_acc]):
        axes[0].text(i, v + 1, f"{v:.2f}%", ha='center', fontweight='bold')

    axes[1].bar(models, [t_speed, s_speed], color=colors, alpha=0.8)
    axes[1].set_title("Inference Speed (Throughput)")
    axes[1].set_ylabel("Images / Second")
    for i, v in enumerate([t_speed, s_speed]):
        axes[1].text(i, v + 10, f"{int(v)} img/s", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_performance.png"))
    plt.close()
    print(green(f"Saved Performance Comparison graphs to {save_dir}"))

def visualize_pruning_on_images(
    student_model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str,
    num_images: int = 8,
    pruned_color: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> None:
    """
        For each image, save a figure showing the original alongside the pruning mask
        applied at each pruning layer. Pruned patches are filled with a solid color.
        
        The model is temporarily set to train() mode so patches are not physically
        dropped, keeping all mask tensors full-sized for easy spatial plotting.
        Dropout layers are frozen manually to avoid corrupting activations.
        Args:
            student_model   (nn.Module):                    The DynamicVisionTransformer to visualize.
            loader          (DataLoader):                   DataLoader yielding (imgs, labels) batches.
            device          (torch.device):                 Device to run inference on.
            save_dir        (str):                          Directory where PNG files are written.
            num_images      (int):                          Number of images to visualize. Default: 8.
            pruned_color    (Tuple[float, float, float]):   RGB fill color for pruned patches,
                                                            values in [0, 1]. Default: (0.5, 0.5, 0.5).
        Notes:
            Expects student_model(imgs) to return a 4-tuple where index 2 is a list of
            binary mask tensors, one per pruning layer, each of shape [B, N+1] (CLS token
            included at position 0).
        """
    print("\nGenerating Pruning Visualizations...")
    os.makedirs(save_dir, exist_ok=True)
    
    student_model.train() # Put in train mode so patches aren't physically deleted
    for m in student_model.modules():
        if isinstance(m, nn.Dropout):
            m.eval() # manually freeze dropout so the image isn't ruined

    images_done = 0
    # ImageNet normalization stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            outputs = student_model(imgs)

            if not isinstance(outputs, tuple) or len(outputs) < 3:
                print("Student model does not return pruning masks.")
                return

            _, _, all_masks, _ = outputs

            B, C, H, W = imgs.shape
            ph, pw = student_model.patch_size
            n_h, n_w = H // ph, W // pw
            num_patches = n_h * n_w

            for i in range(B):
                if images_done >= num_images:
                    return

                img = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img = img * std + mean
                img = np.clip(img, 0, 1)

                fig, axes = plt.subplots(1, len(student_model.pruning_index) + 1, 
                                         figsize=(3 * (len(student_model.pruning_index) + 1), 3))
                axes[0].imshow(img)
                axes[0].set_title("Original")
                axes[0].axis("off")

                for j, layer_id in enumerate(student_model.pruning_index):
                    # Mask is [B, N+1]. Skip CLS token at index 0.
                    mask = all_masks[j][i].cpu().numpy()[1:1 + num_patches]
                    mask = mask.reshape(n_h, n_w)

                    pruned_img = img.copy()
                    for h in range(n_h):
                        for w in range(n_w):
                            if mask[h, w] == 0:
                                y0, y1 = h * ph, (h + 1) * ph
                                x0, x1 = w * pw, (w + 1) * pw
                                pruned_img[y0:y1, x0:x1, :] = pruned_color

                    keep_ratio = mask.mean()
                    axes[j + 1].imshow(pruned_img)
                    axes[j + 1].set_title(f"L{layer_id} | keep={keep_ratio:.2f}")
                    axes[j + 1].axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(save_dir, f"pruning_vis_{images_done}.png"))
                plt.close()
                images_done += 1
    
    print(green(f"Saved {num_images} pruning visualizations to {save_dir}"))

# ----------------------------------------- Main Execution -----------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="imagenet", choices=['cifar10', 'imagenet'], help='Choose the dataset')
    parser.add_argument('--test_teacher', action='store_true', default=True, help='Flag to test the teacher model')
    parser.add_argument('--test_student', action='store_true', default=True, help='Flag to test the student model')
    parser.add_argument('--teacher_checkpoint', type=str, default=None, help='Override teacher checkpoint path')
    parser.add_argument('--student_checkpoint', type=str, default=None, help='Override student checkpoint path')
    parser.add_argument('--visualize', action='store_true', default=True, help='Generate pruning visualization images')
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--n_heads', type=int, default=None)
    parser.add_argument('--device', type=str, default=None, choices = ["cpu", "cuda", "mps"], help='Choose your device')
    parser.add_argument('--num_images', type=int, default=None)
    args = parser.parse_args()
    
    if args.device :
        device = args.device
        print(flash(bold(f"Using device: {device}")))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(bold(f"Using device: {device}"))

    if args.dataset == "cifar10":
        from data.load.load_data import load_CIFAR
        from configs.train_cifar10 import * 
        base_dir = "cifar10"
        class_names = [str(i) for i in range(10)] 
        print(blue(f"Loading {args.dataset} data..."))
        _, _, test_loader = load_CIFAR(CIFAR=10) 
    else:
        from data.load.imagenet_loader import load_imagenet1k
        from configs.train_imagenet1k import * 
        base_dir = "imagenet"
        class_names = None # Will auto-generate Top K
        print(blue(f"Loading {args.dataset} data..."))
        _, _, test_loader = load_imagenet1k()

    for param in ['d_model', 'n_layers', 'batch_size', 'n_heads']:
        val = getattr(args, param)
        if val is not None:
            globals()[param] = val

    results_dir = f"logs/{base_dir}/evaluation_results"
    os.makedirs(results_dir, exist_ok=True)
    pruning_vis_dir = f"logs/{base_dir}/pruning_visualizations"
    os.makedirs(pruning_vis_dir, exist_ok=True)
    
    default_teacher_ckpt = f"checkpoints/{base_dir}/teacher_checkpoint_best.pth"
    default_student_ckpt = f"checkpoints/{base_dir}/student_best.pth"
    
    t_ckpt_path = args.teacher_checkpoint if (args.teacher_checkpoint and os.path.exists(args.teacher_checkpoint)) else default_teacher_ckpt
    s_ckpt_path = args.student_checkpoint if (args.student_checkpoint and os.path.exists(args.student_checkpoint)) else default_student_ckpt

    t_acc, s_acc, t_speed, s_speed = 0.0, 0.0, 0.0, 0.0
    t_preds, t_labels, s_preds, s_labels = [], [], [], []

    if args.test_teacher:
        print("\nLoading Teacher...")
        teacher = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
        
        if not os.path.exists(t_ckpt_path):
            raise FileNotFoundError(red(f"Teacher checkpoint missing: {t_ckpt_path}"))
            
        ckpt = torch.load(t_ckpt_path, map_location=device)
        teacher.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        
        t_acc, _, t_speed, t_preds, t_labels = evaluate_model(teacher, test_loader, device, "Teacher")

    if args.test_student:
        print("\nLoading Student...")
        student = DynamicVisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index, rho).to(device)
        
        if not os.path.exists(s_ckpt_path):
            raise FileNotFoundError(red(f"Student checkpoint missing: {s_ckpt_path}"))
            
        ckpt = torch.load(s_ckpt_path, map_location=device)
        student.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        
        s_acc, _, s_speed, s_preds, s_labels = evaluate_model(student, test_loader, device, "Student")
        
        if args.visualize:
            if args.num_images:
                print(f"visualize the student's prunning of {args.num_images} images" )
                visualize_pruning_on_images(student, test_loader, device, pruning_vis_dir, args.num_images)
            else:
                print("visualize the student's prunning of 8 images")
                visualize_pruning_on_images(student, test_loader, device, pruning_vis_dir)

    if args.test_teacher and args.test_student:
        print(cyan("\nGenerating Comparison Graphs..."))
        
        # Determine Top K for Confusion Matrix (10 for CIFAR, 20 for ImageNet)
        top_k = 10 if args.dataset == "cifar10" else 20
        
        plot_confusion_matrices(
            confusion_matrix(t_labels, t_preds),
            confusion_matrix(s_labels, s_preds),
            class_names, results_dir, top_k=top_k
        )
        
        plot_performance_comparison(t_acc, s_acc, t_speed, s_speed, results_dir)
        
        diff_speed = ((s_speed - t_speed) / t_speed) * 100
        print(bold(f"\n Student Speed-Up: {diff_speed:.2f}%"))
        print(bold(f" Accuracy Drop: {t_acc - s_acc:.2f}%"))
