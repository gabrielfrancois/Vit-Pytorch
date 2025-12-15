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
from models.dynamicViT import DynamicVisionTransformer
from data.load_data import load_CIFAR
from configs.train_cifar10 import *
from helper_function.print import *

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
device = torch.device("cpu")
print(bold(f"Using device: {device}"))

checkpoint_dir = "checkpoints"
teacher_path = f"{checkpoint_dir}/teacher_checkpoint_best.pth"
student_path = f"{checkpoint_dir}/student_best.pth"

results_dir = "testing/log/Evaluation_Graphs_Test"
pruning_vis_dir = "testing/log/Pruning_Images"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(pruning_vis_dir, exist_ok=True)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate_model(model, loader, device, model_name="Model"):
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
def plot_confusion_matrices(teacher_cm, student_cm, class_names, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    sns.heatmap(teacher_cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title("Teacher Confusion Matrix")
    sns.heatmap(student_cm, annot=True, fmt="d", cmap="Oranges",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title("Student (DynamicViT) Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_confusion_matrices.png"))
    plt.close()

def plot_performance_comparison(t_acc, s_acc, t_speed, s_speed, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    models = ["Teacher", "Student"]
    axes[0].bar(models, [t_acc, s_acc])
    axes[0].set_title("Accuracy (%)")
    axes[1].bar(models, [t_speed, s_speed])
    axes[1].set_title("Throughput (img/sec)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_performance.png"))
    plt.close()

def plot_per_class_accuracy(t_cm, s_cm, class_names, save_dir):
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

# ------------------------------------------------------------------
# Pruning Visualization
# ------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import os
import torch
mean_cifar = [0.4914, 0.4822, 0.4465]
std_cifar  = [0.2023, 0.1994, 0.2010]
def visualize_pruning_on_images(student_model, loader, device, pruning_layers=[4, 7, 10],
                                num_images=10, pruned_color=(0.5, 0.5, 0.5),
                                mean=mean_cifar, std=std_cifar, pruning_vis_dir="./pruning_vis"):
    """
    Visualise les effets du pruning sur les images. 
    Images normalisées avant passage dans le modèle, puis dénormalisées pour affichage.
    
    Args:
        student_model : modèle patch-based renvoyant les masques de pruning
        loader : DataLoader
        device : "cuda" ou "cpu"
        pruning_layers : indices des couches à visualiser
        num_images : nombre d’images à visualiser
        pruned_color : couleur des patches supprimés
        mean : liste/np.array des moyennes par canal pour la normalisation
        std : liste/np.array des écarts-types par canal pour la normalisation
        pruning_vis_dir : dossier de sauvegarde des images
    """
    os.makedirs(pruning_vis_dir, exist_ok=True)
    student_model.eval()
    pruned_layers = list(student_model.pruning_index)
    images_done = 0

    mean = np.array(mean) if mean is not None else np.zeros(3)
    std = np.array(std) if std is not None else np.ones(3)

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            # Normalisation si pas déjà faite
            mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32).view(1, -1, 1, 1)
            std_tensor = torch.tensor(std, device=device, dtype=torch.float32).view(1, -1, 1, 1)
            imgs_norm = (imgs - mean_tensor) / std_tensor


            B, C, H_img, W_img = imgs_norm.shape
            ph, pw = student_model.patch_size
            n_h, n_w = H_img // ph, W_img // pw
            num_patches = n_h * n_w

            outputs = student_model(imgs_norm)
            if not isinstance(outputs, tuple) or len(outputs) < 3:
                print("Student model did not return masks, aborting pruning viz.")
                return
            _, _, all_masks, _ = outputs

            for i in range(B):
                if images_done >= num_images:
                    return

                # Dénormalisation pour affichage
                img_denorm = imgs_norm[i].cpu().numpy().transpose(1, 2, 0) * std + mean
                img_denorm = np.clip(img_denorm, 0, 1)

                fig, axes = plt.subplots(1, len(pruning_layers) + 1,
                                         figsize=(3 * (len(pruning_layers) + 1), 3))

                axes[0].imshow(img_denorm)
                axes[0].set_title("Original")
                axes[0].axis("off")

                for j, layer_id in enumerate(pruning_layers):
                    if layer_id not in pruned_layers:
                        axes[j + 1].axis("off")
                        continue
                    real_idx = pruned_layers.index(layer_id)
                    mask = all_masks[real_idx][i].cpu().numpy()[1:1 + num_patches]
                    mask_2d = mask.reshape(n_h, n_w)
                    pruned_img = img_denorm.copy()

                    for h in range(n_h):
                        for w in range(n_w):
                            if mask_2d[h, w] == 0:
                                y0, y1 = h * ph, (h + 1) * ph
                                x0, x1 = w * pw, (w + 1) * pw
                                pruned_img[y0:y1, x0:x1, :] = np.array(pruned_color)

                    keep_ratio = float(mask_2d.mean())
                    axes[j + 1].imshow(np.clip(pruned_img, 0, 1))
                    axes[j + 1].set_title(f"Layer {layer_id}\nKeep: {keep_ratio:.2f}")
                    axes[j + 1].axis("off")

                plt.tight_layout()
                plt.savefig(os.path.join(pruning_vis_dir, f"image_{images_done}.png"))
                plt.close()
                images_done += 1


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer',
                   'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

    print(yellow("Loading Test Data..."))
    _, _, test_loader = load_CIFAR("/home/onyxia/work/Vit-Pytorch/data", CIFAR=10)

    print(yellow("Loading Teacher Model..."))
    teacher = VisionTransformer(
        d_model, n_classes, img_size, patch_size,
        n_channels, n_heads, n_layers
    ).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))

    print(yellow("Loading Student Model..."))
    student = DynamicVisionTransformer(
        d_model, n_classes, img_size, patch_size,
        n_channels, n_heads, n_layers,
        pruning_index=pruning_index
    ).to(device)
    student.load_state_dict(torch.load(student_path, map_location=device))

    # Evaluation
    t_acc, t_loss, t_speed, t_preds, t_labels = evaluate_model(
        teacher, test_loader, device, "Teacher"
    )
    s_acc, s_loss, s_speed, s_preds, s_labels = evaluate_model(
        student, test_loader, device, "Student"
    )

    print(yellow("Generating graphs..."))
    plot_confusion_matrices(
        confusion_matrix(t_labels, t_preds),
        confusion_matrix(s_labels, s_preds),
        class_names, results_dir
    )
    plot_performance_comparison(
        t_acc, s_acc, t_speed, s_speed, results_dir
    )
    plot_per_class_accuracy(
        confusion_matrix(t_labels, t_preds),
        confusion_matrix(s_labels, s_preds),
        class_names, results_dir
    )

    diff_speed = ((s_speed - t_speed) / t_speed) * 100
    print(bold(f"Student speed-up: {diff_speed:.2f}%"))

    print(yellow("Visualizing pruning..."))
    visualize_pruning_on_images(student, test_loader, device)
    print(green("Done."))
