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
from data.load_data import load_imagenet1k
from configs.train_imagenet1k import *
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
def plot_confusion_matrices(teacher_cm, student_cm, class_names=None, save_dir, top_k=10):
    # Si pas de noms de classes, créer par défaut
    if class_names is None:
        num_classes = teacher_cm.shape[0]
        class_names = [f"Class {i}" for i in range(num_classes)]
    # 1. Calculer la somme par classe pour déterminer les plus fréquentes
    freq = np.sum(teacher_cm, axis=1)  # ou axis=0 selon ton besoin
    top_indices = np.argsort(freq)[-top_k:][::-1]  # top_k indices triés décroissant

    # 2. Sous-échantillonner les matrices et noms de classes
    teacher_cm_top = teacher_cm[np.ix_(top_indices, top_indices)]
    student_cm_top = student_cm[np.ix_(top_indices, top_indices)]
    class_names_top = [class_names[i] for i in top_indices]

    # 3. Plot
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

def plot_per_class_accuracy(t_cm, s_cm, class_names=None, save_dir):
    # Si pas de noms de classes, créer par défaut
    if class_names is None:
        num_classes = teacher_cm.shape[0]
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

# ------------------------------------------------------------------
# Pruning Visualization
# ------------------------------------------------------------------



def visualize_pruning_on_images(
    student_model,
    loader,
    device,
    num_images=8,
    pruning_layers=pruning_index,
    pruned_color=(0.5, 0.5, 0.5),
    mean=mean_norm_imagenet,
    std=std_norm_imagenet,
):
    os.makedirs(pruning_vis_dir, exist_ok=True)
    student_model.eval()
    images_done = 0

    mean = np.array(mean)
    std = np.array(std)

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

                fig, axes = plt.subplots(1, len(pruning_layers) + 1,
                                         figsize=(3 * (len(pruning_layers) + 1), 3))
                axes[0].imshow(img)
                axes[0].set_title("Original")
                axes[0].axis("off")

                for j, layer_id in enumerate(pruning_layers):
                    real_idx = student_model.pruning_index.index(layer_id)
                    mask = all_masks[real_idx][i].cpu().numpy()[1:1 + num_patches]
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
                plt.savefig(os.path.join(pruning_vis_dir, f"img_{images_done}.png"))
                plt.close()
                images_done += 1


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    class_names = None

    print(yellow("Loading Test Data..."))
    _, _, test_loader = load_imagenet1k()

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
