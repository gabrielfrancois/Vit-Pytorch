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


# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
device = torch.device("cpu")
print(bold(f"Using device: {device}"))

checkpoint_dir = "checkpoints/imagenet1K"
teacher_path = f"{checkpoint_dir}/teacher_checkpoint_best.pth"
student_path = f"{checkpoint_dir}/student_best.pth"

results_dir = "testing/log/Imagenet/best/Evaluation_Graphs_Test"
pruning_vis_dir = "testing/log/Imagenet/best/Pruning_Images"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(pruning_vis_dir, exist_ok=True)

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate_model(
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
    save_dir: str
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
    models = ["Teacher", "Student"]
    axes[0].bar(models, [t_acc, s_acc])
    axes[0].set_title("Accuracy (%)")
    axes[1].bar(models, [t_speed, s_speed])
    axes[1].set_title("Throughput (img/sec)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "compare_performance.png"))
    plt.close()

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

# ------------------------------------------------------------------
# Pruning Visualization
# ------------------------------------------------------------------



def visualize_pruning_on_images(
    student_model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_images: int = 8,
    pruning_layers: List[int] = pruning_index,
    pruned_color: Tuple[float, float, float] = (0.5, 0.5, 0.5),
    mean: List[float] = mean_norm_imagenet,
    std: List[float] = std_norm_imagenet,
) -> None:
    """
    Visualize pruning masks applied by the student model on input images.

    Args:
        student_model: Student DynamicViT model.
        loader: DataLoader providing the images.
        device: Device to run the visualization on.
        num_images: Maximum number of images to visualize.
        pruning_layers: List of layer indices to visualize pruning.
        pruned_color: RGB color used to mark pruned patches.
        mean: Mean values for image de-normalization.
        std: Standard deviation values for image de-normalization.
    """

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
        pruning_index=pruning_index,rho = 0.709
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
