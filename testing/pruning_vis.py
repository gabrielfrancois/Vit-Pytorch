import os
import time
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.vision_transformer import VisionTransformer
from models.dynamicViT import DynamicVisionTransformer
from data.imagenet_loader import load_imagenet1k
from helper_function.print import *
from configs.train_imagenet1k import *

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(bold(f"Using device: {device}"))

checkpoint_dir = "checkpoints/imagenet1K"
teacher_path = f"{checkpoint_dir}/teacher_checkpoint_best.pth"
student_path = f"{checkpoint_dir}/student_best.pth"

results_dir = "testing/log/imagenet1k/Evaluation_Graphs_ImageNet"
pruning_vis_dir = "testing/log/imagenet1k/Pruning_Images_ImageNet"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(pruning_vis_dir, exist_ok=True)


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
def evaluate_model(model, loader, device, model_name="Model"):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    correct = 0
    total = 0

    start_time = time.time()

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Testing {model_name}"):
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs

            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)

            _, preds = torch.max(logits, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    elapsed = time.time() - start_time
    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    throughput = total / elapsed

    print(orange("-" * 50))
    print(bold(f"{model_name} results"))
    print(bold(f"Accuracy   : {accuracy:.2f}%"))
    print(bold(f"Loss       : {avg_loss:.4f}"))
    print(bold(f"Throughput : {throughput:.2f} img/s"))
    print(orange("-" * 50))

    return accuracy, avg_loss, throughput

# ------------------------------------------------------------------
# Performance plot (simple et utile)
# ------------------------------------------------------------------
def plot_performance(t_acc, s_acc, t_speed, s_speed, save_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(["Teacher", "Student"], [t_acc, s_acc])
    axes[0].set_title("Top-1 Accuracy (%)")

    axes[1].bar(["Teacher", "Student"], [t_speed, s_speed])
    axes[1].set_title("Throughput (img/sec)")

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "imagenet_performance.png"))
    plt.close()

# ------------------------------------------------------------------
# Pruning visualization (ImageNet normalization)
# ------------------------------------------------------------------
mean_imagenet = [0.485, 0.456, 0.406]
std_imagenet = [0.229, 0.224, 0.225]

def visualize_pruning_on_images(
    student_model,
    loader,
    device,
    num_images=8,
    pruning_layers=[4, 7, 10],
    pruned_color=(0.5, 0.5, 0.5)
):
    os.makedirs(pruning_vis_dir, exist_ok=True)
    student_model.eval()
    images_done = 0

    mean = np.array(mean_imagenet)
    std = np.array(std_imagenet)

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

    print(yellow("Loading ImageNet test data..."))
    _, _, test_loader = load_imagenet1k(
        batch_size=64,
        img_size=img_size[0],
        num_workers=4
    )

    print(yellow("Loading Teacher model..."))
    teacher = VisionTransformer(
        d_model, n_classes, img_size, patch_size,
        n_channels, n_heads, n_layers
    ).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))

    print(yellow("Loading Student model..."))
    student = DynamicVisionTransformer(
        d_model, n_classes, img_size, patch_size,
        n_channels, n_heads, n_layers,
        pruning_index=pruning_index
    ).to(device)
    student.load_state_dict(torch.load(student_path, map_location=device))

    t_acc, t_loss, t_speed = evaluate_model(teacher, test_loader, device, "Teacher")
    s_acc, s_loss, s_speed = evaluate_model(student, test_loader, device, "Student")

    plot_performance(t_acc, s_acc, t_speed, s_speed, results_dir)

    speedup = 100 * (s_speed - t_speed) / t_speed
    print(bold(f"Student speed-up: {speedup:.2f}%"))

    print(yellow("Visualizing pruning..."))
    visualize_pruning_on_images(student, test_loader, device)

 