import os
import time
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Imports
from models.vision_transformer import VisionTransformer
from models.dynamicViT import DynamicVisionTransformer
from data.load_data import load_CIFAR
from configs.train_cifar10 import *
from helper_function.print import *

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def visualize_pruning_on_images(
    student_model,
    loader,
    device,
    pruning_layers=[4, 7, 10],
    num_images=10,
    alpha=0.6,
    overlay_color=(1.0, 0.0, 0.0)  # rouge = tokens supprimés
):
    """
    Visualize Dynamic ViT token pruning using alpha overlays.

    For each selected image, this function displays:
    - the original (de-normalized) image
    - the same image with pruned patches overlaid (semi-transparent)
      for selected transformer layers.

    Args:
        student_model (nn.Module): Trained Dynamic ViT model.
        loader (DataLoader): DataLoader providing images.
        device (torch.device): CPU or CUDA device.
        pruning_layers (list[int]): Transformer layer indices to visualize.
        num_images (int): Number of images to visualize.
        alpha (float): Transparency factor for pruned patches (0-1).
        overlay_color (tuple): RGB color used for pruned patch overlay.
    """

    student_model.eval()
    pruned_layers = student_model.pruning_index
    images_done = 0

    # Patch / image geometry
    ph, pw = student_model.patch_size
    H, W = student_model.img_size
    n_h, n_w = H // ph, W // pw

    # ImageNet normalization (assumed)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)

            _, _, all_masks, _ = student_model(imgs)

            for i in range(imgs.size(0)):
                if images_done >= num_images:
                    return

                # ---- De-normalize image ----
                img = imgs[i].cpu().numpy().transpose(1, 2, 0)
                img = (img * std) + mean
                img = np.clip(img, 0, 1)

                fig, axes = plt.subplots(
                    1, len(pruning_layers) + 1,
                    figsize=(3 * (len(pruning_layers) + 1), 3)
                )

                axes[0].imshow(img)
                axes[0].set_title("Original")
                axes[0].axis("off")

                for j, layer_id in enumerate(pruning_layers):
                    if layer_id not in pruned_layers:
                        axes[j + 1].axis("off")
                        continue

                    real_idx = pruned_layers.index(layer_id)
                    mask = all_masks[real_idx][i].cpu().numpy()
                    mask_2d = mask[1:].reshape(n_h, n_w)

                    pruned_img = img.copy()

                    for h in range(n_h):
                        for w in range(n_w):
                            if mask_2d[h, w] == 0:
                                y0, y1 = h * ph, (h + 1) * ph
                                x0, x1 = w * pw, (w + 1) * pw

                                pruned_img[y0:y1, x0:x1, :] = (
                                    (1 - alpha) * pruned_img[y0:y1, x0:x1, :]
                                    + alpha * np.array(overlay_color)
                                )

                    keep_ratio = mask_2d.mean()

                    axes[j + 1].imshow(pruned_img)
                    axes[j + 1].set_title(
                        f"Layer {layer_id}\nKeep: {keep_ratio:.2f}"
                    )
                    axes[j + 1].axis("off")

                plt.tight_layout()
                plt.savefig(
                    os.path.join(pruning_vis_dir, f"image_{images_done}.png")
                )
                plt.close()

                images_done += 1
