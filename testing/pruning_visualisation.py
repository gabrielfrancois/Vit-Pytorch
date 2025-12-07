import logging
from typing import List, Tuple, Dict

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

from data.imagenet_loader import load_imagenet1k
from models.dynamicViT import DynamicVisionTransformer


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_dynamicvit_collect_masks(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Run DynamicViT once and return masks for a single image.
    Returns:
        Tensor [L, T] where L = number of pruning stages, T = tokens kept (binary mask).
    """
    model.to(device)
    images = images.to(device)

    with torch.no_grad():
        _, _, all_masks, _ = model(images)

    # On ne garde que l'image index 0
    masks = torch.stack([m[0] for m in all_masks], dim=0)  # [L, T]
    return masks.cpu()


def mask_to_image_visual(
    img: torch.Tensor,
    mask: torch.Tensor,
    patch_size: int = 16,
    grey_factor: float = 0.25
) -> torch.Tensor:
    """
    Apply a pruning mask visually:
    - Tokens 1 restent normaux
    - Tokens 0 deviennent gris

    img: [3,H,W]
    mask: [T]
    """
    C, H, W = img.shape
    num_patches = (H // patch_size) * (W // patch_size)

    assert mask.numel() == num_patches, "Mask size mismatch"

    img = img.clone()

    # reshape en grille
    mask_grid = mask.view(H // patch_size, W // patch_size)

    for i in range(H // patch_size):
        for j in range(W // patch_size):
            if mask_grid[i, j] == 0:
                y0, y1 = i * patch_size, (i + 1) * patch_size
                x0, x1 = j * patch_size, (j + 1) * patch_size
                img[:, y0:y1, x0:x1] *= grey_factor

    return img


def visualize_pruning_epochs(
    original_img: torch.Tensor,
    masks_per_epoch: Dict[int, torch.Tensor],
    patch_size: int = 16,
    save_path: str = "pruning_evolution_epochs.png"
):
    """
    original_img: [C,H,W]
    masks_per_epoch: {epoch: [L,T]}
    """
    epochs = sorted(masks_per_epoch.keys())
    num_rows = 1 + len(epochs)
    num_cols = masks_per_epoch[epochs[0]].shape[0] + 1

    plt.figure(figsize=(3*num_cols, 3*num_rows))

    # Ligne 0 : image originale seule
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(original_img.permute(1,2,0).clamp(0,1))
    plt.title("Original")
    plt.axis("off")

    # Visualisation pour chaque epoch
    for r, epoch in enumerate(epochs):
        masks = masks_per_epoch[epoch]  # [L,T]
        for c in range(masks.shape[0]):
            vis = mask_to_image_visual(original_img, masks[c], patch_size)

            idx = r * num_cols + (c + 2)  # +2 car original prend col1 de ligne 1
            plt.subplot(num_rows, num_cols, idx)
            plt.imshow(vis.permute(1,2,0).clamp(0,1))
            plt.title(f"Epoch {epoch}, prune {c}")
            plt.axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    logger.info(f"Saved full pruning visualization: {save_path}")
    plt.close()


if __name__ == "__main__":
    logger.info("Loading ImageNet samples...")
    _, _, test_loader = load_imagenet1k(batch_size=32, img_size=128, max_items_val=50)

    imgs, labels = next(iter(test_loader))
    img = imgs[0]  # On prend juste la 1ère image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    target_epochs = [1, 5, 10, 15, 20]
    masks_per_epoch = {}

    for ep in target_epochs:
        ckpt_path = f"checkpoints/dynamicvit_epoch_{ep}.pth"
        logger.info(f"Loading checkpoint {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device)

        model = DynamicVisionTransformer(
            d_model=192,
            n_classes=1000,
            img_size=(128, 128),
            patch_size=(16, 16),
            n_channels=3,
            n_heads=3,
            n_layers=12,
            pruning_index=[2,5,8],
            base_keep_rate=0.7,
        )
        model.load_state_dict(ckpt)

        masks = run_dynamicvit_collect_masks(model, img.unsqueeze(0), device)
        masks_per_epoch[ep] = masks

    visualize_pruning_epochs(img, masks_per_epoch)
