import matplotlib.pyplot as plt
import numpy as np
from models.dynamicVIT import DynamicVisionTransformer

def visualize_pruning(images, all_masks, patch_size, stage_names=None, idx=0):
    """
    images: batch d'images (B,C,H,W)
    all_masks: list de masques [(B,N), ...] retourné par le forward
    patch_size: tuple (patch_h, patch_w)
    stage_names: liste optionnelle des noms des stages
    idx: index de l'image dans le batch à afficher
    """
    img = images[idx].permute(1,2,0).cpu().numpy()  # H,W,C
    n_stages = len(all_masks)
    n_patches_h = img.shape[0] // patch_size[0]
    n_patches_w = img.shape[1] // patch_size[1]

    plt.figure(figsize=(4*n_stages,4))
    for i, mask in enumerate(all_masks):
        stage_mask = mask[idx,1:].cpu().numpy()  # on ignore CLS token
        stage_mask = stage_mask.reshape(n_patches_h, n_patches_w)

        plt.subplot(1, n_stages, i+1)
        plt.imshow(img)
        # overlay des patches gardés
        plt.imshow(np.kron(1-stage_mask, np.ones(patch_size)), cmap='Reds', alpha=0.5)
        plt.axis('off')
        if stage_names:
            plt.title(stage_names[i])
        else:
            plt.title(f"Stage {i+1}")
    plt.show()
