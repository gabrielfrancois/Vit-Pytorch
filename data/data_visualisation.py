# data/visualisation.py
import os
import numpy as np
import matplotlib.pyplot as plt
import torch

def visualise_sample(loader, output_dir="samples", name="sample"):
    os.makedirs(output_dir, exist_ok=True)
    images, labels = next(iter(loader))
    img = images[0].permute(1, 2, 0).cpu().numpy()
    if img.max() > 1.0:
        img /= 255.0
    plt.imshow(np.clip(img, 0, 1))
    plt.title(f"Label: {labels[0].item()}")
    plt.axis("off")
    out_path = os.path.join(output_dir, f"{name}.png")
    plt.savefig(out_path)
    plt.close()
    print(f" Échantillon sauvegardé dans {out_path}")
