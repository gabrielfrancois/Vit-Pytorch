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

import calflops

device = torch.device("cpu")
print(bold(f"Using device: {device}"))

checkpoint_dir = "checkpoints/imagenet1K"
teacher_path = f"{checkpoint_dir}/teacher_checkpoint_best.pth"
student_path = f"{checkpoint_dir}/student_best.pth"

results_dir = "logs/imagenet/student/graphs/best/Evaluation_Graphs_Test"
pruning_vis_dir = "logs/imagenet/student/pruning/Pruning_Images"
os.makedirs(results_dir, exist_ok=True)
os.makedirs(pruning_vis_dir, exist_ok=True)

# -------------------- FLOPs Calculation --------------------

from typing import Tuple
import torch.nn as nn

def compute_model_flops(
    model: nn.Module,
    img_size: Tuple[int, int],
    batch_size: int = 1
) -> Tuple[float, float, float]:
    """
    Compute and display the computational cost of a vision model.

    This function estimates the number of parameters, FLOPs, and MACs
    for a single forward pass given an input image resolution. It also
    reports scaled FLOPs for a full batch and provides a rough estimate
    of training cost by approximating the backward pass as twice the
    forward cost.

    FLOPs and MACs are computed using a dummy input tensor and rely on
    the underlying FLOPs calculation utility.

    Notes:
        - FLOPs correspond to a single forward pass unless stated otherwise.
        - Results may vary slightly depending on the FLOPs estimation backend.

    Args:
        model (nn.Module):
            PyTorch model to be analyzed.
        img_size (Tuple[int, int]):
            Input image size as (height, width).
        batch_size (int, optional):
            Batch size used to scale FLOPs estimation. Defaults to 1.

    Returns:
        Tuple[float, float, float]:
            - flops: Number of floating-point operations per image.
            - macs: Number of multiply–accumulate operations per image.
            - params: Number of trainable parameters.
    """

    input_shape = (1, n_channels, img_size[0], img_size[1])

    flops, macs, params = calflops.calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=False,
        output_precision=4
    )

    flops_g: float = flops / 1e9
    macs_g: float = macs / 1e9
    params_m: float = params / 1e6

    print(red(f"Model Parameters: {params_m:.2f} M"))
    print(cyan(f"FLOPs per image (forward): {flops_g:.2f} GFLOPs"))
    print(green(f"MACs per image: {macs_g:.2f} GMacs"))

    print("-" * 40)

    return flops, macs, params



if __name__ == "__main__":
    class_names = None

    print(yellow("Loading Test Data..."))
    _, _, test_loader = load_imagenet1k()

    # -------------------- Teacher --------------------
    print(yellow("Loading Teacher Model..."))
    teacher = VisionTransformer(
        d_model, n_classes, img_size, patch_size,
        n_channels, n_heads, n_layers
    ).to(device)
    teacher.load_state_dict(torch.load(teacher_path, map_location=device))

    # -------------------- Student --------------------
    print(yellow("Loading Student Model..."))
    student = DynamicVisionTransformer(
        d_model, n_classes, img_size, patch_size,
        n_channels, n_heads, n_layers,
        pruning_index=pruning_index, rho=0.709
    ).to(device)
    student.load_state_dict(torch.load(student_path, map_location=device))

    print(yellow("Computing FLOPs for Teacher..."))
    compute_model_flops(teacher, img_size=img_size, batch_size=batch_size)

    print(yellow("Computing FLOPs for Student..."))
    compute_model_flops(student, img_size=img_size, batch_size=batch_size)

