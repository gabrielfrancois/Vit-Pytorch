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

# -------------------- FLOPs Calculation --------------------

def compute_model_flops(model: nn.Module, img_size: Tuple[int, int], batch_size: int = 1):
    input_shape = (1, n_channels, img_size[0], img_size[1])
    flops, macs, params = calflops.calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=False,
        output_precision=4
    )

    flops_g = flops / 1e9
    macs_g = macs / 1e9
    params_m = params / 1e6

    print(bold(f"Model Parameters: {params_m:.2f} M"))
    print(bold(f"FLOPs per image: {flops_g:.2f} GFLOPs"))
    print(bold(f"MACs per image: {macs_g:.2f} GMacs"))

    flops_batch = flops * batch_size / 1e9
    print(bold(f"FLOPs per batch ({batch_size} imgs) forward: {flops_batch:.2f} GFLOPs"))
    print(bold(f"FLOPs per batch forward+backward: {flops_batch*3:.2f} GFLOPs"))
    print("-"*40)
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

