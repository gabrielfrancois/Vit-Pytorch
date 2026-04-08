import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
import calflops 

from helper_function.print import *
from helper_function.load_model import verbose_load
from src.models.vision_transformer import VisionTransformer
from src.models.dynamicViT import DynamicVisionTransformer

# ----------------------------------------- Device setup -----------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate FLOPs and MACs for Teacher vs Student")
    parser.add_argument('--dataset', type=str, default="imagenet", choices=['cifar10', 'imagenet'])
    parser.add_argument('--teacher-checkpoint', type=str, default=None, help='Override teacher checkpoint path')
    parser.add_argument('--student-checkpoint', type=str, default=None, help='Override student checkpoint path')
    parser.add_argument('--device', type=str, default=None, choices=["cpu", "cuda", "mps"], help='Choose your device')
    args = parser.parse_args()
    return args

# ----------------------------------------- FLOPs Calculation -----------------------------------------

def compute_model_flops(
    model: nn.Module,
    img_size: Tuple[int, int],
    n_channels: int = 3,
    device: torch.device = torch.device('cpu')
) -> Tuple[float, float, float]:
    """
    Compute and display the computational cost of a vision model.
    """
    model.eval()
    model.to(device)
    
    input_shape = (1, n_channels, img_size[0], img_size[1])

    print(blue(f"Calculating FLOPs for input shape: {input_shape}..."))
    
    flops, macs, params = calflops.calculate_flops(
        model=model,
        input_shape=input_shape,
        output_as_string=False,
        output_precision=4,
        print_results=False 
    )

    flops_g: float = flops / 1e9
    macs_g: float = macs / 1e9
    params_m: float = params / 1e6

    print(blue(f" ↳ Model Parameters:       {params_m:.2f} M"))
    print(bold(f" ↳ FLOPs per img (forward): {flops_g:.2f} GFLOPs"))
    print(bold(f" ↳ MACs per image:         {macs_g:.2f} GMACs"))
    print("-" * 50)

    return flops, macs, params

# ----------------------------------------- Main Execution -----------------------------------------

if __name__ == "__main__":
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(bold(f"Using device: {device}"))

    base_dir = "cifar10" if args.dataset == "cifar10" else "imagenet"
    
    default_teacher_ckpt = f"checkpoints/{base_dir}/teacher_checkpoint_best.pth"
    default_student_ckpt = f"checkpoints/{base_dir}/student_best.pth"

    t_ckpt_path = args.teacher_checkpoint if args.teacher_checkpoint else default_teacher_ckpt
    s_ckpt_path = args.student_checkpoint if args.student_checkpoint else default_student_ckpt

    if not os.path.exists(t_ckpt_path):
        raise FileNotFoundError(red(f"Teacher checkpoint missing: {t_ckpt_path}"))
    if not os.path.exists(s_ckpt_path):
        raise FileNotFoundError(red(f"Student checkpoint missing: {s_ckpt_path}"))

    print(blue("\n--- Loading Teacher Model ---"))
    t_ckpt = torch.load(t_ckpt_path, map_location=device)
    t_hparams = t_ckpt["hyperparameters"]
    teacher = VisionTransformer(**t_hparams).to(device)
    
    t_state_dict = t_ckpt.get('model_state_dict', t_ckpt)
    verbose_load(teacher, t_state_dict)

    print(blue("\n--- Loading Student Model ---"))
    s_ckpt = torch.load(s_ckpt_path, map_location=device)
    s_hparams = s_ckpt["hyperparameters"]
    student = DynamicVisionTransformer(**s_hparams).to(device)

    s_state_dict = s_ckpt.get('model_state_dict', s_ckpt)
    s_state_dict = {k.replace("_orig_mod.", "").replace("transformer_encoderss", "transformer_encoders"): v for k, v in s_state_dict.items()}
    verbose_load(student, s_state_dict)

    print(bold("\n=== Teacher Computational Cost ==="))
    compute_model_flops(teacher, img_size=t_hparams['img_size'], n_channels=t_hparams['n_channels'], device=device)

    print(bold("=== Student Computational Cost ==="))
    compute_model_flops(student, img_size=s_hparams['img_size'], n_channels=s_hparams['n_channels'], device=device)