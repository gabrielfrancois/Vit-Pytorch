import argparse
import os
import time

import torch
import torch.amp
import numpy as np
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, List

from helper_function.print import *
from helper_function.MAE_tools import random_masking, patchify
from helper_function.load_model import verbose_load
from src.models.vision_transformer import VisionTransformer
from configs.train_imagenet1k import std_norm_imagenet, mean_norm_imagenet

# ----------------------------------------- Device setup -----------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="imagenet", choices=['cifar10', 'imagenet'])
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to SSL Teacher checkpoint')
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Mask ratio for evaluation')
    parser.add_argument('--threshold', type=float, default=0.10, help='Max pixel variance for Threshold Accuracy')
    parser.add_argument('--num_images', type=int, default=5, help='Images to visualize')
    args = parser.parse_args()
    return args

# ----------------------------------------- Core Evaluation -----------------------------------------

def evaluate_ssl(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    mask_ratio: float,
    threshold: float,
    patch_size: Tuple
) -> Tuple[float, float, float, float]:
    
    model.eval()
    total_mse, total_mae = 0.0, 0.0
    total_psnr, total_threshold_acc = 0.0, 0.0
    total_batches = 0

    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Testing SSL Reconstruction"):
            imgs = imgs.to(device)
            B, C, H, W = imgs.shape
            N = (H // patch_size[0]) * (W // patch_size[1])
            
            bool_masked_pos = random_masking(B, N, mask_ratio, device)
            
            with torch.amp.autocast(device.type):
                pixel_preds = model(imgs, bool_masked_pos=bool_masked_pos)
                target_patches = patchify(imgs, patch_size)

            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1e-6).sqrt()

            mask_expanded = bool_masked_pos.unsqueeze(-1).expand_as(pixel_preds)
            
            preds_masked = pixel_preds[mask_expanded] # Where the model was blind
            targets_masked = target_patches[mask_expanded]
            
            # Loss
            mse = F.mse_loss(preds_masked, targets_masked).item()
            mae = F.l1_loss(preds_masked, targets_masked).item()
            psnr = 10 * np.log10((2.6 ** 2) / (mse + 1e-8))

            abs_diff = torch.abs(preds_masked - targets_masked)
            good_pixels = (abs_diff < threshold).float().mean().item() * 100

            total_mse += mse
            total_mae += mae
            total_psnr += psnr
            total_threshold_acc += good_pixels
            total_batches += 1

    return total_mse/total_batches, total_mae/total_batches, total_psnr/total_batches, total_threshold_acc/total_batches

def visualize_reconstruction(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    save_dir: str,
    mask_ratio: float,
    num_images: int = 5
) -> None:
    print("\nGenerating Reconstruction Visualizations...")
    model.eval()
    images_done = 0

    # ImageNet stats for numpy denormalization
    mean_np = np.array(mean_norm_imagenet)
    std_np = np.array(std_norm_imagenet)

    os.makedirs(save_dir, exist_ok=True)

    def custom_unpatchify(x_patches, ph, pw, H, W):
        B = x_patches.shape[0]
        h, w = H // ph, W // pw
        C = 3
        x = x_patches.reshape(B, h, w, C, ph, pw)
        x = x.permute(0, 3, 1, 4, 2, 5).contiguous()
        x = x.reshape(B, C, H, W)
        return x

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            B, C, H, W = imgs.shape
            
            ph, pw = model.patch_size if hasattr(model, 'patch_size') else (16, 16)
            N = (H // ph) * (W // pw)

            bool_masked_pos = random_masking(B, N, mask_ratio, device)
            # Use explicit float mask to avoid PyTorch AMP boolean casting bugs
            mask_float = bool_masked_pos.float().unsqueeze(-1)
            
            with torch.amp.autocast(device.type):
                pixel_preds = model(imgs, bool_masked_pos=bool_masked_pos)
                target_patches = patchify(imgs, (ph, pw))

            patch_mean = target_patches.mean(dim=-1, keepdim=True)
            patch_var = target_patches.var(dim=-1, keepdim=True)
            patch_std = (patch_var + 1e-6).sqrt()
            
            unnormalized_preds = pixel_preds * patch_std + patch_mean
            

            reconstructed_patches = target_patches * (1.0 - mask_float) + unnormalized_preds * mask_float
            masked_input_patches = target_patches * (1.0 - mask_float) 

            reconstructed_imgs = custom_unpatchify(reconstructed_patches, ph, pw, H, W)
            masked_imgs = custom_unpatchify(masked_input_patches, ph, pw, H, W)

            for i in range(B):
                if images_done >= num_images:
                    return

                # Convert to float32 before numpy to avoid float16 rendering artifacts
                orig_img = imgs[i].cpu().float().numpy().transpose(1, 2, 0)
                orig_img = np.clip(orig_img * std_np + mean_np, 0, 1)

                recon_img = reconstructed_imgs[i].cpu().float().numpy().transpose(1, 2, 0)
                recon_img = np.clip(recon_img * std_np + mean_np, 0, 1)

                mask_img = masked_imgs[i].cpu().float().numpy().transpose(1, 2, 0)
                mask_img = np.clip(mask_img * std_np + mean_np, 0, 1)

                h_grid, w_grid = H // ph, W // pw
                mask_2d = bool_masked_pos[i].reshape(h_grid, w_grid).cpu().numpy()
                mask_2d_upscaled = np.repeat(np.repeat(mask_2d, ph, axis=0), pw, axis=1)
                mask_img[mask_2d_upscaled] = 0.5 

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(orig_img, interpolation='nearest')
                axes[0].set_title("Original Ground Truth")
                axes[0].axis("off")

                axes[1].imshow(mask_img, interpolation='nearest')
                axes[1].set_title(f"Masked Input ({mask_ratio*100:.1f}% hidden)")
                axes[1].axis("off")

                axes[2].imshow(recon_img, interpolation='nearest')
                axes[2].set_title("Model Reconstruction")
                axes[2].axis("off")

                plt.tight_layout(pad=2.0)
                plt.savefig(os.path.join(save_dir, f"ssl_reconstruction_{images_done}.png"), bbox_inches='tight')
                plt.close()
                images_done += 1
                
    print(f"Saved {num_images} reconstruction visualizations to {save_dir}")

# ----------------------------------------- Main Execution -----------------------------------------

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(bold(f"Using device: {device}"))

    if args.dataset == "cifar10":
        from data.load.load_data import load_CIFAR
        from configs.train_cifar10 import * 
        base_dir = "cifar10"
        _, _, test_loader = load_CIFAR(CIFAR=10) 
    else:
        from data.load.imagenet_loader import load_imagenet1k
        from configs.train_imagenet1k import * 
        base_dir = "imagenet"
        _, _, test_loader = load_imagenet1k()

    vis_dir = f"logs/{base_dir}/ssl_teacher/visualizations"
    default_ckpt = f"checkpoints/{base_dir}/ssl_teacher/ssl_teacher_best.pth"
    ckpt_path = args.checkpoint if args.checkpoint else default_ckpt
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(red(f"SSL Teacher checkpoint not found: {ckpt_path}"))

    print(blue("\nLoading SSL Teacher..."))
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get('model_state_dict', ckpt)
    hparams = ckpt["hyperparameters"]
    patch_size = hparams["patch_size"]
    print(f'd_model : {hparams["d_model"]}')
    print(f'patch_size : {patch_size}')
    model = VisionTransformer(**hparams).to(device)
    verbose_load(model, state_dict)

    mse, mae, psnr, thresh_acc = evaluate_ssl(model, test_loader, device, args.mask_ratio, args.threshold, patch_size)
    
    print(blue("-" * 40))
    print(blue("SSL Reconstruction Metrics (Hidden Patches Only):"))
    print(bold(f"  MSE Loss:        {mse:.4f}"))
    print(bold(f"  MAE Loss:        {mae:.4f}"))
    print(bold(f"  PSNR:            {psnr:.2f} dB"))
    print(bold(f"  Threshold Acc:   {thresh_acc:.2f}% (tol < {args.threshold})"))
    print(blue("-" * 40))

    visualize_reconstruction(model, test_loader, device, vis_dir, args.mask_ratio, args.num_images)