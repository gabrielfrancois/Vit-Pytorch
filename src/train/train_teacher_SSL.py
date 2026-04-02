import argparse
import os
import time
import torch
from torch import nn
import torch.amp
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Tuple, List
from torch.utils.data import DataLoader
import numpy as np

from helper_function.print import *
from src.models.vision_transformer import VisionTransformer

def random_masking(B: int, N: int, mask_ratio: float, device: torch.device) -> torch.Tensor:
    """
    Generates a random boolean mask. True = Masked (hidden), False = Visible.
    """
    noise = torch.rand(B, N, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    len_keep = int(N * (1 - mask_ratio))
    
    mask = torch.ones(B, N, device=device)
    mask[:, :len_keep] = 0
    
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask.bool()

def patchify(imgs: torch.Tensor, patch_size: Tuple[int, int]) -> torch.Tensor:
    """
    Converts 2D images (B, C, H, W) into a flat sequence of patches (B, N, patch_dim)
    so we can calculate MSE loss against the Transformer's output.
    """
    p_h, p_w = patch_size
    B, C, H, W = imgs.shape
    x = imgs.reshape(B, C, H // p_h, p_h, W // p_w, p_w)
    x = x.permute(0, 2, 4, 1, 3, 5)
    x = x.reshape(B, (H // p_h) * (W // p_w), C * p_h * p_w)
    return x

# ----------------------------------------- Training Functions -----------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch_index: int,
    scaler: torch.amp.GradScaler,
    mask_ratio: float
) -> float:
    """
    Self-Supervised MAE Pre-training loop.
    """
    model.train()
    running_loss: float = 0.0
    loop = tqdm(loader, desc=f"SSL Train Epoch {epoch_index}")

    for imgs, _ in loop: 
        imgs = imgs.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        B, C, H, W = imgs.shape
        N = (H // patch_size[0]) * (W // patch_size[1])
        bool_masked_pos = random_masking(B, N, mask_ratio, device)

        with torch.amp.autocast(device.type):
            pixel_preds = model(imgs, bool_masked_pos=bool_masked_pos)
            target_patches = patchify(imgs, patch_size)
            
            # Compute MSE Loss ONLY on masked patches
            loss = (pixel_preds - target_patches).pow(2)
            loss = loss.mean(dim=-1)
            loss = (loss * bool_masked_pos).sum() / (bool_masked_pos.sum() + 1e-6)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        loop.set_postfix(mse_loss=loss.item())

    return running_loss / len(loader.dataset)

def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    desc: str,
    mask_ratio: float
) -> float:
    """
    Evaluates MSE loss on validation set.
    """
    model.eval()
    running_loss: float = 0.0
    loop = tqdm(loader, desc=desc)

    with torch.no_grad():
        for imgs, _ in loop:
            imgs = imgs.to(device)
            B, C, H, W = imgs.shape
            N = (H // patch_size[0]) * (W // patch_size[1])
            bool_masked_pos = random_masking(B, N, mask_ratio, device)

            pixel_preds = model(imgs, bool_masked_pos=bool_masked_pos)
            target_patches = patchify(imgs, patch_size)

            loss = (pixel_preds - target_patches) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * bool_masked_pos).sum() / (bool_masked_pos.sum() + 1e-6)

            running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)

# ----------------------------------------- Plot Functions -----------------------------------------

def save_training_plots(train_losses: List[float], val_losses: List[float], lrs: List[float], save_dir: str) -> None:
    print(blue(f"Saving SSL training graphs to {save_dir}..."))

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train MSE Loss', color='tab:blue')
    plt.plot(val_losses, label='Validation MSE Loss', color='tab:orange')
    plt.title('SSL Teacher Pre-Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "ssl_loss_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, label='Learning Rate', color='purple')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "ssl_lr_curve.png"))
    plt.close()

# ----------------------------------------- run Functions -----------------------------------------

def run_training(args, device, train_loader, val_loader, checkpoint_dir, graph_dir, writer):
    start_time = time.time()
    print(blue("Initializing SSL Teacher ViT..."))
    
    teacher = VisionTransformer(
        d_model=d_model, n_classes=n_classes, img_size=img_size, 
        patch_size=patch_size, n_channels=n_channels, n_heads=n_heads, n_layers=n_layers
    ).to(device)

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=alpha, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler()

    history = {'train_loss': [], 'val_loss': [], 'lrs': []}
    best_val_loss = float('inf') 
    start_epoch = 0 

    if args.resume_from is not None and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        teacher.load_state_dict(clean_state_dict, strict=False)
        
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            scaler.load_state_dict(checkpoint['scaler_state'])
            history = checkpoint.get('history', history)
            start_epoch = checkpoint['epoch']
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            print(green(f"--> Resumed SSL model already trained for {start_epoch} epochs with best val loss: {best_val_loss:.4f}"))
    
    teacher = torch.compile(teacher)

    print(blue("Starting SSL teacher training..."))
    for epoch in range(start_epoch, epochs):
        first_time_epoch = time.time()
        
        train_loss = train_one_epoch(teacher, train_loader, optimizer, device, epoch, scaler, args.mask_ratio)
        val_loss = validate_one_epoch(teacher, val_loader, device, desc='Validating SSL Teacher', mask_ratio=args.mask_ratio)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lrs'].append(optimizer.param_groups[0]['lr'])

        scheduler.step()
        
        print(bold(f"Epoch {epoch+1}/{epochs} | Train MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f}"))
        
        writer.add_scalar('SSL_Teacher/Loss/train', train_loss, epoch)
        writer.add_scalar('SSL_Teacher/Loss/val', val_loss, epoch)
        writer.add_scalar('SSL_Teacher/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss if val_loss >= best_val_loss else val_loss,
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'history': history,
            'hyperparameters': {
                'd_model': d_model, 'n_classes': n_classes, 'img_size': img_size,
                'patch_size': patch_size, 'n_channels': n_channels, 'n_heads': n_heads, 'n_layers': n_layers
            }
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"{checkpoint_dir}/ssl_teacher_best.pth"
            torch.save(checkpoint_dict, save_path)
            print(green(f"--> New Best SSL Teacher Model saved at {save_path}"))
        elif (epoch + 1) % 10 == 0:
            torch.save(checkpoint_dict, f"{checkpoint_dir}/ssl_teacher_epoch_{epoch+1}.pth")
            
        epoch_time = time.time() - first_time_epoch
        print(blue('Time for 1 epoch:'), blue(time.strftime("%H:%M:%S", time.gmtime(epoch_time))))
        
    print(green("\nSSL Training complete!"))
    save_training_plots(history['train_loss'], history['val_loss'], history['lrs'], graph_dir)
    
    seconds = time.time() - start_time
    print(blue('Total Time taken:'), blue(time.strftime("%H:%M:%S", time.gmtime(seconds))))
    writer.close()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(bold(f"Using device: {device}"))
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--dataset', type=str, default="cifar10", choices=['cifar10', 'imagenet'])
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--patch_size', type=int, nargs=2, default=None)
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--n_heads', type=int, default=None)
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Percentage of image to mask out')
    args = parser.parse_args()

    if args.n_heads is not None and args.d_model is not None:
        assert args.d_model % args.n_heads == 0, "d_model must be divisible by n_heads"

    if args.dataset == "cifar10":
        from data.load.load_data import load_CIFAR
        from configs.train_cifar10 import * 
        base_dir = "cifar10"
        print(blue(f"Loading {args.dataset} Data..."))
        train_loader, test_loader, val_loader = load_CIFAR(CIFAR=10) 
    else:
        from data.load.imagenet_loader import load_imagenet1k
        from configs.train_imagenet1k import * 
        base_dir = "imagenet"
        print(blue(f"Loading {args.dataset} Data...")) 
        train_loader, test_loader, val_loader = load_imagenet1k()
    
    log_dir = f"./logs/{base_dir}/ssl_teacher/"
    checkpoint_dir = f"checkpoints/{base_dir}/ssl_teacher"
    graph_dir = f"./logs/{base_dir}/ssl_teacher/graphs"
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    param_selected = ['epochs', 'd_model', 'n_layers', 'batch_size', 'patch_size', 'alpha', 'n_heads']
    for param in param_selected: 
        value = getattr(args, param)
        if value is not None:
            if param == 'patch_size':
                value = tuple(value)
            globals()[param] = value

    run_training(
        args=args, device=device,
        train_loader=train_loader, val_loader=val_loader,
        checkpoint_dir=checkpoint_dir, graph_dir=graph_dir, writer=writer
    )