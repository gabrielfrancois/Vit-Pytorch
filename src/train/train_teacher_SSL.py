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
from helper_function.layer_wise import decreasing_llrd
from helper_function.load_model import verbose_load
from helper_function.MAE_tools import random_masking, patchify
from src.models.vision_transformer import VisionTransformer

# ----------------------------------------- Device setup -----------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--d_model', type=int, default=None)
    parser.add_argument('--dataset', type=str, default="imagenet", choices=['cifar10', 'imagenet'])
    parser.add_argument('--resume-from', type=str, default=None)
    parser.add_argument('--n_layers', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--patch_size', type=int, nargs=2, default=None)
    parser.add_argument('--alpha', type=float, default=None, help='Learning rate')
    parser.add_argument('--layer-decay', type=float, default=None, help='choose the layer decay.')
    parser.add_argument('--n_heads', type=int, default=None)
    parser.add_argument('--mask_ratio', type=float, default=0.75, help='Percentage of image to mask out')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of epochs for learning rate warmup')
    args = parser.parse_args()
    return args

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
    device = torch.device(device) if isinstance(device, str) else device
    use_amp = device.type == "cuda"
    model.train()
    running_loss: float = 0.0
    loop = tqdm(loader, desc=f"SSL Train Epoch {epoch_index+1}")

    for imgs, _ in loop: 
        imgs = imgs.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        B, C, H, W = imgs.shape
        N = (H // patch_size[0]) * (W // patch_size[1])
        bool_masked_pos = random_masking(B, N, mask_ratio, device)

        with torch.amp.autocast(device.type, enabled=use_amp):
            pixel_preds = model(imgs, bool_masked_pos=bool_masked_pos)
            target_patches = patchify(imgs, patch_size)
            # Normalise to prevent the model to just predict the mean 
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1e-6).sqrt()
            
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
            mean = target_patches.mean(dim=-1, keepdim=True)
            var = target_patches.var(dim=-1, keepdim=True)
            target_patches = (target_patches - mean) / (var + 1e-6).sqrt()

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
    args_ssl = {
            'd_model': d_model, 'n_classes': n_classes, 'img_size': img_size,
            'patch_size': patch_size, 'n_channels': n_channels, 'n_heads': n_heads, 'n_layers': n_layers
        }

    print(blue("Initializing SSL Teacher ViT..."))
    
    use_amp = device.type == "cuda"
    history = {'train_loss': [], 'val_loss': [], 'lrs': []}
    best_val_loss = float('inf')
    start_epoch = 0
    

    checkpoint = None
    if args.resume_from is not None and os.path.exists(args.resume_from):
        print(blue(f"Loading checkpoint from {args.resume_from}..."))
        checkpoint = torch.load(args.resume_from, map_location=device)
        args_ssl = checkpoint['hyperparameters']
        
        print(blue(f"=== Resuming with parameters: {args_ssl} ==="))
        
    teacher = VisionTransformer(**args_ssl).to(device) 
    
    if checkpoint is not None:
        state_dict = checkpoint['model_state_dict']
        verbose_load(teacher, state_dict) # Ensure we keep the same hparams if resume_from is triggered
        
        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        history = checkpoint.get('history', history)
        
    teacher = torch.compile(teacher) # JIT
    
    # LayeWise --> modify especially the firsts layers!
    normalized_alpha = alpha*batch_size/256
    param_groups = decreasing_llrd(teacher, normalized_alpha, layer_decay, num_layers=args_ssl['n_layers'])
    optimizer = torch.optim.AdamW(param_groups)
    
    # Warmup: start at 1% of the target LR and ramp up linearly over 'warmup_epochs'
    warmup_epochs = min(args.warmup_epochs, epochs - 1)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
    )
    
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=(epochs - warmup_epochs)
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    
    scaler = torch.amp.GradScaler(enabled=use_amp)

    if checkpoint is not None and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            scaler.load_state_dict(checkpoint['scaler_state'])
            print(green(f"--> Resumed SSL model already trained for {start_epoch} epochs with best val loss: {best_val_loss:.4f}"))
        except ValueError as e:
            print(orange(f"--> Could not load optimizer state (likely due to added/removed layers). Starting with fresh optimizer. Error: {e}"))

    print(blue("Starting SSL teacher training..."))
    print(bold(f"learning rate : {alpha} === parametter selected : \n warmup epochs: {args.warmup_epochs} | epochs: {epochs} | patch_size {patch_size} | layer_decay: {layer_decay} | n_layers: {n_layers}"), blue("\n [Start Training]"))
    for epoch in range(start_epoch, epochs):
        first_time_epoch = time.time()

        train_loss = train_one_epoch(teacher, train_loader, optimizer, device, epoch, scaler, mask_ratio)
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
            'hyperparameters': args_ssl
        }
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = f"{checkpoint_dir}/ssl_teacher_best.pth"
            torch.save(checkpoint_dict, save_path)
            print(green(f"--> New Best SSL Teacher Model saved at {save_path}"))
        elif (epoch + 1) % 10 == 0:
            torch.save(checkpoint_dict, f"{checkpoint_dir}/ssl_teacher_epoch_{epoch+1}.pth")

        epoch_time = time.time() - first_time_epoch
        print(blue(f'Time for epoch {epoch+1}:'), blue(time.strftime("%H:%M:%S", time.gmtime(epoch_time))))

    print(green("\nSSL Training complete!"))
    save_training_plots(history['train_loss'], history['val_loss'], history['lrs'], graph_dir)

    seconds = time.time() - start_time
    print(blue('Total Time taken:'), blue(time.strftime("%H:%M:%S", time.gmtime(seconds))))
    writer.close()

# ----------------------------------------- Main -----------------------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(bold(f"Using device: {device}"))
    if args.n_heads is not None and args.d_model is not None:
        assert args.d_model % args.n_heads == 0, "d_model must be divisible by n_heads"
    if args.mask_ratio is None or args.mask_ratio <=0 or args.mask_ratio >=1:
        print(orange(f"mask_ratio {args.mask_ratio} argument should be in (0,1) ! --> default 0.75"))
        args.mask_ratio = 0.75

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
        batch_size = args.batch_size if args.batch_size else batch_size
        train_loader, test_loader, val_loader = load_imagenet1k(batch_size=batch_size)

    log_dir = f"./logs/{base_dir}/ssl_teacher/"
    checkpoint_dir = f"checkpoints/{base_dir}/ssl_teacher"
    graph_dir = f"./logs/{base_dir}/ssl_teacher/graphs"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    param_selected = ['epochs', 'd_model', 'n_layers', 'batch_size', 'patch_size', 'alpha', 'n_heads', 'mask_ratio', 'layer_decay']
    for param in param_selected: 
        value = getattr(args, param)
        if value is not None:
            if param == 'patch_size':
                value = tuple(value)
            globals()[param] = value
    assert len(patch_size) == 2 and all(p > 0 for p in patch_size), f"patch_size must be a pair of positive ints, got {patch_size}"
    run_training(
        args=args, device=device,
        train_loader=train_loader, val_loader=val_loader,
        checkpoint_dir=checkpoint_dir, graph_dir=graph_dir, writer=writer
    )