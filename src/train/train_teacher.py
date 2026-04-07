import argparse
import os
import time
import torch
from torch import nn
import torch.amp
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Tuple, List
from torch.utils.data import DataLoader
import numpy as np

from helper_function.print import *
from helper_function.layer_wise import increasing_llrd
from helper_function.load_model import verbose_load
from src.models.vision_transformer import VisionTransformer

# ----------------------------------------- Training Functions -----------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: int,
    scaler: torch.amp.GradScaler,
    dinov1: nn.Module,
) -> Tuple[float, float]:
    """
    Train the model for a single epoch.
    This function performs a standard supervised training loop including
    forward pass, loss computation, backward pass, and parameter update.
    It also tracks average loss and classification accuracy over the epoch.
    Args:
        model (nn.Module):
            Model to be trained.
        loader (DataLoader):
            DataLoader providing the training batches.
        optimizer (torch.optim.Optimizer):
            Optimizer used to update model parameters.
        criterion (nn.Module):
            Loss function used for training.
        device (torch.device):
            Device on which the model and data are located.
        epoch_index (int):
            Index of the current epoch (used for logging).
        scaler (torch.amp.GradScaler): 
            Gradient scaler for mixed precision training.
        dinov1: (nn.Module),
            The model used for aply REPA
    Returns:
        Tuple[float, float]:
            - avg_loss: Average training loss over the epoch.
            - accuracy: Training accuracy in percentage.
    """
    use_amp = device.type == "cuda"
    model.train()
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0
    loop = tqdm(loader, desc=f"Training Teacher Epoch {epoch_index+1}")

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Get target representations from DINOv1
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=use_amp):
                # DINOv1 returns (B, N+1, 384). Drop the CLS token and get patches.
                features = dinov1.get_intermediate_layers(imgs, n=1)[0]
                ssl_features = features[:, 1:, :] # (B, N, 384)
        
        with torch.amp.autocast(device.type):
            outputs, teacher_feats, repa_features = model(imgs)
            loss_cls = criterion(outputs, labels)
            assert repa_features.shape == ssl_features.shape, f"REPA shape mismatch: model output {repa_features.shape} vs DINOv1 {ssl_features.shape}"
            sim = F.cosine_similarity(repa_features, ssl_features, dim=-1)
            loss_repa = (1.0 - sim).mean()
            loss = loss_cls + (lambda_repa * loss_repa)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item())

    avg_loss: float = running_loss / len(loader.dataset)
    accuracy: float = 100.0 * correct / total
    return avg_loss, accuracy

def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Validating"
) -> Tuple[float, float, np.ndarray]:
    """
    Evaluate the model for a single epoch on a validation or test set.
    The model is run in evaluation mode with gradients disabled.
    The function computes average loss, classification accuracy,
    and the confusion matrix over the entire dataset.
    Args:
        model (nn.Module):
            Model to be evaluated.
        loader (DataLoader):
            DataLoader providing the evaluation batches.
        criterion (nn.Module):
            Loss function used for evaluation.
        device (torch.device):
            Device on which the model and data are located.
        desc (str, optional):
            Description displayed in the progress bar.
    Returns:
        Tuple[float, float, np.ndarray]:
            - avg_loss: Average evaluation loss.
            - accuracy: Evaluation accuracy in percentage.
            - cm: Confusion matrix over all classes.
    """
    model.eval()
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0
    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        loop = tqdm(loader, desc=desc)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs, _, _ = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss: float = running_loss / len(loader.dataset)
    accuracy: float = 100.0 * correct / total
    cm: np.ndarray = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, cm

# ----------------------------------------- Plot Functions -----------------------------------------

def save_training_plots(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    lrs: List[float],
    confusion_mat,
    save_dir: str
) -> None:
    """
    Generate and save training and evaluation visualizations.

    This function saves:
      - Training and validation loss curves
      - Training and validation accuracy curves
      - Learning rate schedule
      - Final confusion matrix heatmap

    Args:
        train_losses (List[float]):
            Training loss values per epoch.
        val_losses (List[float]):
            Validation loss values per epoch.
        train_accs (List[float]):
            Training accuracy values per epoch.
        val_accs (List[float]):
            Validation accuracy values per epoch.
        lrs (List[float]):
            Learning rate values per epoch.
        confusion_mat (np.ndarray):
            Confusion matrix computed on the test set.
        save_dir (str):
            Directory where all figures will be saved.
    """
    if isinstance(confusion_mat, torch.Tensor):
        confusion_mat = confusion_mat.cpu().numpy()
    print(blue(f"Saving training graphs to {save_dir}..."))

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='tab:blue')
    plt.plot(val_losses, label='Validation Loss', color='tab:orange')
    plt.title('Teacher Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Accuracy', color='tab:green')
    plt.plot(val_accs, label='Validation Accuracy', color='tab:red')
    plt.title('Teacher Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, label='Learning Rate', color='purple')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "lr_curve.png"))
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('Final Test Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

# ----------------------------------------- run Functions -----------------------------------------
def run_training(args, device, train_loader, val_loader, test_loader, checkpoint_dir, graph_dir, writer):
    """Encapsulates the model initialization, resume logic, and main training loop."""
    start_time = time.time()
    print(blue("Initializing teacher ViT..."))

    use_amp = device.type == "cuda"
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': [], 'lrs': []}
    best_val_acc = 0.0
    start_epoch = 0
    
    teacher = VisionTransformer(
        d_model=d_model, n_classes=n_classes, img_size=img_size, 
        patch_size=patch_size, n_channels=n_channels, n_heads=n_heads, n_layers=n_layers
    ).to(device)
    
    checkpoint=None
    is_ssl_checkpoint = False
    if args.resume_from is not None and os.path.exists(args.resume_from):
        print(blue(f"Loading checkpoint from {args.resume_from}..."))
        checkpoint = torch.load(args.resume_from, map_location=device)
        
        is_ssl_checkpoint = 'best_val_acc' not in checkpoint
        
        hparams = checkpoint['hyperparameters']
        teacher = VisionTransformer(**hparams).to(device) # Ensure we keep the same hparams
        
        state_dict = checkpoint['model_state_dict']
        verbose_load(teacher, state_dict)
        
        if is_ssl_checkpoint:
            print(green("--> SSL Checkpoint detected! Setting up for REPA fine-tuning (Epoch 0, fresh Optimizer)."))
            start_epoch = 0
        else:
            print(green("--> REPA Checkpoint detected! Resuming REPA training."))
            start_epoch = checkpoint.get('epoch', 0)
            best_val_acc = checkpoint.get('best_val_acc', 0.0)
            history = checkpoint.get('history', history)

    teacher = torch.compile(teacher) # Add Just In Time compiler
    
    # Layer-Wise LR 
    param_groups = increasing_llrd(teacher, alpha, layer_decay, num_layers=n_layers)
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
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=use_amp)

    if checkpoint is not None and not is_ssl_checkpoint and 'optimizer_state_dict' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state'])
            scaler.load_state_dict(checkpoint['scaler_state'])
            print(green(f"--> Fully resumed training from epoch {start_epoch} with best val acc: {best_val_acc:.2f}%"))
        except ValueError as e:
            print(orange(f"--> Optimizer mismatch detected. Starting with fresh optimizer. Error: {e}"))

    print(blue("Loading Frozen DINOv1 for REPA..."))
    assert patch_size == (16, 16), \
    f"DINOv1 uses 16x16 patches but your config has patch_size={patch_size}. " \
    f"Switch to dino_vits8 or adjust patch_size."
    dino_v1 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device) # patch 16x16
    dino_v1.eval()
    for param in dino_v1.parameters():
        param.requires_grad = False

    print(blue("Starting teacher training with REPA..."))
    for epoch in range(start_epoch, epochs):
        first_time_epoch = time.time()
        train_loss, train_acc = train_one_epoch(teacher, train_loader, optimizer, criterion, device, epoch, scaler, dino_v1)
        val_loss, val_acc, _ = validate_one_epoch(teacher, val_loader, criterion, device, desc='Validating Teacher')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lrs'].append(optimizer.param_groups[0]['lr'])

        scheduler.step()# Update Learning Rate

        print(bold(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"))

        writer.add_scalar('Teacher/Loss/train', train_loss, epoch)
        writer.add_scalar('Teacher/Loss/val', val_loss, epoch)
        writer.add_scalar('Teacher/Accuracy/train', train_acc, epoch)
        writer.add_scalar('Teacher/Accuracy/val', val_acc, epoch)
        writer.add_scalar('Teacher/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': teacher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_acc': best_val_acc if val_acc <= best_val_acc else val_acc,
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'history': history,
            'hyperparameters': {
                'd_model': d_model, 'n_classes': n_classes, 'img_size': img_size,
                'patch_size': patch_size, 'n_channels': n_channels, 'n_heads': n_heads, 'n_layers': n_layers
            }
        }

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"{checkpoint_dir}/teacher_checkpoint_best.pth"
            torch.save(checkpoint_dict, save_path)
            print(green(f"--> New Best Teacher Model saved at {save_path}"))
        elif (epoch + 1) % 10 == 0:
            torch.save(checkpoint_dict, f"{checkpoint_dir}/teacher_epoch_{epoch+1}.pth")
        epoch_time = time.time()-first_time_epoch
        print(blue(f'Time for epoch {epoch+1}:'), blue(time.strftime("%H:%M:%S", time.gmtime(epoch_time))))
        
    print(green("\nTraining complete. Loading best model for final testing..."))
    checkpoint = torch.load(f"{checkpoint_dir}/teacher_checkpoint_best.pth", map_location=device)
    teacher.load_state_dict(checkpoint['model_state_dict'])

    test_loss, test_acc, cm = validate_one_epoch(teacher, test_loader, criterion, device, desc='Testing Teacher')
    print(bold(f"Final Test Accuracy: {test_acc:.2f}%"))

    save_training_plots(
        history['train_loss'], history['val_loss'], history['train_acc'], 
        history['val_acc'], history['lrs'], cm, graph_dir
    )
    seconds = time.time() - start_time
    print(blue('Time taken:'), blue(time.strftime("%H:%M:%S", time.gmtime(seconds))))
    writer.close()

# ----------------------------------------- Main -----------------------------------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None, help='Choose the number of epochs')
    parser.add_argument('--d_model', type=int, default=None, help='choose the patch-embedding dimension')
    parser.add_argument('--dataset', type=str, default="imagenet", choices=['cifar10', 'imagenet'], help='Choose the dataset on which you want to train the teacher. Possible choices: ["cifar10", "imagenet"]')
    parser.add_argument('--resume-from', type=str, default=None, help='Choose if you want to resume the training of a previous chekpoint')
    parser.add_argument('--n_layers', type=int, default=None, help='Choose the number of layers')
    parser.add_argument('--batch_size', type=int, default=None, help='Choose the batch size')
    parser.add_argument('--patch_size', type=int, nargs=2, default=None,help='choose the patch-size dimension (ex: 8 8)')
    parser.add_argument('--alpha', type=float, default=None, help='choose the learning rate')
    parser.add_argument('--n_heads', type=int, default=None, help='choose the number of attentions head, BE CAREFUL: n_head MUST be a multiple of d_model!')
    parser.add_argument('--lambda_repa', type=float, default=None, help='Choose the representation factor')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--warmup_epochs', type=int, default=10, help='Number of epochs for learning rate warmup')
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(bold(f"Using device: {device}"))
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

    log_dir = f"./logs/{base_dir}/teacher/"
    checkpoint_dir = f"checkpoints/{base_dir}/teacher"
    graph_dir = f"./logs/{base_dir}/teacher/graphs"

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    param_selected = ['epochs', 'd_model', 'n_layers', 'batch_size', 'patch_size', 'alpha', 'n_heads', 'lambda_repa']
    for param in param_selected: # Set up CLI param if specified...
        value = getattr(args, param)
        if value is not None:
            if param == 'patch_size':
                value = tuple(value)
            globals()[param] = value
    if args.lambda_repa is not None and args.lambda_repa < 0:
        print(orange(f'lambda REPA {args.lambda_repa} should be a positive float, fallback solution :2'))
        args.lambda_repa = 2.0
    print(f"Lambda REPA: {args.lambda_repa if args.lambda_repa is not None else lambda_repa}")
    run_training(
        args=args, device=device,
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
        checkpoint_dir=checkpoint_dir, graph_dir=graph_dir, writer=writer
        )