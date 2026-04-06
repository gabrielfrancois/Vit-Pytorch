import argparse
import os
import time

import torch
import torch.amp
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from typing import List, Tuple, Dict, Any

from helper_function.print import *
from src.models.vision_transformer import VisionTransformer
from src.models.dynamicViT import DynamicVisionTransformer
from .dynamic_loss import DynamicViTLoss

# ----------------------------------------- Test Functions -----------------------------------------

def save_training_plots(
    train_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    ratio_losses: List[float],
    distill_loss: List[float],
    kl_loss: List[float],
    lrs: List[float],
    rho: List[float],
    confusion_mat: Any,
    save_dir: str
) -> None:
    """
    Save all training and validation plots for the student model.
    This includes loss curves, accuracy curves, sparsity-related losses,
    pruning ratio evolution, and the final confusion matrix.
    Args:
        train_losses: Total training loss per epoch.
        train_accs: Training accuracy per epoch.
        val_accs: Validation accuracy per epoch.
        ratio_losses: Sparsity (ratio) loss per epoch.
        distill_loss: Distillation loss per epoch.
        kl_loss: KL divergence loss per epoch.
        lrs: Learning rate per epoch.
        rho: Pruning ratio (rho) per epoch.
        confusion_mat: Confusion matrix computed on the test set.
        save_dir: Directory where plots will be saved.
    """

    print(blue(f"Saving student training graphs to {save_dir}..."))

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Total Train Loss', color='tab:blue')
    plt.title('Student Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "student_loss_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Acc', color='tab:green')
    plt.plot(val_accs, label='Validation Acc', color='tab:red')
    plt.title('Student Accuracy (Hard Pruning on Val)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "student_accuracy_curve.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(ratio_losses, label='Sparsity Loss', color='tab:orange')
    plt.title('Sparsity Convergence (Ratio Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "student_ratio_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(distill_loss, label='Sparsity Loss', color='tab:orange')
    plt.title('Sparsity Convergence (Distill Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('distill Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "student_distill_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(kl_loss, label='Sparsity Loss', color='tab:orange')
    plt.title('Sparsity Convergence (kl Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('kl Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "student_kl_loss.png"))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(rho, label='rho', color='tab:orange')
    plt.title('Pruning ratio evolution (rho)')
    plt.xlabel('Epoch')
    plt.ylabel('rho')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "student_rho.png"))
    plt.close()

    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=False, fmt='d', cmap='Oranges')  
    plt.title('Student Test Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, "student_confusion_matrix.png"))
    plt.close()

def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: int,
    scaler: torch.amp.GradScaler
) -> Tuple[float, float, float, float, float]:
    """
    Train the student model for one epoch using a frozen teacher.
    The training uses mixed precision and a composite loss including
    classification, distillation, KL divergence, and sparsity constraints.
    Args:
        student: Student Vision Transformer model.
        teacher: Pre-trained teacher Vision Transformer (frozen).
        loader: Training data loader.
        optimizer: Optimizer used to update student parameters.
        criterion: DynamicViT loss function.
        device: Device used for training (CPU or CUDA).
        epoch_index: Index of the current epoch.
        scaler: Gradient scaler for mixed precision training.
    Returns:
        avg_loss: Average total loss over the epoch.
        avg_ratio_loss: Average sparsity (ratio) loss.
        avg_distill_loss: Average distillation loss.
        avg_kl_loss: Average KL divergence loss.
        accuracy: Training accuracy in percentage.
    """
    use_amp = device.type == "cuda"
    student.train() 
    running_loss = 0.0
    running_ratio_loss = 0.0
    running_distill_loss = 0.0
    running_kl_loss = 0.0
    correct = 0
    total = 0
    loop = tqdm(loader, desc=f'Training Student Epoch {epoch_index+1}')

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            with torch.amp.autocast(device.type, enabled=use_amp):  # forward in float16
                teacher_logits, teacher_feats, _ = teacher(imgs)
                teacher_logits, teacher_feats = teacher_logits.detach(), teacher_feats.detach()

        optimizer.zero_grad() 
        with torch.amp.autocast(device.type, enabled=use_amp):  # forward in float16
            student_logits, student_feats, all_masks, all_scores = student(imgs)
            loss, metrics = criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                student_feats=student_feats,
                teacher_feats=teacher_feats,
                all_masks=all_masks,
            )
        # Multiplies the loss by a huge number before backprop and then divides them back down before update
        scaler.scale(loss).backward() #x1024

        # Prevents the "NaN explosion" by capping massive gradients
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        running_ratio_loss += metrics['ratio'] * imgs.size(0)
        running_distill_loss += metrics["distill"] * imgs.size(0)
        running_kl_loss += metrics["kl"] * imgs.size(0)

        _, predicted = torch.max(student_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item(), ratio=metrics['ratio'])

    avg_loss = running_loss / len(loader.dataset)
    avg_ratio_loss = running_ratio_loss / len(loader.dataset)
    avg_distill_loss = running_distill_loss / len(loader.dataset)
    avg_kl_loss = running_kl_loss / len(loader.dataset)
    accuracy = 100 * correct / total

    return avg_loss, avg_ratio_loss, avg_distill_loss, avg_kl_loss, accuracy

def validate_one_epoch(
    student: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    desc: str = "Validation"
) -> Tuple[float, Any]:
    """
    Evaluate the student model on a validation or test dataset.
    The model is run in evaluation mode with no gradient computation.
    Only classification accuracy and confusion matrix are computed.
    Args:
        student: Student Vision Transformer model.
        loader: Validation or test data loader.
        device: Device used for evaluation.
        desc: Description shown in the progress bar.
    Returns:
        accuracy: Classification accuracy in percentage.
        cm: Confusion matrix over all classes.
    """
    student.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        loop = tqdm(loader, desc=desc)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            student_logits, _, _, _ = student(imgs) # We only need logits here
            
            _, predicted = torch.max(student_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, cm

# ----------------------------------------- run Functions -----------------------------------------
def run_training(args, device, train_loader, val_loader, test_loader, checkpoint_dir, graph_dir, writer, teacher_checkpoint):
    start_time = time.time()

    print(blue("Initializing teacher..."))
    teacher = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)

    if os.path.exists(teacher_checkpoint):
        print(f"Loading Teacher weights from {teacher_checkpoint}")
        checkpoint = torch.load(teacher_checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        # Load the weights (strict=False ignores the missing/new REPA layers)
        teacher.load_state_dict(clean_state_dict, strict=False)
        print(green(f"Teacher {teacher_checkpoint} successfully loaded."))
    else:
        raise FileNotFoundError(red(f"Teacher checkpoint not found at {teacher_checkpoint}. Run train_teacher.py first!"))

    teacher.eval() 
    for param in teacher.parameters():
        param.requires_grad = False # Freeze weights
    teacher = torch.compile(teacher) # Add JIT compiler

    print(blue("Initializing student..."))
    student = DynamicVisionTransformer(
        d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index, rho
    ).to(device)

    print("Copying backbone weights from teacher to student...")
    teacher_dict = teacher.state_dict()
    student_dict = student.state_dict()
    new_student_dict = {}

    for k, v in teacher_dict.items():
        new_key = k.replace('transformer_encoder', 'transformer_encoders')
        if new_key in student_dict:
            new_student_dict[new_key] = v
    assert len(new_student_dict) > 0, "No teacher weights were transferred to student — check layer naming (transformer_encoder vs transformer_encoders)"
    print(green(f"--> Transferred {len(new_student_dict)}/{len(student_dict)} weight tensors from teacher to student."))
    student.load_state_dict(new_student_dict, strict=False)

    target_ratios = [rho**(i+1) for i in range(len(pruning_index))] 
    criterion = DynamicViTLoss(
        lambda_kl=lambda_kl, lambda_distill=lambda_distill, 
        lambda_ratio=lambda_ratio, target_ratios=target_ratios, 
        lambda_class=lambda_class
    )

    use_amp = device.type == "cuda"
    optimizer = torch.optim.AdamW(student.parameters(), lr=alpha, weight_decay=1e-4)
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

    history = {'train_loss': [], 'ratio_loss': [], "distill_loss": [], "kl_loss": [], 'train_acc': [], 'val_acc': [], 'lrs': [], 'rho': []}
    best_val_acc = 0.0
    start_epoch = 0

    if args.resume_from is not None and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        state_dict = checkpoint['model_state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "") 
            new_state_dict[new_key] = v
        student.load_state_dict(new_state_dict, strict=False)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint['scaler_state'])

        history = checkpoint.get('history', history)
        start_epoch = checkpoint['epoch']
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(green(f"--> Resumed model already trained for {start_epoch} epochs with best val acc: {best_val_acc:.2f}%"))
    student = torch.compile(student) # Add JIT

    for epoch in range(start_epoch, epochs):
        first_time_epoch = time.time()
        train_loss, ratio_loss, distill_loss, kl_loss, train_acc = train_one_epoch(
            student, teacher, train_loader,
            optimizer, criterion,
            device, epoch,
            scaler
        )
        val_acc, _ = validate_one_epoch(student, val_loader, device)
        scheduler.step() 

        history['train_loss'].append(train_loss)
        history['ratio_loss'].append(ratio_loss)
        history['distill_loss'].append(distill_loss)
        history['kl_loss'].append(kl_loss)
        history['train_acc'].append(train_acc)
        history['rho'].append(rho)
        history['val_acc'].append(val_acc)
        history['lrs'].append(optimizer.param_groups[0]['lr'])

        print(bold(f"Epoch {epoch+1}/{epochs} | rho {rho:.3f} | Loss: {train_loss:.4f} | Ratio loss: {ratio_loss:.4f} | Distill: {distill_loss:.4f} | KL: {kl_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"))

        writer.add_scalar('Student/Loss/total', train_loss, epoch)
        writer.add_scalar('Student/Loss/ratio', ratio_loss, epoch)
        writer.add_scalar('Student/Loss/distill', distill_loss, epoch)
        writer.add_scalar('Student/Loss/kl', kl_loss, epoch)
        writer.add_scalar('Student/Accuracy/rho', rho, epoch)
        writer.add_scalar('Student/Accuracy/train', train_acc, epoch)
        writer.add_scalar('Student/Accuracy/val', val_acc, epoch)
        writer.add_scalar('Student/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        checkpoint_dict = {
            'epoch': epoch + 1,
            'model_state_dict': student.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'history': history,
            'best_val_acc': best_val_acc if val_acc <= best_val_acc else val_acc,
            'hyperparameters': {
                'd_model': d_model, 'n_classes': n_classes, 'img_size': img_size,
                'patch_size': patch_size, 'n_channels': n_channels, 'n_heads': n_heads, 
                'n_layers': n_layers, 'pruning_index': pruning_index, 'rho': rho 
            }
        }

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint_dict, f"{checkpoint_dir}/student_best.pth")
            print(green(f"--> New Best Student Saved ({val_acc:.2f}%)"))
        elif (epoch + 1) % 10 == 0:
            torch.save(checkpoint_dict, f"{checkpoint_dir}/student_epoch_{epoch+1}.pth")

        epoch_time = time.time()-first_time_epoch
        print(blue('Time for 1 epoch:'), blue(time.strftime("%H:%M:%S", time.gmtime(epoch_time))))

    print(green("\nTraining complete. Loading best student for final testing..."))
    checkpoint = torch.load(f"{checkpoint_dir}/student_best.pth", map_location=device)
    student.load_state_dict(checkpoint['model_state_dict'])

    test_acc, cm = validate_one_epoch(student, test_loader, device, desc="Testing Student")
    print(blue(f"Final Student Test Accuracy: {test_acc:.2f}%"))

    save_training_plots(
        history['train_loss'], history['train_acc'], history['val_acc'],
        history['ratio_loss'], history["distill_loss"], history['kl_loss'],
        history['lrs'], history['rho'], cm, graph_dir
    )

    seconds = time.time() - start_time
    print(blue('Time Taken:'), blue(time.strftime("%H:%M:%S", time.gmtime(seconds))))
    writer.close()

def rho_schedule(
    epoch: int,
    max_epoch: int,
    rho_init: float = 1.0,
    rho_final: float = 0.7,
    steepness: float = 10.0
) -> float:
    """
    Compute a smooth pruning ratio schedule using a sigmoid function.
    The schedule starts close to rho_init, transitions smoothly around
    mid-training, and converges towards rho_final.
    Args:
        epoch: Current epoch index.
        max_epoch: Total number of epochs.
        rho_init: Initial pruning ratio.
        rho_final: Final pruning ratio.
        steepness: Controls how sharp the transition is.

    Returns:
        rho: Pruning ratio for the given epoch.
    """
    x = epoch/ (max_epoch-1)
    s = 1 / (1 + np.exp(-steepness * (x - 0.5)))
    return rho_init + (rho_final - rho_init) * s

# ----------------------------------------- Main -----------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None, help='Choose the number of epochs')
    parser.add_argument('--resume-from', type=str, default=None, help='Choose if you want to resume the training of a previous student chekpoint')
    parser.add_argument('--d_model', type=int, default=None, help='choose the patch-embedding dimension')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=['cifar10', 'imagenet'], help='Choose the dataset on which you want to train the teacher. Possible choices: ["cifar10", "imagenet"]')
    parser.add_argument('--n_layers', type=int, default=None, help='Choose the number of layers')
    parser.add_argument('--batch_size', type=int, default=None, help='Choose the batch size')
    parser.add_argument('--patch_size',type=int,nargs=2,default=None,help='choose the patch-size dimension (ex: 8 8)')
    parser.add_argument('--alpha', type=float, default=None, help='choose the learning rate')
    parser.add_argument('--n_heads', type=int, default=None, help='choose the number of attentions head, BE CAREFUL: n_head MUST be a multiple of d_model!')
    parser.add_argument('--teacher_checkpoint', type=str, default=None, help='Explicit path to the teacher checkpoint')
    parser.add_argument('--device', type=str, default=None, choices=['cuda', 'mps', 'cpu'])
    parser.add_argument('--run_name', type=str, default="student", help='Subfolder name for this run checkpoints')
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

        print(blue(f"Loading {args.dataset} data..."))
        train_loader, test_loader, val_loader = load_CIFAR(CIFAR=10) 
    else:
        from data.load.imagenet_loader import load_imagenet1k
        from configs.train_imagenet1k import * 
        base_dir = "imagenet"
        print(blue(f"Loading {args.dataset} data..."))
        train_loader, test_loader, val_loader = load_imagenet1k()

    checkpoint_dir = f"checkpoints/{base_dir}/{args.run_name}"
    teacher_checkpoint = f"checkpoints/{base_dir}/teacher_2th_try/teacher_checkpoint_best.pth" if base_dir == "cifar10" else f"checkpoints/{base_dir}/teacher_checkpoint_best.pth"
    if args.teacher_checkpoint and os.path.exists(args.teacher_checkpoint):
        teacher_checkpoint = args.teacher_checkpoint
        print(green(f"Overriding default teacher path with: {teacher_checkpoint}"))
    elif args.teacher_checkpoint:
        raise FileNotFoundError(red(f"Provided teacher checkpoint does not exist: {args.teacher_checkpoint}"))

    log_dir = f"./logs/{base_dir}/student/"
    graph_dir = f"./logs/{base_dir}/student/graphs"

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    param_selected = [
        'epochs', 'd_model',
        'dataset','n_layers',
        'batch_size','patch_size',
        'alpha','n_heads'
        ]
    for param in param_selected: # Set up CLI param if specified...
        value = getattr(args, param)
        if value is not None:
            if param == 'patch_size':
                value = tuple(value)
            globals()[param] = value

    run_training(
        args=args, device=device, 
        train_loader=train_loader, val_loader=val_loader, test_loader=test_loader, 
        checkpoint_dir=checkpoint_dir, graph_dir=graph_dir, writer=writer,
        teacher_checkpoint=teacher_checkpoint
    )