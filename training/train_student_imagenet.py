# This Python file gathers all the functions / classes needed to handle training, validation or inference from the ViT model
# python -m training.train_student_imagenet
import os
import time
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from helper_function.print import *
from models.vision_transformer import VisionTransformer
from models.dynamicViT_imagenet import DynamicVisionTransformer
from .dynamic_loss_imagenet import DynamicViTLoss
from data.load.imagenet_loader import load_imagenet1k
from configs.train_imagenet1k import * 
import time
import torch.amp
from typing import List, Tuple, Dict, Any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(bold(f"Using device: {device}"))

# Checkpoint paths
checkpoint_dir = "checkpoints/imagenet1K"

checkpoint_path = f"{checkpoint_dir}/student_checkpoint_last.pth"
teacher_checkpoint = f"{checkpoint_dir}/teacher_checkpoint_best.pth"
os.makedirs(checkpoint_dir, exist_ok=True)

# Logging Directories
log_dir = "training/log/Imagenet/Student_ViT_imagenet1k"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

graph_dir = "training/log/Imagenet/Student_ViT_imagenet1k-graphs"
os.makedirs(graph_dir, exist_ok=True)


# Initialize and Load TEACHER 
print(yellow("Initializing Teacher..."))
teacher = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)

if os.path.exists(teacher_checkpoint):
    print(green(f"Loading Teacher weights from {teacher_checkpoint}"))
    teacher.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
else:
    raise FileNotFoundError(red(f"Teacher checkpoint not found at {teacher_checkpoint}. Please run run_teacher.py first!"))

teacher.eval() # Teacher is always in eval mode, already trained (otherwise, it would be the WORSE teacher ever!)
for param in teacher.parameters():
    param.requires_grad = False # Freeze weights


# Initialize student
print(yellow("Initializing Student..."))
student= DynamicVisionTransformer(
    d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index,rho_init
).to(device)

# Load Teacher weights into Student Backbone, the student should start as a copy of the teacher, then learn to prune.
print(blue("Copying backbone weights from Teacher to Student..."))
teacher_dict = teacher.state_dict()
student_dict = student.state_dict()

new_student_dict = {}

for k, v in teacher_dict.items():
    # Map 'transformer_encoder' -> 'transformer_encoders'
    # Indeed, the keys name changed since we use ModuleList instrad of Sequential, thus, we've to make it match.
    new_key = k.replace('transformer_encoder', 'transformer_encoders')
    if new_key in student_dict:
        new_student_dict[new_key] = v
    else:
        # This might happen for predictors or mismatched layers
        pass

# Update student with available matching weights (Backbone + Classifier)
# strict=False because Student has extra 'predictor' layers that Teacher doesn't have (in PredictorLG)
student.load_state_dict(new_student_dict, strict=False) 

# Optimizer & Loss, note that we only optimize the STUDENT
optimizer = torch.optim.AdamW(student.parameters(), lr=alpha, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Plotting Function
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
    
    # 1. Total Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Total Train Loss', color='tab:blue')
    plt.title('Student Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "student_loss_curve.png"))
    plt.close()

    # 2. Accuracy Curve
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

    # 3. Ratio Loss Curve (Sparsity)
    plt.figure(figsize=(10, 6))
    plt.plot(ratio_losses, label='Sparsity Loss', color='tab:orange')
    plt.title('Sparsity Convergence (Ratio Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Ratio Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "student_ratio_loss.png"))
    plt.close()

    # 4. Distill Loss Curve (Sparsity)
    plt.figure(figsize=(10, 6))
    plt.plot(distill_loss, label='Sparsity Loss', color='tab:orange')
    plt.title('Sparsity Convergence (Distill Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('distill Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "student_distill_loss.png"))
    plt.close()

    # 5. kl Loss Curve (Sparsity)
    plt.figure(figsize=(10, 6))
    plt.plot(kl_loss, label='Sparsity Loss', color='tab:orange')
    plt.title('Sparsity Convergence (kl Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('kl Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "student_kl_loss.png"))
    plt.close()
    # 5. Pruning ratio evolution (rho)
    plt.figure(figsize=(10, 6))
    plt.plot(rho, label='rho', color='tab:orange')
    plt.title('Pruning ratio evolution (rho)')
    plt.xlabel('Epoch')
    plt.ylabel('rho')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "student_rho.png"))
    plt.close()

    # 7. Confusion Matrix without annotation
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=False, fmt='d', cmap='Oranges')  
    plt.title('Student Test Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, "student_confusion_matrix.png"))
    plt.close()

# Core Functions
def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: int,
    rho: float,
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
        rho: Current pruning ratio.
        scaler: Gradient scaler for mixed precision training.

    Returns:
        avg_loss: Average total loss over the epoch.
        avg_ratio_loss: Average sparsity (ratio) loss.
        avg_distill_loss: Average distillation loss.
        avg_kl_loss: Average KL divergence loss.
        accuracy: Training accuracy in percentage.
    """
    student.train() 
    # Teacher is already eval/frozen  

    running_loss = 0.0
    running_ratio_loss = 0.0
    running_distill_loss = 0.0
    running_kl_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f'Training Student Epoch {epoch_index}')
    
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        # --- Teacher ---
        with torch.no_grad():
            teacher_logits, teacher_feats = teacher(imgs)
            teacher_logits, teacher_feats = teacher_logits.detach(), teacher_feats.detach()

        # --- Student in mixed precision ---
        optimizer.zero_grad()  # reset grad

        with torch.amp.autocast("cuda"):  # forward in float16
            student_logits, student_feats, all_masks, all_scores = student(imgs)
            loss, metrics = criterion(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                labels=labels,
                student_feats=student_feats,
                teacher_feats=teacher_feats,
                all_masks=all_masks,
            )

        # --- Backward with scaler ---
        scaler.scale(loss).backward()     # scale before backward
        scaler.step(optimizer)           
        scaler.update()                   

        # Metrics
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


    # Normalising the epoch between 0 and 1
    x = epoch/ (max_epoch-1) # the first epoch 0 then we take max_epoch-1
    # Centered Sigmoid à 0.5
    s = 1 / (1 + np.exp(-steepness * (x - 0.5)))
    # Interpolation between rho_init and rho_final
    return rho_init + (rho_final - rho_init) * s

# Main Execution
if __name__ == "__main__":
    # look at the time
    start_time = time.time()
    # Load Data
    print(blue("Loading Data..."))
    data_path = "/home/onyxia/work/Vit-Pytorch/data" 
    train_loader, test_loader, val_loader = load_imagenet1k() 

    print(yellow("Starting Student Training..."))
    
    start_epoch = 0
    # Metric History
    history = {
        'train_loss': [],
        'ratio_loss': [],
        "distill_loss": [],
        "kl_loss": [],
        'train_acc': [],
        'val_acc': [],
        'lrs': [],
        'rho': []
    }
    checkpoint = {}
    scaler = torch.amp.GradScaler()  # Initialize the scaler for mixed precision

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        last_epoch = ckpt['epoch']
        ans = input(
            f"Checkpoint found at epoch {last_epoch}. "
            "Do you want to resume training from this epoch? [y/n] "
        )
        if ans.lower() == 'y':
            print(f"--> Resuming training from epoch {last_epoch + 1}")
            student.load_state_dict(ckpt['student_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            scaler.load_state_dict(ckpt['scaler_state'])
            history = ckpt['history']
            start_epoch = last_epoch + 1
        else:
            print("--> Starting from scratch, the previous checkpoint will be overwritten.")

    best_val_acc = 0.0
    rho = 1
    for epoch in range(start_epoch,epochs):
        rho = rho_schedule(epoch, epochs)
        # Dynamic loss setup, rho is replaceable
        target_ratios = [rho**(i+1) for i in range(len(pruning_index))] 
        criterion = DynamicViTLoss(
            lambda_class = lambda_class,
            lambda_kl=lambda_kl, 
            lambda_distill=lambda_distill, 
            lambda_ratio=lambda_ratio, 
            target_ratios=target_ratios
        )
        # Train
        train_loss, ratio_loss, distill_loss, kl_loss, train_acc = train_one_epoch(
            student, teacher, train_loader, optimizer, criterion, device, epoch, rho,scaler
        )
        
        # Validate
        val_acc, _ = validate_one_epoch(student, val_loader, device)
        
        # Scheduler Step
        scheduler.step()
        
        # Store History
        history['train_loss'].append(train_loss)
        history['ratio_loss'].append(ratio_loss)
        history['distill_loss'].append(distill_loss)
        history['kl_loss'].append(kl_loss)
        history['train_acc'].append(train_acc)
        history['rho'].append(rho)
        history['val_acc'].append(val_acc)
        history['lrs'].append(optimizer.param_groups[0]['lr'])

        # Logging
        print(red(f"Epoch {epoch+1}/{epochs} | rho  {rho:.3f} | Loss: {train_loss:.4f} | Ratio loss: {ratio_loss:.4f} | Distill loss: {distill_loss:.4f} | kl loss : {kl_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"))
        
        writer.add_scalar('Student/Loss/total', train_loss, epoch)
        writer.add_scalar('Student/Loss/ratio', ratio_loss, epoch)
        writer.add_scalar('Student/Loss/distill', distill_loss, epoch)
        writer.add_scalar('Student/Loss/kl', kl_loss, epoch)
        writer.add_scalar('Student/Accuracy/rho', rho, epoch)
        writer.add_scalar('Student/Accuracy/train', train_acc, epoch)
        writer.add_scalar('Student/Accuracy/val', val_acc, epoch)
        writer.add_scalar('Student/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        checkpoint = {
            'epoch': epoch,
            'student_state': student.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'history': history
        }

        torch.save(checkpoint, checkpoint_path)
        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(student.state_dict(), f"{checkpoint_dir}/student_epoch_{epoch+1}.pth")
            
        # Save Best Student
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), f"{checkpoint_dir}/student_best.pth")
            print(purple(f"--> New Best Student Saved ({val_acc:.2f}%)"))

    # Final Test
    print(yellow("\nTraining Complete. Loading best student for final testing..."))
    student.load_state_dict(torch.load(f"{checkpoint_dir}/student_best.pth"))
    test_acc, cm = validate_one_epoch(student, test_loader, device, desc="Testing Student")
    print(blue(f"Final Student Test Accuracy: {test_acc:.2f}%"))

    # Save Plots
    save_training_plots(
        history['train_loss'],
        history['train_acc'],
        history['val_acc'],
        history['ratio_loss'],
        history["distill_loss"], 
        history['kl_loss'],
        history['lrs'],
        history['rho'],
        cm,
        graph_dir
    )

    # Display the time taken by the student (expected to be much lower)
    seconds = time.time() - start_time
    print(cyan('Time Taken:'), cyan(time.strftime("%H:%M:%S",time.gmtime(seconds))))
    
    writer.close()