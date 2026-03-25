# This Python file gathers all the functions / classes needed to handle training, validation or inference from the ViT model
# python -m training.train
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
from typing import List, Tuple
from torch.utils.data import DataLoader

from helper_function.print import *
from models.vision_transformer import VisionTransformer
from models.dynamicViT import DynamicVisionTransformer
from .dynamic_loss import DynamicViTLoss
from data.load.load_data import load_CIFAR
from configs.train_cifar10 import * 
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(bold(f"Using device: {device}"))

# Checkpoint paths
checkpoint_dir = "checkpoints/cifar10/student_2th_try"
teacher_checkpoint = "checkpoints/cifar10/teacher_2th_try/teacher_checkpoint_best.pth"
os.makedirs(checkpoint_dir, exist_ok=True)

# Logging Directories
log_dir = "./logs/cifar10/student/"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

graph_dir = "./logs/cifar10/student/graphs"
os.makedirs(graph_dir, exist_ok=True)

# Initialize and Load TEACHER 
print(blue("Initializing Teacher..."))
teacher = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)

if os.path.exists(teacher_checkpoint):
    print(f"Loading Teacher weights from {teacher_checkpoint}")
    teacher.load_state_dict(torch.load(teacher_checkpoint, map_location=device))
else:
    raise FileNotFoundError(red(f"Teacher checkpoint not found at {teacher_checkpoint}. Please run run_teacher.py first!"))

teacher.eval() # Teacher is always in eval mode, already trained (otherwise, it would be the WORS teacher ever!)
for param in teacher.parameters():
    param.requires_grad = False # Freeze weights


# Initialize student
print(blue("Initializing Student..."))
student = DynamicVisionTransformer(
    d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index,rho
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

# Dynamic loss setup, pho is replaceable
target_ratios = [rho**(i+1) for i in range(len(pruning_index))] 
criterion = DynamicViTLoss(
    lambda_kl=lambda_kl, 
    lambda_distill=lambda_distill, 
    lambda_ratio=lambda_ratio, 
    target_ratios=target_ratios
)

# Plotting Function

def save_training_plots(
    train_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    ratio_losses: List[float],
    distill_losses: List[float],
    kl_losses: List[float],
    lrs: List[float],
    confusion_mat: np.ndarray,
    save_dir: str
) -> None:
    """
    Generate and save all training visualizations for the student model.

    This includes:
      - Total training loss
      - Training and validation accuracy
      - Sparsity (ratio) loss
      - Distillation loss
      - KL divergence loss
      - Final confusion matrix

    Args:
        train_losses (List[float]):
            Total training loss per epoch.
        train_accs (List[float]):
            Training accuracy per epoch.
        val_accs (List[float]):
            Validation accuracy per epoch.
        ratio_losses (List[float]):
            Sparsity (token ratio) loss per epoch.
        distill_losses (List[float]):
            Feature/logit distillation loss per epoch.
        kl_losses (List[float]):
            KL divergence loss per epoch.
        lrs (List[float]):
            Learning rate values per epoch.
        confusion_mat (np.ndarray):
            Confusion matrix computed on the test set.
        save_dir (str):
            Directory where plots will be saved.
    """

# Main Execution
if __name__ == "__main__":
    start_time = time.time()

    print(blue("Loading Data..."))
    data_path = "./data/raw/cifar10" 
    train_loader, test_loader, val_loader = load_CIFAR(data_path, CIFAR=10) 

    print(yellow("Starting Student Training..."))
    
    # Metric History
    history = {
        'train_loss': [],
        'ratio_loss': [],
        "distill_loss": [],
        "kl_loss": [],
        'train_acc': [],
        'val_acc': [],
        'lrs': []
    }
    
    best_val_acc = 0.0

    for epoch in range(epochs):
        # Train
        train_loss, ratio_loss, distill_loss, kl_loss, train_acc = train_one_epoch(
            student, teacher, train_loader, optimizer, criterion, device, epoch
        )
        
        val_acc, _ = validate_one_epoch(student, val_loader, device)
        scheduler.step()
        
        # Store History
        history['train_loss'].append(train_loss)
        history['ratio_loss'].append(ratio_loss)
        history['distill_loss'].append(distill_loss)
        history['kl_loss'].append(kl_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lrs'].append(optimizer.param_groups[0]['lr'])

        # Logging
        print(red(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Ratio loss: {ratio_loss:.4f} | Distill loss: {distill_loss:.4f} | kl loss : {kl_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"))
        
        writer.add_scalar('Student/Loss/total', train_loss, epoch)
        writer.add_scalar('Student/Loss/ratio', ratio_loss, epoch)
        writer.add_scalar('Student/Loss/distill', distill_loss, epoch)
        writer.add_scalar('Student/Loss/kl', kl_loss, epoch)
        writer.add_scalar('Student/Accuracy/train', train_acc, epoch)
        writer.add_scalar('Student/Accuracy/val', val_acc, epoch)
        writer.add_scalar('Student/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(student.state_dict(), f"{checkpoint_dir}/student_epoch_{epoch+1}.pth")
            
        # Save Best Student
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(student.state_dict(), f"{checkpoint_dir}/student_best.pth")
            print(purple(f"--> New Best Student Saved ({val_acc:.2f}%)"))

    # Final Test
    print(green("\nTraining Complete. Loading best student for final testing..."))
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
        cm,
        graph_dir
    )

    # Display the time taken by the student (expected to be much lower)
    seconds = time.time() - start_time
    print(blue('Time Taken:'), blue(time.strftime("%H:%M:%S",time.gmtime(seconds))))
    
    writer.close()