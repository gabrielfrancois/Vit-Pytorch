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

from helper_function.print import *
from models.vision_transformer import VisionTransformer
from models.dynamicViT import DynamicVisionTransformer
from .dynamic_loss import DynamicViTLoss
from data.load_data import load_CIFAR
from configs.train_cifar10 import * 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize Models
# TEACHER: Standard ViT (Frozen Expert)
print("Initializing Teacher...")
teacher = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
teacher.eval() # Teacher is always in eval mode (no dropout)
for param in teacher.parameters():
    param.requires_grad = False # Freeze weights so we don't train the teacher

# STUDENT: Dynamic ViT (Learner)
print("Initializing Student...")
student = DynamicVisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index=pruning_index).to(device)

# Optimizer & Loss, Note that we only optimize the STUDENT parameters...
optimizer = torch.optim.AdamW(student.parameters(), lr=alpha, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

# Set up the loss
target_ratios = [rho**(i+1) for i in range(len(pruning_index))] 
criterion = DynamicViTLoss(lambda_kl=lambda_kl, lambda_distill=lambda_distill, lambda_ratio=lambda_ratio, target_ratios=target_ratios)

# 4. Logging
log_dir = "training/log/ViT_CIFAR10"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Training Function
def train_one_epoch(student, teacher, loader, optimizer, criterion, device, epoch_index):
    student.train() # Student learns (Dropout on, Masking on)
    # Teacher is already in eval() and frozen globally

    running_loss = 0.0
    running_ratio_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f'Training Epoch {epoch_index}')
    
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        # Get Teacher Output (Ground Truth for Distillation) 
        with torch.no_grad():
            # Returns logits AND features (t'_i)
            teacher_logits, teacher_feats = teacher(imgs)

        #Get Student Output 
        # Returns logits, features (t_i), masks (D), and raw scores
        student_logits, student_feats, all_masks, all_scores = student(imgs)

        # Calculate Compound Loss 
        # Pass all 6 required components to loss.py
        loss, metrics = criterion(
            student_logits, 
            teacher_logits, 
            labels, 
            student_feats, 
            teacher_feats, 
            all_masks
        )

        # Backpropagation 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        running_loss += loss.item() * imgs.size(0)
        running_ratio_loss += metrics['ratio'] * imgs.size(0)
        
        _, predicted = torch.max(student_logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item(), ratio_loss=metrics['ratio'])
    
    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


# Validation Function
def validate_one_epoch(student, loader, device):
    student.eval() # Student stops masking randomly, uses hard pruning logic
    
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        loop = tqdm(loader, desc='Validation')
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # We only care about student logits for accuracy
            student_logits, _, _, _ = student(imgs)
            
            _, predicted = torch.max(student_logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, cm

# Main Execution Loop
if __name__ == "__main__":
    # Load Data
    train_loader, test_loader, val_loader = load_CIFAR(batch_size=batch_size) 

    print("Starting training...")
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = train_one_epoch(
            student, teacher, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_acc, cm = validate_one_epoch(student, test_loader, device)
        
        # Update Learning Rate
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)

        # Save Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(student.state_dict(), f"{checkpoint_dir}/dynamic_vit_epoch_{epoch+1}.pth")

    print("Training Complete.")
    writer.close()
