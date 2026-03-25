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
from src.models.vision_transformer import VisionTransformer
from src.models.dynamicViT_imagenet import DynamicVisionTransformer
from data.load.imagenet_loader import load_imagenet1k
from configs.train_imagenet1k import * 
from typing import Tuple, List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(bold(f"Using device: {device}"))

print("Initializing Teacher ViT...")
teacher = VisionTransformer(
    d_model=d_model, 
    n_classes=n_classes, 
    img_size=img_size, 
    patch_size=patch_size, 
    n_channels=n_channels, 
    n_heads=n_heads, 
    n_layers=n_layers
).to(device)

# Optimizer & Loss
optimizer = torch.optim.AdamW(teacher.parameters(), lr=alpha, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()

# Logging and checkpoints
log_dir = "logs/imagenet/teacher/Teacher_ViT_imagenet1k"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
checkpoint_dir = "checkpoints/imagenet"
os.makedirs(checkpoint_dir, exist_ok=True)
graph_dir = "logs/imagenet/teacher/Teacher_ViT_imagenet1K-graphs"
os.makedirs(graph_dir, exist_ok=True)

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: int
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model to train.
        loader: DataLoader for training data.
        optimizer: Optimizer for model parameters.
        criterion: Loss function.
        device: Device to run computations on.
        epoch_index: Current epoch index (for logging).

    Returns:
        avg_loss: Average loss over the epoch.
        accuracy: Training accuracy in percentage.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc=f'Training Teacher Epoch {epoch_index}')
    
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        # We don't mind 'feats' here 
        outputs, _ = model(imgs) 
        
        # Backprop
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total

    return avg_loss, accuracy

def validate_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = 'Validating'
) -> Tuple[float, float, torch.Tensor]:
    """
    Validate the model on a dataset for one epoch.

    Args:
        model: PyTorch model to evaluate.
        loader: DataLoader for validation or test data.
        criterion: Loss function.
        device: Device to run computations on.
        desc: Description for tqdm progress bar.

    Returns:
        avg_loss: Average loss over the validation set.
        accuracy: Validation accuracy in percentage.
        cm: Confusion matrix of predictions.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        loop = tqdm(loader, desc=desc)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            
            # Unpack tuple here as well
            outputs, _ = model(imgs)
            
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100 * correct / total
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, cm

def save_training_plots(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    lrs: List[float],
    confusion_mat: torch.Tensor,
    save_dir: str
) -> None:
    """
    Generate and save training graphs: Loss, Accuracy, Learning Rate curves, and Confusion Matrix heatmap.

    Args:
        train_losses: List of training loss values per epoch.
        val_losses: List of validation loss values per epoch.
        train_accs: List of training accuracy values per epoch.
        val_accs: List of validation accuracy values per epoch.
        lrs: List of learning rates per epoch.
        confusion_mat: Confusion matrix for the final evaluation.
        save_dir: Directory to save plots.
    """

    print(blue(f"Saving training graphs to {save_dir}..."))
    
    # 1. Loss Curve
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

    # 2. Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Acc', color='tab:green')
    plt.plot(val_accs, label='Validation Acc', color='tab:red')
    plt.title('Teacher Training & Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

    # 3. Learning Rate Curve
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, label='Learning Rate', color='purple')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "lr_curve.png"))
    plt.close()

    # 4. Confusion Matrix Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=False, fmt='d', cmap='Blues')
    plt.title('Final Test Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

if __name__ == "__main__":
    start_time = time.time()
    print(blue("Loading Data..."))
    
    train_loader, test_loader, val_loader = load_imagenet1k() 

    print(blue("Starting Teacher Training..."))
    best_val_acc = 0.0

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': [],
        'lrs': []
    }
    
    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            teacher, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_acc, _ = validate_one_epoch(
            teacher, val_loader, criterion, device, desc='Validating Teacher'
        )
        
        # Store Metrics
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lrs'].append(optimizer.param_groups[0]['lr'])

        # Update Learning Rate
        scheduler.step()
        
        # Logging
        print(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
        writer.add_scalar('Teacher/Loss/train', train_loss, epoch)
        writer.add_scalar('Teacher/Loss/val', val_loss, epoch)
        writer.add_scalar('Teacher/Accuracy/train', train_acc, epoch)
        writer.add_scalar('Teacher/Accuracy/val', val_acc, epoch)
        writer.add_scalar('Teacher/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"{checkpoint_dir}/teacher_checkpoint_best.pth"
            torch.save(teacher.state_dict(), save_path)
            print(purple(f"--> New Best Teacher Model saved at {save_path}"))

        if (epoch + 1) % 5 == 0:
            torch.save(teacher.state_dict(), f"{checkpoint_dir}/teacher_epoch_{epoch+1}.pth")

    # Final Test on Test Set
    # After training is complete, check performance on the hold-out test set using the best model
    print(green("\nTraining Complete. Loading best model for final testing..."))
    teacher.load_state_dict(torch.load(f"{checkpoint_dir}/teacher_checkpoint_best.pth"))
    test_loss, test_acc, cm = validate_one_epoch(teacher, test_loader, criterion, device, desc='Testing Teacher')
    print(bold(f"Final Test Accuracy: {test_acc:.2f}%"))

    save_training_plots(
        history['train_loss'], 
        history['val_loss'], 
        history['train_acc'], 
        history['val_acc'], 
        history['lrs'], 
        cm, 
        graph_dir
    )
    
    # Display the time taken by the student (expected to be much lower)
    seconds = time.time() - start_time
    print(blue('Time Taken:'), blue(time.strftime("%H:%M:%S",time.gmtime(seconds))))

    writer.close()