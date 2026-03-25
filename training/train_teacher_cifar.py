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
from typing import Tuple, List
from torch.utils.data import DataLoader

from helper_function.print import *
from models.vision_transformer import VisionTransformer
from models.dynamicViT import DynamicVisionTransformer
from .dynamic_loss import DynamicViTLoss
from data.load.load_data import load_CIFAR
from configs.train_cifar10 import * 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(bold(f"Using device: {device}"))

# Initialize Teacher Model
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
log_dir = "./logs/cifar10/teacher/"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
graph_dir = "./logs/cifar10/teacher/graphs"
os.makedirs(graph_dir, exist_ok=True)

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: int
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

    Returns:
        Tuple[float, float]:
            - avg_loss: Average training loss over the epoch.
            - accuracy: Training accuracy in percentage.
    """

    model.train()
    running_loss: float = 0.0
    correct: int = 0
    total: int = 0

    loop = tqdm(loader, desc=f"Training Teacher Epoch {epoch_index}")

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs, _ = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

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

            outputs, _ = model(imgs)
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


def save_training_plots(
    train_losses: List[float],
    val_losses: List[float],
    train_accs: List[float],
    val_accs: List[float],
    lrs: List[float],
    confusion_mat: np.ndarray,
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

    print(blue(f"Saving training graphs to {save_dir}..."))

    # 1. Loss curve
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

    # 2. Accuracy curve
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

    # 3. Learning rate curve
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, label='Learning Rate', color='purple')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "lr_curve.png"))
    plt.close()

    # 4. Confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=True, fmt='d', cmap='Blues')
    plt.title('Final Test Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

# Main Execution Loop
if __name__ == "__main__":
    start_time = time.time()
    print(yellow("Loading Data..."))
    
    data_path = "./data/raw/cifar10" 
    train_loader, test_loader, val_loader = load_CIFAR(data_path, CIFAR=10) 

    print(yellow("Starting Teacher Training..."))
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
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lrs'].append(optimizer.param_groups[0]['lr'])

        # Update Learning Rate
        scheduler.step()
        
        print(red(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"))
        
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

        # Save Periodic Checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save(teacher.state_dict(), f"{checkpoint_dir}/teacher_epoch_{epoch+1}.pth")

    # Final Test on Test Set
    # After training is complete, check performance on the hold-out test set using the best model
    print(green("\nTraining Complete. Loading best model for final testing..."))
    teacher.load_state_dict(torch.load(f"{checkpoint_dir}/teacher_checkpoint_best.pth"))
    test_loss, test_acc, cm = validate_one_epoch(teacher, test_loader, criterion, device, desc='Testing Teacher')
    print(bold(f"Final Test Accuracy: {test_acc:.2f}%"))

    # Generate and Save Graphs
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
    print(cyan('Time Taken:'), cyan(time.strftime("%H:%M:%S",time.gmtime(seconds))))

    writer.close()