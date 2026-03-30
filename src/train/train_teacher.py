import argparse
import os
import time
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from typing import Tuple, List
from torch.utils.data import DataLoader
import numpy as np

from helper_function.print import *
from src.models.vision_transformer import VisionTransformer


# ----------------------------------------- Training Functions -----------------------------------------

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(bold(f"Using device: {device}"))

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=None, help='Choose the number of epochs')
    parser.add_argument('--d_model', type=int, default=None, help='choose the patch-embedding dimension')
    parser.add_argument('--dataset', type=str, default="cifar10", choices=['cifar10', 'imagenet'], help='Choose the dataset on which you want to train the teacher. Possible choices: ["cifar10", "imagenet"]')
    parser.add_argument('--n_layers', type=int, default=None, help='Choose the number of layers')
    parser.add_argument('--batch_size', type=int, default=None, help='Choose the batch size')
    parser.add_argument('--patch_size',type=int,nargs=2,default=None,help='choose the patch-size dimension (ex: 8 8)')
    parser.add_argument('--alpha', type=float, default=None, help='choose the learning rate')
    parser.add_argument('--n_heads', type=int, default=None, help='choose the number of attentions head, BE CAREFUL: n_head MUST be a multiple of d_model!')
    subparsers = parser.add_subparsers(dest="dataset", required=False)
    args = parser.parse_args()

    available_dataset = ["cifar10", "imagenet"]
    assert args.dataset in available_dataset, "choose a dataset in the available options: ['cifar10', 'imagenet']"
    if args.n_heads is not None and args.d_model is not None:
        assert args.d_model % args.n_heads == 0, "d_model must be divisible by n_heads"

    param_selected = [
        'epochs', 'd_model',
        'dataset','n_layers',
        'batch_size','patch_size',
        'alpha','n_heads'
        ]

    if args.dataset == "cifar10":
        from data.load.load_data import load_CIFAR
        from configs.train_cifar10 import * 

        log_dir = "./logs/cifar10/teacher/"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        checkpoint_dir = "checkpoints/cifar10/teacher_2th_try"
        os.makedirs(checkpoint_dir, exist_ok=True)
        graph_dir = "./logs/cifar10/teacher/graphs"
        os.makedirs(graph_dir, exist_ok=True)

        print(blue(f"Loading {args.dataset} Data..."))
        train_loader, test_loader, val_loader = load_CIFAR(CIFAR=10) 
    else:
        from data.load.imagenet_loader import load_imagenet1k
        from configs.train_imagenet1k import * 

        log_dir = "logs/imagenet/teacher/Teacher_ViT_imagenet1k"
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir)
        checkpoint_dir = "checkpoints/imagenet"
        os.makedirs(checkpoint_dir, exist_ok=True)
        graph_dir = "logs/imagenet/teacher/Teacher_ViT_imagenet1K-graphs"
        os.makedirs(graph_dir, exist_ok=True)

        print(blue(f"Loading {args.dataset} Data..."))
        train_loader, test_loader, val_loader = load_imagenet1k() 
    
    for param in param_selected: # Set up CLI param if specified...
        value = getattr(args, param)
        if value is not None:
            if param == 'patch_size':
                value = tuple(value)
            globals()[param] = value

    start_time = time.time()
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

    optimizer = torch.optim.AdamW(teacher.parameters(), lr=alpha, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

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
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lrs'].append(optimizer.param_groups[0]['lr'])

        # Update Learning Rate
        scheduler.step()
        
        print(bold(f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%"))
        
        writer.add_scalar('Teacher/Loss/train', train_loss, epoch)
        writer.add_scalar('Teacher/Loss/val', val_loss, epoch)
        writer.add_scalar('Teacher/Accuracy/train', train_acc, epoch)
        writer.add_scalar('Teacher/Accuracy/val', val_acc, epoch)
        writer.add_scalar('Teacher/LearningRate', optimizer.param_groups[0]['lr'], epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"{checkpoint_dir}/teacher_checkpoint_best.pth"
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': teacher.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), # Great for resuming interrupted training!
                'best_val_acc': best_val_acc,
                'hyperparameters': {
                    'd_model': d_model,
                    'n_classes': n_classes,
                    'img_size': img_size,
                    'patch_size': patch_size,
                    'n_channels': n_channels,
                    'n_heads': n_heads,
                    'n_layers': n_layers
                }
            }
            torch.save(checkpoint, save_path)
            print(green(f"--> New Best Teacher Model saved at {save_path}"))

        if (epoch + 1) % 10 == 0:
            torch.save(teacher.state_dict(), f"{checkpoint_dir}/teacher_epoch_{epoch+1}.pth")
        
    print(green("\nTraining Complete. Loading best model for final testing..."))
    checkpoint = torch.load(f"{checkpoint_dir}/teacher_checkpoint_best.pth", map_location=device)
    teacher.load_state_dict(checkpoint['model_state_dict'])

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
    
    seconds = time.time() - start_time
    print(blue('Time Taken:'), blue(time.strftime("%H:%M:%S",time.gmtime(seconds))))
    writer.close()