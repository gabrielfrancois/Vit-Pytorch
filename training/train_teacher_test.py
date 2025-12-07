import os
import time
import logging
from typing import Tuple, List, Dict

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models.vision_transformer import VisionTransformer
from data.load_data import load_CIFAR


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: int
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        model: Neural network model.
        loader: DataLoader for training data.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to run computation on.
        epoch_index: Current epoch index.

    Returns:
        Tuple containing average loss and accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

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

    avg_loss = running_loss / len(loader.dataset)
    acc = 100.0 * correct / total
    return avg_loss, acc


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "Validating"
) -> Tuple[float, float, List[List[int]]]:
    """
    Evaluate the model for one epoch.

    Args:
        model: Neural network model.
        loader: Validation/Test DataLoader.
        criterion: Loss function.
        device: Device used for computation.
        desc: Progress bar description.

    Returns:
        Tuple containing average loss, accuracy, and confusion matrix.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    preds = []
    labels_list = []

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

            preds.extend(predicted.cpu().numpy())
            labels_list.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    acc = 100.0 * correct / total
    cm = confusion_matrix(labels_list, preds)

    return avg_loss, acc, cm


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
    Generate and save Loss, Accuracy, Learning Rate curves and Confusion Matrix.

    Args:
        train_losses: List of training losses.
        val_losses: List of validation losses.
        train_accs: List of training accuracies.
        val_accs: List of validation accuracies.
        lrs: List of learning rates.
        confusion_mat: Final confusion matrix.
        save_dir: Directory where plots will be saved.
    """
    logger.info(f"Saving plots to {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label="Train Acc")
    plt.plot(val_accs, label="Val Acc")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "accuracy_curve.png"))
    plt.close()

    # LR
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, label="Learning Rate")
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("LR")
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "lr_curve.png"))
    plt.close()

    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()


def main(cfg) -> None:
    """
    Main training loop, Hydra-compatible.

    Args:
        cfg: Hydra configuration object.
    """

    logger.info("Loading CIFAR data...")
    train_loader, test_loader, val_loader = load_CIFAR(cfg.dataset.data_dir, CIFAR=10)

    logger.info("Initializing model...")
    teacher = VisionTransformer(
        d_model=cfg.model.d_model,
        n_classes=cfg.model.n_classes,
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        n_channels=cfg.model.n_channels,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers
    ).to(device)

    optimizer = AdamW(teacher.parameters(), lr=cfg.training.alpha, weight_decay=cfg.training.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(cfg.teacher.log_dir)
    os.makedirs(cfg.outputs.checkpoint_dir, exist_ok=True)

    best_val_acc = 0.0
    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "lrs": []
    }

    for epoch in range(cfg.training.epochs):

        train_loss, train_acc = train_one_epoch(
            teacher, train_loader, optimizer, criterion, device, epoch
        )

        val_loss, val_acc, _ = validate_one_epoch(
            teacher, val_loader, criterion, device, desc="Validating Teacher"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lrs"].append(optimizer.param_groups[0]["lr"])

        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{cfg.training.epochs} | "
            f"Loss {train_loss:.4f} | Train Acc {train_acc:.2f}% | Val Acc {val_acc:.2f}%"
        )

        writer.add_scalar("Teacher/Loss/train", train_loss, epoch)
        writer.add_scalar("Teacher/Loss/val", val_loss, epoch)
        writer.add_scalar("Teacher/Accuracy/train", train_acc, epoch)
        writer.add_scalar("Teacher/Accuracy/val", val_acc, epoch)
        writer.add_scalar("Teacher/LearningRate", optimizer.param_groups[0]["lr"], epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = f"{cfg.outputs.checkpoint_dir}/teacher_checkpoint_best.pth"
            torch.save(teacher.state_dict(), save_path)
            logger.info(f"New best model saved at: {save_path}")

    logger.info("Testing best model...")
    teacher.load_state_dict(torch.load(f"{cfg.outputs.checkpoint_dir}/teacher_checkpoint_best.pth"))
    test_loss, test_acc, cm = validate_one_epoch(
        teacher, test_loader, criterion, device, desc="Testing Teacher"
    )

    logger.info(f"Final Test Accuracy : {test_acc:.2f}%")

    save_training_plots(
        history["train_loss"],
        history["val_loss"],
        history["train_acc"],
        history["val_acc"],
        history["lrs"],
        cm,
        cfg.result.result_dir,
    )

    writer.close()
