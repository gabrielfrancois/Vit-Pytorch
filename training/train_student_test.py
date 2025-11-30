import os
import logging
from typing import Tuple, Dict, List, Optional

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from omegaconf import DictConfig

from models.vision_transformer import VisionTransformer
from models.dynamicViT import DynamicVisionTransformer
from training.dynamic_loss import DynamicViTLoss
from data.load_data import load_CIFAR


# ---------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def train_one_epoch(
    student: nn.Module,
    teacher: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch_index: int
) -> Tuple[float, float, float, float, float]:
    """
    Train the student model for one epoch with knowledge distillation from the teacher.

    Args:
        student (nn.Module): Student model to train.
        teacher (nn.Module): Teacher model used for distillation.
        loader: DataLoader for the training dataset.
        optimizer: Optimizer for the student model.
        criterion: Loss function with distillation metrics.
        device: Device to run training on.
        epoch_index: Current epoch index (for progress display).

    Returns:
        Tuple containing:
            - avg_loss: Average total loss.
            - avg_ratio_loss: Average ratio loss.
            - avg_distill_loss: Average distillation loss.
            - avg_kl_loss: Average KL divergence loss.
            - accuracy: Training accuracy (%).
    """
    student.train()
    running_loss = running_ratio = running_distill = running_kl = 0.0
    correct = total = 0

    loop = tqdm(loader, desc=f"Training Student Epoch {epoch_index}")

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        with torch.no_grad():
            t_logits, t_feats = teacher(imgs)

        s_logits, s_feats, masks, scores = student(imgs)

        loss, metrics = criterion(
            student_logits=s_logits,
            teacher_logits=t_logits,
            labels=labels,
            student_feats=s_feats,
            teacher_feats=t_feats,
            all_masks=masks
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        running_ratio += metrics["ratio"] * imgs.size(0)
        running_distill += metrics["distill"] * imgs.size(0)
        running_kl += metrics["kl"] * imgs.size(0)

        _, predicted = s_logits.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = running_loss / len(loader.dataset)
    avg_ratio = running_ratio / len(loader.dataset)
    avg_distill = running_distill / len(loader.dataset)
    avg_kl = running_kl / len(loader.dataset)
    accuracy = 100 * correct / total

    logger.info(
        f"Epoch {epoch_index} training completed | Loss: {avg_loss:.4f} | "
        f"Acc: {accuracy:.2f}%"
    )

    return avg_loss, avg_ratio, avg_distill, avg_kl, accuracy


def validate_one_epoch(
    student: nn.Module,
    loader,
    device: torch.device,
    desc: str = "Validation"
) -> Tuple[float, torch.Tensor]:
    """
    Evaluate the student model on validation or test data.

    Args:
        student (nn.Module): Student model to evaluate.
        loader: DataLoader for validation/test dataset.
        device: Device to run evaluation on.
        desc: Progress description for tqdm.

    Returns:
        Tuple containing:
            - accuracy (%)
            - confusion matrix (torch.Tensor)
    """
    student.eval()
    correct = total = 0
    preds, labels_all = [], []

    with torch.no_grad():
        loop = tqdm(loader, desc=desc)

        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _, _, _ = student(imgs)

            _, predicted = logits.max(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            preds.extend(predicted.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())

    accuracy = 100 * correct / total
    cm = confusion_matrix(labels_all, preds)
    logger.info(f"{desc} accuracy: {accuracy:.2f}%")

    return accuracy, cm


def save_training_plots(
    history: Optional[Dict[str, List[float]]] = None,
    train_losses: Optional[List[float]] = None,
    train_accs: Optional[List[float]] = None,
    val_accs: Optional[List[float]] = None,
    ratio_losses: Optional[List[float]] = None,
    distill_loss: Optional[List[float]] = None,
    kl_loss: Optional[List[float]] = None,
    lrs: Optional[List[float]] = None,
    confusion_mat=None,
    save_dir: str = "."
) -> None:
    """
    Save training curves (losses, accuracies) and confusion matrix to disk.

    Args:
        history: Dictionary of metric lists.
        train_losses, train_accs, val_accs, ratio_losses, distill_loss, kl_loss, lrs:
            Optional separate lists (used if history is not provided).
        confusion_mat: Confusion matrix to save as heatmap.
        save_dir: Directory to save plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving student training graphs to {save_dir}")

    if history is None:
        history = {
            "train_loss": train_losses or [],
            "train_acc": train_accs or [],
            "val_acc": val_accs or [],
            "ratio_loss": ratio_losses or [],
            "distill_loss": distill_loss or [],
            "kl_loss": kl_loss or [],
            "lrs": lrs or [],
        }

    # Total Loss
    plt.figure()
    plt.plot(history["train_loss"])
    plt.title("Total Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(save_dir, "student_loss_curve.png"))
    plt.close()

    # Accuracy
    plt.figure()
    plt.plot(history["train_acc"], label="Train")
    plt.plot(history["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(save_dir, "student_accuracy_curve.png"))
    plt.close()

    # Ratio Loss
    plt.figure()
    plt.plot(history["ratio_loss"])
    plt.title("Ratio Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(save_dir, "student_ratio_loss.png"))
    plt.close()

    # Distillation Loss
    plt.figure()
    plt.plot(history["distill_loss"])
    plt.title("Distillation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(save_dir, "student_distill_loss.png"))
    plt.close()

    # KL Loss
    plt.figure()
    plt.plot(history["kl_loss"])
    plt.title("KL Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=.3)
    plt.savefig(os.path.join(save_dir, "student_kl_loss.png"))
    plt.close()

    # Confusion Matrix
    if confusion_mat is not None:
        plt.figure(figsize=(10, 10))
        sns.heatmap(confusion_mat, annot=True, fmt=".2f", cmap="Oranges")
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(save_dir, "student_confusion_matrix.png"))
        plt.close()
        logger.info("Confusion matrix saved.")


# ---------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------

def train_student(cfg: DictConfig) -> None:
    """
    Train a DynamicViT student model using a ViT teacher model.

    Args:
        cfg: OmegaConf configuration object containing all training parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Directories
    checkpoint_dir = cfg.outputs.checkpoint_dir
    graph_dir = cfg.student.graph_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    writer = SummaryWriter(cfg.student.log_dir)

    # Load teacher
    teacher = VisionTransformer(
        cfg.model.d_model,
        cfg.model.n_classes,
        cfg.model.img_size,
        cfg.model.patch_size,
        cfg.model.n_channels,
        cfg.model.n_heads,
        cfg.model.n_layers,
    ).to(device)

    ckpt_path = cfg.student.teacher_checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing teacher checkpoint: {ckpt_path}")

    teacher.load_state_dict(torch.load(ckpt_path, map_location=device))
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Student
    student = DynamicVisionTransformer(
        cfg.model.d_model,
        cfg.model.n_classes,
        cfg.model.img_size,
        cfg.model.patch_size,
        cfg.model.n_channels,
        cfg.model.n_heads,
        cfg.model.n_layers,
        pruning_index=cfg.dynamicvit.pruning_index
    ).to(device)

    # Copy backbone weights
    t_state = teacher.state_dict()
    s_state = student.state_dict()
    mapping = {k.replace("transformer_encoder", "transformer_encoders"): v
               for k, v in t_state.items() if k.replace("transformer_encoder", "transformer_encoders") in s_state}
    student.load_state_dict(mapping, strict=False)

    # Optimizer / Scheduler / Loss
    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=cfg.training.alpha,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.training.epochs)

    criterion = DynamicViTLoss(
        lambda_kl=cfg.dynamicvit.lambda_kl,
        lambda_distill=cfg.dynamicvit.lambda_distill,
        lambda_ratio=cfg.dynamicvit.lambda_ratio,
        target_ratios=[cfg.dynamicvit.rho ** (i+1) for i in range(len(cfg.dynamicvit.pruning_index))]
    )

    # Load data
    train_loader, test_loader, val_loader = load_CIFAR(cfg.dataset.data_dir, CIFAR=10)

    # Training loop
    history = {k: [] for k in ["train_loss", "ratio_loss", "distill_loss", "kl_loss", "train_acc", "val_acc", "lrs"]}
    best_val = 0

    for epoch in range(cfg.training.epochs):
        train_loss, ratio_loss, distill_loss, kl_loss, train_acc = train_one_epoch(
            student, teacher, train_loader, optimizer, criterion, device, epoch_index=epoch
        )

        val_acc, _ = validate_one_epoch(student, val_loader, device)
        scheduler.step()

        # Log history
        history["train_loss"].append(train_loss)
        history["ratio_loss"].append(ratio_loss)
        history["distill_loss"].append(distill_loss)
        history["kl_loss"].append(kl_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lrs"].append(optimizer.param_groups[0]["lr"])

        writer.add_scalar("Loss/total", train_loss, epoch)
        writer.add_scalar("Acc/train", train_acc, epoch)
        writer.add_scalar("Acc/val", val_acc, epoch)

        logger.info(
            f"Epoch {epoch+1}/{cfg.training.epochs} | Loss {train_loss:.4f} | "
            f"Acc {train_acc:.2f}% | Val {val_acc:.2f}%"
        )

        if val_acc > best_val:
            best_val = val_acc
            torch.save(student.state_dict(), f"{checkpoint_dir}/student_best.pth")
            logger.info("New best student model saved.")

    # Final test
    student.load_state_dict(torch.load(f"{checkpoint_dir}/student_best.pth"))
    test_acc, cm = validate_one_epoch(student, test_loader, device, desc="Testing Student")
    logger.info(f"Final Student Test Accuracy: {test_acc:.2f}%")

    save_training_plots(history, confusion_mat=cm, save_dir=graph_dir)
    writer.close()


# ---------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    from omegaconf import OmegaConf
    cfg = OmegaConf.load("configs/cifar.yaml")
    train_student(cfg)
