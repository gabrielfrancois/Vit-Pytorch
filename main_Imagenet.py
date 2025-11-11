import os
import logging
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from training.trainer import ViTTrainer

# ============================================================
# LOGGER
# ============================================================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/train_vit.log", mode="w"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("ViT_Trainer")

# ============================================================
# DATASET
# ============================================================
dataset_name = "imagenet"  # ou "CIFAR100_d32_reg"
data_dir = "/home/onyxia/work/Vit-Pytorch/data"

if dataset_name == "CIFAR100_d32_reg":
    from configs.train_cifar10 import * 
    from data.load_data import load_CIFAR
    train_loader, val_loader, test_loader = load_CIFAR(CIFAR=10, data_dir=data_dir)
elif dataset_name == "imagenet":
    from configs.train_imagenet1k import *
    from data.imagenet_loader import load_imagenet1k
    train_loader, val_loader, test_loader = load_imagenet1k(
        batch_size=4, max_items_train=None, max_items_val=None
    )
else:
    raise ValueError(f"Dataset inconnu: {dataset_name}")

log.info(f"Dataset chargé: {dataset_name.upper()}")
log.info(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)} | Test: {len(test_loader.dataset)}")

# ============================================================
# MODEL PARAMS
# ============================================================
model_params = {
    "d_model": d_model,
    "n_classes": n_classes,
    "img_size": img_size,
    "patch_size": patch_size,
    "n_channels": n_channels,
    "n_heads": n_heads,
    "n_layers": n_layers
}

train_params = {
    "lr": alpha,
    "weight_decay": weight_decay,
    "epochs": epochs,
    "label_smoothing": 0.1,
    "log_dir": f"runs/{dataset_name}_ViT"
}

# ============================================================
# INIT TRAINER
# ============================================================
trainer = ViTTrainer(
    model_params=model_params,
    train_params=train_params,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    checkpoint_dir="checkpoints",
    plot_dir="plots"
)

name = f"{dataset_name}_ViT_d{d_model}_p{patch_size}"
best_val_acc = 0.0
cm_max = None

# ============================================================
# TRAINING LOOP
# ============================================================
for epoch in range(1, epochs + 1):
    log.info(f"\nEpoch {epoch}/{epochs}")
    
    train_loss, train_acc = trainer.train_one_epoch(train_loader)
    val_loss, val_acc, cm = trainer.validate_one_epoch(val_loader)
    lr = trainer.step_scheduler()

    log.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    log.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {lr:.6f}")

    # TensorBoard
    trainer.writer.add_scalar("Loss/train", train_loss, epoch)
    trainer.writer.add_scalar("Loss/val", val_loss, epoch)
    trainer.writer.add_scalar("Accuracy/train", train_acc, epoch)
    trainer.writer.add_scalar("Accuracy/val", val_acc, epoch)
    trainer.writer.add_scalar("LR", lr, epoch)

    # Checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        cm_max = cm
        trainer.save_checkpoint(f"{name}_best")
        log.info(f"Nouveau meilleur modèle sauvegardé ({val_acc:.2f}%)")

trainer.writer.close()
log.info("Entraînement terminé avec succès.")

# ============================================================
# PLOTS
# ============================================================
plt.figure(figsize=(8, 8))

plt.subplot(2, 2, 1)
plt.plot(trainer.train_losses, label='Train Loss')
plt.plot(trainer.val_losses, label='Val Loss')
plt.title(f'Loss (patch size={patch_size})')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(trainer.train_accs, label='Train Acc')
plt.plot(trainer.val_accs, label='Val Acc')
plt.title(f'Accuracy ({n_heads} heads, {n_layers} layers)')
plt.legend()

plt.subplot(2, 2, 3)
if cm_max is not None:
    sns.heatmap(cm_max, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Ground truth")
    plt.title("Validation Confusion Matrix")

plt.subplot(2, 2, 4)
plt.plot(trainer.lrs)
plt.title(f'Learning Rate: lr={alpha}, epochs={epochs}')

plt.tight_layout()
plt.savefig(f"plots/{name}_training.png")
plt.close()
log.info("Figures enregistrées dans le dossier plots/")

# ============================================================
# TEST FINAL
# ============================================================
trainer.test_model(test_loader, f"{name}_best")
log.info("Test terminé.")
