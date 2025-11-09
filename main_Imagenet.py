import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from models.vision_transformer import VisionTransformer
from data.imagenet_loader import load_imagenet1k
from configs.train_imagenet1k import *

# -------------------
# Paramètres
# -------------------
name = "Imagenet1k_reg"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_dir = "checkpoints"
plot_dir = "plotsImagenet1k"
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# -------------------
# Data
# -------------------
train_loader, val_loader, test_loader = load_imagenet1k(batch_size=batch_size)

# -------------------
# Model
# -------------------
model = VisionTransformer(
    d_model=d_model,
    n_classes=n_classes,
    img_size=img_size,
    patch_size=patch_size,
    n_channels=n_channels,
    n_heads=n_heads,
    n_layers=n_layers
).to(device)
model = torch.compile(model)

# -------------------
# Loss, optimizer, scheduler
# -------------------
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=alpha, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# -------------------
# AMP
# -------------------
from torch.amp import autocast, GradScaler
scaler = GradScaler()  # pas de device_type

# -------------------
# TensorBoard
# -------------------
writer = SummaryWriter(log_dir=f"runs/{name}")

# -------------------
# Métriques
# -------------------
train_losses, val_losses = [], []
train_accs, val_accs = [], []
lrs = []
best_val_acc = 0.0

# -------------------
# Training loop
# -------------------
for epoch in range(1, epochs + 1):
    print(f"\nEpoch {epoch}/{epochs}")
    model.train()
    running_loss, running_corrects = 0.0, 0
    for images, labels in tqdm(train_loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        # AMP forward + backward
        with autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)
        running_corrects += (outputs.argmax(1) == labels).sum().item()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = 100 * running_corrects / len(train_loader.dataset)

    # ---- Validation ----
    model.eval()
    val_loss, val_corrects = 0.0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            with autocast(dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            val_corrects += (outputs.argmax(1) == labels).sum().item()
            all_preds.append(outputs.argmax(1).cpu())
            all_labels.append(labels.cpu())

    val_loss /= len(val_loader.dataset)
    val_acc = 100 * val_corrects / len(val_loader.dataset)
    cm = confusion_matrix(torch.cat(all_labels), torch.cat(all_preds))

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
    print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

    scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])

    # Logging
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        cm_max = cm
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, f"{name}.pth"))
        print("New best model saved")

writer.close()

# -------------------
# PLOTS
# -------------------
plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title(f'Loss: patch size={patch_size}')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.title(f'Accuracy: {n_heads} heads & {n_layers} layers')
plt.legend()

plt.subplot(2, 2, 3)
sns.heatmap(cm_max, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Ground truth")
plt.title("Validation confusion matrix")

plt.subplot(2, 2, 4)
plt.plot(lrs)
plt.title(f'Learning Rate: lr = {alpha} & epoch = {epochs}')

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"{name}_training.png"))
plt.close()

# -------------------
# TEST
# -------------------
model.load_state_dict(torch.load(os.path.join(checkpoint_dir, f"{name}.pth")))
model.eval()
test_loss, test_corrects = 0.0, 0
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing"):
        images, labels = images.to(device), labels.to(device)
        with autocast(dtype=torch.float16):
            outputs = model(images)
            loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
        test_corrects += (outputs.argmax(1) == labels).sum().item()
        all_preds.append(outputs.argmax(1).cpu())
        all_labels.append(labels.cpu())

test_loss /= len(test_loader.dataset)
test_acc = 100 * test_corrects / len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
