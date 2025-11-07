import os
import time
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from helper_function.print import *
from models.vision_transformer import VisionTransformer
# Ne pas importer load_CIFAR ici, on ne l'utilise plus

from training.train import *
from datasets import load_dataset
from torch.utils.data import DataLoader

# Hyperparams à adapter à ImageNet ou à charger via un .py config
batch_size = 32
num_workers = 2
epochs = 30
name = "ViT_imagenet1k_128x128"
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# 1. Load ImageNet-1k-128x128 en streaming HF
def load_imagenet_hf(batch_size=32, num_workers=2):
    trainset = load_dataset("benjamin-paine/imagenet-1k-128x128", split="train", streaming=True).with_format("torch")
    valset   = load_dataset("benjamin-paine/imagenet-1k-128x128", split="validation", streaming=True).with_format("torch")
    train_loader = DataLoader(trainset, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(valset,   batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader

train_loader, val_loader = load_imagenet_hf(batch_size=batch_size, num_workers=num_workers)

# 2. Model & Training params (adapte selon ton besoin ViT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(...).to(device) # adapte selon config

optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

writer = SummaryWriter(f"training/log/{name}")
train_losses, val_losses, train_accs, val_accs, lrs = [], [], [], [], []

# 3. Training loop
best_val_acc = 0.0

for epoch in range(1, epochs+1):
    print(f"\nEpoch {epoch}/{epochs}")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc, cm = validate_one_epoch(model, val_loader, criterion, device)
    scheduler.step()

    print(f"Train Loss: {train_loss: .4f} | Train Acc: {train_acc: .2f}%")
    print(f"Val Loss: {val_loss: .4f} | Val Acc: {val_acc: .2f}%")

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)
    lrs.append(optimizer.param_groups[0]["lr"])

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        cm_max = cm
        save_path = os.path.join(checkpoint_dir, f"{name}.pth")
        torch.save(model.state_dict(), save_path)
        print("New best model saved")

writer.close()
print("Training complete")

# 4. Sauvegarde figures d'apprentissage
plot_dir = "training/log/plots_imagenet"
os.makedirs(plot_dir, exist_ok=True)
plt.figure(figsize=(8, 8))
plt.subplot(2, 2, 1)
plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss')
plt.title('Loss'); plt.legend()
plt.subplot(2, 2, 2)
plt.plot(train_accs, label='Train Acc'); plt.plot(val_accs, label='Val Acc')
plt.title('Accuracy'); plt.legend()
plt.subplot(2, 2, 3)
sns.heatmap(cm_max, cmap="Blues")
plt.xlabel("Predicted"); plt.ylabel("Ground truth"); plt.title("Validation confusion matrix")
plt.subplot(2, 2, 4)
plt.plot(lrs); plt.title('Learning Rate')
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f"{name}_training.png"))
plt.close()

# 5. Test possible si tu prépares un DataLoader test (sinon sur 'validation')


valset = load_dataset("benjamin-paine/imagenet-1k-128x128", split="validation", streaming=True).with_format("torch")
val_loader = DataLoader(valset, batch_size=32, num_workers=2, pin_memory=True)

# Puis appel de ta fonction de test
test_model(name=name, loader=val_loader, criterion=criterion, device=device)


