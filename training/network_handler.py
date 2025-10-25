# This Python file gathres all the functions / classes needed to handle training, validation or inference from the ViT model

import os
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models.vision_transformer import VisionTransformer
from data.load_data_leo import load_CIFAR10_data
from configs.train_cifar10 import * #contains some constants

# TODO: make connection to config files + use torch.compile(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants of the training (model, optimizer, ...)
model = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
optimizer = Adam(model.parameters(), lr=alpha)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()

# Tensorboard
log_dir = "runs/ViT_CIFAR10"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Training and Validation
def train_one_epoch(model, loader, optimizer, criterion, device): #maybe remove some variables if fixed
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(loader, desc='Training')
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0) #multiplication by batch size cancels out when divided by total nb of data
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loop.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100*correct/total

    return avg_loss, accuracy

def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0) #multiplication by batch size cancels out when divided by total nb of data
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100*correct/total

    return avg_loss, accuracy


### Training loop

data_dir = "/home/onyxia/work/Vit-Pytorch/data"
train_loader, val_loader, _ = load_CIFAR10_data(data_dir)

best_val_acc = 0.0

for epoch in range(epochs):
    print(f"\nEpoch {epoch}/{epochs}")

    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
    scheduler.step() #based on validation performance

    print(f"Train Loss: {train_loss: .4f} | Train Acc: {train_acc: .2f}%")
    print(f"Val Loss: {val_loss: .4f} | Val Acc: {val_acc: .2f}%")

    #tensorboard logging
    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    writer.add_scalar("Accuracy/train", train_acc, epoch)
    writer.add_scalar("Accuracy/val", val_acc, epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    # save checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth"))
        print("New best model saved")
writer.close()
print("Training complete")

#run this in terminal: tensorboard --logdir runs
