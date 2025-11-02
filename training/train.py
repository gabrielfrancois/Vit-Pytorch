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
from data.load_data import load_CIFAR10_data
from configs.train_cifar10 import * #contains some constants

# TODO: make connection to config files + use torch.compile(model) + add custom name to distinguish different experiments


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants of the training (model, optimizer, ...)
model = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
optimizer = Adam(model.parameters(), lr=alpha)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss()

# Tensorboard
log_dir = "training/log/ViT_CIFAR10"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir)

# checkpoints
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# configure plot
train_losses, val_losses = [], []
train_accs, val_accs = [], []
lrs = []

# save plots
plot_dir = "training/log/plotsCIFAR10"
os.makedirs(plot_dir, exist_ok=True)

# Training, Validation, and test
def train_one_epoch(model, loader, optimizer, criterion, device): # maybe remove some variables if fixed
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
        loop = tqdm(loader, desc='Validation')
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0) # multiplication by batch size cancels out when divided by total nb of data
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100*correct/total

    return avg_loss, accuracy

def test_model(name, loader, criterion, device):

    # Define model
    model = VisionTransformer(
            d_model=d_model,
            n_classes=n_classes,
            img_size=img_size,
            patch_size=patch_size,
            n_channels=n_channels,
            n_heads=n_heads,
            n_layers=n_layers,
        ).to(device)

    # Load model
    model_path = f"/home/onyxia/work/Vit-Pytorch/checkpoints/{name}.pth"
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    start_time = time.time()
    
    with torch.no_grad():
        loop = tqdm(loader, desc="Testing")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * imgs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loop.set_postfix(loss=loss.item())
        
        total_time = time.time() - start_time
        avg_loss = running_loss / len(loader.dataset)
        accuracy = 100 * correct / total
        time_per_image = total_time / len(loader.dataset)

        print('Test results:')
        print(f"Device: {device}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Total time: {total_time}s")
        print(f"Time per image: {time_per_image*1000:.2f} ms/image")

        # confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Ground truth")
        plt.title("Confusion matrix")

        path_cm = os.path.join(plot_dir, "confusion_matrix.png")
        plt.savefig(path_cm)
        plt.close()

        report = classification_report(all_labels, all_preds, digits=3)
        print("Classficiation report: \n", report)

        return {
            "loss": avg_loss,
            "acuracy": accuracy,
            "total_time": total_time,
            "time_per_image": time_per_image,
            "confusion_matrix": cm,
            "classification_report": report
            }



### Training loop

data_dir = "/home/onyxia/work/Vit-Pytorch/data"
train_loader, val_loader, test_loader = load_CIFAR10_data(data_dir)

#name of the model
name = "baseline"

best_val_acc = 0.0

train = True
if train:

    for epoch in range(epochs):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step() # based on validation performance

        print(f"Train Loss: {train_loss: .4f} | Train Acc: {train_acc: .2f}%")
        print(f"Val Loss: {val_loss: .4f} | Val Acc: {val_acc: .2f}%")

        # Save progression
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        lrs.append(optimizer.param_groups[0]["lr"])

        # Tensorboard logging
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

        # save checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(checkpoint_dir, name, ".pth")
            torch.save(model.state_dict(), save_path)
            print("New best model saved")
    writer.close()
    print("Training complete")

    # save all figures in logs
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title(f'Loss: patch size={patch_size}')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title(f'Accuracy: n_heads={n_heads} & {n_layers} layers')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(lrs)
    plt.title(f'Learning Rate: lr = {alpha} & epoch = {epochs}')

    plt.tight_layout()
    plt.savefig(get_next_checkpoint_path(base_dir=plot_dir, extension="", base_name=f"logs_train_dmodel={d_model}"))
    plt.close()


    # run this in terminal: tensorboard --logdir runs


test_model(name="best_model", loader=test_loader, criterion=criterion, device=device)


