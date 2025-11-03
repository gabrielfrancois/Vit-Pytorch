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
from data.load_data import load_CIFAR
from configs.train_cifar10 import * #contains some constants

# TODO: use torch.compile(model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants of the training (model, optimizer, ...)
model = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
#optimizer = Adam(model.parameters(), lr=alpha)
optimizer = torch.optim.AdamW(model.parameters(), lr=alpha, weight_decay=1e-4) # regularisation
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #regularisation

# Tensorboard
log_dir = "training/log/ViT_CIFAR100"
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
plot_dir = "training/log/plotsCIFAR100"
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

    all_labels = []
    all_preds = []

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

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            loop.set_postfix(loss=loss.item())
    
    avg_loss = running_loss / len(loader.dataset)
    accuracy = 100*correct/total
    cm = confusion_matrix(all_labels, all_preds)

    return avg_loss, accuracy, cm

def test_model(name, loader, criterion, device):

    # Define model
    model = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
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
        print(f"Total time: {total_time:.2f}s")
        print(f"Time per image: {time_per_image*1000:.2f} ms/image")

        # confusion matrix
        cm = confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(6, 6))
        sns.heatmap(cm, cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Ground truth")
        plt.title(f"Test confusion matrix for {name}")

        path_cm = os.path.join(plot_dir, f"{name}_test_cm.png")
        plt.savefig(path_cm)
        plt.close()

        report = classification_report(all_labels, all_preds, zero_division=0, digits=3)
        print("Classficiation report: \n", report)

    
        # Create a text file to store the metrics simply
        metrics_path = os.path.join(plot_dir, f"{name}_metrics.txt")

        with open(metrics_path, "w") as f:
            f.write(f"Device: {device}\n")
            f.write(f"Loss: {avg_loss:.4f}\n")
            f.write(f"Accuracy: {accuracy:.2f}\n")
            f.write(f"Total time: {total_time:.2f}s\n")
            f.write(f"Time per image: {time_per_image*1000:.2f} ms/image\n")
            f.write("\nClassification report:\n")
            f.write(classification_report(all_labels, all_preds, digits=3))
            f.write("\n\nHyperparameters:\n")
            f.write(f"d_model: {d_model}\n")
            f.write(f"n_classes: {n_classes}\n")
            f.write(f"img_size: {img_size}\n")
            f.write(f"patch_size: {patch_size}\n")
            f.write(f"n_channels: {n_channels}\n")
            f.write(f"n_heads: {n_heads}\n")
            f.write(f"n_layers: {n_layers}\n")
            f.write(f"batch_size: {batch_size}\n")
            f.write(f"alpha: {alpha}\n")
            f.write(f"epochs: {epochs}\n")
            
            
        return {
            "loss": avg_loss,
            "acuracy": accuracy,
            "total_time": total_time,
            "time_per_image": time_per_image,
            "confusion_matrix": cm,
            "classification_report": report
            }



