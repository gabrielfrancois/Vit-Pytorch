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
from training.train import *


### Training loop

#name of the model
name = "CIFAR100_d32_reg"

data_dir = "/home/onyxia/work/Vit-Pytorch/data"


train_loader_100, val_loader_100, test_loader_100 = load_CIFAR(CIFAR=100, data_dir = data_dir)





train = True
if train:
    
    print(f" !! LAUNCHING TRAINING FOR {name} !! ")
    
    best_val_acc = 0.0

    for epoch in range(1, epochs+1):
        print(f"\nEpoch {epoch}/{epochs}")

        train_loss, train_acc = train_one_epoch(model, train_loader_100, optimizer, criterion, device)
        val_loss, val_acc, cm = validate_one_epoch(model, val_loader_100, criterion, device)
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
            cm_max = cm
            save_path = os.path.join(checkpoint_dir, f"{name}.pth")
            torch.save(model.state_dict(), save_path)
            print("New best model saved")
    writer.close()
    print("Training complete")

    # save all figures in logs
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


    # run this in terminal: tensorboard --logdir runs



test_model(name=name, loader=test_loader_100, criterion=criterion, device=device)


