"""
Goal: Fine-tune a pretrained CIFAR-10 Vision Transformer on STL-10 dataset with LORA

STL-10 dataset: https://cs.stanford.edu/~acoates/stl10/
- 10 classes (like CIFAR-10)
- 96x96 images (resized to 32x32 to match CIFAR-10 pretrained model) 
[this is just for test]
"""

import torch
from torch import nn
from torchvision import transforms as T
from torchvision.datasets import STL10
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ours
from models.vision_transformer import VisionTransformer
from models.dynamicViT_imagenet import DynamicVisionTransformer
from models.finetune import inject_lora, propor_params
#from configs.train_cifar10 import *
from configs.finetune_STL import *
from finetuning.STL_data import load_STL10



def load_pretrained(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    if 'model_state' in state: state = state['model_state']
    model.load_state_dict(state, strict=True)
    print("loaded")


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    loop = tqdm(loader, desc='Training')
    
    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)[0] #to return only the logits and not the teacher features
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()

        # to show the improve in live
        current_acc = 100 * correct / total
        loop.set_postfix(loss=loss.item(), acc=f"{current_acc:.2f}%")
 
    avg_loss = running_loss / total
    accuracy = 100*correct/total
    
    return avg_loss, accuracy 


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct, total = 0, 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            
            outputs = model(imgs)[0]
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * imgs.size(0)
            _, pred = outputs.max(1)
            
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    avg_loss = running_loss / total
    accuracy = 100 * correct / total

    cm = confusion_matrix(all_labels, all_preds)
    
    return avg_loss, accuracy, cm




def save_finetune_plots(train_losses, val_losses, cm, train_accs, val_accs, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # 1. Loss Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val loss', color='orange')
    plt.title(f'Fine-tuning cross entropy loss (LORA rank = {rank})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "finetune_loss.png"))
    plt.close()

    # 2. Accuracy Curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label='Train Acc', color='blue')
    plt.plot(val_accs, label='Val Acc', color='orange')
    plt.title(f'Fine-tuning Accuracy (LORA rank = {rank})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, "finetune_accuracy.png"))
    plt.close()
    print(f"Plots saved in {save_dir}")

    # 3. Confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Validation Confusion Matrix (LORA rank = {rank})')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(save_dir, "finetune_confusion_matrix.png"))
    plt.close()


def test_model(model, loader, device, save_dir):
        model.eval()
        
        correct, total = 0, 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for imgs, labels in tqdm(loader):
                imgs, labels = imgs.to(device), labels.to(device)

                #print(labels)
                
                outputs = model(imgs)[0]
                
                _, pred = outputs.max(1)
                #print(pred)
                
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        accuracy = 100 * correct / total

        cm = confusion_matrix(all_labels, all_preds)

        # plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Test Confusion Matrix (accuracy = {accuracy})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(save_dir, "test_confusion_matrix.png"))
        plt.close()

        print(accuracy)
        
        return accuracy


Train = True

if Train:

    plot_dir = "/home/onyxia/work/Vit-Pytorch/plots/plot_finetune"
    checkpoint_dir = "/home/onyxia/work/Vit-Pytorch/checkpoints/fine_tune"
    
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    best_val_acc = 0.0

    # load data
    data = load_STL10(data_dir="./data", batch_size=batch_size)
    train_loader = data[0]
    val_loader = data[1]
    test_loader = data[2]

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    student = DynamicVisionTransformer(
        d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index,rho_init
    ).to(device)

    checkpoint = "/home/onyxia/work/Vit-Pytorch/checkpoints/imagenet1K/student_best.pth"
    load_pretrained(student, checkpoint)
    print(student)

    #(classifier): Sequential((0): Linear(in_features=96, out_features=1000, bias=True))

    # We change the last layer for two reasons: 1) STL has only 10 classes, 2) we finetune the model
    student.classifier = nn.Linear(d_model, 10).to(device)

    print(student)
    #(classifier): Linear(in_features=96, out_features=10, bias=True)

    # Applying lora
    student = inject_lora(student, rank, alpha=1)
    print("injected")

    # only train Lora params
    lora_params = [p for p in student.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        train_loss, train_acc = train_one_epoch(student, train_loader, optimizer, criterion, device)
        val_loss, val_acc, cm = validate_one_epoch(student, val_loader, criterion, device)
    
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")

        
        # save most recent model every 5 epochs and delete the previous version
        if (epoch + 1) % 5 == 0:
            last_model_path = os.path.join(checkpoint_dir, "finetune_last.pth")
            torch.save({
                'epoch': epoch,
                'model_state': student.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'history': history
            }, last_model_path)
            print(f"--> last checkpoint saved (Epoch {epoch+1})")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_path = os.path.join(checkpoint_dir, "finetune_best.pth")
            torch.save(student.state_dict(), best_model_path)
            print(f"New best accuracy : {val_acc:.2f}% ! Model saved.")

        save_finetune_plots(history['train_loss'], history['val_loss'], cm, history['train_acc'], history['val_acc'], plot_dir)


        torch.save(student.state_dict(), "/home/onyxia/work/Vit-Pytorch/checkpoints/student_finetune_STL.pth")


        # TEST

        # load data

        finetuned = DynamicVisionTransformer(
            d_model, 10, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index, rho_init
        ).to(device)

        finetuned = inject_lora(finetuned, rank, alpha=1)

        checkpoint = "/home/onyxia/work/Vit-Pytorch/checkpoints/fine_tune/finetune_best.pth"
        load_pretrained(finetuned, checkpoint)
        print(finetuned)
        plot_dir = "/home/onyxia/work/Vit-Pytorch/plots/plot_finetune/"
        test_model(finetuned, test_loader, device, plot_dir)


    












        