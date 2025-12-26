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

# ours
from models.vision_transformer import VisionTransformer
from models.dynamicViT_imagenet import DynamicVisionTransformer
from models.finetune import inject_lora, propor_params
#from configs.train_cifar10 import *
from configs.finetune_STL import *
from finetuning.STL_data import load_STL10



def load_pretrained(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
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


if __name__ == "__main__":

    # load data
    train_loader = load_STL10(data_dir="./data", batch_size=batch_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

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
    student = inject_lora(student, rank=4, alpha=1)
    print("injected")

    # only train Lora params
    lora_params = [p for p in student.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):

        loss, acc = train_one_epoch(student, train_loader, optimizer, criterion, device)
        print(loss, acc)


    torch.save(student.state_dict(), "/home/onyxia/work/Vit-Pytorch/checkpoints/student_finetune_STL.pth")









    