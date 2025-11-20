# faire une version jouet avec le model cifar qui a été loadé
# télécharger le dataset (attention GITIGNORE)
# resize images at la taille de cifar !

import torch
from torch import nn
import torchvision
from torchvision import transforms as T
from torchvision.datasets import STL10
from torch.utils.data import Dataset, DataLoader, random_split
from models.finetune import LORA

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from helper_function.print import *
from models.vision_transformer import VisionTransformer
from data.load_data import load_CIFAR
from configs.train_cifar10 import * #contains some constants





# STL-10 dataset
# https://cs.stanford.edu/~acoates/stl10/ (images acquired from ImageNet)
# "It is inspired by the CIFAR-10 dataset but with some modifications. In particular, each class 
# has fewer labeled training examples than in CIFAR-10, but a very large set of unlabeled examples 
# is provided to learn image models prior to supervised training. 
# images to fintune, from STL-10, dim = 96*96, 10 classes --> need to change the architecture 
# (final layer)


transform = T.Compose([
    T.Resize((32,32)), # CIFAR dimensions
    T.ToTensor()
])

num_workers = 4

class STL10Dataset(Dataset):
    """Simple wrapper around torchvision STL10 with optional custom transforms."""

    def __init__(self, split: str, transform=None):
        super().__init__()
        self.dataset = STL10(
            root="./data",
            split=split,
            download=True,
            transform=transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

def load_STL10(data_dir: str, img_size: int = 128, batch_size: int = 64):
    
    train_dataset = STL10Dataset(split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"[STL10] Train: {len(train_dataset)}")

    return train_loader


# Function to inject LORA in all linear layers

def inject_lora(model, rank, alpha=1):
    """ wrap every nn.Linear inside the model with LORA finetuning """
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            setattr(model, name, LORA(module, rank, alpha))
        else:
            inject_lora(module, rank, alpha) # recursivity
    return model

def load_pretrained(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print("loaded")


def train_one_epoch(model, loader, potimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += (pred == labels).sum().item()
    
    avg_loss = running_loss / total
    accuracy = 100*correct/total
    
    return avg_loss, accuracy 


if __name__ == "__main__":


    # load data
    #train_ds = STL10_data(split="train")
    #train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)
    train_loader = load_STL10(data_dir="./data", img_size=128, batch_size=64)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)

    checkpoint = "/home/onyxia/work/Vit-Pytorch/checkpoints/baseline_CIFAR10_reg.pth"
    load_pretrained(model, checkpoint)

    # Applying lora
    model = inject_lora(model, rank=4, alpha=1)
    print("injected")

    # only train Lora params
    lora_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    print(loss, acc)


    torch.save(model.state_dict(), "/home/onyxia/work/Vit-Pytorch/checkpoints/CIFAR_finetune_STL.pth")