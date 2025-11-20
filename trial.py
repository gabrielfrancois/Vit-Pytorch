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
from models.finetune import LORA
from configs.train_cifar10 import *



# Remark : we have to find a way to handle data classes number imbalance for other datasets (changing the last layer)


transform = T.Compose([
    T.Resize((32,32)), # CIFAR dimensions
    T.ToTensor()
])



# we consider only the training dataset
class STL10Dataset(Dataset):
    """Simple wrapper around torchvision STL10 with optional custom transforms."""

    def __init__(self, split, transform=None):
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


def load_STL10(data_dir, batch_size):
    train_dataset = STL10Dataset(split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    print(f"STL10 train size: {len(train_dataset)}")

    return train_loader


# Function to inject LORA in all linear layers

def inject_lora(model, rank, alpha=1):
    
    """ wrap every nn.Linear inside the model with LORA finetuning """
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear): 
            # very useful trick --> it will replace all the linear layers of the model 
            # by LORA layers, no need to rewrite all the models architecture code
            lora_layer = LORA(module, rank, alpha).to(device)

            setattr(model, name, lora_layer)
        else:
            inject_lora(module, rank, alpha) # recursivity
    return model


def load_pretrained(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=False)
    print("loaded")


def propor_params(model):
    """
    prints the proportion of trainable weights
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total
    
    print(f"Proporition of trainable weights: {trainable} / {total} = ({percentage:.2f}%)")
    #return trainable # in case the freezing from finetune.py didnt work


def train_one_epoch(model, loader, potimizer, criterion, device):
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    
    loop = tqdm(loader, desc='Training')
    
    for imgs, labels in loop:
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

    model = VisionTransformer(d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers).to(device)
    propor_params(model)

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