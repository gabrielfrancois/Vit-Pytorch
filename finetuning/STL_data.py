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
    T.Resize((128,128)), # upscale to match pre-trained ViT on imagenet (128*128), STL is 96*96
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