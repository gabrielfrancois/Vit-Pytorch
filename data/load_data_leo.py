# Data loading for CIFAR 10

# Requirements
import os
import pickle 
import numpy as np 
from typing import Tuple

import torch
from torch.utiles.data import Dataset, DataLoader, random_split #maybe we should avoid random split to make it more reproducible
from torchvision import transforms as T


# main code to define the data objects

class CIFAR10Datasets(Dataset): #inherit Dataset object properties from PyTorch
    """Custom dataset for CIFAR-10"""

    def __init__(self, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = torch.tensor(X, dtype=torch.uint8)
        self.y = torch.tensor(y, dtype=torch.long) #int
        self.transform = transform #for modularity

    def __len__(self):
        return len(self.y) #the smaller object for more efficient memory usage

    def __getitem__(self, idx): #for easy access
        img = self.X[idx]
        label = self.y[idx]
        if self.transform: #if not None
            img = self.transform(img)
        return img, label
    

def unpickle(file_path: str) -> dict:
    """Utility function to unpickle CIFAR-10 data"""
    with open(file_path, "rb") as f: #read binary
        return pickle.load(f, encoding="bytes")
    

