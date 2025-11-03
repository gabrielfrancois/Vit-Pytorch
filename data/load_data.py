# Data loading for CIFAR 10

# Requirements
import os
import pickle 
import numpy as np 
from typing import Tuple
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, random_split #maybe we should avoid random split to make it more reproducible
from torchvision import transforms as T
from torchvision.datasets import CIFAR100

from configs.train_cifar10 import *

# main code to define the data objects

class CIFAR_Datasets(Dataset): # inherit Dataset object properties from PyTorch
    """Custom dataset for CIFAR-10 or CIFAR-100"""

    def __init__(self, CIFAR: int, X: np.ndarray, y: np.ndarray, transform=None):
        self.X = torch.tensor(X, dtype=torch.uint8)
        self.y = torch.tensor(y, dtype=torch.long) # int
        self.CIFAR = CIFAR
        self.transform = transform # for modularity

    def __len__(self):
        return len(self.y) # the smaller object for more efficient memory usage

    def __getitem__(self, idx): # for easy access
        img = self.X[idx]
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, label
    

#utility function
def unpickle(file_path: str) -> dict:
    """Utility function to unpickle CIFAR-10 data"""
    with open(file_path, "rb") as f: 
        return pickle.load(f, encoding="bytes")

    
def load_CIFAR(data_dir: str, CIFAR: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Loads CIFAR-10 or CIFAR-100 data and returns train / validation / test datasets"""

    train_transform  = T.Compose([
        T.Resize((32, 32)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=6),
        T.ToTensor(),
        ])

    test_transform = T.Compose([ # absolutely no data augmentation here!
        T.Resize((32, 32)),
        T.ToTensor(),
        ])

    if CIFAR == 10:
        # Train data
        train_batches = [unpickle(os.path.join(data_dir, f"data_batch_{i}")) for i in range(1,5+1)]
        X_train = np.concatenate([b[b'data'] for b in train_batches]) 
        y_train = np.concatenate([b[b'labels'] for b in train_batches])

        # Test data
        test_batch = unpickle(os.path.join(data_dir, "test_batch"))
        X_test = test_batch[b'data']
        y_test = np.array(test_batch[b'labels'])

        # Reshape to (N, 3, 32, 32)
        X_train = X_train.reshape(-1, 3, 32, 32)
        X_test = X_test.reshape(-1, 3, 32, 32)

        # Refine transforms according to Gabriel François

        #build datasets
        train_transform = T.Compose([T.ToPILImage(), train_transform]) #necessary for CIFAR 10
        test_transform = T.Compose([T.ToPILImage(), test_transform])
        full_train_dataset = CIFAR_Datasets(CIFAR = 10, X = X_train, y = y_train, transform=train_transform)
        test_dataset = CIFAR_Datasets(CIFAR = 10, X = X_test, y = y_test, transform=test_transform)
    
    else: #CIFAR-100
        full_train_dataset = CIFAR100(root=data_dir, train=True, transform=train_transform, download=True)
        test_dataset = CIFAR100(root=data_dir, train=False, transform=test_transform, download=True)

    
    # split full_train_dataset into train and val datasets
    val_ratio = 0.1 #10% of the data will be used for validation
    val_size = int(val_ratio * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator = torch.Generator().manual_seed(42) #reproducibility
    )

    # Create loaders
    # maybe adjust the params later!
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True, num_workers = 2, pin_memory = True)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, num_workers = 2, pin_memory = True)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = 2, pin_memory = True)

    print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader

def visualise(loader, output_dir='images'):
    """ Visualise examples """

    os.makedirs(output_dir, exist_ok=True)

    images, labels = next(iter(loader)) # take a batch of images

    # Here we take the first image but could be modify for more
    img = images[0]
    label = str(labels[0].item())

    # convert to numpy for plotting
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    # Plot
    plt.figure(figsize=(3, 3))
    plt.title(f"Label: {label}")
    plt.imshow(img_np)
    plt.axis("off")
    output_path = os.path.join(output_dir, f"sample_{label}.png") #pdf for vectorised images
    plt.savefig(output_path)
    plt.close()
    
    print('finished')




if __name__ == '__main__':
    data_dir = "/home/onyxia/work/Vit-Pytorch/data"
    #train_loader_10, val_loader_10, test_loader_10 = load_CIFAR(CIFAR = 10, data_dir = data_dir)
    #visualise(train_loader_10)

    train_loader_100, val_loader_100, test_loader_100 = load_CIFAR(CIFAR=100, data_dir = data_dir)
    visualise(train_loader_100)
