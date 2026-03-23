import os
import numpy as np 
from typing import Tuple
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from torchvision.datasets import CIFAR100, CIFAR10

from configs.train_cifar10 import * 

def load_CIFAR(CIFAR: int) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Loads CIFAR-10 or CIFAR-100 data and returns train / validation / test dataloaders"""

    train_transform  = T.Compose([
        T.Resize((32, 32)),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=6),
        T.ToTensor(),
    ])

    test_transform = T.Compose([ # No data augmentation for validation/test!
        T.Resize((32, 32)),
        T.ToTensor(),
    ])

    if CIFAR == 10:
        dataset_path = "./data/raw/cifar10"
        os.makedirs(dataset_path, exist_ok=True)
        DatasetClass = CIFAR10
    elif CIFAR == 100:
        dataset_path = "./data/raw/cifar100"
        os.makedirs(dataset_path, exist_ok=True)
        DatasetClass = CIFAR100
    else:
        raise ValueError("CIFAR argument must be 10 or 100")

    full_train_dataset = DatasetClass(root=dataset_path, train=True, transform=train_transform, download=True)
    full_val_dataset   = DatasetClass(root=dataset_path, train=True, transform=test_transform, download=True)
    
    test_dataset = DatasetClass(root=dataset_path, train=False, transform=test_transform, download=True)

    # Split into train and val datasets cleanly
    val_ratio = 0.1
    num_train_samples = len(full_train_dataset)
    val_size = int(val_ratio * num_train_samples)
    train_size = num_train_samples - val_size

    indices = torch.randperm(num_train_samples, generator=torch.Generator().manual_seed(50)).tolist()

    # Create subsets using the specific indices and the correctly transformed base datasets
    train_dataset = Subset(full_train_dataset, indices[:train_size])
    val_dataset = Subset(full_val_dataset, indices[train_size:])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    print(f"Train: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def visualise(loader, output_dir):
    """ Visualise examples """
    os.makedirs(output_dir, exist_ok=True)

    images, labels = next(iter(loader)) 
    img = images[0]
    label = str(labels[0].item())
    print(f"Visualising label: {label}")

    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    plt.figure(figsize=(3, 3))
    plt.title(f"Label: {label}")
    plt.imshow(img_np)
    plt.axis("off")
    output_path = os.path.join(output_dir, f"sample_{label}.png") 
    plt.savefig(output_path)
    plt.close()
    
    print(f"Saved image to {output_path}")

if __name__ == '__main__':
    data_dir = "./data/images"
    train_loader, val_loader, test_loader = load_CIFAR(CIFAR=10)
    visualise(train_loader, output_dir=data_dir)