import os
from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms as T
from datasets import load_dataset
import numpy as np
import matplotlib.pyplot as plt

from configs.train_imagenet1k import *

class ImageNet1kDataset(Dataset):
    """
    Dataset wrapper for 'benjamin-paine/imagenet-1k-128x128' (Hugging Face)
    Compatible with both streaming and local loading.
    """

    def __init__(self, split: str = "train", transform=None, streaming: bool = False):
        self.ds = load_dataset(
            "benjamin-paine/imagenet-1k-128x128",
            split=split,
            streaming=streaming
        )
        self.transform = transform
        self.streaming = streaming

        if not streaming:
            # Preload in-memory structure for indexing
            self.ds = self.ds.with_format("torch")

    def __len__(self):
        if self.streaming:
            raise TypeError("Streaming mode: length not available.")
        return len(self.ds)

    def __getitem__(self, idx):
        if self.streaming:
            # Hugging Face streaming yields iterators, so idx is ignored
            raise TypeError("Cannot index a streaming dataset directly.")
        sample = self.ds[idx]
        img, label = sample["image"], sample["label"]
        if self.transform:
            img = self.transform(img)
        return img, label


def load_imagenet1k(
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 50,
    streaming: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns train, validation and test dataloaders for ImageNet-1k (128x128).
    """

    # Transforms
    train_transform = T.Compose([
    T.RandomResizedCrop(128, scale=(0.7, 1.0)),
    T.RandomHorizontalFlip(),
    T.ConvertImageDtype(torch.float32),  # conversion float32
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]),
    ])

    test_transform = T.Compose([
        T.Resize((128, 128)),
        T.ConvertImageDtype(torch.float32),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])


    # Streaming mode: no splitting possible
    if streaming:
        train_ds = ImageNet1kDataset(split="train", transform=train_transform, streaming=True)
        val_ds = ImageNet1kDataset(split="validation", transform=test_transform, streaming=True)

        train_loader = DataLoader(train_ds, batch_size=batch_size)
        val_loader = DataLoader(val_ds, batch_size=batch_size)
        test_loader = val_loader  # same as val for now
        print("Streaming mode: no explicit train/val split.")
        return train_loader, val_loader, test_loader

    # Local (non-streaming) mode
    full_train_dataset = ImageNet1kDataset(split="train", transform=train_transform)
    test_dataset = ImageNet1kDataset(split="validation", transform=test_transform)

    # Split train into train/val
    val_size = int(val_ratio * len(full_train_dataset))
    train_size = len(full_train_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_train_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    return train_loader, val_loader, test_loader


def visualise_sample(loader, output_dir: str = "samples"):
    """
    Saves one sample image from a DataLoader for sanity check.
    """

    os.makedirs(output_dir, exist_ok=True)

    images, labels = next(iter(loader))
    img = images[0].cpu().permute(1, 2, 0).numpy()
    label = str(labels[0].item())

    # Denormalize for correct visualization
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    plt.figure(figsize=(3, 3))
    plt.imshow(img)
    plt.title(f"Label: {label}")
    plt.axis("off")
    out_path = os.path.join(output_dir, f"imagenet_sample_{label}.png")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    print(f"Saved sample to {out_path}")


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_imagenet1k(batch_size=32)
    visualise_sample(train_loader)
