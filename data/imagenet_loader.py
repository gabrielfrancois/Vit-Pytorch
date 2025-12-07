
import os
import logging
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as T

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_imagenet1k(
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 4,
    seed: int = 50,
    max_items_train: int = None,
    max_items_val: int = None,
    img_size: int = 128,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Charge le dataset ImageNet-1k (version 128x128 sur Hugging Face)
    et renvoie train, val, test loaders.
    """

    logger.info("Chargement du dataset ImageNet-1k (benjamin-paine/imagenet-1k-128x128)...")
    pin_memory = torch.cuda.is_available()

    train_transform = T.Compose(
        [
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds_train = load_dataset(
        "benjamin-paine/imagenet-1k-128x128",
        split=f"train[:{max_items_train}]" if max_items_train else "train",
    )
    ds_val = load_dataset(
        "benjamin-paine/imagenet-1k-128x128",
        split=f"validation[:{max_items_val}]" if max_items_val else "validation",
    )

    class HFDataset(Dataset):
        def __init__(self, hf_ds, transform):
            self.ds = hf_ds
            self.transform = transform

        def __len__(self) -> int:
            return len(self.ds)

        def __getitem__(self, idx: int):
            sample = self.ds[idx]
            img = sample["image"].convert("RGB")
            label = sample["label"]
            return self.transform(img), label

    train_dataset_full = HFDataset(ds_train, train_transform)
    test_dataset = HFDataset(ds_val, test_transform)

    val_size = int(val_ratio * len(train_dataset_full))
    train_size = len(train_dataset_full) - val_size

    train_dataset, val_dataset = random_split(
        train_dataset_full,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(
        f"ImageNet-1k chargé : train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader
