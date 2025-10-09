# Importing necessary libraries
import os
import random
import numpy as np
from tqdm.notebook import tqdm, trange
import torch
from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from torchinfo import summary
import torchvision
from PIL import Image
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt


# Load data
def unpickle(file : str):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
data_batch_1 = unpickle("/home/onyxia/work/Vit-Pytorch/data/data_batch_1")
data_batch_2 = unpickle("/home/onyxia/work/Vit-Pytorch/data/data_batch_2")
data_batch_3 = unpickle("/home/onyxia/work/Vit-Pytorch/data/data_batch_3")
data_batch_4 = unpickle("/home/onyxia/work/Vit-Pytorch/data/data_batch_4")
data_batch_5 = unpickle("/home/onyxia/work/Vit-Pytorch/data/data_batch_5")


# Each of the batch files contains a dictionary with the following elements:
# data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the i-th image in the array data.

train_transform = T.Compose([
    T.Resize((32, 32)),
    T.RandomHorizontalFlip(),
    T.RandAugment(num_ops=2, magnitude=6),
    T.ToTensor()
])

# Loading training data directly using ImageFolder
train_dataloader = DataLoader(data_batch_1, batch_size=64, shuffle=True, num_workers=8)
print(train_dataloader)