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

"""
Each of the batch files contains a dictionary with the following elements:
data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. 
The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the i-th image in the array data.
"""

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
data_batch_5 = unpickle("/home/onyxia/work/Vit-Pytorch/data/data_batch_5") # data batches are dict with : dict_keys([b'batch_label', b'labels', b'data', b'filenames'])

# unificate all train batches
batches = [data_batch_1, data_batch_2, data_batch_3, data_batch_4, data_batch_5]

X_train = np.concatenate([b[b'data'] for b in batches])
y_train = np.concatenate([b[b'labels'] for b in batches])
print(X_train.shape)

# Prepare the train dataset
data_test = unpickle("/home/onyxia/work/Vit-Pytorch/data/test_batch")
X_test = data_test[b'data']
y_test = np.array(data_test[b'labels'])

# Make again images in 3D
X_train = X_train.reshape(-1, 3, 32, 32)
X_test = X_test.reshape(-1, 3, 32, 32)

train_transform = T.Compose([
    T.ToPILImage(), # convert tansors into PIL format for the following operation
    T.Resize((32, 32)),# transforms each image in the batch to a fixed size of 32x32 pixels (useless here because there are already in 32x32))
    T.RandomHorizontalFlip(), # Applies a random horizontal flip to each image with probability 0.5 (default behavior) to increase data diversity
    T.RandAugment(num_ops=2, magnitude=6), # add to each channel a random 2 noise of 6 deviation (Example: If an RGB pixel has value [255,0,126] --> [260, -4, 144])
    T.ToTensor() # Converts the transformed image tensor into a PyTorch tensor 
])



