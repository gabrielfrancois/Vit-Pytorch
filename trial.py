# faire une version jouet avec le model cifar qui a été loadé
# télécharger le dataset (attention GITIGNORE)
# resize images at la taille de cifar !

import torch
from torch import nn
import torchvision
from torchvision import transforms
from torchvision.datasets import STL10
from torch.utils.data import Dataset, DataLoader
from models.finetune import LORA
from models.vision_transformer import VisionTransformer


transform = transforms.Compose([
    transforms.Resize((32,32)), # CIFAR dimensions
    transforms.ToTensor()
])

# STL-10 dataset
# https://cs.stanford.edu/~acoates/stl10/ (images acquired from ImageNet)
# "It is inspired by the CIFAR-10 dataset but with some modifications. In particular, each class 
# has fewer labeled training examples than in CIFAR-10, but a very large set of unlabeled examples 
# is provided to learn image models prior to supervised training. 
# images to fintune, from STL-10, dim = 96*96, 10 classes --> need to change the architecture 
# (final layer)

train = STL10(root="./data", split="train", download=True, transform=transform) #2.64G
test  = STL10(root="./data", split="test", download=True, transform=transform)


