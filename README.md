# Vit-Pytorch

Implement a visual recognition with vision transformers (Vit) using CIFAR-10 dataset (https://www.cs.toronto.edu/~kriz/cifar.html).
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
Here are the classes in the dataset, as well as 10 random images from each:
    - airplaine
    - automobile
    - bird
    - cat
    - deer
    - dog
    - frog
    - horse
    - ship
    - truck
The classes are completely mutually exclusive. There is no overlap between automobiles and trucks. "Automobile" includes sedans, SUVs, things of that sort. "Truck" includes only big trucks. Neither includes pickup trucks.

The goal is to implement a tiny ViT running on small devices (laptop, smartphone...) with small computacional capacities. 