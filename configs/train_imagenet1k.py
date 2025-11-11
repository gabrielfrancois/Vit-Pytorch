# train_imagenet1k.py
d_model = 128        # plus gros que CIFAR (patches + classes)
n_classes = 1000
img_size = (128, 128)
patch_size = (16, 16)
n_channels = 3
n_heads = 8
n_layers = 12
batch_size = 64          # Risque de ne pas tenir si les batchs sont trop gros
epochs = 3
alpha = 0.0005
weight_decay = 1e-4
