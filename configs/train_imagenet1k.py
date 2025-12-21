# train_imagenet1k.py
d_model = 96
n_classes = 1000
img_size = (128, 128)
patch_size = (16, 16)
n_channels = 3
n_heads = 4
n_layers = 12
batch_size = 256
epochs = 50
alpha = 0.005


# dataset

data_dir = "/home/onyxia/work/Vit-Pytorch/data"

# checkpoints and plots directories.

plot_dir = "plots/plots_test"

checkpoint_dir = "checkpoints/checkpoint_test"

# other

resume = True  # permet soit de continuer à partir du checkpoint enregistré soit de repartir de 0.

pruning_index = [4,7,10]
rho = 0.98
lambda_class = 0.001
lambda_kl = 1
lambda_ratio = 1/2
lambda_distill = 0.0001