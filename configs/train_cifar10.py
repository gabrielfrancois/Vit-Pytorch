# setting up the hyperparametters

d_model = 32  # 12 before
n_classes = 10
img_size = (32, 32)
patch_size = (4, 4)
n_channels = 3  # before it was 1
n_heads = 4
n_layers = 12
batch_size = 100
epochs = 100
alpha = 0.005

# dataset


data_dir = "/home/onyxia/work/Vit-Pytorch/data"

# plots etc.

plot_dir = "plots/plots_CIFAR"

checkpoint_dir = "checkpoints"

# other

resume = True  # permet soit de continuer à partir du checkpoint enregistré soit de repartir de 0

pruning_index = [4, 7, 10]
rho = 0.7
lambda_kl = 1 / 2
lambda_ratio = 2
lambda_distill = 1 / 2
