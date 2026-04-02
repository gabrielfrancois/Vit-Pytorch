# train_imagenet1k.py
d_model = 96
n_classes = 1000
img_size = (128, 128)
patch_size = (16, 16)
n_channels = 3
n_heads = 4
n_layers = 12
batch_size = 256
epochs = 100
alpha = 0.001

# dynamic_Vit parameters
pruning_index = [4,7,10]
rho = 0.7
rho_init = 1
steepness = 10.0
lambda_class = 1
lambda_kl = 1
lambda_ratio = (1/2)
lambda_distill = 2

# Normalisation parameters
mean_norm_imagenet = [0.485, 0.456, 0.406]
std_norm_imagenet = [0.229, 0.224, 0.225]

# finetuneing option
finetuning = 'LORA' # Enter: False, 'LORA'
rank = 4

# REPA
lambda_repa = 1.0