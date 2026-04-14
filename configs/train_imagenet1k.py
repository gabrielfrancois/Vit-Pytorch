# train_imagenet1k.py
d_model = 192
n_classes = 1000
img_size = (128, 128)
patch_size = (8, 8)
n_channels = 3
n_heads = 6
n_layers = 12
batch_size = 256
epochs = 200
alpha = 0.00015
layer_decay = 0.9

# dynamic_Vit parameters
pruning_index = [4,7,10]
rho = 0.7
rho_init = 1
steepness = 10.0
lambda_class = 1.0
lambda_kl = 1.0
lambda_ratio = 2.0
lambda_distill = 0.2

# Normalisation parameters
mean_norm_imagenet = [0.485, 0.456, 0.406]
std_norm_imagenet = [0.229, 0.224, 0.225]

# finetuneing option
finetuning = 'LORA' # Enter: False, 'LORA'
rank = 4

# REPA
lambda_repa = 2.0