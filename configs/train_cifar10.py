# setting up the hyperparametters

d_model = 32  # 12 before
n_classes = 100
img_size = (32,32)
patch_size = (4,4)
n_channels = 3 # before it was 1
n_heads = 4 
n_layers = 12
batch_size = 100
epochs = 3
alpha = 0.005

# dataset


data_dir = "/home/onyxia/work/Vit-Pytorch/data"



pruning_index = [4,7,10]
rho_final = 0.7
rho_init = 1
lambda_class = 1
lambda_kl = 1/2
lambda_ratio = 2
lambda_distill = 1/2

# finetuneing option
finetuning = 'LORA' # Enter: False, 'LORA'
rank = 4