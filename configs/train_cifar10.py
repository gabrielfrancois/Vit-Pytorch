# setting up the hyperparameters
d_model = 15    # Changed from 32 back to 12
n_classes = 10  # Changed from 100 back to 10 
img_size = (32, 32)
patch_size = (4, 4)
n_channels = 3 
n_heads = 3     # Changed from 4 to 3 (Checkpoint has 12 dims / 4 per head = 3 heads)
n_layers = 12
batch_size = 100
epochs = 100
alpha = 0.005

# dataset
data_dir = "data/raw/cifar10"

# dynamic_Vit parameters
pruning_index = [4,7,10]
rho = 0.7
rho_init = 1
lambda_class = 1
lambda_kl = 1/2
lambda_ratio = 2
lambda_distill = (1/2)

# finetuneing option
finetuning = 'LORA' # Enter: False, 'LORA'
rank = 4