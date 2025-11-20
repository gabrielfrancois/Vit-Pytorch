# setting up the hyperparametters

d_model = 12 
n_classes = 10
img_size = (32,32)
patch_size = (4,4)
n_channels = 3 # before it was 1
n_heads = 3
n_layers = 12
batch_size = 100
epochs = 30
alpha = 0.005

# finetuneing option
finetuning = 'LORA' # Enter: False, 'LORA', or 'QLORA'
rank = 4