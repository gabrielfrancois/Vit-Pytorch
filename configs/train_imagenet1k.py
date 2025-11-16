# train_imagenet1k.py
d_model = 128        
n_classes = 1000
img_size = (128, 128)
patch_size = (16, 16)
n_channels = 3
n_heads = 8
n_layers = 3 
batch_size = 256          
epochs = 6
alpha = 0.0005
weight_decay = 1e-4

# dataset 

data_dir = "/home/onyxia/work/Vit-Pytorch/data"

# plots etc.

plot_dir = "plots_test"

#other

resume = True  # permet soit de continuer à partir du checkpoint enregistré soit de repartir de 0