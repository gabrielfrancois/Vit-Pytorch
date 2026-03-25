import torch
from torch import nn

class LORA(nn.Module):
    """ Class to fine-tune the model using Low Raw Adaptation"""

    def __init__(self, original_layer: nn.Linear, rank: int, alpha: int = 1):
        #r = 4 for test

        super().__init__()

        self.original_layer = original_layer
        in_dim = original_layer.in_features
        out_dim = original_layer.out_features

        self.to(original_layer.weight.device) #very important to be on the same device!
        
        self.r = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LORA matrices
        self.A = nn.Linear(in_dim, rank, bias=False) # A is size dim_model*r
        self.B = nn.Linear(rank, out_dim, bias=False) # B is size r*dim_model

        # initialisation
        nn.init.kaiming_uniform_(self.A.weight) #kaiming scheme recommended
        nn.init.zeros_(self.B.weight)

        # Freeze original model's weight
        for param in self.original_layer.parameters(): 
            param.requires_grad = False

    def forward(self, x):
        result = self.original_layer(x) #original layer, frozen
        lora_update = self.B(self.A(x))

        return result + lora_update * self.scaling

# Function to inject LORA in all linear layers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def inject_lora(model, rank, alpha=1):
    
    """ wrap every nn.Linear inside the model with LORA finetuning """
    
    for name, module in model.named_children():
        if isinstance(module, nn.Linear): 
            # very useful trick --> it will replace all the linear layers of the model 
            # by LORA layers, no need to rewrite all the models architecture code
            lora_layer = LORA(module, rank, alpha).to(device)

            setattr(model, name, lora_layer)
        else:
            inject_lora(module, rank, alpha) # recursivity
    return model


def propor_params(model):
    """
    prints the proportion of trainable weights
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    percentage = 100 * trainable / total
    
    print(f"Proporition of trainable weights: {trainable} / {total} = ({percentage:.2f}%)")
    #return trainable # in case the freezing from finetune.py didnt work

if __name__ == "__main__":

    # quick test
    
    original = nn.Linear(10, 10)
    lora_layer = LORA(original, rank=2)
    x = torch.randn(1, 10)
    output = lora_layer(x)
    print("Output shape:", output.shape)
