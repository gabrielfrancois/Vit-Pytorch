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

if __name__ == "__main__":

    # quick test
    
    original = nn.Linear(10, 10)
    lora_layer = LORA(original, rank=2)
    x = torch.randn(1, 10)
    output = lora_layer(x)
    print("Output shape:", output.shape)
