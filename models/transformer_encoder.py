from torch import nn as nn
from .multi_head_attention import MultiHeadAttention

# r_mlp correspond to the degre of expansion (and compression) of our MLP succeding to the multi head attention. Try to change this, but no longer too big :) 

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(d_model, n_heads)

        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(d_model)

        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp), # expansion
            nn.GELU(),
            nn.Linear(d_model*r_mlp, d_model) # compression to come back to d_model
        )

    def forward(self, x):
        # Residual Connection After Sub-Layer 1
        out = x + self.mha(self.ln1(x))
        # Residual Connection After Sub-Layer 2
        out = out + self.mlp(self.ln2(out))
        return out