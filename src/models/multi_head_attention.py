import torch
from torch import nn as nn

from .attention_head import AttentionHead


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.head_size = d_model // n_heads

        self.W_o = nn.Linear(d_model, d_model)

        self.heads = nn.ModuleList([AttentionHead(d_model, self.head_size) for _ in range(n_heads)])

    def forward(self, x, mask=None):
        """
        input:
        ---------------------
            - x: Features (B, N, C)
            - mask: (B, N) where 1=keep, 0=drop
        output:
        ---------------------
            - out: New features (B, N, C) 
        Note: in the case of dynamic ViT (the student) N actually may not be N if it pass through the mask, it could be an integer N' <= N 
        Only during INFERENCE we've N', during training, it would be always N !
        """
        out = torch.cat([head(x, mask=mask) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out 
