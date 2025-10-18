import torch
from torch import nn as nn
import numpy as np


class PositionalEmbeeding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model)) # Classification Token added for every image in the batch

        # Creating positional encoding to keep hte informations about the position.
        pe = torch.zeros(max_seq_length, d_model)
        for pos in range(max_seq_length):
            for i in range(d_model):
                if i % 2 == 0:
                    pe[pos][i] = np.sin(pos/(10000 ** (i/d_model)))
                else:
                    pe[pos][i] = np.cos(pos/(10000 ** ((i-1)/d_model)))
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # Expand to have class token for every image in batch
        tokens_batch = self.cls_token.expand(x.size()[0], -1, -1)

        # Adding class tokens to the beginning of each embedding
        x = torch.cat((tokens_batch,x), dim=1)

        # Add positional encoding to embeddings
        x = x + self.pe
        return x

if __name__ == "__main__":
    test = PositionalEmbeeding(d_model = 4, max_seq_length = 5)
    print(test.cls_token)
