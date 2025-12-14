import torch 
from torch import nn as nn 

class AttentionHead(nn.Module):
    def __init__(self, d_model, head_size):
        super().__init__()
        self.head_size = head_size
        self.query = nn.Linear(d_model, head_size)
        self.key = nn.Linear(d_model, head_size)
        self.value = nn.Linear(d_model, head_size)
        self.dropout = nn.Dropout(0.1) # regularisation

    def forward(self, x, mask=None):
        """
        input:
        ---------------
            - x : Features :(batch_size, nb_patch_by_images, d_model) = (B, N, d_model)
            - mask : (B, N)
        output:
        ---------------
            - out : new attention head: (B, N, head_size)
        """
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x) # (B, N, head_size)

        # Dot Product of Queries and Keys
        attention = Q @ K.transpose(-2,-1) # (B, N, head_size) @ (B, head_size, N) -> (B, N, N) 

        # Scaling
        attention = attention / (self.head_size ** 0.5)

        # Apply attention masking if provided
        if mask is not None:
            # mask shape: (B, N) (Batch_size, nb_of patch token before pruning) where 0 means 'kept', 1 means 'pruned'.
            # A token is pruned by blocking its attention TO ALL other tokens.
            # So, for a pruned token at position i, all attention scores A[:, i, :] should be -inf.
            mask = mask.unsqueeze(1)  # (B, 1, N)
            # Assuming mask is 0 for pruned tokens and 1 for kept token, we'd set scores to -inf where mask is 0.
            # Use -1e9 instead of -inf to avoid  unexpected Nan...
            attention = attention.masked_fill(mask == 0, -1e9) # broadcasting --> (B, N, N)

        attention = torch.softmax(attention, dim=-1)
        attention = self.dropout(attention)
        out = attention @ V  # (B, N, head_size)

        return out