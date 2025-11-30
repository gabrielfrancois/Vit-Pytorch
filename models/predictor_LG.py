"""
predictor_LG (LG for Local-Global) compute the importance score of any tokens, the model select then the top_k most important tokens to keep.
predictor_LG return the policy of keeping or not each token which'll be computed by un grumberl softmax/argmax in the dynamic transformer encoder.
We state for the sake of simplicity and for match with the paper C = d_model (embeed_dim) here, (recall B = batch_size, N = nb of patch)
The input tokens are first procced in a sequence of layer norm and linear layer + MLP
Layer norm apply for a specific token vector x of size d_model = C the formula here: https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
This formula is computed on accross the last dimension C. The input tokens x comes from previous transformer block.
Depending on the depth of the network, the magnitude of the values in x could vary significantly.
After, we split the output of in_conv (for input convolutional block) into 2 blocks: local and global.
Local :
    - keep (arbitrarly) C//2 first tokens for it
    - just apply an MLP on it  (done in "in_conv")
Global:
    - more complicated function : Agg(MLP(x), policy)
and return the new policy to compute the net mask.
"""

import torch
from torch import nn as nn


class PredictorLG(nn.Module):
    """
    Lightweight module to predict token importance scores.
    """

    def __init__(self, embed_dim=32):
        super().__init__()
        # Local modeling: a small MLP to process token features
        self.in_conv = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, embed_dim), nn.GELU()
        )
        # Output layer: predicts a 2D vector (drop/keep probabilities) for each token
        self.out_conv = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),  # 2 outputs for (drop, keep)
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, x, policy):
        """
        input :
            - x : Features of size (B, N, C) (input tokens)
            - policy: (B, N) current mask, 1 for kept tokens, 0 for pruned
        output :
            -  new policy: D : (B, N, 2)
        Note that because of the mask, only on TRAINING, x features could be of size (B, N', C), with N' <= N
        This is all the point of the dynamic ViT: reduce the dimension of patch_size!
        """

        x = self.in_conv(x)  # (B, N, C)
        B, N, C = x.size()
        # Split features into local and global parts
        local_x = x[:, :, : C // 2]  # (B, N, C//2) for local info (C' = C//2)

        # Global pooling over kept tokens only, using the policy mask
        epsilon = 1e-6  # Could change
        policy_sum = torch.sum(policy, dim=1, keepdim=True) + epsilon  # (B,1)
        masked_x = x[:, :, C // 2 :] * policy.unsqueeze(
            -1
        )  # (B, N, C//2) (Hadamar product)
        sum_x = masked_x.sum(dim=1, keepdim=True)  # (B,1,C//2)
        policy_sum = torch.sum(policy, dim=1, keepdim=True) + epsilon  # (B, 1)
        global_x = sum_x / policy_sum.unsqueeze(-1)  # (B, 1, C//2)

        # Concatenate local and global features
        x = torch.cat([local_x, global_x.expand(B, N, C // 2)], dim=-1)  # (B, N, C)

        return self.out_conv(x)  # (B, N, 2)
        # These two dimensions correspond to the log-probabilities of the two possible actions for each token: dropping or keeping.
