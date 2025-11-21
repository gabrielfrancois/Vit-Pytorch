import torch
from torch import nn as nn
import torch.nn.functional as F
from .multi_head_attention import MultiHeadAttention
from .predictor_LG import PredictorLG

# r_mlp correspond to the degre of expansion (and compression) of our MLP succeding to the multi head attention. Try to change this, but no longer too big :) 

class DynamicTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4, has_predictor=False):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Sub-Layer 1 Normalization
        self.ln1 = nn.LayerNorm(d_model)

        # Multi-Head Attention
        self.mha = MultiHeadAttention(d_model, n_heads)

        # post-attention dropout
        self.dropout1 = nn.Dropout(0.1)

        # Sub-Layer 2 Normalization
        self.ln2 = nn.LayerNorm(d_model)

        # Multilayer Perception
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model*r_mlp), # expansion
            nn.GELU(),
            nn.Dropout(0.1), #regularisation
            nn.Linear(d_model*r_mlp, d_model), # compression to come back to d_model
            nn.Dropout(0.1)
        )

        # True ==> pruning 
        self.has_predictor = has_predictor
        if self.has_predictor:
            self.predictor = PredictorLG(embed_dim=d_model)

    def forward(self, x, policy):
        """
        input:
        ---------------
            - x: Features (B, N, C)
            - policy: Current binary policy (mask) (B, N) where 1=keep, 0=drop
        output:
        ---------------
            - x: Features (B, N, C)
            - policy: New binary policy (mask) (B, N)
            - pred_score: new proba of keeping each patch: (B, N)
        """
        # Standard Transformer Operations (Pass policy/mask to Attention)
        
        attn_out = self.mha(self.ln1(x), mask=policy) 
        x = x + self.dropout1(attn_out)
        x = x + self.mlp(self.ln2(x)) 

        # Dynamic Token Sparsification
        new_policy = policy
        pred_score = None

        if self.has_predictor:
            # Predict logits for dropping/keeping: Output (B, N, 2)
            pred_logits = self.predictor(x, policy) 
            
            # Extract "keep" probability (index 1) for the loss function, pred_logits are log-probs, so exp() to get the probabilities 
            pred_score = pred_logits.exp()[:, :, 1] 

            if self.training: # automatically set by torch 
                # TRAINING: Use Gumbel-Softmax to sample a binary mask differentiably, this allows gradients to flow back into the predictor
                hard_keep_decision = F.gumbel_softmax(pred_logits, tau=1, hard=True)[:, :, 1]
            else:
                # INFERENCE: Simple argmax or thresholding
                hard_keep_decision = torch.argmax(pred_logits, dim=-1)

            # Update the policy: 
            # If a token was already dropped (policy=0), it stays dropped.
            # If a token was kept (policy=1), it takes the new decision.
            new_policy = policy * hard_keep_decision # Formula: D_new = D_old * decision (Hadamard product)
        return x, new_policy, pred_score