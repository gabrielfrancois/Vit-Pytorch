import torch
from torch import nn as nn
import torch.nn.functional as F
from .multi_head_attention import MultiHeadAttention
from .predictor_LG import PredictorLG
from helper_function.print import *

# r_mlp correspond to the degre of expansion (and compression) of our MLP succeding to the multi head attention. Try to change this, but no longer too big :) 

class DynamicTransformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, r_mlp=4, has_predictor=False, keep_ratio=0.7):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.keep_ratio = keep_ratio

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
        Dynamic ViT logic: prune useless token to gain speedness, ONLY in the INFERENCE, we litterally cut the patch 
        (over the dimension N) to improove computational speedness. 
        During TRAINING, we train the mask and keep all patches to keep GPU computationnal advantages. 
        input:
        ---------------
            - x: Features (B, N, C)
            - policy: Current binary policy (mask) (B, N) where 1=keep, 0=drop
        output:
        ---------------
            - x: Features (B, N, C)
            - policy: New binary policy (mask) (B, N)
            - pred_score: new proba of keeping each patch: (B, N)
        Note: We predict importance BEFORE doing MHA/MLP.
        """

        new_policy = policy
        pred_score = None

        if self.has_predictor:
            # Predict scores
            pred_logits = self.predictor(x, policy) # (B, N, 2)
            pred_score = pred_logits.exp()[:, :, 1] # (B, N) Prob of keeping, computed from the begining to gain efficiency

            if self.training: # automatically set by torch 
                 # TRAINING: Use Gumbel-Softmax to sample a binary mask differentiably, this allows gradients to flow back into the predictor
                hard_keep_decision = F.gumbel_softmax(pred_logits, tau=1, hard=True)[:, :, 1]
                # FORCE CLS TOKEN: Always keep index 0 during training mask update
                new_policy[:, 0] = 1.0 

                # Calculate attention with MASK, x is still (B, N, C) to keep GPU computational advantages
                attn_out = self.mha(self.ln1(x), mask=new_policy)
                x = x + self.dropout1(attn_out)
                x = x + self.mlp(self.ln2(x))

                return x, new_policy, pred_score

            else: # INFERENCE, hard pruning
                # Here, N will definitely reduce to N' <= N
                B, N, C = x.shape
                keep = int(N*self.keep_ratio)
                if keep < 1:
                    keep = 1
                # Force CLS score to infinity (1e9 to avoid nan) so it is ALWAYS selected in Top-K
                pred_score[:, 0] = 1e9
                _, keep_indices = torch.topk(pred_score, k=keep, dim=1) # dim = 1 to compute over N ! 
                # Sorted by position (0, 5, 12, 99) to maintain the sequence flow (top-left to bottom-right)
                keep_indices, _ = torch.sort(keep_indices, dim=1)

                # Physical pruning, actual speedup !
                batch_indices = torch.arange(B).unsqueeze(-1).expand(-1, keep).to(x.device)  # Create batch indices: [[0, 0...], [1, 1...]...]
                x = x[batch_indices, keep_indices] 
                new_policy = policy[batch_indices, keep_indices]

                # x is now (B, N', C) (with N' <= N). No mask needed.
                attn_out = self.mha(self.ln1(x), mask=None)
                x = x + self.dropout1(attn_out)
                x = x + self.mlp(self.ln2(x))
                return x, new_policy, pred_score, keep_indices

        else: # no predictor, process whatever x we received (could be full or already pruned)
            attn_out = self.mha(self.ln1(x), mask=None) # no mask 
            x = x + self.dropout1(attn_out)
            x = x + self.mlp(self.ln2(x)) 
            return x, policy, None, None