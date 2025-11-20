import torch
from torch import nn as nn 

from .dynamic_transformer_encoder import DynamicTransformerEncoder
from .predictor_LG import PredictorLG
from .patch_embed import PatchEmbedding
from .positional_embeeding import PositionalEmbeeding
from .transformer_encoder import TransformerEncoder
from helper_function.print import *


class DynamicVisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index):
        super().__init__()
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads. Actually, I think we could relax this assumption, we'll need to adapt the code though..."

        self.d_model = d_model # Dimensionality of model
        self.n_classes = n_classes # Number of classes
        self.img_size = img_size # Image size
        self.patch_size = patch_size # Patch size
        self.n_channels = n_channels # Number of channels
        self.n_heads = n_heads # Number of attention heads
        self.pruning_index = pruning_index # index where patch are prunned

        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1]) # The number of patches can be found by dividing the product of the height and width of the input image by the product of the height and width of the patch size.
        self.max_seq_length = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEmbeeding(self.d_model, self.max_seq_length)
        self.transformer_encoders = nn.ModuleList() # To get iterable (sequential isn't yet possible due to the masks).
        self.dropout = nn.Dropout(0.1) #regularisation

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
        )

        for i in range(n_layers):
            if i in pruning_index:
                has_pred = True
            else:
                has_pred = False
            self.transformer_encoders.append(
                DynamicTransformerEncoder(self.d_model, self.n_heads, has_predictor=has_pred)
            )
    
    def forward(self, images):
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Initialize Policy: Keep all tokens  
        B, N, C = x.shape
        current_policy = torch.ones(B, N, device=x.device)

        # To store prediction scores for calculating the Sparsity Loss later
        all_pred_scores = []
        all_masks = []       # binary decisions (D)

        # Manual Loop over layers
        for layer in self.transformer_encoders:
            x, current_policy, pred_score = layer(x, current_policy)
            if pred_score is not None:
                all_pred_scores.append(pred_score)
                # Force CLS token (index 0) to always be 1. If we don't do this, the predictor might "prune" the CLS token
                # DynamicViT usually relies on the predictor learning to keep it, thus, we'll force it by: current_policy[:, 0] = 1
                current_policy[:, 0] = 1.0
                all_masks.append(current_policy)

        # Final Classifier (Only use the CLS token)
        cls_token = x[:, 0]
        logits = self.classifier(cls_token)

        # Store the t_i's of the paper
        student_feats = x

        return logits, student_feats, all_masks, all_pred_scores