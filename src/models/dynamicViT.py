import torch
from torch import nn as nn 

from src.models.dynamic_transformer_encoder import DynamicTransformerEncoder
from src.models.predictor_LG import PredictorLG
from src.models.patch_embed import PatchEmbedding
from .positional_embedding import PositionalEmbedding  
from .transformer_encoder import TransformerEncoder
from helper_function.print import *

class DynamicVisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, pruning_index, base_keep_rate=0.7):
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

        # Calculate number of patches
        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1]) 
        self.max_seq_length = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEmbedding(self.d_model, self.max_seq_length)
        self.transformer_encoders = nn.ModuleList() # To get iterable (sequential isn't yet possible due to the masks).
        self.dropout = nn.Dropout(0.1) # Regularisation

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
        )

        # Map layer index to the specific ratio it should enforce
        layer_to_ratio = {}
        current_cumulative_ratio = 1.0
        
        # Create a sorted list to iterate in order
        sorted_pruning_locs = sorted(pruning_index)
        
        for loc in sorted_pruning_locs:
            current_cumulative_ratio *= base_keep_rate # exact rate of the paper
            layer_to_ratio[loc] = current_cumulative_ratio

        for i in range(n_layers):
            if i in pruning_index:
                has_pred = True
                ratio_for_this_layer = layer_to_ratio[i] # Get the calculated ratio for this specific layer
            else:
                has_pred = False
                ratio_for_this_layer = 1.0 # No pruning happens here anyway
            self.transformer_encoders.append(
                DynamicTransformerEncoder(
                    self.d_model, 
                    self.n_heads, 
                    has_predictor=has_pred, 
                    keep_ratio=ratio_for_this_layer # Pass the specific ratio
                )
            )
    
    def forward(self, images):
        """
        input :
        ------------------
            - images: (batch size, image channel, image height, image width)
        output :
        ------------------
            - logits: tensor: (B, nb_classes) (for instance, in cifar-10 = 10)
            - student_feats: tensor: (B, N, d_model) (recall that d_model is actually d_model + 1, due to the cls token added)
            - all_masks: list of mask = [(B,N), ..., (B,N)]
            - all_pred_scores: list of proba of keeping each mask = [(B,N), ... , (B,N)]
        """
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
            x, current_policy, pred_score, keep_indices = layer(x, current_policy)
            if pred_score is not None:
                all_pred_scores.append(pred_score)
                # Force CLS token (index 0) to always be 1. If we don't do this, the predictor might "prune" the CLS token
                # DynamicViT usually relies on the predictor learning to keep it => current_policy[:, 0] = 1
                current_policy[:, 0] = 1.0
                all_masks.append(current_policy)

        # Final Classifier (Only use the CLS token)
        cls_token = x[:, 0]
        logits = self.classifier(cls_token)

        student_feats = x # Store the t_i's of the paper

        return logits, student_feats, all_masks, all_pred_scores