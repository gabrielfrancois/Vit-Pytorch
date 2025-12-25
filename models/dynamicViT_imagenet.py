import torch
from torch import nn as nn 

from .dynamic_transformer_encoder import DynamicTransformerEncoder
from .predictor_LG import PredictorLG
from .patch_embed import PatchEmbedding
from .positional_embedding import PositionalEmbedding  
from .transformer_encoder import TransformerEncoder
from helper_function.print import *
from typing import List, Tuple, Dict, Optional

class DynamicVisionTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_classes: int,
        img_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        n_channels: int,
        n_heads: int,
        n_layers: int,
        pruning_index: List[int],
        rho: float
    ) -> None:
        """
        Initialize a Dynamic Vision Transformer (DynamicViT) model.

        Args:
            d_model: Dimensionality of the model (embedding size).
            n_classes: Number of output classes.
            img_size: Tuple (H, W) of input image size.
            patch_size: Tuple (ph, pw) of patch size.
            n_channels: Number of input channels (e.g., 3 for RGB).
            n_heads: Number of attention heads.
            n_layers: Number of transformer layers.
            pruning_index: List of layer indices where pruning occurs.
            rho: Base keep rate for pruning (fraction of tokens kept per pruned layer).
        """
        super().__init__()
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_classes = n_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_channels = n_channels
        self.n_heads = n_heads
        self.pruning_index = pruning_index
        self.base_keep_rate = rho

        # Calculate number of patches and sequence length (+1 for CLS token)
        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1])
        self.max_seq_length = self.n_patches + 1

        # Layers
        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEmbedding(self.d_model, self.max_seq_length)
        self.transformer_encoders = nn.ModuleList()
        self.dropout = nn.Dropout(0.1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
        )

        # Map layer index to cumulative keep ratio
        layer_to_ratio: Dict[int, float] = {}
        current_cumulative_ratio = 1.0
        sorted_pruning_locs = sorted(pruning_index)
        for loc in sorted_pruning_locs:
            current_cumulative_ratio *= self.base_keep_rate
            layer_to_ratio[loc] = current_cumulative_ratio

        # Build transformer layers
        for i in range(n_layers):
            if i in pruning_index:
                has_pred = True
                ratio_for_this_layer = layer_to_ratio[i]
            else:
                has_pred = False
                ratio_for_this_layer = 1.0
            self.transformer_encoders.append(
                DynamicTransformerEncoder(
                    self.d_model,
                    self.n_heads,
                    has_predictor=has_pred,
                    keep_ratio=ratio_for_this_layer
                )
            )

    def forward(
        self,
        images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """
        Forward pass of the DynamicViT student model.

        Args:
            images: Input images tensor of shape (B, C, H, W).

        Returns:
            logits: Class logits tensor of shape (B, n_classes).
            student_feats: Features tensor from transformer layers, shape (B, N, d_model) including CLS token.
            all_masks: List of binary masks applied at pruning layers [(B, N), ...].
            all_pred_scores: List of prediction scores of keeping each token at pruning layers [(B, N), ...].
        """
        x = self.patch_embedding(images)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        B, N, C = x.shape
        current_policy = torch.ones(B, N, device=x.device)
        all_pred_scores: List[torch.Tensor] = []
        all_masks: List[torch.Tensor] = []

        for layer in self.transformer_encoders:
            x, current_policy, pred_score, keep_indices = layer(x, current_policy)
            if pred_score is not None:
                all_pred_scores.append(pred_score)
                current_policy[:, 0] = 1.0  # CLS token is always kept
                all_masks.append(current_policy)

        cls_token = x[:, 0]
        logits = self.classifier(cls_token)
        student_feats = x

        return logits, student_feats, all_masks, all_pred_scores
