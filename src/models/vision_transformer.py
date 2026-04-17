"""
This Vision transformer will be the 'teacher' model, he'll trained the dynamic ViT to fetch a faster model almost as effiscient as the teacher (as much as possible).
"""

import torch
from torch import nn as nn

from helper_function.print import *

from .patch_embed import PatchEmbedding
from .positional_embedding import PositionalEmbedding
from .transformer_encoder import TransformerEncoder


class VisionTransformer(nn.Module):
    def __init__(self, d_model, n_classes, img_size, patch_size, n_channels, n_heads, n_layers, repa_layer_index:int=7):
        super().__init__()

        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, "img_size dimensions must be divisible by patch_size dimensions"
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads. Actually, I think we could relax this assumption, we'll need to adapt the code though..."

        self.d_model = d_model # Dimensionality of model
        self.n_classes = n_classes # Number of classes
        self.img_size = img_size # Image size
        self.patch_size = patch_size # Patch size
        self.n_channels = n_channels # Number of channels
        self.n_heads = n_heads # Number of attention heads
        

        self.n_patches = (self.img_size[0] * self.img_size[1]) // (self.patch_size[0] * self.patch_size[1]) # The number of patches can be found by dividing the product of the height and width of the input image by the product of the height and width of the patch size.
        self.max_seq_length = self.n_patches + 1

        self.patch_embedding = PatchEmbedding(self.d_model, self.img_size, self.patch_size, self.n_channels)
        self.positional_encoding = PositionalEmbedding(self.d_model, self.max_seq_length)
        self.dropout = nn.Dropout(0.1) #regularisation

        self.transformer_encoder = nn.Sequential(*[TransformerEncoder(self.d_model, self.n_heads) for _ in range(n_layers)]) 
        # The vision transformer will also need to be able to have multiple encoder modules. 
        # This can be achieved by putting a list of encoder layers inside of a sequential wrapper.
        
        # Self Supervised Learning PRE-TRAINING COMPONENTS (MAE Style)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        pixels_per_patch = patch_size[0] * patch_size[1] * n_channels
        self.pretrain_head = nn.Linear(self.d_model, pixels_per_patch)

        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(self.d_model, self.n_classes),
        )

        # Initialize the mask token
        torch.nn.init.normal_(self.mask_token, std=.02)

        # Choose the explicit layer to align (e.g., 7)
        self.repa_layer_index = repa_layer_index 
        
        self.ssl_dim = 384 # DINOv2-vits14 dimension
        self.repa_proj = nn.Sequential(
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Linear(self.d_model * 2, self.ssl_dim)
        )
    def forward(self, images, bool_masked_pos=None): 
        """
        input :
        ------------------
            - images: (batch size (=B), image channel, image height, image width)
            - bool_masked_pos: (B, N) boolean tensor. True means the patch is masked. 
                               If None, the model acts as a standard classifier.
        output :
        ------------------
            - logits: tensor: (B, nb_classes) (for instance, in cifar-10 = 10)
            - teacher_feats: tensor: (B, N, d_model) (recall that d_model is actually N + 1, due to the cls token added)
        """
        x = self.patch_embedding(images)
        B, N, D = x.shape

        if bool_masked_pos is not None:
            mask_tokens = self.mask_token.expand(B, N, -1) # broadcasting
            
            # Reshape boolean mask so we can multiply it with the features
            # True (1) = the mask token, False (0) = original image patch
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            x = x * (1 - w) + mask_tokens * w

        x = self.positional_encoding(x) # Add CLS AFTERWARD
        x = self.dropout(x)
        repa_features = None

        if bool_masked_pos is not None: # predict the pixels of the spatial patches.
            x = self.transformer_encoder(x)
            # Strip off the CLS token (index 0)
            spatial_features = x[:, 1:, :] # (B, N, d_model)

            # Project d_model back into raw pixel dimension
            pixel_predictions = self.pretrain_head(spatial_features) 
            return pixel_predictions
        else: # Apply REPA
            for i, layer in enumerate(self.transformer_encoder):
                x = layer(x) # Pass through the current Transformer layer
                if i == self.repa_layer_index: # here we apply REPA
                    repa_features = self.repa_proj(x[:, 1:, :])# Ignore CLS
                    
            teacher_feats = x  # Capture the features (t_i' in the paper) before classification
            logits = self.classifier(x[:, 0]) # Classify using only the CLS token
            return logits, teacher_feats, repa_features