import numpy as np
import torch
from torch import Tensor, nn


class PositionalEmbedding(nn.Module):
    """
    Positional embedding module with a learnable classification token and
    deterministic sinusoidal positional encodings.

    This module:
    1. Creates a learnable `[CLS]` token (`cls_token`)
    2. Generates sinusoidal positional encodings of shape `(1, max_seq_length, d_model)`
    3. Prepends the `[CLS]` token to the sequence
    4. Adds positional encodings to the resulting embeddings

    Args:
        d_model (int): Dimensionality of each embedding vector.
        max_seq_length (int): Maximum number of tokens (excluding the CLS token).

    Shape:
        - Input:  `(B, N, d_model)`
        - Output: `(B, N + 1, d_model)`
          where +1 corresponds to the added CLS token.
    """

    def __init__(self, d_model: int, max_seq_length: int) -> None:
        super().__init__()

        self.d_model: int = d_model
        self.max_seq_length: int = max_seq_length

        # Learnable CLS token: shape (1, 1, d_model)
        self.cls_token: nn.Parameter = nn.Parameter(torch.randn(1, 1, d_model))

        # Build sinusoidal positional encoding
        pe = torch.zeros(max_seq_length + 1, d_model)  # +1 for CLS token position

        for pos in range(max_seq_length + 1):
            for i in range(0, d_model, 2):
                pe[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pe[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))

        # Store as non-trainable buffer
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_seq_length+1, d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Prepend the CLS token and add positional encodings.

        Args:
            x (Tensor): Input embeddings of shape (B, N, d_model).

        Returns:
            Tensor: Output embeddings with CLS token and positional encodings,
            shape (B, N + 1, d_model).
        """
        batch_size: int = x.size(0)

        # Expand CLS token across the batch: (B, 1, d_model)
        cls_tokens: Tensor = self.cls_token.expand(batch_size, -1, -1)

        # Prepend CLS token
        x = torch.cat((cls_tokens, x), dim=1)  # (B, N+1, d_model)

        # Add positional encoding (automatically broadcasts from (1, N+1, d_model))
        x = x + self.pe[:, : x.size(1)]

        return x
