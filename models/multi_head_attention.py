from typing import Optional

import torch
from torch import Tensor, nn

from .attention_head import AttentionHead


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention module.
    Args:
        d_model (int): Input/output feature dimension.
        n_heads (int): Number of parallel attention heads to learn.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.head_size: int = d_model // n_heads
        self.heads: nn.ModuleList = nn.ModuleList(
            [AttentionHead(d_model, self.head_size) for _ in range(n_heads)]
        )
        self.W_o: nn.Linear = nn.Linear(d_model, d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Forward pass for multi-head attention.
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, d_model).
            mask (Optional[Tensor]): Mask tensor of shape (batch_size, seq_len)
                where 1=kept, 0=pruned.
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, d_model).
        Notes:
            At inference, the mask may reduce the sequence length N to N' <= N (dynamic ViT).
            During training, N is constant.
        """
        out: Tensor = torch.cat([head(x, mask=mask) for head in self.heads], dim=-1)
        out = self.W_o(out)
        return out
