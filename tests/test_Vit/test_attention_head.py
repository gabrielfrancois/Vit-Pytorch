# tests/test_attention_head.py
import torch
from torch import Tensor
from models.attention_head import AttentionHead


def test_attention_head_output_shape() -> None:
    """
    Test that AttentionHead produces the expected output shape.
    """
    batch_size, seq_len, d_model, head_size = 4, 8, 16, 4
    x: Tensor = torch.randn(batch_size, seq_len, d_model)
    layer = AttentionHead(d_model=d_model, head_size=head_size)
    out: Tensor = layer(x)

    assert out.shape == (batch_size, seq_len, head_size)


def test_attention_head_forward_runs_without_error() -> None:
    """Ensure forward pass executes without errors."""
    x: Tensor = torch.randn(3, 10, 32)
    layer = AttentionHead(d_model=32, head_size=8)
    _ = layer(x)


def test_attention_head_example() -> None:
    """
    Small example to verify values roughly match expected pattern (shape only).
    """
    x: Tensor = torch.ones(1, 2, 4)  # batch=1, seq_len=2, d_model=4
    layer = AttentionHead(d_model=4, head_size=2)
    out: Tensor = layer(x)

    assert out.shape == (1, 2, 2)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
