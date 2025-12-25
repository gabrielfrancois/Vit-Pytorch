import torch
from torch import Tensor

from models.attention_head import AttentionHead


def test_attention_head_output_shape() -> None:
    """
    Check output shape of AttentionHead layer.
    Verifies if the output matches expected (batch_size, seq_len, head_size).
    """
    batch_size, seq_len, d_model, head_size = 4, 8, 16, 4
    x: Tensor = torch.randn(batch_size, seq_len, d_model)
    layer: AttentionHead = AttentionHead(d_model=d_model, head_size=head_size)
    out: Tensor = layer(x)
    assert out.shape == (batch_size, seq_len, head_size)


def test_attention_head_forward_runs_without_error() -> None:
    """
    Ensure AttentionHead forward pass runs without exceptions.
    """
    x: Tensor = torch.randn(3, 10, 32)
    layer: AttentionHead = AttentionHead(d_model=32, head_size=8)
    _ = layer(x)  # Check doesn't throw


def test_attention_head_example_shape_and_no_nan_inf() -> None:
    """
    Run AttentionHead on simple input; check shape and numeric stability.
    """
    x: Tensor = torch.ones(1, 2, 4)
    layer: AttentionHead = AttentionHead(d_model=4, head_size=2)
    out: Tensor = layer(x)
    assert out.shape == (1, 2, 2)
    assert not torch.isnan(out).any(), "output contains NaN"
    assert not torch.isinf(out).any(), "output contains Inf"


def test_attention_head_with_mask_keeps_shape_and_no_nan_inf() -> None:
    """
    Test AttentionHead forward with mask (mix of pruned/kept tokens).
    Verifies output shape and absence of NaN/Inf.
    """
    batch_size, seq_len, d_model, head_size = 2, 4, 8, 2
    x: Tensor = torch.randn(batch_size, seq_len, d_model)
    mask: Tensor = torch.tensor([[1, 0, 1, 1], [1, 1, 0, 1]])
    layer: AttentionHead = AttentionHead(d_model=d_model, head_size=head_size)
    out: Tensor = layer(x, mask)
    assert out.shape == (batch_size, seq_len, head_size)
    assert not torch.isnan(out).any(), "output contains NaN"
    assert not torch.isinf(out).any(), "output contains Inf"
