# tests/test_transformer_encoder.py
import torch
from torch import Tensor
from models.transformer_encoder import TransformerEncoder


def test_transformer_encoder_output_shape() -> None:
    """
    Test that TransformerEncoder produces the correct output shape:
    (batch_size, seq_len, d_model)
    """
    batch_size = 2
    seq_len = 8
    d_model = 16
    n_heads = 4

    x: Tensor = torch.randn(batch_size, seq_len, d_model)
    layer = TransformerEncoder(d_model=d_model, n_heads=n_heads)

    out: Tensor = layer(x)
    assert out.shape == (batch_size, seq_len, d_model)


def test_transformer_encoder_forward_runs_without_error() -> None:
    """Ensure forward pass executes without crashing."""
    x: Tensor = torch.randn(1, 10, 32)
    layer = TransformerEncoder(d_model=32, n_heads=4)
    _ = layer(x)


def test_transformer_encoder_example() -> None:
    """
    Small deterministic example to validate residual connections and MLP behavior.
    """
    batch_size = 1
    seq_len = 2
    d_model = 4
    n_heads = 2

    x: Tensor = torch.ones(batch_size, seq_len, d_model)
    layer = TransformerEncoder(d_model=d_model, n_heads=n_heads, r_mlp=2)
    out: Tensor = layer(x)

    # Shape check
    assert out.shape == (batch_size, seq_len, d_model)

    # Output should not be identical to input due to MHA + MLP + residuals
    assert not torch.allclose(out, x)

    # No NaN or Inf values
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
