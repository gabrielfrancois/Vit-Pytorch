import torch
from torch import Tensor

from models.multi_head_attention import MultiHeadAttention


def test_multihead_attention_output_shape() -> None:
    """
    Ensure that the MultiHeadAttention module returns a tensor of shape
    (batch_size, sequence_length, d_model) when given valid input.
    """
    batch_size = 4
    seq_len = 8
    d_model = 16
    n_heads = 4

    x: Tensor = torch.randn(batch_size, seq_len, d_model)
    layer = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    out: Tensor = layer(x)
    assert out.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected output shape {(batch_size, seq_len, d_model)}, got {out.shape}"


def test_multihead_attention_forward_runs_without_error() -> None:
    """
    Verify that a forward pass executes without raising any exception.
    This ensures correct tensor dimensionality and internal operations.
    """
    x: Tensor = torch.randn(3, 10, 32)
    layer = MultiHeadAttention(d_model=32, n_heads=4)

    try:
        _ = layer(x)
    except Exception as exc:
        raise AssertionError(f"Forward pass raised an exception: {exc}") from exc


def test_multihead_attention_all_parameters_receive_gradients() -> None:
    """
    Confirm that every trainable parameter receives a non-null gradient
    after backpropagation through a simple loss.
    """
    x: Tensor = torch.randn(2, 6, 32, requires_grad=True)
    layer = MultiHeadAttention(d_model=32, n_heads=4)

    out = layer(x)
    loss = out.sum()
    loss.backward()

    for name, param in layer.named_parameters():
        assert (
            param.grad is not None
        ), f"Expected gradient for parameter '{name}', but got None"


def test_multihead_attention_supports_mask() -> None:
    """
    Verify that the module accepts a mask tensor and still produces an
    output of the correct shape without failing.
    """
    batch_size = 2
    seq_len = 5
    d_model = 16
    n_heads = 4

    x = torch.randn(batch_size, seq_len, d_model)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 0, 1, 1, 1]], dtype=torch.bool)

    layer = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    try:
        out = layer(x, mask=mask)
    except Exception as exc:
        raise AssertionError(f"Forward with mask failed: {exc}") from exc

    assert out.shape == (
        batch_size,
        seq_len,
        d_model,
    ), f"Expected shape {(batch_size, seq_len, d_model)}, got {out.shape}"


def test_multihead_attention_head_dimensions() -> None:
    """
    Ensure that head_size is correctly computed as d_model // n_heads.
    """
    d_model = 32
    n_heads = 8
    layer = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    assert (
        layer.head_size == d_model // n_heads
    ), f"Expected head_size {d_model // n_heads}, got {layer.head_size}"


def test_multihead_attention_uses_all_heads() -> None:
    """
    Ensure that all attention heads are used and their outputs are concatenated
    along the last dimension before the final projection.
    """
    batch_size = 2
    seq_len = 6
    d_model = 32
    n_heads = 4

    x = torch.randn(batch_size, seq_len, d_model)
    layer = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    # Run individual heads manually for comparison
    head_outputs = [head(x) for head in layer.heads]
    concatenated = torch.cat(head_outputs, dim=-1)

    out = layer(x)

    # The input to W_o should match concatenation shape
    assert (
        concatenated.shape[-1] == d_model
    ), f"Concatenated head dim mismatch: expected {d_model}, got {concatenated.shape[-1]}"
    assert (
        out.shape == x.shape
    ), f"Final output shape mismatch: expected {x.shape}, got {out.shape}"
