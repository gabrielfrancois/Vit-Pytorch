import torch
from torch import Tensor
from models.multi_head_attention import MultiHeadAttention


def test_multihead_attention_output_shape() -> None:
    """
    Test that `MultiHeadAttention` produces an output tensor with the expected shape.

    The expected output shape is `(batch_size, sequence_length, d_model)`.
    """
    batch_size = 4
    sequence_length = 8
    d_model = 16
    n_heads = 4

    x: Tensor = torch.randn(batch_size, sequence_length, d_model)
    layer = MultiHeadAttention(d_model=d_model, n_heads=n_heads)

    out: Tensor = layer(x)
    assert out.shape == (batch_size, sequence_length, d_model), (
        f"Expected output shape {(batch_size, sequence_length, d_model)}, "
        f"but got {out.shape}."
    )


def test_multihead_attention_forward_runs_without_error() -> None:
    """
    Test that the forward method executes without raising exceptions.

    Ensures that the computational graph can be built for common shapes.
    """
    x: Tensor = torch.randn(3, 10, 32)
    layer = MultiHeadAttention(d_model=32, n_heads=4)

    try:
        _ = layer(x)
    except Exception as exc:
        raise AssertionError(
            f"Forward pass should not raise an exception, but got: {exc}"
        ) from exc


def test_multihead_attention_all_parameters_receive_gradients() -> None:
    """
    Test that all trainable parameters receive gradients during backpropagation.

    This ensures that the module is trainable and backpropagation flows correctly.
    """
    x: Tensor = torch.randn(2, 6, 32, requires_grad=True)
    layer = MultiHeadAttention(d_model=32, n_heads=4)

    out: Tensor = layer(x)
    loss: Tensor = out.sum()
    loss.backward()

    for name, param in layer.named_parameters():
        assert param.grad is not None, (
            f"Expected gradient for parameter '{name}', but got None."
        )
