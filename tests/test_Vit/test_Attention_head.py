import torch
from torch import Tensor
from attention_head import AttentionHead


def test_attention_head_output_shape() -> None:
    """
    Test that ``AttentionHead`` produces an output tensor with the expected shape.

    The expected output shape is ``(batch_size, sequence_length, head_size)``.
    """
    batch_size = 4
    sequence_length = 8
    d_model = 16
    head_size = 4

    x: Tensor = torch.randn(batch_size, sequence_length, d_model)
    layer = AttentionHead(d_model=d_model, head_size=head_size)

    out: Tensor = layer(x)

    assert out.shape == (batch_size, sequence_length, head_size), (
        f"Expected output shape {(batch_size, sequence_length, head_size)}, "
        f"but got {out.shape}."
    )


def test_attention_head_softmax_stability() -> None:
    """
    Test that the attention weight matrix produced inside ``AttentionHead``
    forms valid probability distributions.

    For each query position, the attention weights should sum to 1 along
    the last dimension after the softmax operation.
    """
    batch_size = 2
    sequence_length = 5
    d_model = 12
    head_size = 6

    x: Tensor = torch.randn(batch_size, sequence_length, d_model)
    layer = AttentionHead(d_model=d_model, head_size=head_size)

    # Manual reproduction of internal QK^T softmax
    Q: Tensor = layer.query(x)
    K: Tensor = layer.key(x)

    attn_scores: Tensor = Q @ K.transpose(-2, -1)
    attn_scores = attn_scores / (head_size ** 0.5)

    attn_softmax: Tensor = torch.softmax(attn_scores, dim=-1)

    row_sums: Tensor = attn_softmax.sum(dim=-1)

    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), (
        "Softmax rows are expected to sum to 1, "
        "but they do not within the given tolerance."
    )


def test_attention_head_forward_runs_without_error() -> None:
    """
    Test that the ``forward`` method executes without raising exceptions.

    The goal is simply to validate the computational graph for common
    shapes and ensure no runtime errors occur.
    """
    x: Tensor = torch.randn(3, 10, 32)
    layer = AttentionHead(d_model=32, head_size=8)

    try:
        _ = layer(x)
    except Exception as exc:
        raise AssertionError(
            f"Forward pass should not raise an exception, but got: {exc}"
        ) from exc


def test_attention_head_all_parameters_receive_gradients() -> None:
    """
    Test that all trainable parameters of ``AttentionHead`` receive gradients.

    This ensures that backpropagation flows properly through the module.
    """
    x: Tensor = torch.randn(2, 6, 32, requires_grad=True)
    layer = AttentionHead(d_model=32, head_size=8)

    out: Tensor = layer(x)
    loss: Tensor = out.sum()
    loss.backward()

    for name, param in layer.named_parameters():
        assert param.grad is not None, (
            f"Expected gradient for parameter '{name}', but got None."
        )
