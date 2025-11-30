import torch
from torch import Tensor

from models.positional_embedding import PositionalEmbedding


def test_cls_token_shape() -> None:
    """
    Check that the CLS token has the correct shape: (1, 1, d_model).
    """
    d_model: int = 4
    max_seq: int = 5
    pe = PositionalEmbedding(d_model, max_seq)

    assert pe.cls_token.shape == (1, 1, d_model)


def test_positional_encoding_shape() -> None:
    """
    Check that the positional encoding buffer has the expected shape:
    (1, max_seq_length + 1, d_model).
    """
    d_model: int = 4
    max_seq: int = 5
    pe = PositionalEmbedding(d_model, max_seq)

    assert pe.pe.shape == (1, max_seq + 1, d_model)


def test_forward_output_shape() -> None:
    """
    Check that the forward pass returns a tensor of shape (B, N + 1, d_model),
    where the CLS token is prepended to the sequence.
    """
    d_model: int = 4
    max_seq: int = 5
    batch: int = 2

    model = PositionalEmbedding(d_model, max_seq)
    x: Tensor = torch.randn(batch, max_seq, d_model)

    out: Tensor = model(x)

    assert out.shape == (batch, max_seq + 1, d_model)


def test_cls_token_is_trainable() -> None:
    """
    Ensure that the CLS token is a learnable parameter.
    """
    d_model: int = 4
    max_seq: int = 5
    pe = PositionalEmbedding(d_model, max_seq)

    assert pe.cls_token.requires_grad is True


def test_positional_encoding_not_trainable() -> None:
    """
    Ensure that the positional encoding buffer is not trainable.
    """
    d_model: int = 4
    max_seq: int = 5
    pe = PositionalEmbedding(d_model, max_seq)

    assert pe.pe.requires_grad is False
