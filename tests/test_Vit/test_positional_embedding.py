import torch
import pytest
from torch import nn

from models.positional_embedding import PositionalEmbedding


def test_cls_token_shape():
    d_model = 4
    max_seq = 5
    pe = PositionalEmbedding(d_model, max_seq)
    assert pe.cls_token.shape == (1, 1, d_model)


def test_positional_encoding_shape():
    d_model = 4
    max_seq = 5
    pe = PositionalEmbedding(d_model, max_seq)
   
    assert pe.pe.shape == (1, max_seq, d_model)


def test_forward_raises_shape_error():
    d_model = 4
    max_seq = 5
    batch = 2

    model = PositionalEmbedding(d_model, max_seq)
    x = torch.randn(batch, max_seq, d_model)

    # Ton code est censé planter ici vu le mismatch
    with pytest.raises(RuntimeError):
        model(x)


def test_cls_token_is_trainable():
    d_model = 4
    max_seq = 5
    pe = PositionalEmbedding(d_model, max_seq)
    assert pe.cls_token.requires_grad is True


def test_positional_encoding_not_trainable():
    d_model = 4
    max_seq = 5
    pe = PositionalEmbedding(d_model, max_seq)
    assert pe.pe.requires_grad is False
