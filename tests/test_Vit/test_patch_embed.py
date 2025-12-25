# tests/test_patch_embedding.py
import torch
from torch import Tensor
from models.patch_embed import PatchEmbedding


def test_patch_embedding_output_shape() -> None:
    """
    Test that PatchEmbedding produces the correct output shape.
    """
    batch_size = 2
    img_size = 32
    patch_size = 16
    n_channels = 3
    d_model = 64

    x: Tensor = torch.randn(batch_size, n_channels, img_size, img_size)
    layer = PatchEmbedding(d_model=d_model, img_size=img_size, patch_size=patch_size, n_channels=n_channels)

    out: Tensor = layer(x)
    n_patches = (img_size // patch_size) ** 2
    assert out.shape == (batch_size, n_patches, d_model)


def test_patch_embedding_forward_runs_without_error() -> None:
    """Ensure forward pass runs without crashing."""
    x: Tensor = torch.randn(1, 3, 32, 32)
    layer = PatchEmbedding(d_model=128, img_size=32, patch_size=16, n_channels=3)
    _ = layer(x)  # will raise if broken


def test_patch_embedding_example() -> None:
    """
    Test PatchEmbedding on a small deterministic example to validate patch output.
    """
    x = torch.ones(1, 1, 4, 4)  # 1 image, 1 channel, 4x4
    layer = PatchEmbedding(d_model=2, img_size=4, patch_size=2, n_channels=1)
    out = layer(x)

    # 4 patches of 2 features each
    assert out.shape == (1, 4, 2)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
