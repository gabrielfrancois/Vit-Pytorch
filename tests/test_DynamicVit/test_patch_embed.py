import torch
from torch import Tensor

from models.patch_embed import PatchEmbedding


def test_patch_embedding_output_shape() -> None:
    """
    Verify that PatchEmbedding returns a tensor of shape
    (batch_size, n_patches, d_model).
    """
    batch_size = 2
    img_size = 32
    patch_size = 16
    n_channels = 3
    d_model = 64

    x: Tensor = torch.randn(batch_size, n_channels, img_size, img_size)
    layer = PatchEmbedding(
        d_model=d_model, img_size=img_size, patch_size=patch_size, n_channels=n_channels
    )

    out: Tensor = layer(x)
    n_patches = (img_size // patch_size) ** 2

    assert out.shape == (
        batch_size,
        n_patches,
        d_model,
    ), f"Expected {(batch_size, n_patches, d_model)}, got {out.shape}"


def test_patch_embedding_forward_runs_without_error() -> None:
    """
    Ensure that a forward pass executes without raising exceptions.
    """
    x: Tensor = torch.randn(1, 3, 32, 32)
    layer = PatchEmbedding(d_model=128, img_size=32, patch_size=16, n_channels=3)

    try:
        _ = layer(x)
    except Exception as exc:
        raise AssertionError(f"Forward raised an exception: {exc}") from exc


def test_patch_embedding_small_deterministic_example() -> None:
    """
    Validate embedding output using a simple deterministic input.
    4 patches of 2-dim embeddings expected for a 4x4 image with patch 2x2.
    """
    x: Tensor = torch.ones(1, 1, 4, 4)
    layer = PatchEmbedding(d_model=2, img_size=4, patch_size=2, n_channels=1)

    out: Tensor = layer(x)

    assert out.shape == (1, 4, 2)
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_patch_embedding_computes_correct_number_of_patches() -> None:
    """
    Confirm that the number of extracted patches matches
    (img_size // patch_size) squared.
    """
    batch_size = 1
    img_size = 64
    patch_size = 8
    n_channels = 3
    d_model = 32

    x = torch.randn(batch_size, n_channels, img_size, img_size)
    layer = PatchEmbedding(
        d_model=d_model, img_size=img_size, patch_size=patch_size, n_channels=n_channels
    )

    out = layer(x)
    expected = (img_size // patch_size) ** 2

    assert out.size(1) == expected, f"Expected {expected} patches, got {out.size(1)}"


def test_patch_embedding_projection_weights_receive_grad() -> None:
    """
    Check that Conv2D projection parameters receive gradients after backward().
    """
    x = torch.randn(2, 3, 32, 32, requires_grad=False)
    layer = PatchEmbedding(d_model=64, img_size=32, patch_size=16, n_channels=3)

    out = layer(x)
    loss = out.sum()
    loss.backward()

    for name, param in layer.named_parameters():
        assert param.grad is not None, f"Expected gradient for '{name}', got None"
