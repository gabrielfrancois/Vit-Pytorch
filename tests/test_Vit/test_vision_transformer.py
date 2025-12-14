# tests/test_vision_transformer.py
import torch
from torch import Tensor
from models.vision_transformer import VisionTransformer


def test_vit_output_shape() -> None:
    """
    Test that VisionTransformer produces output of shape (batch_size, n_classes).
    """
    batch_size = 2
    img_size = (32, 32)
    patch_size = (16, 16)
    n_channels = 3
    d_model = 64
    n_heads = 4
    n_layers = 2
    n_classes = 10

    images: Tensor = torch.randn(batch_size, n_channels, *img_size)
    model = VisionTransformer(
        d_model=d_model,
        n_classes=n_classes,
        img_size=img_size,
        patch_size=patch_size,
        n_channels=n_channels,
        n_heads=n_heads,
        n_layers=n_layers
    )

    out: Tensor = model(images)
    assert out.shape == (batch_size, n_classes)


def test_vit_forward_runs_without_error() -> None:
    """Ensure forward pass executes without crashing."""
    images: Tensor = torch.randn(1, 3, 32, 32)
    model = VisionTransformer(
        d_model=32,
        n_classes=5,
        img_size=(32, 32),
        patch_size=(16, 16),
        n_channels=3,
        n_heads=4,
        n_layers=1
    )
    _ = model(images)


def test_vit_example() -> None:
    """
    Small deterministic example to validate the full VisionTransformer pipeline.
    """
    batch_size = 1
    img_size = (4, 4)
    patch_size = (2, 2)
    n_channels = 1
    d_model = 4
    n_heads = 2
    n_layers = 1
    n_classes = 3

    images: Tensor = torch.ones(batch_size, n_channels, *img_size)
    model = VisionTransformer(
        d_model=d_model,
        n_classes=n_classes,
        img_size=img_size,
        patch_size=patch_size,
        n_channels=n_channels,
        n_heads=n_heads,
        n_layers=n_layers
    )

    out: Tensor = model(images)
    # Shape check
    assert out.shape == (batch_size, n_classes)
    # Output should not be identical to zero
    assert not torch.allclose(out, torch.zeros_like(out))
    # No NaN or Inf
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
