import torch
from torch import Tensor

from models.vision_transformer import VisionTransformer


def test_vit_output_shape() -> None:
    """
    Ensure that VisionTransformer produces output logits of shape
    (batch_size, n_classes) and teacher features of shape (batch_size, n_patches+1, d_model).
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
        n_layers=n_layers,
    )

    logits, teacher_feats = model(images)

    n_patches = (img_size[0] * img_size[1]) // (patch_size[0] * patch_size[1])
    max_seq_length = n_patches + 1  # CLS token added

    assert logits.shape == (
        batch_size,
        n_classes,
    ), f"Expected logits shape {(batch_size, n_classes)}, got {logits.shape}"
    assert teacher_feats.shape == (
        batch_size,
        max_seq_length,
        d_model,
    ), f"Expected teacher_feats shape {(batch_size, max_seq_length, d_model)}, got {teacher_feats.shape}"


def test_vit_forward_runs_without_error() -> None:
    """
    Ensure forward pass executes without crashing for a small batch.
    """
    images: Tensor = torch.randn(1, 3, 32, 32)
    model = VisionTransformer(
        d_model=32,
        n_classes=5,
        img_size=(32, 32),
        patch_size=(16, 16),
        n_channels=3,
        n_heads=4,
        n_layers=1,
    )
    _logits, _teacher_feats = model(images)  # Should not raise


def test_vit_example() -> None:
    """
    Small deterministic example to validate the VisionTransformer pipeline.
    Checks shapes, non-zero outputs, and numerical stability.
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
        n_layers=n_layers,
    )

    logits, teacher_feats = model(images)

    n_patches = (img_size[0] * img_size[1]) // (patch_size[0] * patch_size[1])
    max_seq_length = n_patches + 1

    # Shape checks
    assert logits.shape == (batch_size, n_classes)
    assert teacher_feats.shape == (batch_size, max_seq_length, d_model)

    # Output should not be identical to zeros
    assert not torch.allclose(logits, torch.zeros_like(logits))
    assert not torch.allclose(teacher_feats, torch.zeros_like(teacher_feats))

    # No NaN or Inf
    assert not torch.isnan(logits).any()
    assert not torch.isinf(logits).any()
    assert not torch.isnan(teacher_feats).any()
    assert not torch.isinf(teacher_feats).any()
