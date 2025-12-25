import pytest
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from sklearn.metrics import confusion_matrix
from omegaconf import OmegaConf

cfg = OmegaConf.load("configs/imagenet.yaml")
# Si tu veux voir les valeurs finales fusionnées
cfg = OmegaConf.merge(OmegaConf.load("configs/base.yaml"), cfg)


@pytest.fixture
def dummy_data() -> DataLoader:
    """
    Creates a small dummy CIFAR-like dataset for testing
    using the batch size defined in the config.
    """
    x = torch.randn(cfg.training.batch_size, 3, cfg.model.img_size[0], cfg.model.img_size[1])
    y = torch.randint(0, 5, (cfg.training.batch_size,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=cfg.training.batch_size)


def test_teacher_train_validate(dummy_data: DataLoader, tmp_path: Path) -> None:
    """
    Tests Teacher ViT training, validation, and plot saving.
    Uses hyperparameters from configs/cifar.yaml
    """
    from training.train_teacher_test import train_one_epoch, validate_one_epoch, save_training_plots
    from models.vision_transformer import VisionTransformer

    device = "cpu"

    # Initialize model with hyperparameters
    model = VisionTransformer(
        d_model=cfg.model.d_model,
        n_classes=5,  # pour un test rapide
        img_size=cfg.model.img_size,
        patch_size=cfg.model.patch_size,
        n_channels=cfg.model.n_channels,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers
    ).to(device)

    # Use learning rate from YAML
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.alpha)
    criterion = nn.CrossEntropyLoss()

    # Train one epoch
    train_loss, train_acc = train_one_epoch(model, dummy_data, optimizer, criterion, device, epoch_index=0)
    assert isinstance(train_loss, float)
    assert 0 <= train_acc <= 100

    # Validate one epoch
    val_loss, val_acc, cm = validate_one_epoch(model, dummy_data, criterion, device)
    assert isinstance(val_loss, float)
    assert 0 <= val_acc <= 100
    assert cm.shape[0] == cm.shape[1]

    # Save plots
    save_training_plots(
        [train_loss], [val_loss], [train_acc], [val_acc], [cfg.training.alpha], cm, tmp_path
    )

    # Check that plot files are created
    files = list(tmp_path.glob("*.png"))
    assert len(files) >= 4
