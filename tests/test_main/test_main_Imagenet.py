# tests/test_main_script.py
from pathlib import Path
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from training.trainer import ViTTrainer


@pytest.fixture
def small_dataset() -> DataLoader:
    """
    Creates a small dummy dataset to simulate training, validation, and test loaders.

    Returns:
        DataLoader: PyTorch DataLoader for testing.
    """
    x = torch.randn(16, 3, 16, 16)  # 16 samples, 3 channels, 16x16 images
    y = torch.randint(0, 2, (16,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=4)
    return loader


@pytest.fixture
def trainer_fixture(tmp_path: Path) -> ViTTrainer:
    """
    Creates a ViTTrainer instance with minimal parameters for testing.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Returns:
        ViTTrainer: Trainer instance ready for testing.
    """
    model_params = {
        "d_model": 64,
        "n_classes": 2,
        "img_size": (16, 16),
        "patch_size": (8, 8),
        "n_channels": 3,
        "n_heads": 2,
        "n_layers": 1
    }
    train_params = {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "epochs": 1,
        "label_smoothing": 0.0,
        "log_dir": str(tmp_path / "logs")
    }
    trainer = ViTTrainer(
        model_params=model_params,
        train_params=train_params,
        checkpoint_dir=str(tmp_path / "checkpoints"),
        plot_dir=str(tmp_path / "plots")
    )
    return trainer


def test_training_loop(trainer_fixture: ViTTrainer, small_dataset: DataLoader) -> None:
    """
    Tests a simplified training loop with one epoch to ensure train_one_epoch,
    validate_one_epoch, and scheduler steps run without errors.

    Args:
        trainer_fixture (ViTTrainer): Trainer instance.
        small_dataset (DataLoader): Dummy dataset loader.
    """
    # Train
    train_loss, train_acc = trainer_fixture.train_one_epoch(small_dataset)
    assert isinstance(train_loss, float)
    assert 0 <= train_acc <= 100

    # Validate
    val_loss, val_acc, cm = trainer_fixture.validate_one_epoch(small_dataset)
    assert isinstance(val_loss, float)
    assert 0 <= val_acc <= 100
    assert cm.shape[0] == cm.shape[1]

    # Scheduler step
    lr = trainer_fixture.step_scheduler()
    assert lr == trainer_fixture.scheduler.get_last_lr()[0]


def test_checkpoint_and_test(
        trainer_fixture: ViTTrainer,
        small_dataset: DataLoader,
        tmp_path: Path) -> None:

    """
    Tests saving/loading checkpoints and running test_model.

    Args:
        trainer_fixture (ViTTrainer): Trainer instance.
        small_dataset (DataLoader): Dummy dataset loader.
        tmp_path (Path): Temporary directory for files.
    """
    # Run one epoch to populate metrics
    trainer_fixture.train_one_epoch(small_dataset)

    # Save checkpoint
    trainer_fixture.save_checkpoint("test_ckpt", epoch=0)
    ckpt_files = list((tmp_path / "checkpoints").glob("*.pth"))
    assert len(ckpt_files) == 1

    # Load checkpoint
    checkpoint, epoch = trainer_fixture.load_checkpoint("test_ckpt")
    assert checkpoint is not None
    assert epoch == 0

    # Run test_model
    avg_loss, acc, cm, report = trainer_fixture.test_model(small_dataset, "test_ckpt_epoch0")
    assert isinstance(avg_loss, float)
    assert 0 <= acc <= 100
    assert cm.shape[0] == cm.shape[1]
    assert isinstance(report, str)

    # Check that plot and metrics files were created
    plot_file = tmp_path / "plots" / "test_ckpt_epoch0_cm.png"
    metrics_file = tmp_path / "plots" / "test_ckpt_epoch0_metrics.txt"
    assert plot_file.exists()
    assert metrics_file.exists()
