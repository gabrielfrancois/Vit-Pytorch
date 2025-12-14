import os
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from training.trainer import ViTTrainer

# The fixture creates a small dummy dataset and DataLoader for testing.
# Pytest automatically injects it into tests that include `dummy_data` as a parameter.

@pytest.fixture 
def dummy_data():
    """
    Creates a small dummy dataset and DataLoader for testing purposes. 
    Pytest automatically injects it into tests that include `dummy_data` as a parameter.

    Returns:
        DataLoader: PyTorch DataLoader with 8 samples of 3x16x16 images and binary labels.
    """
    x = torch.randn(8, 3, 16, 16)  # 8 samples, 3 channels, 16x16 pixels
    y = torch.randint(0, 2, (8,))
    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=2)
    return loader

@pytest.fixture
def trainer(tmp_path):
    """
    Initializes a ViTTrainer instance with small model parameters for testing.

    Args:
        tmp_path (Path): Temporary directory provided by pytest.

    Returns:
        ViTTrainer: Trainer instance ready for tests.
    """
    model_params = {
        "d_model": 128,
        "n_classes": 2,           # matches dummy dataset
        "img_size": (16, 16),
        "patch_size": (8, 8),
        "n_channels": 3,
        "n_heads": 4,
        "n_layers": 2
    }
    train_params = {
        "lr": 0.001,
        "weight_decay": 1e-4,
        "step_size": 5,
        "gamma": 0.5,
        "label_smoothing": 0.0,
        "log_dir": str(tmp_path / "logs")
    }
    trainer = ViTTrainer(model_params, train_params,
                         checkpoint_dir=str(tmp_path / "ckpts"),
                         plot_dir=str(tmp_path / "plots"))
    return trainer

def test_train_one_epoch(trainer, dummy_data):
    """
    Tests that training for one epoch runs and updates metrics.

    Asserts:
        - loss is a float
        - accuracy is between 0 and 100
        - train_losses and train_accs lists are updated
    """
    loss, acc = trainer.train_one_epoch(dummy_data)
    assert isinstance(loss, float)
    assert 0 <= acc <= 100
    assert len(trainer.train_losses) == 1
    assert len(trainer.train_accs) == 1

def test_validate_one_epoch(trainer, dummy_data):
    """
    Tests that validation for one epoch runs correctly.

    Asserts:
        - loss is a float
        - accuracy is between 0 and 100
        - confusion matrix is square
    """
    loss, acc, cm = trainer.validate_one_epoch(dummy_data)
    assert isinstance(loss, float)
    assert 0 <= acc <= 100
    assert cm.shape[0] == cm.shape[1]

def test_step_scheduler(trainer):
    """
    Tests that the learning rate scheduler step updates the optimizer's LR correctly.
    """
    lr_before = trainer.optimizer.param_groups[0]["lr"]
    lr_after = trainer.step_scheduler()
    assert lr_after == trainer.scheduler.get_last_lr()[0]

def test_save_and_load_checkpoint(trainer, dummy_data, tmp_path):
    """
    Tests saving and loading a checkpoint.

    Asserts:
        - checkpoint file is created
        - checkpoint contains model_state_dict
        - loaded epoch matches saved epoch
    """
    trainer.train_one_epoch(dummy_data)
    trainer.save_checkpoint("dummy", epoch=1, keep_last=1)
    ckpt_files = list((tmp_path / "ckpts").glob("*.pth"))
    assert len(ckpt_files) == 1

    checkpoint, epoch = trainer.load_checkpoint("dummy")
    assert epoch == 1
    assert "model_state_dict" in checkpoint

def test_test_model(trainer, dummy_data, tmp_path):
    """
    Tests the test_model method: runs evaluation, generates metrics, and creates plots.

    Asserts:
        - avg_loss is a float
        - accuracy is between 0 and 100
        - confusion matrix is square
        - classification report is a string
        - plot and metrics files exist
    """
    trainer.train_one_epoch(dummy_data)
    trainer.save_checkpoint("dummy_test", epoch=0)
    avg_loss, acc, cm, report = trainer.test_model(dummy_data, "dummy_test_epoch0")
    assert isinstance(avg_loss, float)
    assert 0 <= acc <= 100
    assert cm.shape[0] == cm.shape[1]
    assert isinstance(report, str)

    plot_file = tmp_path / "plots" / "dummy_test_epoch0_cm.png"
    metrics_file = tmp_path / "plots" / "dummy_test_epoch0_metrics.txt"
    assert plot_file.exists()
    assert metrics_file.exists()
