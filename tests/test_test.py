import os
import logging
from pathlib import Path
from typing import Tuple, List, Any

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix

from models.vision_transformer import VisionTransformer
from models.dynamicViT import DynamicVisionTransformer
from testing.run_test import evaluate_model, plot_confusion_matrices, plot_performance_comparison, plot_per_class_accuracy

# --- Logger Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EvaluationTest")

# --- Load configuration ---
cfg_base = OmegaConf.load("configs/base.yaml")
cfg_cifar = OmegaConf.load("configs/cifar.yaml")
cfg = OmegaConf.merge(cfg_base, cfg_cifar)

# --- Fixtures ---
@pytest.fixture
def dummy_data() -> DataLoader:
    """Creates a small dummy CIFAR-like dataset for testing"""
    batch_size = cfg.training.batch_size
    n_classes = min(cfg.model.n_classes, 10)  # limit classes to 10 for dummy
    img_size = cfg.model.img_size
    x = torch.randn(batch_size, 3, img_size[0], img_size[1])
    y = torch.randint(0, n_classes, (batch_size,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size)


@pytest.fixture
def teacher_model() -> VisionTransformer:
    """Initializes a Teacher ViT model with config parameters"""
    m_cfg = cfg.model
    return VisionTransformer(
        d_model=m_cfg.d_model,
        n_classes=min(m_cfg.n_classes, 10),
        img_size=m_cfg.img_size,
        patch_size=m_cfg.patch_size,
        n_channels=m_cfg.n_channels,
        n_heads=m_cfg.n_heads,
        n_layers=m_cfg.n_layers
    )


@pytest.fixture
def student_model() -> DynamicVisionTransformer:
    """Initializes a Student DynamicViT model with config parameters"""
    m_cfg = cfg.model
    pruning_idx = cfg.dynamicvit.get("pruning_index", 0)
    return DynamicVisionTransformer(
        d_model=m_cfg.d_model,
        n_classes=min(m_cfg.n_classes, 10),
        img_size=m_cfg.img_size,
        patch_size=m_cfg.patch_size,
        n_channels=m_cfg.n_channels,
        n_heads=m_cfg.n_heads,
        n_layers=m_cfg.n_layers,
        pruning_index=pruning_idx
    )


# --- Test Function ---
def test_evaluate_models(
    dummy_data: DataLoader,
    teacher_model: VisionTransformer,
    student_model: DynamicVisionTransformer,
    tmp_path: Path
) -> None:
    """Tests evaluation functions for Teacher and Student models"""
    device = "cpu"
    logger.info("Starting dummy evaluation of Teacher model...")
    t_acc, t_loss, t_speed, t_preds, t_labels = evaluate_model(
        teacher_model, dummy_data, device, model_name="Teacher"
    )

    logger.info("Starting dummy evaluation of Student model...")
    s_acc, s_loss, s_speed, s_preds, s_labels = evaluate_model(
        student_model, dummy_data, device, model_name="Student"
    )

    # Generate plots in tmp_path
    logger.info("Generating plots...")
    t_cm = confusion_matrix(t_labels, t_preds, labels=list(range(min(cfg.model.n_classes, 10))))
    s_cm = confusion_matrix(s_labels, s_preds, labels=list(range(min(cfg.model.n_classes, 10))))
    class_names = [f"class_{i}" for i in range(min(cfg.model.n_classes, 10))]

    plot_confusion_matrices(t_cm, s_cm, class_names, tmp_path)
    plot_performance_comparison(t_acc, s_acc, t_speed, s_speed, tmp_path)
    plot_per_class_accuracy(t_cm, s_cm, class_names, tmp_path)

    # Verify files are created
    png_files: List[Path] = list(tmp_path.glob("*.png"))
    assert len(png_files) >= 3, "Expected at least 3 plot files to be created"
    logger.info(f"All evaluation plots generated in {tmp_path}")
