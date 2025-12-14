import logging
from pathlib import Path
from typing import List

import pytest
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from data.imagenet_loader import load_imagenet1k
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

# --- Fixtures ---
@pytest.fixture
def mini_imagenet_loader() -> DataLoader:
    """Loads a tiny subset of ImageNet-1k for quick tests"""
    _, val_loader, _ = load_imagenet1k(
        batch_size=8,
        max_items_train=16,
        max_items_val=16,
        img_size=32
    )
    return val_loader

@pytest.fixture
def teacher_model() -> VisionTransformer:
    """Teacher ViT model for mini-ImageNet"""
    return VisionTransformer(
        d_model=64, n_classes=5, img_size=(32, 32),
        patch_size=(8, 8), n_channels=3, n_heads=2, n_layers=1
    )

@pytest.fixture
def student_model() -> DynamicVisionTransformer:
    """Student DynamicViT model for mini-ImageNet"""
    return DynamicVisionTransformer(
        d_model=64, n_classes=5, img_size=(32, 32),
        patch_size=(8, 8), n_channels=3, n_heads=2, n_layers=1,
        pruning_index=[0]
    )

# --- Test Function ---
def test_evaluate_models_on_mini_imagenet(
    mini_imagenet_loader: DataLoader,
    teacher_model: VisionTransformer,
    student_model: DynamicVisionTransformer,
    tmp_path: Path
) -> None:
    device = "cpu"

    # --- Teacher ---
    t_acc, t_loss, t_speed, t_preds, t_labels = evaluate_model(
        teacher_model, mini_imagenet_loader, device, model_name="Teacher"
    )

    # --- Student ---
    s_acc, s_loss, s_speed, s_preds, s_labels = evaluate_model(
        student_model, mini_imagenet_loader, device, model_name="Student"
    )

    # --- Plots ---
    class_names = [f"class_{i}" for i in range(5)]
    t_cm = confusion_matrix(t_labels, t_preds, labels=list(range(5)))
    s_cm = confusion_matrix(s_labels, s_preds, labels=list(range(5)))

    plot_confusion_matrices(t_cm, s_cm, class_names, tmp_path)
    plot_performance_comparison(t_acc, s_acc, t_speed, s_speed, tmp_path)
    plot_per_class_accuracy(t_cm, s_cm, class_names, tmp_path)

    # --- Check plots created ---
    png_files: List[Path] = list(tmp_path.glob("*.png"))
    assert len(png_files) >= 3, "Expected at least 3 plot files to be created"
    logger.info(f"All evaluation plots generated in {tmp_path}")
