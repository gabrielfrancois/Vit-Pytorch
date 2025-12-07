import logging
import torch
import pytest
import yaml
from torch.utils.data import DataLoader, TensorDataset

from training.train_student_test import (
    train_one_epoch,
    validate_one_epoch,
    save_training_plots
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Config YAML ---
with open("configs/imagenet.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PRUNING_INDEX = cfg.get("dynamicvit", {}).get("pruning_index", [4,7,10])
D_MODEL = cfg["model"]["d_model"]
N_CLASSES = 5  # on teste uniquement sur 5 classes
IMG_SHAPE = cfg["model"]["img_size"]
BATCH_SIZE = cfg["training"]["batch_size"]

# --- Dummy models ---
class DummyTeacher(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3*IMG_SHAPE[0]*IMG_SHAPE[1], N_CLASSES)

    def forward(self, x):
        b = x.size(0)
        logits = self.linear(x.view(b, -1))
        feats = torch.randn(b, 128)
        return logits, feats

class DummyStudent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3*IMG_SHAPE[0]*IMG_SHAPE[1], N_CLASSES)

    def forward(self, x):
        b = x.size(0)
        logits = self.linear(x.view(b, -1))
        feats = torch.randn(b, 128)
        masks = [torch.ones(b, len(PRUNING_INDEX))]
        scores = [torch.rand(b, len(PRUNING_INDEX))]
        return logits, feats, masks, scores

class DummyCriterion:
    def __call__(self, student_logits, teacher_logits, labels, student_feats, teacher_feats, all_masks):
        loss = torch.nn.functional.cross_entropy(student_logits, labels)
        return loss, {"ratio":0.1,"distill":0.2,"kl":0.05}

# --- Fixtures ---
@pytest.fixture
def dummy_loader():
    # On génère au moins un échantillon par classe pour que cm soit complet
    imgs = torch.randn(N_CLASSES, 3, IMG_SHAPE[0], IMG_SHAPE[1])
    labels = torch.arange(N_CLASSES)
    dataset = TensorDataset(imgs, labels)
    return DataLoader(dataset, batch_size=16)

# --- Tests ---
def test_train_one_epoch(dummy_loader):
    student = DummyStudent()
    teacher = DummyTeacher()
    criterion = DummyCriterion()
    optimizer = torch.optim.Adam(student.parameters(), lr=cfg["training"]["alpha"])

    avg_loss, ratio_loss, distill_loss, kl_loss, acc = train_one_epoch(
        student, teacher, dummy_loader, optimizer, criterion, device="cpu", epoch_index=0
    )
    assert avg_loss > 0
    assert 0 <= acc <= 100

def test_validate_one_epoch(dummy_loader):
    student = DummyStudent()
    acc, cm = validate_one_epoch(student, dummy_loader, device="cpu")
    assert 0 <= acc <= 100
    # cm doit avoir N_CLASSES x N_CLASSES
    assert cm.shape == (N_CLASSES, N_CLASSES)

def test_save_training_plots(tmp_path):
    # On convertit confusion_mat en float et fmt='.2f'
    save_training_plots(
        train_losses=[1.0,0.8],
        train_accs=[40,50],
        val_accs=[38,48],
        ratio_losses=[0.1, 0.08],
        distill_loss=[0.2, 0.15],
        kl_loss=[0.05, 0.04],
        lrs=[1e-3,9e-4],
        confusion_mat=torch.zeros((N_CLASSES, N_CLASSES),dtype=torch.float),
        save_dir=tmp_path
    )

    expected_files = [
        "student_loss_curve.png",
        "student_accuracy_curve.png",
        "student_ratio_loss.png",
        "student_distill_loss.png",
        "student_kl_loss.png",
        "student_confusion_matrix.png",
    ]
    for f in expected_files:
        assert (tmp_path / f).exists()
