from unittest.mock import patch
from PIL import Image
import torch
from data.imagenet_loader import load_imagenet1k

@patch("data.imagenet_loader.load_dataset")
def test_load_imagenet_basic(mock_load):
    # Dataset factice de 4 images
    fake_data = [
        {"image": Image.new("RGB", (128, 128)), "label": 0},
        {"image": Image.new("RGB", (128, 128)), "label": 1},
        {"image": Image.new("RGB", (128, 128)), "label": 2},
        {"image": Image.new("RGB", (128, 128)), "label": 3},
    ]

    mock_load.return_value = fake_data

    train_loader, val_loader, test_loader = load_imagenet1k(
        batch_size=2,
        max_items_train=4,
        max_items_val=2,
        img_size=64,
    )

    x, y = next(iter(train_loader))
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape[0] == 2
