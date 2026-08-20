import torch

from src.data_loader import build_dataloaders
from src.models import MLP


CONFIG = {
    "seed": 42,
    "model": {"input_dim": 10, "hidden_dim": 8, "output_dim": 1},
    "training": {"batch_size": 16},
    "validation": {"split": 0.2},
    "data": {"source": "synthetic", "n_samples": 40, "noise": 0.1},
}


def test_dataloaders_have_expected_shapes():
    train_loader, validation_loader = build_dataloaders(CONFIG)
    features, targets = next(iter(train_loader))
    assert features.shape[1] == 10
    assert targets.shape[1] == 1
    assert len(train_loader.dataset) + len(validation_loader.dataset) == 40


def test_model_forward_shape():
    model = MLP(input_dim=10, hidden_dim=8, output_dim=1)
    predictions = model(torch.randn(4, 10))
    assert predictions.shape == (4, 1)
