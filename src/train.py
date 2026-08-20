import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml

try:
    from src.data_loader import build_dataloaders
    from src.models import MLP
except ModuleNotFoundError:
    from data_loader import build_dataloaders
    from models import MLP


def load_config(path):
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_items = 0
    for features, targets in loader:
        if training:
            optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, targets)
        if training:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(features)
        total_items += len(features)
    return total_loss / total_items


def train(config):
    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    train_loader, validation_loader = build_dataloaders(config)
    model_config = config["model"]
    model = MLP(**{key: int(value) for key, value in model_config.items()})
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"].get("learning_rate", 0.001)),
    )
    best_validation_loss = float("inf")
    best_state_dict = None
    epochs = int(config["training"].get("epochs", 5))

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer)
        validation_loss = _run_epoch(model, validation_loader, criterion)
        print(
            f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f}"
        )
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state_dict = {
                key: value.detach().clone()
                for key, value in model.state_dict().items()
            }

    model_path = Path("models/best_model.pt")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": best_state_dict,
            "config": config,
            "best_validation_loss": best_validation_loss,
        },
        model_path,
    )
    print(f"Modelo guardado en {model_path}")
    return model_path


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo base")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    train(load_config(args.config))


if __name__ == "__main__":
    main()
