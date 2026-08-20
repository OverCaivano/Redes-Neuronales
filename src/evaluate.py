"""Evaluacion del checkpoint generado por train.py."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

def evaluate(model_path):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    try:
        from src.data_loader import build_dataloaders
        from src.models import MLP
    except ModuleNotFoundError:
        from data_loader import build_dataloaders
        from models import MLP

    _, validation_loader = build_dataloaders(config)
    model_config = config["model"]
    model = MLP(**{key: int(value) for key, value in model_config.items()})
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for features, targets in validation_loader:
            total_loss += criterion(model(features), targets).item() * len(features)
            total_items += len(features)
    loss = total_loss / total_items
    print(f"Validation MSE: {loss:.4f}")
    return loss


def main():
    parser = argparse.ArgumentParser(description="Evalua un modelo entrenado")
    parser.add_argument("--model_path", default="models/best_model.pt")
    args = parser.parse_args()
    if not Path(args.model_path).exists():
        raise FileNotFoundError(
            f"No existe el checkpoint {args.model_path}. Ejecuta primero train.py."
        )
    evaluate(args.model_path)


if __name__ == "__main__":
    main()
