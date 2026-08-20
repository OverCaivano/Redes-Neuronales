"""Carga de datos sintéticos o tabulares para el pipeline inicial."""

from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset


def _synthetic_data(data_config, input_dim, seed):
    generator = torch.Generator().manual_seed(seed)
    features = torch.randn(
        int(data_config.get("n_samples", 256)),
        input_dim,
        generator=generator,
    )
    weights = torch.randn(input_dim, 1, generator=generator)
    noise = float(data_config.get("noise", 0.1))
    targets = features @ weights + noise * torch.randn(
        features.shape[0], 1,
        generator=generator,
    )
    return features.float(), targets.float()


def _csv_data(data_config, input_dim):
    path = Path(data_config["path"])
    if not path.exists():
        raise FileNotFoundError(f"No existe el dataset configurado: {path}")

    frame = pd.read_csv(path)
    target_column = data_config.get("target_column", "target")
    if target_column not in frame.columns:
        raise ValueError(f"La columna objetivo '{target_column}' no existe en {path}")

    feature_frame = frame.drop(columns=[target_column])
    if input_dim != feature_frame.shape[1]:
        raise ValueError(
            f"input_dim={input_dim} no coincide con las {feature_frame.shape[1]} columnas de entrada"
        )
    features = torch.tensor(feature_frame.to_numpy(), dtype=torch.float32)
    targets = torch.tensor(frame[[target_column]].to_numpy(), dtype=torch.float32)
    return features, targets


def build_dataloaders(config):
    """Devuelve loaders de entrenamiento y validacion junto con sus dimensiones."""
    seed = int(config.get("seed", 42))
    model_config = config["model"]
    data_config = config["data"]
    input_dim = int(model_config["input_dim"])

    if data_config.get("source", "synthetic") == "csv":
        features, targets = _csv_data(data_config, input_dim)
    else:
        features, targets = _synthetic_data(data_config, input_dim, seed)

    split = float(config.get("validation", {}).get("split", 0.2))
    if not 0 < split < 1:
        raise ValueError("validation.split debe estar entre 0 y 1")

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(features), generator=generator)
    validation_size = max(1, int(len(features) * split))
    validation_indices = indices[:validation_size]
    training_indices = indices[validation_size:]

    train_dataset = TensorDataset(features[training_indices], targets[training_indices])
    validation_dataset = TensorDataset(
        features[validation_indices], targets[validation_indices]
    )
    batch_size = int(config["training"].get("batch_size", 32))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size)
    return train_loader, validation_loader
