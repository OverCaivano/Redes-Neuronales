from pathlib import Path

import yaml

from src.train import train


def test_training_writes_checkpoint(tmp_path, monkeypatch):
    config = {
        "seed": 42,
        "model": {"input_dim": 4, "hidden_dim": 8, "output_dim": 1},
        "training": {"epochs": 1, "batch_size": 8, "learning_rate": 0.01},
        "validation": {"split": 0.2},
        "data": {"source": "synthetic", "n_samples": 20, "noise": 0.1},
    }
    monkeypatch.chdir(tmp_path)
    checkpoint_path = train(config)
    assert Path(checkpoint_path).exists()
    assert yaml.safe_load(yaml.safe_dump(config)) == config
