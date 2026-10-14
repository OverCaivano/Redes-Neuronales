"""Entrenamiento minimo con fixtures; sin CSV reales ni evaluacion de test."""

import hashlib
import importlib
import json

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.metricas import MetricasRegresion
from src.train import train, ejecutar_epoca
from src.data_loader import build_dataloaders
from src.models import MLP
from src.loaders_favorita import construir_loaders_favorita

modulo = importlib.import_module("src.train")


@pytest.fixture
def config_favorita(tmp_path):
    fechas = pd.date_range("2017-06-01", "2017-08-15")
    tabla = pd.DataFrame({
        "id": range(len(fechas)), "date": fechas.strftime("%Y-%m-%d"),
        "store_nbr": 1, "family": "BEAUTY", "sales": np.arange(len(fechas)) % 9,
        "onpromotion": np.arange(len(fechas)) % 2,
    })
    ruta = tmp_path / "train.csv"
    tabla.to_csv(ruta, index=False)
    return {
        "seed": 42, "device": "cpu", "model": {"hidden_dim": 8, "output_dim": 1},
        "training": {"epochs": 2, "batch_size": 7, "learning_rate": 0.001, "num_workers": 0},
        "data": {"source": "favorita", "path": str(ruta)},
        "output": {"directory": str(tmp_path / "run")},
    }


def test_metricas_por_muestra_no_promedio_de_rmse():
    metricas = MetricasRegresion()
    metricas.actualizar(torch.tensor([[3.], [4.]]), torch.zeros(2, 1))
    metricas.actualizar(torch.tensor([[0.]]), torch.zeros(1, 1))
    resultado = metricas.resultado()
    assert resultado["mae"] == pytest.approx(7 / 3)
    assert resultado["mse"] == pytest.approx(25 / 3)
    assert resultado["rmse"] == pytest.approx(np.sqrt(25 / 3))
    assert resultado["n"] == 3


@pytest.mark.parametrize("predicciones,objetivos", [
    (torch.ones(2, 1), torch.ones(2)),
    (torch.ones(2, 2), torch.ones(2, 1)),
    (torch.tensor([[float("nan")]]), torch.ones(1, 1)),
    (torch.ones(1, 1), torch.tensor([[float("inf")]])),
])
def test_formas_y_finitud(predicciones, objetivos):
    with pytest.raises(ValueError):
        MetricasRegresion().actualizar(predicciones, objetivos)


def test_validacion_sin_gradientes_y_baseline_en_escala_original():
    class Modelo(nn.Module):
        def forward(self, x):
            assert not torch.is_grad_enabled()
            assert not self.training
            return torch.zeros(len(x), 1)
    # Rezago escalado [-1, 0, 1], media 10 y escala 2 => [8, 10, 12].
    loader = DataLoader(TensorDataset(torch.tensor([[-1.], [0.], [1.]]),
                                     torch.tensor([[7.], [12.], [12.]])), batch_size=2)
    resultado = ejecutar_epoca(Modelo(), loader, nn.MSELoss(), baseline=(0, 10., 2.))
    assert resultado["baseline"]["mae"] == pytest.approx(1.)
    assert resultado["baseline"]["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert resultado["baseline"]["n"] == resultado["n"] == 3


def test_perdida_y_gradientes_no_finitos():
    loader = DataLoader(TensorDataset(torch.ones(2, 1), torch.ones(2, 1)), batch_size=2)
    modelo = nn.Linear(1, 1)
    with pytest.raises(ValueError, match="Perdida no finita"):
        ejecutar_epoca(modelo, loader, lambda p, y: p.sum() * float("inf"))
    optimizer = torch.optim.Adam(modelo.parameters())
    original = modelo.weight.detach().clone()
    hook = modelo.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
    with pytest.raises(ValueError, match="Gradientes"):
        ejecutar_epoca(modelo, loader, nn.MSELoss(), optimizer)
    hook.remove()
    assert torch.equal(modelo.weight, original)
    assert not optimizer.state


def test_persistencia_y_test_no_utilizado(config_favorita, monkeypatch):
    original_builder = construir_loaders_favorita

    class TestProhibido:
        def __iter__(self):
            pytest.fail("No se permite evaluar test durante entrenamiento")

    loaders_modulo = importlib.import_module("src.loaders_favorita")
    preparados = []
    def construir(*args, **kwargs):
        resultado = original_builder(*args, **kwargs)
        resultado.loaders["test"] = TestProhibido()
        preparados.append(resultado)
        return resultado
    monkeypatch.setattr(loaders_modulo, "construir_loaders_favorita", construir)
    ruta = train(config_favorita)
    directorio = ruta.parent
    assert set(p.name for p in directorio.iterdir()) == {
        "best.pt", "last.pt", "metricas.csv", "resumen.json", "configuracion.yaml"
    }
    checkpoint = torch.load(directorio / "last.pt", map_location="cpu", weights_only=False)
    assert checkpoint["epoch"] == 2
    assert checkpoint["input_dim"] == 10
    assert checkpoint["optimizer_state_dict"]["state"]
    assert len(checkpoint["nombres_columnas"]) == 10
    assert checkpoint["config"]["training"]["epochs"] == 2
    assert checkpoint["preparador"].nombres_columnas == preparados[0].nombres_columnas
    for clave, valor in preparados[0].preparador.parametros_escalado.items():
        np.testing.assert_array_equal(checkpoint["preparador"].parametros_escalado[clave], valor)
    modelo = MLP(input_dim=10, hidden_dim=8)
    modelo.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.Adam(modelo.parameters())
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    assert optimizer.state
    x, y = next(iter(preparados[0].loaders["validacion"]))
    with torch.no_grad():
        assert torch.isfinite(modelo(x)).all()
    metricas = pd.read_csv(directorio / "metricas.csv")
    assert len(metricas) == 2
    assert metricas["n_validacion"].eq(30).all()
    resumen = json.loads((directorio / "resumen.json").read_text())
    assert resumen["test_evaluado"] is False
    assert resumen["mejor_val_rmse"] == pytest.approx(metricas.val_rmse.min())


def test_mejor_epoca_intermedia_no_ultima(config_favorita, monkeypatch):
    config_favorita["training"]["epochs"] = 3
    original = modulo.ejecutar_epoca
    valores = iter([3., 1., 2.])
    estados = []
    def ejecutar(modelo, loader, criterio, optimizer=None, **kwargs):
        resultado = original(modelo, loader, criterio, optimizer, **kwargs)
        if optimizer is None:
            resultado["rmse"] = next(valores)
            resultado["mse"] = resultado["rmse"] ** 2
            estados.append({k: v.detach().clone() for k, v in modelo.state_dict().items()})
        return resultado
    monkeypatch.setattr(modulo, "ejecutar_epoca", ejecutar)
    ruta = train(config_favorita)
    best = torch.load(ruta, weights_only=False)
    last = torch.load(ruta.parent / "last.pt", weights_only=False)
    assert best["epoch"] == 2 and last["epoch"] == 3
    assert last["best_epoch"] == 2
    for clave, valor in best["model_state_dict"].items():
        assert torch.equal(valor, estados[1][clave])
    assert best["optimizer_state_dict"]["state"]
    assert last["best_validation_rmse"] == 1.


def test_fallo_segunda_epoca_preserva_checkpoint(config_favorita, monkeypatch):
    original = modulo.ejecutar_epoca
    llamadas = 0
    huella = None
    from pathlib import Path
    ruta = Path(config_favorita["output"]["directory"]) / "last.pt"
    def ejecutar(modelo, loader, criterio, optimizer=None, **kwargs):
        nonlocal llamadas, huella
        if optimizer is not None:
            llamadas += 1
            if llamadas == 2:
                huella = hashlib.sha256(ruta.read_bytes()).hexdigest()
                raise ValueError("Gradientes no finitos")
        return original(modelo, loader, criterio, optimizer, **kwargs)
    monkeypatch.setattr(modulo, "ejecutar_epoca", ejecutar)
    with pytest.raises(ValueError, match="Gradientes"):
        train(config_favorita)
    assert hashlib.sha256(ruta.read_bytes()).hexdigest() == huella
    assert torch.load(ruta, weights_only=False)["epoch"] == 1
    assert len(pd.read_csv(ruta.parent / "metricas.csv")) == 1


def test_fuente_desconocida_rechazada():
    config = {"data": {"source": "error"}, "model": {"input_dim": 2}}
    with pytest.raises(ValueError, match="source"):
        train(config)
    with pytest.raises(ValueError, match="source"):
        build_dataloaders(config)


def test_no_sobrescribe_corrida(config_favorita):
    ruta = train(config_favorita)
    contenido = ruta.read_bytes()
    with pytest.raises(FileExistsError):
        train(config_favorita)
    assert ruta.read_bytes() == contenido
