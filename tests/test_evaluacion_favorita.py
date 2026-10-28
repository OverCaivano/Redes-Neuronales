"""Checkpoints de prueba sin entrenar: pesos conocidos y CSV pequenos."""

import copy
import importlib
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch

from src.evaluacion_favorita import evaluar_favorita, cargar_best_favorita
from src.loaders_favorita import construir_loaders_favorita
from src.models import MLP
from src.preprocesamiento_temporal import PERIODOS
from src.reanudacion import huella_dataset

modulo = importlib.import_module("src.evaluacion_favorita")


@pytest.fixture
def checkpoint(tmp_path):
    fechas = pd.date_range("2017-06-01", "2017-08-15")
    tabla = pd.DataFrame({"id": range(len(fechas)), "date": fechas, "store_nbr": 1,
                          "family": "BEAUTY", "sales": np.arange(len(fechas), dtype=float),
                          "onpromotion": 0})
    csv = tmp_path / "train.csv"
    tabla.to_csv(csv, index=False)
    datos = construir_loaders_favorita(csv, batch_size=7)
    modelo = MLP(input_dim=datos.input_dim, hidden_dim=4)
    with torch.no_grad():
        for p in modelo.parameters():
            p.zero_()
        modelo.network[-1].bias.fill_(1.)
    estado = {
        "format_version": 2, "config": {
            "seed": 42, "model": {"input_dim": datos.input_dim, "hidden_dim": 4, "output_dim": 1},
            "training": {"batch_size": 7},
            "data": {"source": "favorita", "path": str(csv), "tiendas": [1], "familias": ["BEAUTY"]}},
        "model_state_dict": modelo.state_dict(), "preparador": datos.preparador,
        "input_dim": datos.input_dim, "nombres_columnas": datos.nombres_columnas,
        "dataset_sha256": huella_dataset(csv), "periodos": {k: list(v) for k, v in PERIODOS.items()},
        "epoch": 2, "best_epoch": 2, "best_validation_rmse": 1.,
        "historial": [{"epoca": 1, "val_rmse": 2.}, {"epoca": 2, "val_rmse": 1.}],
        "versiones": {"python": platform.python_version(), "torch": str(torch.__version__),
                      "sklearn": sklearn.__version__},
    }
    ruta = tmp_path / "corrida" / "best.pt"
    ruta.parent.mkdir()
    torch.save(estado, ruta)
    return ruta, csv, tabla


def test_metricas_test_sin_fit_ni_entrenamiento(checkpoint, monkeypatch):
    ruta, csv, tabla = checkpoint
    antes = huella_dataset(ruta), huella_dataset(csv)
    def prohibido(*a, **kw):
        pytest.fail("No se debe ajustar ni entrenar durante evaluacion")
    monkeypatch.setattr(OneHotEncoder, "fit", prohibido)
    monkeypatch.setattr(StandardScaler, "fit", prohibido)
    monkeypatch.setattr(StandardScaler, "partial_fit", prohibido)
    monkeypatch.setattr(torch.optim.Adam, "__init__", prohibido)
    original = modulo.construir_loaders_favorita
    class Prohibido:
        def __iter__(self):
            pytest.fail("Solo se puede recorrer test")
    def construir(*a, **kw):
        assert kw["preparador"] is not None
        datos = original(*a, **kw)
        datos.loaders["entrenamiento"] = Prohibido()
        datos.loaders["validacion"] = Prohibido()
        return datos
    monkeypatch.setattr(modulo, "construir_loaders_favorita", construir)
    forward = MLP.forward
    def comprobar(self, x):
        assert not self.training and not torch.is_grad_enabled()
        assert x.shape[1] == 10 and x.dtype == torch.float32
        return forward(self, x)
    monkeypatch.setattr(MLP, "forward", comprobar)
    resultado = evaluar_favorita(ruta)
    y = tabla.loc[tabla.date >= "2017-07-16", "sales"].to_numpy()
    assert resultado["mlp"]["n"] == resultado["baseline_semanal"]["n"] == 31
    assert resultado["mlp"]["mae"] == pytest.approx(np.abs(1 - y).mean())
    assert resultado["mlp"]["rmse"] == pytest.approx(np.sqrt(((1 - y) ** 2).mean()))
    assert resultado["baseline_semanal"]["mae"] == pytest.approx(7., abs=1e-5)
    assert resultado["baseline_semanal"]["rmse"] == pytest.approx(7., abs=1e-5)
    assert resultado["fecha_min"] == "2017-07-16"
    assert resultado["fecha_max"] == "2017-08-15"
    assert antes == (huella_dataset(ruta), huella_dataset(csv))


@pytest.mark.parametrize("cambio,patron", [
    ("input_dim", "Dimension"), ("columnas", "columnas"), ("periodos", "temporales"),
    ("legacy", "v2"), ("versiones", "Versiones"), ("pesos", "Pesos"),
    ("incompleto", "incompleto"), ("best", "validacion"), ("no_finito", "finitos"),
])
def test_checkpoint_incompatible(checkpoint, cambio, patron):
    ruta, _, _ = checkpoint
    estado = torch.load(ruta, weights_only=False)
    if cambio == "input_dim": estado["input_dim"] += 1
    elif cambio == "columnas": estado["nombres_columnas"] = tuple(reversed(estado["nombres_columnas"]))
    elif cambio == "periodos": estado["periodos"]["test"][0] = "2017-07-17"
    elif cambio == "legacy": estado["format_version"] = 1
    elif cambio == "versiones": estado["versiones"]["sklearn"] = "0.0"
    elif cambio == "pesos": estado["model_state_dict"]["network.0.weight"] = torch.zeros(1, 1)
    elif cambio == "incompleto": del estado["preparador"]
    elif cambio == "best": estado["best_epoch"] = 1
    else: estado["model_state_dict"]["network.0.weight"].fill_(float("nan"))
    torch.save(estado, ruta)
    with pytest.raises(ValueError, match=patron):
        evaluar_favorita(ruta)


def test_csv_incompatible_rechazado_antes_de_loaders(checkpoint, monkeypatch):
    ruta, csv, _ = checkpoint
    csv.write_text(csv.read_text() + "\n")
    monkeypatch.setattr(modulo, "construir_loaders_favorita", lambda *a, **k: pytest.fail("No cargar datos diferentes"))
    with pytest.raises(ValueError, match="SHA-256"):
        evaluar_favorita(ruta)


def test_otra_ruta_mismo_csv_y_salida_nueva(checkpoint, tmp_path, monkeypatch):
    ruta, csv, _ = checkpoint
    copia = tmp_path / "otra_ubicacion.csv"
    copia.write_bytes(csv.read_bytes())
    raiz = tmp_path / "runs"
    monkeypatch.setattr(modulo, "RAIZ_RESULTADOS", raiz)
    salida = raiz / "evaluacion"
    resultado = evaluar_favorita(ruta, data_path=copia, output_dir=salida)
    import json
    assert json.loads((salida / "evaluacion.json").read_text()) == resultado
    with pytest.raises(FileExistsError):
        evaluar_favorita(ruta, output_dir=salida)
    with pytest.raises(ValueError, match="fuera"):
        evaluar_favorita(ruta, output_dir=ruta.parent)


def test_rechaza_checkpoint_ilegible_y_last(tmp_path):
    ruta = tmp_path / "best.pt"
    ruta.write_bytes(b"roto")
    with pytest.raises(ValueError, match="ilegible"):
        cargar_best_favorita(ruta)
    with pytest.raises(ValueError, match="best.pt"):
        cargar_best_favorita(tmp_path / "last.pt")


def test_predicciones_forma_incorrecta(checkpoint, monkeypatch):
    monkeypatch.setattr(MLP, "forward", lambda self, x: torch.ones(len(x)))
    with pytest.raises(ValueError, match="forma"):
        evaluar_favorita(checkpoint[0])


def test_evaluador_sintetico_preservado(tmp_path):
    from src.evaluate import evaluate
    from src.data_loader import build_dataloaders
    config = {"seed": 42, "model": {"input_dim": 2, "hidden_dim": 4, "output_dim": 1},
              "training": {"batch_size": 3}, "validation": {"split": .3},
              "data": {"source": "synthetic", "n_samples": 20, "noise": .1}}
    modelo = MLP(**config["model"])
    ruta = tmp_path / "sintetico.pt"
    torch.save({"config": config, "model_state_dict": modelo.state_dict()}, ruta)
    _, loader = build_dataloaders(config)
    with torch.no_grad():
        errores = torch.cat([(modelo(x) - y).square() for x, y in loader])
    assert evaluate(ruta) == pytest.approx(errores.mean().item())


def test_cli_favorita(checkpoint, monkeypatch):
    from src.evaluate import main
    recibido = {}
    monkeypatch.setattr(modulo, "evaluar_favorita", lambda ruta, **kw: recibido.update(ruta=ruta, **kw))
    monkeypatch.setattr("sys.argv", ["evaluate", "--source", "favorita", "--model_path", str(checkpoint[0])])
    main()
    assert recibido == {"ruta": str(checkpoint[0]), "data_path": None, "output_dir": None}
