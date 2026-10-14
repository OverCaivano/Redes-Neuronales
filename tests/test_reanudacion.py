"""Reanudacion entre epocas con CSV pequenos y sin evaluar test."""

import copy
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.train import train
from src.reanudacion import cargar_last
from src.checkpoints import guardar_epoca

modulo = importlib.import_module("src.train")


@pytest.fixture
def config(tmp_path):
    fechas = pd.date_range("2017-06-01", "2017-08-15")
    ruta = tmp_path / "datos.csv"
    pd.DataFrame({
        "id": range(len(fechas)), "date": fechas.strftime("%Y-%m-%d"),
        "store_nbr": 1, "family": "BEAUTY", "sales": np.arange(len(fechas)) % 7,
        "onpromotion": np.arange(len(fechas)) % 3,
    }).to_csv(ruta, index=False)
    return {
        "seed": 42, "device": "cpu", "model": {"hidden_dim": 8, "output_dim": 1},
        "training": {"epochs": 2, "batch_size": 3, "learning_rate": 0.001, "num_workers": 0},
        "data": {"source": "favorita", "path": str(ruta)},
        "output": {"directory": str(tmp_path / "interrumpida")},
    }


def huellas(directorio):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directorio.iterdir() if p.is_file()}


def iguales(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for clave in a:
            iguales(a[clave], b[clave])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            iguales(x, y)
    else:
        assert a == b


def test_continua_vs_interrumpida_sin_fit(config, monkeypatch):
    continua = copy.deepcopy(config)
    continua["training"]["epochs"] = 4
    continua["output"]["directory"] += "_continua"
    train(continua)
    train(config)
    directorio = Path(config["output"]["directory"])
    anterior = cargar_last(directorio / "last.pt")
    historial = copy.deepcopy(anterior["historial"])

    def prohibido(*args, **kwargs):
        pytest.fail("No debe reajustar transformaciones")
    monkeypatch.setattr(OneHotEncoder, "fit", prohibido)
    monkeypatch.setattr(StandardScaler, "fit", prohibido)
    monkeypatch.setattr(StandardScaler, "partial_fit", prohibido)
    config["training"]["epochs"] = 4
    train(config, resume=directorio / "last.pt")
    final = cargar_last(directorio / "last.pt")
    referencia = cargar_last(Path(continua["output"]["directory"]) / "last.pt")
    assert final["epoch"] == 4
    assert final["historial"][:2] == historial
    iguales(final["model_state_dict"], referencia["model_state_dict"])
    iguales(final["optimizer_state_dict"], referencia["optimizer_state_dict"])
    iguales(final["loader_rng_state"], referencia["loader_rng_state"])
    iguales(final["torch_rng_state"], referencia["torch_rng_state"])
    assert final["best_epoch"] == referencia["best_epoch"]
    for real, esperado in zip(final["historial"], referencia["historial"]):
        assert {k: v for k, v in real.items() if k != "segundos_epoca"} == {
            k: v for k, v in esperado.items() if k != "segundos_epoca"
        }
    assert pd.read_csv(directorio / "metricas.csv").epoca.tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize("total", [1, 2])
def test_total_alcanzado_no_escribe_ni_entrena(config, monkeypatch, capsys, total):
    train(config)
    directorio = Path(config["output"]["directory"])
    antes = huellas(directorio)
    def prohibido(*args, **kwargs):
        pytest.fail("No debe ejecutar otra epoca")
    monkeypatch.setattr(modulo, "ejecutar_epoca", prohibido)
    config["training"]["epochs"] = total
    train(config, resume=directorio / "last.pt")
    assert huellas(directorio) == antes
    assert "Sin epocas nuevas" in capsys.readouterr().out


@pytest.mark.parametrize("seccion,clave,valor", [
    ("model", "hidden_dim", 16), ("data", "tiendas", [2]),
    ("data", "familias", ["NUEVA"]), ("training", "batch_size", 9),
    ("training", "learning_rate", 0.1), ("model", "input_dim", 99),
])
def test_config_incompatible_no_escribe(config, seccion, clave, valor):
    train(config)
    directorio = Path(config["output"]["directory"])
    antes = huellas(directorio)
    config[seccion][clave] = valor
    config["training"]["epochs"] = 3
    with pytest.raises(ValueError, match="incompatible"):
        train(config, resume=directorio / "last.pt")
    assert huellas(directorio) == antes


def test_dataset_modificado_rechazado(config):
    train(config)
    ruta = Path(config["output"]["directory"]) / "last.pt"
    tabla = pd.read_csv(config["data"]["path"])
    tabla.loc[0, "sales"] += 1
    tabla.to_csv(config["data"]["path"], index=False)
    config["training"]["epochs"] = 3
    with pytest.raises(ValueError, match="SHA-256"):
        train(config, resume=ruta)


def test_periodos_modificados_rechazados(config, monkeypatch):
    train(config)
    from src.preprocesamiento_temporal import PERIODOS
    monkeypatch.setitem(PERIODOS, "entrenamiento", ("2013-01-01", "2017-06-14"))
    with pytest.raises(ValueError, match="Particiones"):
        train(config, resume=Path(config["output"]["directory"]) / "last.pt")


@pytest.mark.parametrize("fallo", ["best", "incompleto", "ilegible", "columnas", "historial", "rng"])
def test_checkpoints_invalidos(config, fallo):
    train(config)
    ruta = Path(config["output"]["directory"]) / "last.pt"
    config["training"]["epochs"] = 3
    if fallo == "best":
        with pytest.raises(ValueError, match="last.pt"):
            train(config, resume=ruta.parent / "best.pt")
        return
    if fallo == "ilegible":
        ruta.write_bytes(b"checkpoint roto")
    else:
        estado = cargar_last(ruta)
        if fallo == "incompleto":
            del estado["optimizer_state_dict"]
        elif fallo == "columnas":
            estado["nombres_columnas"] = tuple(reversed(estado["nombres_columnas"]))
        elif fallo == "historial":
            estado["historial"][0]["epoca"] = 5
        else:
            estado["torch_rng_state"] = torch.ones(2)
        torch.save(estado, ruta)
    with pytest.raises(ValueError):
        train(config, resume=ruta)


def test_legacy_migra_con_advertencia_y_sin_fit(config, monkeypatch):
    train(config)
    ruta = Path(config["output"]["directory"]) / "last.pt"
    estado = cargar_last(ruta)
    estado["format_version"] = 1
    for clave in ("dataset_sha256", "periodos", "python_rng_state", "numpy_rng_state", "loaders_rng_state"):
        estado.pop(clave)
    torch.save(estado, ruta)
    def prohibido(*args, **kwargs):
        pytest.fail("Legacy no debe reajustar")
    monkeypatch.setattr(OneHotEncoder, "fit", prohibido)
    monkeypatch.setattr(StandardScaler, "fit", prohibido)
    config["training"]["epochs"] = 3
    with pytest.warns(UserWarning, match="sin SHA-256"):
        train(config, resume=ruta)
    assert cargar_last(ruta)["format_version"] == 2


def test_mejor_anterior_se_conserva_si_no_mejora(config, monkeypatch):
    train(config)
    ruta = Path(config["output"]["directory"]) / "last.pt"
    anterior = cargar_last(ruta)
    best_bytes = (ruta.parent / "best.pt").read_bytes()
    ejecutar = modulo.ejecutar_epoca
    def empeorar(modelo, loader, criterion, optimizer=None, **kwargs):
        metricas = ejecutar(modelo, loader, criterion, optimizer, **kwargs)
        if optimizer is None:
            metricas["rmse"] = 1e6
            metricas["mse"] = 1e12
        return metricas
    monkeypatch.setattr(modulo, "ejecutar_epoca", empeorar)
    config["training"]["epochs"] = 3
    train(config, resume=ruta)
    assert (ruta.parent / "best.pt").read_bytes() == best_bytes
    assert cargar_last(ruta)["best_epoch"] == anterior["best_epoch"]


def test_fallo_reanudando_conserva_todos_los_archivos(config, monkeypatch):
    train(config)
    directorio = Path(config["output"]["directory"])
    antes = huellas(directorio)
    def fallo(*args, **kwargs):
        raise ValueError("Gradientes no finitos")
    monkeypatch.setattr(modulo, "ejecutar_epoca", fallo)
    config["training"]["epochs"] = 4
    with pytest.raises(ValueError, match="Gradientes"):
        train(config, resume=directorio / "last.pt")
    assert huellas(directorio) == antes


def test_fallo_publicacion_restaura_archivos(config, monkeypatch):
    train(config)
    directorio = Path(config["output"]["directory"])
    antes = huellas(directorio)
    checkpoints = importlib.import_module("src.checkpoints")
    reemplazar = checkpoints.os.replace
    def fallo(origen, destino):
        if Path(destino).name == "last.pt" and str(origen).endswith("last.pt"):
            raise OSError("fallo simulado")
        return reemplazar(origen, destino)
    monkeypatch.setattr(checkpoints.os, "replace", fallo)
    config["training"]["epochs"] = 3
    with pytest.raises(OSError, match="simulado"):
        train(config, resume=directorio / "last.pt")
    assert huellas(directorio) == antes
    assert not list(directorio.glob(".epoca-*"))
