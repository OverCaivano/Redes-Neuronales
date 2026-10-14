"""Validacion de checkpoints propios antes de reanudar una corrida."""

import copy
import csv
import hashlib
import json
import math
from pathlib import Path

import torch

from src.preprocesamiento_temporal import PERIODOS


def huella_dataset(ruta):
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def cargar_last(ruta):
    """Solo archivos propios/confiables: contienen objetos sklearn serializados."""
    ruta = Path(ruta)
    if ruta.name != "last.pt":
        raise ValueError("Reanudar requiere last.pt, no best.pt")
    try:
        estado = torch.load(ruta, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        raise
    except Exception as error:
        raise ValueError(f"Checkpoint ilegible o incompatible: {ruta}") from error
    requeridas = {
        "format_version", "model_state_dict", "optimizer_state_dict", "epoch",
        "config", "input_dim", "nombres_columnas", "preparador", "best_epoch",
        "best_validation_rmse", "historial", "torch_rng_state", "loader_rng_state",
    }
    if not isinstance(estado, dict) or not requeridas.issubset(estado):
        raise ValueError("Checkpoint incompleto: faltan campos obligatorios")
    if estado["format_version"] not in (1, 2):
        raise ValueError("Version de checkpoint no soportada")
    if estado["format_version"] == 2 and not {
        "dataset_sha256", "periodos", "python_rng_state", "numpy_rng_state",
        "loaders_rng_state", "segundos_entrenamiento",
    }.issubset(estado):
        raise ValueError("Checkpoint v2 incompleto")
    try:
        historial = estado["historial"]
        epoca = estado["epoch"]
        if not isinstance(epoca, int) or epoca < 1 or len(historial) != epoca:
            raise ValueError("Epoca e historial incompatibles")
        if [fila["epoca"] for fila in historial] != list(range(1, epoca + 1)):
            raise ValueError("Historial de epocas discontinuo")
        if not all(math.isfinite(v) for fila in historial for v in fila.values()):
            raise ValueError("Historial no finito")
        mejor = min(historial, key=lambda fila: fila["val_rmse"])
        if estado["best_epoch"] != mejor["epoca"] or estado["best_validation_rmse"] != mejor["val_rmse"]:
            raise ValueError("Mejor epoca incompatible con historial")
        if len(estado["nombres_columnas"]) != estado["input_dim"]:
            raise ValueError("Dimension incompatible con columnas")
        if tuple(estado["preparador"].nombres_columnas) != tuple(estado["nombres_columnas"]):
            raise ValueError("Columnas incompatibles con preparador")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Estructura del checkpoint incompatible") from error
    return estado


def _config_comparable(config):
    resultado = copy.deepcopy(config)
    resultado.pop("output", None)
    resultado["training"].pop("epochs", None)
    resultado["model"].pop("input_dim", None)
    resultado["data"].pop("path", None)
    return resultado


def validar_reanudacion(ruta, estado, config, digest):
    """Comprueba configuracion, datos y archivos hermanos sin escribir."""
    if _config_comparable(config) != _config_comparable(estado["config"]):
        raise ValueError("Configuracion incompatible: arquitectura, filtros, dispositivo o entrenamiento")
    if config["model"].get("input_dim", estado["input_dim"]) != estado["input_dim"]:
        raise ValueError("input_dim incompatible")
    if estado.get("dataset_sha256", digest) != digest:
        raise ValueError("Dataset incompatible: SHA-256 diferente")
    directorio = Path(ruta).parent
    try:
        resumen = json.loads((directorio / "resumen.json").read_text(encoding="utf-8"))
        with (directorio / "metricas.csv").open(encoding="utf-8", newline="") as archivo:
            filas = list(csv.DictReader(archivo))
        best = torch.load(directorio / "best.pt", map_location="cpu", weights_only=False)
    except Exception as error:
        raise ValueError("Faltan resultados o best.pt es ilegible; restaure el respaldo") from error
    periodos = {nombre: list(limites) for nombre, limites in PERIODOS.items()}
    registrados = estado.get("periodos", resumen.get("periodos"))
    if registrados != periodos or resumen.get("periodos") != periodos:
        raise ValueError("Particiones temporales incompatibles")
    if resumen.get("epocas_completadas") != estado["epoch"]:
        raise ValueError("Resumen y checkpoint desincronizados; restaure el respaldo")
    if len(filas) != len(estado["historial"]):
        raise ValueError("metricas.csv e historial desincronizados")
    for fila, registrada in zip(filas, estado["historial"]):
        if set(fila) != set(registrada) or any(
            not math.isclose(float(fila[k]), float(v), rel_tol=1e-12, abs_tol=1e-12)
            for k, v in registrada.items()
        ):
            raise ValueError("metricas.csv incompatible con historial")
    if (
        not isinstance(best, dict) or best.get("epoch") != estado["best_epoch"]
        or best.get("best_validation_rmse") != estado["best_validation_rmse"]
        or best.get("input_dim") != estado["input_dim"]
        or tuple(best.get("nombres_columnas", ())) != tuple(estado["nombres_columnas"])
    ):
        raise ValueError("best.pt incompatible con last.pt; restaure el respaldo")
    return resumen
