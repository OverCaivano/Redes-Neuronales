"""Evaluacion final diaria a un paso, sin fit, optimizador ni seleccion en test."""

import json
import math
from pathlib import Path
import platform
from time import perf_counter
import warnings

import sklearn
from sklearn.exceptions import InconsistentVersionWarning
import torch
from torch import nn

from src.loaders_favorita import construir_loaders_favorita
from src.models import MLP
from src.preprocesamiento_temporal import PERIODOS
from src.reanudacion import huella_dataset
from src.train import ejecutar_epoca
from src.transformaciones import PreparadorEntradas, COLUMNAS_NUMERICAS

RAIZ_RESULTADOS = Path(__file__).resolve().parents[1] / "runs"


def cargar_best_favorita(ruta):
    """Solo checkpoints propios/confiables: weights_only=False carga objetos Python."""
    ruta = Path(ruta)
    if ruta.name != "best.pt":
        raise ValueError("La evaluacion final requiere best.pt seleccionado por validacion")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", InconsistentVersionWarning)
            estado = torch.load(ruta, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        raise
    except Exception as error:
        raise ValueError("Checkpoint ilegible o entorno incompatible") from error
    requeridas = {"format_version", "config", "model_state_dict", "preparador", "input_dim",
                  "nombres_columnas", "dataset_sha256", "periodos", "epoch", "best_epoch",
                  "historial", "best_validation_rmse", "versiones"}
    if not isinstance(estado, dict) or not requeridas.issubset(estado):
        raise ValueError("Checkpoint incompleto: se requiere v2 con huella del dataset")
    try:
        if estado["format_version"] != 2:
            raise ValueError("Se requiere checkpoint v2 con huella del dataset")
        config = estado["config"]
        if config["data"]["source"] != "favorita":
            raise ValueError("El checkpoint no corresponde a Favorita")
        versiones = estado["versiones"]
        if (versiones["sklearn"] != sklearn.__version__
            or versiones["torch"].split("+")[0] != torch.__version__.split("+")[0]
            or versiones["python"].split(".")[:2] != platform.python_version().split(".")[:2]):
            raise ValueError("Versiones del entorno incompatibles con el checkpoint")
        if estado["periodos"] != {k: list(v) for k, v in PERIODOS.items()}:
            raise ValueError("Particiones temporales incompatibles")
        preparador = estado["preparador"]
        if not isinstance(preparador, PreparadorEntradas):
            raise ValueError("Preparador guardado invalido")
        columnas = tuple(estado["nombres_columnas"])
        if (columnas != preparador.nombres_columnas or len(columnas) != estado["input_dim"]
            or config["model"]["input_dim"] != estado["input_dim"]):
            raise ValueError("Dimension u orden de columnas incompatible")
        historial = estado["historial"]
        if (not historial or [f["epoca"] for f in historial] != list(range(1, estado["epoch"] + 1))
            or any(not math.isfinite(f["val_rmse"]) for f in historial)):
            raise ValueError("Historial de validacion invalido")
        mejor = min(historial, key=lambda f: f["val_rmse"])
        if (estado["epoch"] != estado["best_epoch"] or mejor["epoca"] != estado["epoch"]
            or mejor["val_rmse"] != estado["best_validation_rmse"]):
            raise ValueError("best.pt no coincide con la seleccion por validacion")
        if config["model"]["output_dim"] != 1:
            raise ValueError("Favorita requiere output_dim=1")
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Estructura del checkpoint incompatible") from error
    return estado


def reconstruir_modelo_favorita(estado):
    """Recupera pesos finitos en CPU sin optimizador; compartido con consultas."""
    modelo = MLP(**estado["config"]["model"])
    try:
        modelo.load_state_dict(estado["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("Pesos incompatibles con la arquitectura guardada") from error
    if any(not torch.isfinite(p).all() for p in modelo.parameters()):
        raise ValueError("Pesos no finitos")
    modelo.eval()
    return modelo


def evaluar_favorita(model_path, *, data_path=None, output_dir=None):
    """Reconstruye con el preparador guardado y recorre SOLO test, en CPU.

    Los rezagos se calculan antes del corte: evaluacion diaria a un paso con
    historia observada hasta t-1. No es un pronostico multidia de origen fijo.
    output_dir, si se indica, debe ser una carpeta NUEVA dentro de runs/ y
    fuera de la corrida original. No escribe nada por defecto.
    """
    inicio = perf_counter()
    model_path = Path(model_path)
    destino = None
    if output_dir is not None:
        destino = Path(output_dir).resolve()
        if (not destino.is_relative_to(RAIZ_RESULTADOS.resolve())
            or destino == RAIZ_RESULTADOS.resolve()
            or destino.is_relative_to(model_path.resolve().parent)):
            raise ValueError("La salida debe estar en runs/, fuera de la corrida original")
        if destino.exists():
            raise FileExistsError("La carpeta de evaluacion ya existe; use una nueva")
    estado = cargar_best_favorita(model_path)
    config = estado["config"]
    ruta_csv = Path(data_path or config["data"]["path"])
    digest = huella_dataset(ruta_csv)
    if digest != estado["dataset_sha256"]:
        raise ValueError("Dataset incompatible: SHA-256 diferente; usar el train.csv original")
    modelo = reconstruir_modelo_favorita(estado)
    preparado = construir_loaders_favorita(
        ruta_csv, tiendas=config["data"].get("tiendas"), familias=config["data"].get("familias"),
        batch_size=int(config["training"]["batch_size"]), semilla=int(config.get("seed", 42)),
        num_workers=0, preparador=estado["preparador"],
    )
    if (preparado.input_dim != estado["input_dim"]
        or tuple(preparado.nombres_columnas) != tuple(estado["nombres_columnas"])):
        raise ValueError("Dimension u orden de columnas incompatible al reconstruir datos")
    indice = COLUMNAS_NUMERICAS.index("ventas_hace_7_dias")
    parametros = preparado.preparador.parametros_escalado
    baseline = (preparado.nombres_columnas.index("ventas_hace_7_dias"),
                float(parametros["media"][indice]), float(parametros["escala"][indice]))
    resultado = ejecutar_epoca(modelo, preparado.loaders["test"], nn.MSELoss(), baseline=baseline)
    ingenuo = resultado.pop("baseline")
    if resultado["n"] != preparado.filas_por_particion["test"] or ingenuo["n"] != resultado["n"]:
        raise ValueError("Cantidad de filas de evaluacion incompatible")
    fechas = preparado.metadatos["test"]["fecha"]
    informe = {
        "particion": "test_interno", "protocolo": "diario_a_un_paso_historia_observada_hasta_t-1",
        "checkpoint": str(model_path.resolve()), "checkpoint_sha256": huella_dataset(model_path),
        "epoca_seleccionada_por_validacion": estado["epoch"], "device": "cpu",
        "dataset_sha256": digest, "input_dim": preparado.input_dim,
        "nombres_columnas": list(preparado.nombres_columnas),
        "tiendas": config["data"].get("tiendas"), "familias": config["data"].get("familias"),
        "periodos": estado["periodos"], "filas_por_particion": preparado.filas_por_particion,
        "fecha_min": str(fechas.min().date()), "fecha_max": str(fechas.max().date()),
        "mlp": resultado, "baseline_semanal": ingenuo, "segundos_total": perf_counter() - inicio,
    }
    if destino is not None:
        destino.mkdir(parents=True, exist_ok=False)
        (destino / "evaluacion.json").write_text(json.dumps(informe, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Test interno {informe['fecha_min']} a {informe['fecha_max']}: {resultado['n']} ejemplos; "
          f"epoca {estado['epoch']} seleccionada por validacion; CPU")
    print(f"MLP: MAE={resultado['mae']:.6f}, RMSE={resultado['rmse']:.6f}")
    print(f"Baseline semanal: MAE={ingenuo['mae']:.6f}, RMSE={ingenuo['rmse']:.6f}")
    print(f"Duracion: {informe['segundos_total']:.3f} s. Sin entrenamiento ni reajuste.")
    if destino is not None:
        print(f"Informe: {destino / 'evaluacion.json'}")
    return informe
