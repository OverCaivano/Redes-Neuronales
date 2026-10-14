"""Entrenamiento sintetico o Favorita; test no participa en seleccion."""

import argparse
import copy
import csv
import io
import json
import math
from pathlib import Path
import platform
import random
import warnings

import numpy as np
import sys
from time import perf_counter

import torch
import torch.nn as nn
import yaml

# Conserva tanto python src/train.py como python -m src.train.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import build_dataloaders
from src.models import MLP
from src.metricas import MetricasRegresion, validar_predicciones
from src.checkpoints import guardar_checkpoint, guardar_epoca
from src.reanudacion import cargar_last, huella_dataset, validar_reanudacion


def load_config(path):
    with Path(path).open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _comprobar_tensores_finitos(valor, contexto):
    if isinstance(valor, torch.Tensor):
        if not torch.isfinite(valor).all():
            raise ValueError(f"{contexto} no finitos; no se guardara esta epoca")
    elif isinstance(valor, dict):
        for elemento in valor.values():
            _comprobar_tensores_finitos(elemento, contexto)
    elif isinstance(valor, (list, tuple)):
        for elemento in valor:
            _comprobar_tensores_finitos(elemento, contexto)


def ejecutar_epoca(model, loader, criterion, optimizer=None, *, device="cpu", baseline=None):
    """baseline = (indice de entrada, media, escala) del rezago semanal."""
    training = optimizer is not None
    model.train(training)
    metricas = MetricasRegresion()
    ingenuas = MetricasRegresion() if baseline is not None else None
    with torch.set_grad_enabled(training):
        for features, targets in loader:
            features, targets = features.to(device), targets.to(device)
            if not torch.isfinite(features).all():
                raise ValueError("Entradas no finitas")
            if training:
                optimizer.zero_grad(set_to_none=True)
            predictions = model(features)
            validar_predicciones(predictions, targets)
            loss = criterion(predictions, targets)
            if not torch.isfinite(loss):
                raise ValueError("Perdida no finita; no se guardara esta epoca")
            metricas.actualizar(predictions, targets)
            if ingenuas is not None:
                indice, media, escala = baseline
                prediccion_ingenua = features[:, indice:indice + 1].double() * escala + media
                ingenuas.actualizar(prediccion_ingenua, targets)
            if training:
                loss.backward()
                for parametro in model.parameters():
                    if parametro.grad is not None:
                        _comprobar_tensores_finitos(parametro.grad, "Gradientes")
                optimizer.step()
                _comprobar_tensores_finitos(model.state_dict(), "Pesos")
                _comprobar_tensores_finitos(optimizer.state, "Estado del optimizador")
    resultado = metricas.resultado()
    if ingenuas is not None:
        resultado["baseline"] = ingenuas.resultado()
    return resultado


def _run_epoch(model, loader, criterion, optimizer=None):
    """Contrato anterior del pipeline sintetico: devuelve MSE."""
    return ejecutar_epoca(model, loader, criterion, optimizer)["mse"]


def _train_synthetic(config):
    # Conserva tambien el CSV tabular numerico anterior.
    torch.manual_seed(int(config.get("seed", 42)))
    train_loader, validation_loader = build_dataloaders(config)
    model = MLP(**{key: int(value) for key, value in config["model"].items()})
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(config["training"].get("learning_rate", 0.001))
    )
    best_validation_loss, best_state_dict = float("inf"), None
    epochs = int(config["training"].get("epochs", 5))
    if epochs <= 0:
        raise ValueError("epochs debe ser positivo")
    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer)
        validation_loss = _run_epoch(model, validation_loader, criterion)
        print(f"Epoch {epoch}/{epochs} - train_loss={train_loss:.4f} val_loss={validation_loss:.4f}")
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_state_dict = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model_path = Path("models/best_model.pt")
    guardar_checkpoint(model_path, {
        "model_state_dict": best_state_dict, "config": config,
        "best_validation_loss": best_validation_loss,
    })
    return model_path


def _train_favorita(config, resume=None):
    from src.loaders_favorita import construir_loaders_favorita
    from src.transformaciones import COLUMNAS_NUMERICAS
    from src.preprocesamiento_temporal import PERIODOS
    import sklearn

    config = copy.deepcopy(config)
    epochs = int(config["training"]["epochs"])
    learning_rate = float(config["training"]["learning_rate"])
    if epochs <= 0 or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("epochs y learning_rate deben ser positivos y finitos")
    device = torch.device(config.get("device", "cpu"))
    if device.type not in ("cpu", "cuda"):
        raise ValueError("device debe ser cpu o cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA no esta disponible")
    if int(config["model"].get("output_dim", 1)) != 1:
        raise ValueError("Favorita requiere output_dim=1")
    estado_previo = cargar_last(resume) if resume is not None else None
    digest = huella_dataset(config["data"]["path"])
    resumen_anterior = {}
    directorio = Path(resume).parent if resume is not None else Path(config["output"]["directory"])
    if estado_previo is not None:
        resumen_anterior = validar_reanudacion(resume, estado_previo, config, digest)
        config["output"]["directory"] = str(directorio)
        if epochs <= estado_previo["epoch"]:
            print(f"Sin epocas nuevas: completadas {estado_previo['epoch']}, total solicitado {epochs}.")
            return directorio / "best.pt"
    if estado_previo is None and directorio.exists() and any(directorio.iterdir()):
        raise FileExistsError(f"La carpeta de resultados no esta vacia: {directorio}; use otra ruta")

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    inicio = perf_counter()
    datos = config["data"]
    preparado = construir_loaders_favorita(
        datos["path"], tiendas=datos.get("tiendas"), familias=datos.get("familias"),
        batch_size=int(config["training"]["batch_size"]), semilla=seed,
        num_workers=int(config["training"].get("num_workers", 0)),
        preparador=estado_previo["preparador"] if estado_previo else None,
    )
    if estado_previo is not None:
        if (
            preparado.input_dim != estado_previo["input_dim"]
            or tuple(preparado.nombres_columnas) != tuple(estado_previo["nombres_columnas"])
            or preparado.filas_por_particion != resumen_anterior["filas_por_particion"]
        ):
            raise ValueError("Dataset, dimensiones o columnas incompatibles al reconstruir loaders")
    config["model"]["input_dim"] = preparado.input_dim
    config["model"]["output_dim"] = 1
    modelo = MLP(
        input_dim=preparado.input_dim, hidden_dim=int(config["model"]["hidden_dim"]), output_dim=1
    ).to(device)
    optimizer = torch.optim.Adam(modelo.parameters(), lr=learning_rate)
    criterio = nn.MSELoss()
    rezago = "ventas_hace_7_dias"
    parametros = preparado.preparador.parametros_escalado
    indice_numerico = COLUMNAS_NUMERICAS.index(rezago)
    baseline = (
        preparado.nombres_columnas.index(rezago),
        float(parametros["media"][indice_numerico]),
        float(parametros["escala"][indice_numerico]),
    )
    directorio.mkdir(parents=True, exist_ok=True)
    segundos_preparacion = perf_counter() - inicio
    historial = []
    mejor_rmse, mejor_epoca = float("inf"), None
    primera_epoca = 1
    if estado_previo is not None:
        _comprobar_tensores_finitos(estado_previo["model_state_dict"], "Pesos guardados")
        _comprobar_tensores_finitos(estado_previo["optimizer_state_dict"], "Optimizador guardado")
        try:
            modelo.load_state_dict(estado_previo["model_state_dict"])
            optimizer.load_state_dict(estado_previo["optimizer_state_dict"])
        except (ValueError, KeyError, RuntimeError) as error:
            raise ValueError("Pesos u optimizador incompatibles con la arquitectura") from error
        historial = copy.deepcopy(estado_previo["historial"])
        mejor_rmse = estado_previo["best_validation_rmse"]
        mejor_epoca = estado_previo["best_epoch"]
        primera_epoca = estado_previo["epoch"] + 1
        if estado_previo["format_version"] == 1:
            comprobacion = ejecutar_epoca(
                modelo, preparado.loaders["validacion"], criterio,
                device=device, baseline=baseline,
            )
            ultima = historial[-1]
            if not all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-6) for a, b in (
                (comprobacion["mae"], ultima["val_mae"]),
                (comprobacion["rmse"], ultima["val_rmse"]),
                (comprobacion["baseline"]["mae"], ultima["baseline_mae"]),
                (comprobacion["baseline"]["rmse"], ultima["baseline_rmse"]),
            )):
                raise ValueError("Dataset/checkpoint legado incompatible: validacion no reproducida")
            warnings.warn(
                "Checkpoint legado sin SHA-256 historico: conteos y validacion verificados; "
                "no se puede demostrar identidad historica del CSV. Se registra su huella actual.",
                UserWarning,
            )
        # Restaurar despues de construir el modelo/loaders y verificar el legado.
        try:
            torch.set_rng_state(estado_previo["torch_rng_state"])
            preparado.loaders["entrenamiento"].generator.set_state(estado_previo["loader_rng_state"])
            for nombre, rng in estado_previo.get("loaders_rng_state", {}).items():
                preparado.loaders[nombre].generator.set_state(rng)
            if estado_previo.get("cuda_rng_state") is not None and device.type == "cuda":
                torch.cuda.set_rng_state_all(estado_previo["cuda_rng_state"])
            if "python_rng_state" in estado_previo:
                random.setstate(estado_previo["python_rng_state"])
                np.random.set_state(estado_previo["numpy_rng_state"])
        except (ValueError, TypeError, RuntimeError) as error:
            raise ValueError("Estados aleatorios del checkpoint incompatibles") from error
    inicio_entrenamiento = perf_counter()
    for epoch in range(primera_epoca, epochs + 1):
        inicio_epoca = perf_counter()
        entrenamiento = ejecutar_epoca(
            modelo, preparado.loaders["entrenamiento"], criterio, optimizer, device=device
        )
        validacion = ejecutar_epoca(
            modelo, preparado.loaders["validacion"], criterio, device=device, baseline=baseline
        )
        fila = {
            "epoca": epoch, "train_mse": entrenamiento["mse"],
            "train_mae": entrenamiento["mae"], "train_rmse": entrenamiento["rmse"],
            "val_mse": validacion["mse"], "val_mae": validacion["mae"],
            "val_rmse": validacion["rmse"],
            "baseline_mae": validacion["baseline"]["mae"],
            "baseline_rmse": validacion["baseline"]["rmse"],
            "n_train": entrenamiento["n"], "n_validacion": validacion["n"],
            "learning_rate": learning_rate, "segundos_epoca": perf_counter() - inicio_epoca,
        }
        if not all(math.isfinite(valor) for valor in fila.values()):
            raise ValueError("Metricas no finitas; se conserva el ultimo checkpoint valido")
        mejora = fila["val_rmse"] < mejor_rmse
        if mejora:
            mejor_rmse, mejor_epoca = fila["val_rmse"], epoch
        historial.append(fila)
        segundos_sesion = perf_counter() - inicio_entrenamiento
        segundos = resumen_anterior.get("segundos_entrenamiento", 0.0) + segundos_sesion
        resumen = {
            "epocas_completadas": epoch, "mejor_epoca": mejor_epoca,
            "mejor_val_rmse": mejor_rmse, "mejor_val_mae": historial[mejor_epoca - 1]["val_mae"],
            "baseline_validacion": validacion["baseline"],
            "input_dim": preparado.input_dim, "device": str(device),
            "filas_por_particion": preparado.filas_por_particion,
            "filas_descartadas_historia": preparado.filas_descartadas_historia,
            "periodos": PERIODOS, "segundos_preparacion": segundos_preparacion,
            "segundos_entrenamiento": segundos, "test_evaluado": False,
            "segundos_ultima_sesion": segundos_sesion,
            "reanudado_desde_epoca": primera_epoca - 1,
            "dataset_sha256": digest,
            "metricas_entrenamiento": "Predicciones durante las actualizaciones de cada epoca",
        }
        estado = {
            "format_version": 2, "model_state_dict": modelo.state_dict(),
            "dataset_sha256": digest,
            "periodos": {nombre: list(limites) for nombre, limites in PERIODOS.items()},
            "python_rng_state": random.getstate(), "numpy_rng_state": np.random.get_state(),
            "segundos_entrenamiento": segundos,
            "loaders_rng_state": {
                nombre: loader.generator.get_state()
                for nombre, loader in preparado.loaders.items() if hasattr(loader, "generator")
            },
            "optimizer_state_dict": optimizer.state_dict(), "epoch": epoch,
            "config": config, "input_dim": preparado.input_dim,
            "nombres_columnas": preparado.nombres_columnas,
            "preparador": preparado.preparador, "best_epoch": mejor_epoca,
            "best_validation_rmse": mejor_rmse, "historial": historial,
            "torch_rng_state": torch.get_rng_state(),
            "loader_rng_state": preparado.loaders["entrenamiento"].generator.get_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            "versiones": {"python": platform.python_version(), "torch": str(torch.__version__),
                         "sklearn": sklearn.__version__},
        }
        # Solo se llega aqui tras completar y validar la epoca entera.
        buffer = io.StringIO(newline="")
        escritor = csv.DictWriter(buffer, fieldnames=list(fila))
        escritor.writeheader()
        escritor.writerows(historial)
        guardar_epoca(
            directorio, estado, mejora=mejora, metricas_csv=buffer.getvalue(),
            resumen_json=json.dumps(resumen, indent=2, allow_nan=False),
            configuracion_yaml=yaml.safe_dump(config, sort_keys=False),
        )
        print(
            f"Epoch {epoch}/{epochs} - train_MSE={fila['train_mse']:.6f} "
            f"val_MAE={fila['val_mae']:.6f} val_RMSE={fila['val_rmse']:.6f} "
            f"baseline_MAE={fila['baseline_mae']:.6f} baseline_RMSE={fila['baseline_rmse']:.6f}",
            flush=True,
        )
    return directorio / "best.pt"


def train(config, *, resume=None):
    source = config["data"].get("source", "synthetic")
    if source == "favorita":
        return _train_favorita(config, resume=resume)
    if resume is not None:
        raise ValueError("La reanudacion solo admite data.source=favorita")
    if source in ("synthetic", "csv"):
        return _train_synthetic(config)
    raise ValueError(f"data.source desconocida: {source}")


def main():
    parser = argparse.ArgumentParser(description="Entrena el MLP sintetico o Favorita")
    parser.add_argument("--config", help="YAML; al reanudar se usa el checkpoint si se omite")
    parser.add_argument("--epochs", type=int, help="Numero TOTAL de epocas")
    parser.add_argument("--resume", help="Ruta a last.pt de una corrida propia")
    args = parser.parse_args()
    if args.resume and not args.config:
        config = copy.deepcopy(cargar_last(args.resume)["config"])
    else:
        config = load_config(args.config or "configs/default.yaml")
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    train(config, resume=args.resume)


if __name__ == "__main__":
    main()
