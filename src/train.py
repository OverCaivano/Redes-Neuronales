"""Entrenamiento sintetico o Favorita; test no participa en seleccion."""

import argparse
import copy
import csv
import io
import json
import math
from pathlib import Path
import platform
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
from src.checkpoints import guardar_checkpoint, guardar_texto


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


def _train_favorita(config):
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
    directorio = Path(config["output"]["directory"])
    if directorio.exists() and any(directorio.iterdir()):
        raise FileExistsError(f"La carpeta de resultados no esta vacia: {directorio}; use otra ruta")

    seed = int(config.get("seed", 42))
    torch.manual_seed(seed)
    inicio = perf_counter()
    datos = config["data"]
    preparado = construir_loaders_favorita(
        datos["path"], tiendas=datos.get("tiendas"), familias=datos.get("familias"),
        batch_size=int(config["training"]["batch_size"]), semilla=seed,
        num_workers=int(config["training"].get("num_workers", 0)),
    )
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
    guardar_texto(directorio / "configuracion.yaml", yaml.safe_dump(config, sort_keys=False))
    segundos_preparacion = perf_counter() - inicio
    historial = []
    mejor_rmse, mejor_epoca = float("inf"), None
    inicio_entrenamiento = perf_counter()
    for epoch in range(1, epochs + 1):
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
        segundos = perf_counter() - inicio_entrenamiento
        resumen = {
            "epocas_completadas": epoch, "mejor_epoca": mejor_epoca,
            "mejor_val_rmse": mejor_rmse, "mejor_val_mae": historial[mejor_epoca - 1]["val_mae"],
            "baseline_validacion": validacion["baseline"],
            "input_dim": preparado.input_dim, "device": str(device),
            "filas_por_particion": preparado.filas_por_particion,
            "filas_descartadas_historia": preparado.filas_descartadas_historia,
            "periodos": PERIODOS, "segundos_preparacion": segundos_preparacion,
            "segundos_entrenamiento": segundos, "test_evaluado": False,
            "metricas_entrenamiento": "Predicciones durante las actualizaciones de cada epoca",
        }
        estado = {
            "format_version": 1, "model_state_dict": modelo.state_dict(),
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
        guardar_checkpoint(directorio / "last.pt", estado)
        if mejora:
            guardar_checkpoint(directorio / "best.pt", estado)
        buffer = io.StringIO(newline="")
        escritor = csv.DictWriter(buffer, fieldnames=list(fila))
        escritor.writeheader()
        escritor.writerows(historial)
        guardar_texto(directorio / "metricas.csv", buffer.getvalue())
        guardar_texto(directorio / "resumen.json", json.dumps(resumen, indent=2, allow_nan=False))
        print(
            f"Epoch {epoch}/{epochs} - train_MSE={fila['train_mse']:.6f} "
            f"val_MAE={fila['val_mae']:.6f} val_RMSE={fila['val_rmse']:.6f} "
            f"baseline_MAE={fila['baseline_mae']:.6f} baseline_RMSE={fila['baseline_rmse']:.6f}",
            flush=True,
        )
    return directorio / "best.pt"


def train(config):
    source = config["data"].get("source", "synthetic")
    if source == "favorita":
        return _train_favorita(config)
    if source in ("synthetic", "csv"):
        return _train_synthetic(config)
    raise ValueError(f"data.source desconocida: {source}")


def main():
    parser = argparse.ArgumentParser(description="Entrena el MLP sintetico o Favorita")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, help="Epocas efectivas de esta corrida (sin editar YAML)")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    train(config)


if __name__ == "__main__":
    main()
