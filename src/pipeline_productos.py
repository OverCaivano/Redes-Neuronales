"""Entrenamiento y evaluación cronológicos para ventas sintéticas por producto."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.checkpoints import guardar_checkpoint, guardar_texto
from src.datos_sinteticos import (
    FECHA_FIN,
    FECHA_INICIO,
    validar_catalogo_productos,
    validar_ventas,
)
from src.metricas import MetricasRegresion
from src.models import MLP


RAIZ = Path(__file__).resolve().parents[1]
DIRECTORIO_DATOS = RAIZ / "data" / "synthetic"
DIRECTORIO_SALIDA = RAIZ / "runs" / "synthetic_product_sales_v1"
RUTA_VENTAS = DIRECTORIO_DATOS / "ventas_sinteticas.csv"
RUTA_CATALOGO = DIRECTORIO_DATOS / "catalogo_productos_sinteticos.csv"
RUTA_CHECKPOINT = DIRECTORIO_SALIDA / "best_model.pt"
RUTA_METRICAS = DIRECTORIO_SALIDA / "metricas.json"

SEMILLA_PREDETERMINADA = 42
FIN_ENTRENAMIENTO = pd.Timestamp("2025-12-31")
FIN_VALIDACION = pd.Timestamp("2026-06-30")
FEATURES_CATEGORICAS = ("producto_id", "tienda_id", "categoria", "mes")
FEATURES_TEMPORALES = (
    "dia_semana_seno", "dia_semana_coseno", "dia_anio_seno", "dia_anio_coseno",
)
DIMENSION_OCULTA = 64
EPOCAS_PREDETERMINADAS = 12
TAMANO_LOTE_PREDETERMINADO = 2048
TASA_APRENDIZAJE_PREDETERMINADA = 0.001
VERSION_CHECKPOINT = 1


def _huella(ruta):
    digest = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            digest.update(bloque)
    return digest.hexdigest()


def cargar_dataset(ruta_ventas=RUTA_VENTAS, ruta_catalogo=RUTA_CATALOGO):
    """Carga y valida el panel sintético y su catálogo, sin usar datos de fuera."""
    ruta_ventas, ruta_catalogo = Path(ruta_ventas), Path(ruta_catalogo)
    if not ruta_ventas.is_file():
        raise FileNotFoundError(f"No existe el dataset de ventas sintéticas: {ruta_ventas}")
    if not ruta_catalogo.is_file():
        raise FileNotFoundError(f"No existe el catálogo sintético: {ruta_catalogo}")
    catalogo = pd.read_csv(ruta_catalogo, dtype={"producto_id": str})
    ventas = pd.read_csv(ruta_ventas, dtype={"producto_id": str})
    validar_catalogo_productos(catalogo)
    validar_ventas(ventas, catalogo)
    ventas["fecha"] = pd.to_datetime(ventas["fecha"], format="%Y-%m-%d", errors="raise")
    ventas["mes"] = ventas["fecha"].dt.month
    ventas = ventas.merge(
        catalogo.loc[:, ["producto_id", "categoria"]],
        on="producto_id",
        how="left",
        validate="many_to_one",
    )
    if ventas["categoria"].isna().any():
        raise ValueError("Hay ventas sin una categoría de producto")
    return ventas, catalogo


def separar_por_fecha(ventas):
    """Separa fechas cronológicamente; test no participa del ajuste ni selección."""
    fecha = pd.to_datetime(ventas["fecha"], errors="raise")
    particiones = {
        "entrenamiento": ventas.loc[fecha.le(FIN_ENTRENAMIENTO)].copy(),
        "validacion": ventas.loc[
            fecha.gt(FIN_ENTRENAMIENTO) & fecha.le(FIN_VALIDACION)
        ].copy(),
        "prueba": ventas.loc[
            fecha.gt(FIN_VALIDACION) & fecha.le(FECHA_FIN)
        ].copy(),
    }
    if any(parte.empty for parte in particiones.values()):
        raise ValueError("Entrenamiento, validación y prueba deben contener observaciones")
    return particiones


def _caracteristicas_temporales(fechas):
    fechas = pd.to_datetime(fechas, errors="raise")
    dia_semana = fechas.dt.day_of_week.to_numpy(dtype=np.float64)
    dia_anio = fechas.dt.day_of_year.to_numpy(dtype=np.float64) - 1
    dias_del_anio = np.where(fechas.dt.is_leap_year.to_numpy(), 366.0, 365.0)
    angulo_semana = 2 * np.pi * dia_semana / 7
    angulo_anio = 2 * np.pi * dia_anio / dias_del_anio
    return np.column_stack((
        np.sin(angulo_semana),
        np.cos(angulo_semana),
        np.sin(angulo_anio),
        np.cos(angulo_anio),
    )).astype(np.float64)


class PreparadorCaracteristicas:
    """Ajusta vocabularios y escalas usando solo las filas de entrenamiento."""

    def __init__(self):
        self.categorias = None
        self.media_temporal = None
        self.escala_temporal = None
        self.nombres_features = None

    def ajustar(self, entrenamiento):
        if entrenamiento.empty:
            raise ValueError("No se puede ajustar con entrenamiento vacío")
        self.categorias = {
            columna: sorted(entrenamiento[columna].unique().tolist())
            for columna in FEATURES_CATEGORICAS
        }
        temporal = _caracteristicas_temporales(entrenamiento["fecha"])
        self.media_temporal = temporal.mean(axis=0)
        self.escala_temporal = temporal.std(axis=0)
        self.escala_temporal[self.escala_temporal == 0] = 1.0
        nombres_categoricos = [
            f"{columna}_{valor}"
            for columna in FEATURES_CATEGORICAS
            for valor in self.categorias[columna]
        ]
        self.nombres_features = tuple(nombres_categoricos) + FEATURES_TEMPORALES
        return self

    def transformar(self, datos):
        if self.categorias is None:
            raise RuntimeError("El preparador debe ajustarse con entrenamiento")
        cantidad = len(datos)
        bloques = []
        for columna in FEATURES_CATEGORICAS:
            categorias = self.categorias[columna]
            indices = {valor: indice for indice, valor in enumerate(categorias)}
            valores = datos[columna].tolist()
            desconocidos = sorted(set(valores) - set(categorias))
            if desconocidos:
                raise ValueError(f"Categorías desconocidas en {columna}: {desconocidos}")
            matriz = np.zeros((cantidad, len(categorias)), dtype=np.float32)
            if cantidad:
                matriz[np.arange(cantidad), [indices[valor] for valor in valores]] = 1.0
            bloques.append(matriz)
        temporal = _caracteristicas_temporales(datos["fecha"])
        temporal = (temporal - self.media_temporal) / self.escala_temporal
        if not np.isfinite(temporal).all():
            raise ValueError("Las características temporales no son finitas")
        bloques.append(temporal.astype(np.float32))
        resultado = np.concatenate(bloques, axis=1)
        if resultado.shape != (cantidad, len(self.nombres_features)):
            raise ValueError("Dimensiones de características incompatibles")
        return resultado

    def estado(self):
        if self.categorias is None:
            raise RuntimeError("El preparador debe ajustarse antes de guardar")
        return {
            "categorias": self.categorias,
            "media_temporal": self.media_temporal.tolist(),
            "escala_temporal": self.escala_temporal.tolist(),
            "nombres_features": list(self.nombres_features),
        }

    @classmethod
    def desde_estado(cls, estado):
        preparador = cls()
        preparador.categorias = estado["categorias"]
        preparador.media_temporal = np.asarray(estado["media_temporal"], dtype=np.float64)
        preparador.escala_temporal = np.asarray(estado["escala_temporal"], dtype=np.float64)
        preparador.nombres_features = tuple(estado["nombres_features"])
        esperados = tuple(
            f"{columna}_{valor}"
            for columna in FEATURES_CATEGORICAS
            for valor in preparador.categorias[columna]
        ) + FEATURES_TEMPORALES
        if (
            preparador.nombres_features != esperados
            or preparador.media_temporal.shape != (len(FEATURES_TEMPORALES),)
            or preparador.escala_temporal.shape != (len(FEATURES_TEMPORALES),)
            or not np.isfinite(preparador.media_temporal).all()
            or not np.isfinite(preparador.escala_temporal).all()
            or (preparador.escala_temporal <= 0).any()
        ):
            raise ValueError("Estado del preparador incompatible")
        return preparador


def _matriz_y(datos, preparador):
    caracteristicas = preparador.transformar(datos)
    objetivos = np.log1p(
        datos["unidades_vendidas"].to_numpy(dtype=np.float32)
    ).reshape(-1, 1)
    return caracteristicas, objetivos


def _metricas_regresion(predicciones, objetivos):
    metricas = MetricasRegresion()
    metricas.actualizar(
        torch.as_tensor(np.asarray(predicciones, dtype=np.float32).reshape(-1, 1)),
        torch.as_tensor(np.asarray(objetivos, dtype=np.float32).reshape(-1, 1)),
    )
    return metricas.resultado()


def _predicciones_modelo(modelo, caracteristicas):
    modelo.eval()
    with torch.inference_mode():
        salida_log = modelo(torch.from_numpy(caracteristicas))
        predicciones = torch.expm1(salida_log).clamp_min(0)
    if not torch.isfinite(predicciones).all():
        raise ValueError("El modelo produjo predicciones no finitas")
    return predicciones.cpu().numpy().reshape(-1)


def _predicciones_baseline(entrenamiento, datos):
    medias = entrenamiento.groupby(
        ["tienda_id", "producto_id"], sort=True
    )["unidades_vendidas"].mean()
    claves = zip(datos["tienda_id"].astype(int), datos["producto_id"].astype(str))
    try:
        return np.asarray([medias.loc[clave] for clave in claves], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"El baseline no tiene historia para la serie {error}") from error


def _evaluar(modelo, preparador, entrenamiento, datos):
    caracteristicas, _ = _matriz_y(datos, preparador)
    reales = datos["unidades_vendidas"].to_numpy(dtype=np.float64)
    predicciones_modelo = _predicciones_modelo(modelo, caracteristicas)
    predicciones_baseline = _predicciones_baseline(entrenamiento, datos)
    return {
        "modelo": _metricas_regresion(predicciones_modelo, reales),
        "baseline_media_producto_tienda": _metricas_regresion(
            predicciones_baseline, reales
        ),
    }


def entrenar_pipeline(
    ruta_ventas=RUTA_VENTAS,
    ruta_catalogo=RUTA_CATALOGO,
    directorio_salida=DIRECTORIO_SALIDA,
    *,
    semilla=SEMILLA_PREDETERMINADA,
    epocas=EPOCAS_PREDETERMINADAS,
    tamano_lote=TAMANO_LOTE_PREDETERMINADO,
):
    """Entrena el MLP, selecciona por validación y evalúa test una sola vez."""
    if isinstance(epocas, bool) or not isinstance(epocas, int) or epocas <= 0:
        raise ValueError("epocas debe ser un entero positivo")
    if (
        isinstance(tamano_lote, bool)
        or not isinstance(tamano_lote, int)
        or tamano_lote <= 0
    ):
        raise ValueError("tamano_lote debe ser un entero positivo")
    if isinstance(semilla, bool) or not isinstance(semilla, int):
        raise ValueError("semilla debe ser un entero")

    directorio_salida = Path(directorio_salida)
    checkpoint_path = directorio_salida / "best_model.pt"
    metricas_path = directorio_salida / "metricas.json"
    if checkpoint_path.exists() or metricas_path.exists():
        raise FileExistsError(
            f"La corrida ya contiene artefactos: {directorio_salida}; use otra carpeta"
        )

    ventas, catalogo = cargar_dataset(ruta_ventas, ruta_catalogo)
    particiones = separar_por_fecha(ventas)
    entrenamiento = particiones["entrenamiento"]
    validacion = particiones["validacion"]
    prueba = particiones["prueba"]

    preparador = PreparadorCaracteristicas().ajustar(entrenamiento)
    x_entrenamiento, y_entrenamiento = _matriz_y(entrenamiento, preparador)

    torch.manual_seed(semilla)
    generador = torch.Generator().manual_seed(semilla)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_entrenamiento),
            torch.from_numpy(y_entrenamiento),
        ),
        batch_size=tamano_lote,
        shuffle=True,
        generator=generador,
        num_workers=0,
    )
    modelo_config = {
        "input_dim": len(preparador.nombres_features),
        "hidden_dim": DIMENSION_OCULTA,
        "output_dim": 1,
    }
    modelo = MLP(**modelo_config)
    optimizador = torch.optim.Adam(
        modelo.parameters(), lr=TASA_APRENDIZAJE_PREDETERMINADA
    )
    criterio = nn.MSELoss()
    mejores_pesos = None
    mejor_rmse = float("inf")
    mejor_epoca = None
    historial = []

    for epoca in range(1, epocas + 1):
        modelo.train()
        perdida_total = 0.0
        muestras = 0
        for caracteristicas, objetivos in loader:
            optimizador.zero_grad(set_to_none=True)
            salida = modelo(caracteristicas)
            perdida = criterio(salida, objetivos)
            if not torch.isfinite(perdida):
                raise ValueError("La pérdida no es finita; entrenamiento detenido")
            perdida.backward()
            optimizador.step()
            cantidad = len(caracteristicas)
            perdida_total += float(perdida.item()) * cantidad
            muestras += cantidad

        validacion_actual = _evaluar(
            modelo, preparador, entrenamiento, validacion
        )
        rmse_validacion = validacion_actual["modelo"]["rmse"]
        historial.append({
            "epoca": epoca,
            "train_mse_log1p": perdida_total / muestras,
            "val_mae": validacion_actual["modelo"]["mae"],
            "val_rmse": rmse_validacion,
        })
        if rmse_validacion < mejor_rmse:
            mejor_rmse = rmse_validacion
            mejor_epoca = epoca
            mejores_pesos = {
                nombre: peso.detach().clone()
                for nombre, peso in modelo.state_dict().items()
            }

    if mejores_pesos is None:
        raise RuntimeError("No se obtuvo un checkpoint válido")
    modelo.load_state_dict(mejores_pesos, strict=True)
    metricas = {
        "validacion": _evaluar(modelo, preparador, entrenamiento, validacion),
        "prueba": _evaluar(modelo, preparador, entrenamiento, prueba),
    }
    configuracion = {
        "semilla": semilla,
        "epocas_solicitadas": epocas,
        "epoca_seleccionada": mejor_epoca,
        "tamano_lote": tamano_lote,
        "optimizador": "Adam",
        "tasa_aprendizaje": TASA_APRENDIZAJE_PREDETERMINADA,
        "perdida": "MSE sobre log1p(unidades_vendidas)",
        "transformacion_objetivo": "log1p / expm1",
        "arquitectura": modelo_config,
        "device": "cpu",
    }
    metadatos = {
        "origen": "dataset_sintetico",
        "datos_sinteticos": True,
        "semilla_datos": 42,
        "sha256_ventas": _huella(ruta_ventas),
        "sha256_catalogo": _huella(ruta_catalogo),
        "fecha_minima": FECHA_INICIO.strftime("%Y-%m-%d"),
        "fecha_maxima_observada": FECHA_FIN.strftime("%Y-%m-%d"),
        "periodos": {
            "entrenamiento": [FECHA_INICIO.strftime("%Y-%m-%d"), FIN_ENTRENAMIENTO.strftime("%Y-%m-%d")],
            "validacion": [
                (FIN_ENTRENAMIENTO + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                FIN_VALIDACION.strftime("%Y-%m-%d"),
            ],
            "prueba": [
                (FIN_VALIDACION + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                FECHA_FIN.strftime("%Y-%m-%d"),
            ],
        },
        "columnas_omitidas_a_futuro": [
            "unidades_vendidas", "precio", "promocion",
        ],
        "supuesto_inferencia": (
            "No se suministran precio ni promoción; las predicciones son "
            "estimaciones medias que no condicionan por esas variables."
        ),
    }
    checkpoint = {
        "format_version": VERSION_CHECKPOINT,
        "config": configuracion,
        "metadata": metadatos,
        "preparador": preparador.estado(),
        "feature_names": list(preparador.nombres_features),
        "catalogo": catalogo.to_dict(orient="records"),
        "model_state_dict": mejores_pesos,
    }
    reporte = {
        "config": configuracion,
        "metadata": metadatos,
        "filas_por_particion": {
            nombre: len(parte) for nombre, parte in particiones.items()
        },
        "features": list(preparador.nombres_features),
        "metricas": metricas,
        "mejora_rmse_en_prueba": (
            metricas["prueba"]["baseline_media_producto_tienda"]["rmse"]
            - metricas["prueba"]["modelo"]["rmse"]
        ),
        "modelo_supera_baseline_rmse": (
            metricas["prueba"]["modelo"]["rmse"]
            < metricas["prueba"]["baseline_media_producto_tienda"]["rmse"]
        ),
        "historial_validacion": historial,
    }
    guardar_checkpoint(checkpoint_path, checkpoint)
    guardar_texto(
        metricas_path,
        json.dumps(reporte, indent=2, ensure_ascii=False, allow_nan=False),
    )
    return {
        "checkpoint": checkpoint_path,
        "metricas_path": metricas_path,
        "reporte": reporte,
    }


def _cargar_checkpoint(ruta):
    ruta = Path(ruta)
    try:
        checkpoint = torch.load(ruta, map_location="cpu", weights_only=True)
    except FileNotFoundError:
        raise
    except Exception as error:
        raise ValueError("Checkpoint sintético ilegible o incompatible") from error
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format_version") != VERSION_CHECKPOINT
        or checkpoint.get("metadata", {}).get("origen") != "dataset_sintetico"
        or checkpoint.get("metadata", {}).get("datos_sinteticos") is not True
    ):
        raise ValueError("El checkpoint no pertenece al pipeline sintético de productos")
    preparador = PreparadorCaracteristicas.desde_estado(checkpoint["preparador"])
    if checkpoint.get("feature_names") != list(preparador.nombres_features):
        raise ValueError("Features guardadas incompatibles con el preparador")
    modelo = MLP(**checkpoint["config"]["arquitectura"])
    try:
        modelo.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, TypeError) as error:
        raise ValueError("Pesos incompatibles con la arquitectura del checkpoint") from error
    if any(not torch.isfinite(parametro).all() for parametro in modelo.parameters()):
        raise ValueError("El checkpoint contiene pesos no finitos")
    modelo.eval()
    return checkpoint, preparador, modelo


def evaluar_checkpoint(
    checkpoint_path=RUTA_CHECKPOINT,
    ruta_ventas=RUTA_VENTAS,
    ruta_catalogo=RUTA_CATALOGO,
):
    """Evalúa un checkpoint existente sobre test, sin ajustar ni entrenar."""
    checkpoint, preparador, modelo = _cargar_checkpoint(checkpoint_path)
    if (
        _huella(ruta_ventas) != checkpoint["metadata"]["sha256_ventas"]
        or _huella(ruta_catalogo) != checkpoint["metadata"]["sha256_catalogo"]
    ):
        raise ValueError("Los datos no coinciden con las huellas del checkpoint")
    ventas, _ = cargar_dataset(ruta_ventas, ruta_catalogo)
    particiones = separar_por_fecha(ventas)
    return _evaluar(
        modelo,
        preparador,
        particiones["entrenamiento"],
        particiones["prueba"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Entrena o evalúa el MLP de ventas sintéticas por producto."
    )
    subcomandos = parser.add_subparsers(dest="comando", required=True)
    entrenamiento = subcomandos.add_parser("train", help="Entrenar y evaluar el modelo.")
    entrenamiento.add_argument("--data-dir", type=Path, default=DIRECTORIO_DATOS)
    entrenamiento.add_argument("--output-dir", type=Path, default=DIRECTORIO_SALIDA)
    entrenamiento.add_argument("--epochs", type=int, default=EPOCAS_PREDETERMINADAS)
    entrenamiento.add_argument("--batch-size", type=int, default=TAMANO_LOTE_PREDETERMINADO)
    entrenamiento.add_argument("--seed", type=int, default=SEMILLA_PREDETERMINADA)
    evaluacion = subcomandos.add_parser("evaluate", help="Evaluar el checkpoint guardado.")
    evaluacion.add_argument("--checkpoint", type=Path, default=RUTA_CHECKPOINT)
    evaluacion.add_argument("--data-dir", type=Path, default=DIRECTORIO_DATOS)
    args = parser.parse_args()

    if args.comando == "train":
        resultado = entrenar_pipeline(
            args.data_dir / "ventas_sinteticas.csv",
            args.data_dir / "catalogo_productos_sinteticos.csv",
            args.output_dir,
            semilla=args.seed,
            epocas=args.epochs,
            tamano_lote=args.batch_size,
        )
        reporte = resultado["reporte"]
        print(f"Epoca seleccionada por validacion: {reporte['config']['epoca_seleccionada']}")
        print(f"Checkpoint: {resultado['checkpoint']}")
        for particion, datos in reporte["metricas"].items():
            modelo = datos["modelo"]
            baseline = datos["baseline_media_producto_tienda"]
            print(
                f"{particion}: modelo MAE={modelo['mae']:.4f}, RMSE={modelo['rmse']:.4f}; "
                f"baseline MAE={baseline['mae']:.4f}, RMSE={baseline['rmse']:.4f}"
            )
        if not reporte["modelo_supera_baseline_rmse"]:
            print("Advertencia: el modelo no supera al baseline en RMSE de prueba.")
    else:
        metricas = evaluar_checkpoint(
            args.checkpoint,
            args.data_dir / "ventas_sinteticas.csv",
            args.data_dir / "catalogo_productos_sinteticos.csv",
        )
        modelo = metricas["modelo"]
        baseline = metricas["baseline_media_producto_tienda"]
        print(
            f"Prueba: modelo MAE={modelo['mae']:.4f}, RMSE={modelo['rmse']:.4f}; "
            f"baseline MAE={baseline['mae']:.4f}, RMSE={baseline['rmse']:.4f}"
        )
        if modelo["rmse"] >= baseline["rmse"]:
            print("Advertencia: el modelo no supera al baseline en RMSE de prueba.")


if __name__ == "__main__":
    main()
