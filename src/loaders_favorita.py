"""Construccion de loaders Favorita sin entrenar ni escribir artefactos."""

from dataclasses import dataclass

import pandas as pd
from torch.utils.data import DataLoader

from src.datos_favorita import leer_datos_favorita
from src.preprocesamiento_temporal import (
    PERIODOS, generar_variables_historicas, dividir_por_fecha,
)
from src.transformaciones import PreparadorEntradas
from src.dataset_favorita import DatasetFavorita, crear_dataloader


@dataclass(frozen=True)
class ResultadoLoadersFavorita:
    """Metadatos en orden posicional del Dataset, no del sampler mezclado.

    Validacion/test se recorren en ese mismo orden. En entrenamiento el
    sampler mezcla conjuntamente X e y; los metadatos no deben asociarse
    secuencialmente a los lotes mezclados. No modificar datasets ni metadatos.
    """
    loaders: dict[str, DataLoader]
    preparador: PreparadorEntradas
    nombres_columnas: tuple[str, ...]
    input_dim: int
    metadatos: dict[str, pd.DataFrame]
    filas_entrada: int
    filas_descartadas_historia: int
    filas_por_particion: dict[str, int]


def construir_loaders_favorita(
    ruta="data/raw/train.csv", *, tiendas=None, familias=None,
    batch_size=256, semilla=42, num_workers=0,
):
    """Lee historia completa, divide por fecha, ajusta en train y crea loaders.

    Reutiliza el pipeline existente. Transforma una particion por vez y libera
    su tabla al terminar. Conserva solo CSR, objetivos float32 [N,1] y metadatos
    compactos (id, fecha, tienda, familia). No densifica las matrices.
    """
    datos = leer_datos_favorita(ruta, tiendas=tiendas, familias=familias)
    temporal = generar_variables_historicas(datos)
    del datos
    filas_entrada = temporal.filas_entrada
    descartadas = temporal.filas_descartadas_historia
    particiones = dividir_por_fecha(temporal)
    del temporal

    fin_anterior = None
    for nombre, (inicio, fin) in PERIODOS.items():
        parte = particiones[nombre]
        if parte.empty:
            raise ValueError(f"La particion {nombre} esta vacia")
        fechas = parte["fecha"]
        if not fechas.between(inicio, fin).all():
            raise ValueError(f"Fechas fuera del periodo de {nombre}")
        if fin_anterior is not None and fechas.min() <= fin_anterior:
            raise ValueError("Las fechas de las particiones se superponen")
        fin_anterior = fechas.max()
    del parte, fechas

    preparador = PreparadorEntradas(particiones["entrenamiento"])
    nombres_columnas = preparador.nombres_columnas
    input_dim = len(nombres_columnas)
    loaders, metadatos, cantidades = {}, {}, {}
    for nombre in PERIODOS:
        parte = particiones.pop(nombre)
        matriz = preparador.transformar(parte)
        if matriz.shape != (len(parte), input_dim):
            raise ValueError(f"Dimensiones incompatibles en {nombre}")
        dataset = DatasetFavorita(matriz, parte["ventas_objetivo"])
        # DatasetFavorita valida CSR float32, finitud y objetivos alineados.
        metadata = parte[["id", "fecha", "tienda", "familia"]].copy()
        metadata["familia"] = metadata["familia"].astype("category")
        metadata.reset_index(drop=True, inplace=True)
        if len(metadata) != len(dataset):
            raise ValueError(f"Metadatos desalineados en {nombre}")
        metadatos[nombre] = metadata
        cantidades[nombre] = len(dataset)
        loaders[nombre] = crear_dataloader(
            dataset, batch_size=batch_size, shuffle=nombre == "entrenamiento",
            semilla=semilla, num_workers=num_workers,
        )
        del parte, matriz, dataset, metadata

    return ResultadoLoadersFavorita(
        loaders=loaders, preparador=preparador, nombres_columnas=nombres_columnas,
        input_dim=input_dim, metadatos=metadatos, filas_entrada=filas_entrada,
        filas_descartadas_historia=descartadas, filas_por_particion=cantidades,
    )
