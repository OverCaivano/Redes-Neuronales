"""Lectura de Favorita sin modificar los CSV originales.

Correspondencias: id -> id, date -> fecha, store_nbr -> tienda,
family -> familia, sales -> ventas_objetivo,
onpromotion -> cantidad_en_promocion.

Las promociones del dia objetivo se consideran planificadas y conocidas al
cierre del dia anterior. Este modulo no codifica categorias ni escala valores.
"""

from pathlib import Path

import numpy as np
import pandas as pd


COLUMNAS_ORIGINALES = {
    "id": "id",
    "date": "fecha",
    "store_nbr": "tienda",
    "family": "familia",
    "sales": "ventas_objetivo",
    "onpromotion": "cantidad_en_promocion",
}
CLAVE = ["fecha", "tienda", "familia"]
ORDEN = ["tienda", "familia", "fecha"]


def validar_datos(datos):
    """Valida el contrato interno; no transforma ni modifica el DataFrame."""
    faltantes = set(COLUMNAS_ORIGINALES.values()) - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas internas: {sorted(faltantes)}")
    if datos[list(COLUMNAS_ORIGINALES.values())].isna().any().any():
        raise ValueError("Las filas observadas contienen valores faltantes")
    if not pd.api.types.is_datetime64_any_dtype(datos["fecha"]):
        raise ValueError("fecha debe tener tipo datetime")
    if datos["fecha"].dt.tz is not None:
        raise ValueError("fecha debe representar dias sin zona horaria")
    if not datos["fecha"].eq(datos["fecha"].dt.normalize()).all():
        raise ValueError("fecha debe representar dias completos, sin hora")
    for columna in ("id", "tienda", "ventas_objetivo", "cantidad_en_promocion"):
        if not pd.api.types.is_numeric_dtype(datos[columna]):
            raise ValueError(f"{columna} debe ser numerica")
        if not np.isfinite(datos[columna].to_numpy()).all():
            raise ValueError(f"{columna} contiene valores no finitos")
    if datos.duplicated(CLAVE).any():
        raise ValueError("Claves duplicadas por fecha + tienda + familia")


def leer_datos_favorita(
    ruta="data/raw/train.csv", *, tiendas=None, familias=None, tamano_bloque=100_000
):
    """Lee las seis columnas obligatorias y devuelve filas ordenadas.

    Los filtros son listas opcionales de identificadores/nombres originales.
    None conserva todos; una lista vacia selecciona cero filas.
    La validacion de valores y claves se aplica al subconjunto seleccionado.
    Se rechazan filas observadas incompletas; los dias ausentes se manejan
    posteriormente como huecos de calendario, nunca como ventas cero.
    """
    if tamano_bloque <= 0:
        raise ValueError("tamano_bloque debe ser positivo")
    ruta = Path(ruta)
    encabezado = pd.read_csv(ruta, nrows=0)
    faltantes = set(COLUMNAS_ORIGINALES) - set(encabezado.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas obligatorias: {sorted(faltantes)}")

    tiendas = None if tiendas is None else tuple(tiendas)
    familias = None if familias is None else tuple(familias)
    bloques = []
    with pd.read_csv(
        ruta, usecols=list(COLUMNAS_ORIGINALES), chunksize=tamano_bloque
    ) as lector:
        for bloque in lector:
            if tiendas is not None:
                bloque = bloque.loc[bloque["store_nbr"].isin(tiendas)]
            if familias is not None:
                bloque = bloque.loc[bloque["family"].isin(familias)]
            if not bloque.empty:
                bloques.append(bloque)

    if bloques:
        datos = pd.concat(bloques, ignore_index=True)
    else:
        datos = pd.read_csv(ruta, usecols=list(COLUMNAS_ORIGINALES), nrows=0)
        for columna in ("id", "store_nbr", "sales", "onpromotion"):
            datos[columna] = pd.to_numeric(datos[columna])

    datos = datos.rename(columns=COLUMNAS_ORIGINALES)
    datos["fecha"] = pd.to_datetime(datos["fecha"], format="%Y-%m-%d", errors="raise")
    validar_datos(datos)
    return datos.sort_values(ORDEN).reset_index(drop=True)
