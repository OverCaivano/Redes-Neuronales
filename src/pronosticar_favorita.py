"""Pronóstico recursivo Favorita con un checkpoint y entradas compatibles."""

from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.evaluacion_favorita import cargar_best_favorita, reconstruir_modelo_favorita
from src.predecir_favorita import validar_consulta
from src.preprocesamiento_temporal import VARIABLES_CICLICAS
from src.transformaciones import COLUMNAS_NUMERICAS


class PronosticoNoDisponible(LookupError):
    """La combinación no tiene historia o promociones completas para el horizonte."""


def leer_promociones_favorita(ruta):
    """Lee y normaliza solo el calendario de promociones del CSV de test."""
    columnas = ["date", "store_nbr", "family", "onpromotion"]
    datos = pd.read_csv(Path(ruta), usecols=columnas)
    return datos.rename(columns={
        "date": "fecha",
        "store_nbr": "tienda",
        "family": "familia",
        "onpromotion": "cantidad_en_promocion",
    })


def _fechas_normalizadas(datos, columna):
    try:
        fechas = pd.to_datetime(datos[columna], errors="raise")
    except (TypeError, ValueError):
        raise ValueError(f"{columna} debe contener fechas válidas") from None
    if fechas.isna().any() or fechas.dt.tz is not None or not fechas.eq(fechas.dt.normalize()).all():
        raise ValueError(f"{columna} debe contener días completos sin zona horaria")
    return fechas


def _ventas_conocidas(historial, *, tienda, familia, origen):
    requeridas = {"fecha", "tienda", "familia", "ventas_objetivo"}
    if not isinstance(historial, pd.DataFrame) or not requeridas.issubset(historial.columns):
        raise ValueError(f"El historial requiere las columnas: {sorted(requeridas)}")

    datos = historial.loc[:, list(requeridas)].copy()
    datos["fecha"] = _fechas_normalizadas(datos, "fecha")
    datos = datos.loc[
        datos["tienda"].eq(tienda) & datos["familia"].eq(familia)
        & datos["fecha"].le(origen)
    ].copy()
    if datos.empty:
        raise PronosticoNoDisponible("No hay ventas observadas para la serie hasta la fecha de origen.")
    if datos["fecha"].duplicated().any():
        raise ValueError("El historial tiene fechas duplicadas para tienda y familia")
    ventas = pd.to_numeric(datos["ventas_objetivo"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(ventas).all():
        raise ValueError("El historial conocido contiene ventas no finitas")
    datos["ventas_objetivo"] = ventas

    esperadas = pd.date_range(origen - pd.Timedelta(days=6), origen, freq="D")
    disponibles = datos.set_index("fecha")["ventas_objetivo"]
    if origen not in disponibles.index or not esperadas.isin(disponibles.index).all():
        raise PronosticoNoDisponible(
            "Se requieren siete ventas diarias observadas hasta la fecha de origen, sin huecos."
        )
    return disponibles.to_dict()


def _promociones_futuras(promociones, *, tienda, familia, fechas):
    requeridas = {"fecha", "tienda", "familia", "cantidad_en_promocion"}
    if not isinstance(promociones, pd.DataFrame) or not requeridas.issubset(promociones.columns):
        raise ValueError(f"El calendario requiere las columnas: {sorted(requeridas)}")

    datos = promociones.loc[:, list(requeridas)].copy()
    datos["fecha"] = _fechas_normalizadas(datos, "fecha")
    datos = datos.loc[
        datos["tienda"].eq(tienda) & datos["familia"].eq(familia)
        & datos["fecha"].isin(fechas)
    ].copy()
    if datos["fecha"].duplicated().any():
        raise ValueError("El calendario tiene promociones duplicadas para una fecha")
    valores = pd.to_numeric(datos["cantidad_en_promocion"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(valores).all() or (valores < 0).any():
        raise ValueError("Las promociones deben ser cantidades finitas y no negativas")
    datos["cantidad_en_promocion"] = valores

    disponibles = datos.set_index("fecha")["cantidad_en_promocion"]
    if not fechas.isin(disponibles.index).all():
        raise PronosticoNoDisponible("Faltan promociones para una o más fechas del horizonte.")
    return disponibles.to_dict()


def pronosticar_favorita(
    checkpoint,
    historial,
    promociones,
    *,
    tienda,
    familia,
    fecha_origen,
    horizonte=7,
):
    """Predice hasta siete días; solo las ventas conocidas o ya pronosticadas alimentan rezagos.

    ``historial`` y ``promociones`` son DataFrames canónicos. El pronóstico
    ignora todas las ventas posteriores al origen y modifica solo su estado
    temporal en memoria.
    """
    origen = validar_consulta(tienda, familia, fecha_origen)
    if isinstance(horizonte, bool) or not isinstance(horizonte, Integral) or not 1 <= horizonte <= 7:
        raise ValueError("horizonte debe ser un entero entre 1 y 7")
    estado = cargar_best_favorita(checkpoint)
    nombres = tuple(estado["nombres_columnas"])
    preparador = estado["preparador"]
    columnas_base = tuple(COLUMNAS_NUMERICAS + VARIABLES_CICLICAS)
    if (estado["config"]["data"]["source"] != "favorita"
        or estado["input_dim"] != len(nombres)
        or nombres != preparador.nombres_columnas
        or nombres[:len(columnas_base)] != columnas_base
        or estado["config"]["model"]["input_dim"] != estado["input_dim"]):
        raise ValueError("Checkpoint incompatible con el preparador de entradas Favorita")

    categorias = preparador.codificador.categorias
    if tienda not in categorias["tienda"] or familia not in categorias["familia"]:
        raise PronosticoNoDisponible("Tienda o familia no compatible con el checkpoint completo.")

    fechas = pd.date_range(origen + pd.Timedelta(days=1), periods=int(horizonte), freq="D")
    ventas = _ventas_conocidas(historial, tienda=tienda, familia=familia, origen=origen)
    cantidades = _promociones_futuras(
        promociones, tienda=tienda, familia=familia, fechas=fechas
    )
    modelo = reconstruir_modelo_favorita(estado)
    predicciones = []

    for fecha in fechas:
        anteriores = [ventas.get(fecha - pd.Timedelta(days=dia)) for dia in range(1, 8)]
        if any(valor is None for valor in anteriores):
            raise PronosticoNoDisponible(
                "La historia recursiva no contiene siete días anteriores para una entrada."
            )
        angulo_semana = 2 * np.pi * fecha.dayofweek / 7
        angulo_mes = 2 * np.pi * (fecha.month - 1) / 12
        fila = pd.DataFrame([{
            "ventas_hace_1_dia": anteriores[0],
            "ventas_hace_7_dias": anteriores[6],
            "promedio_ventas_7_dias": float(np.mean(anteriores)),
            "cantidad_en_promocion": cantidades[fecha],
            "dia_semana_seno": np.sin(angulo_semana),
            "dia_semana_coseno": np.cos(angulo_semana),
            "mes_seno": np.sin(angulo_mes),
            "mes_coseno": np.cos(angulo_mes),
            "tienda": tienda,
            "familia": familia,
        }])
        matriz = preparador.transformar(fila)
        if matriz.shape != (1, estado["input_dim"]) or nombres != preparador.nombres_columnas:
            raise ValueError("Dimensiones u orden de características incompatibles con el checkpoint")
        entrada = torch.from_numpy(matriz.toarray())
        with torch.inference_mode():
            salida = modelo(entrada)
        if salida.shape != (1, 1) or not torch.isfinite(salida).all():
            raise ValueError("El pronóstico debe ser finito y tener forma [1, 1]")
        estimacion = float(salida.item())
        ventas[fecha] = estimacion
        predicciones.append({
            "fecha": fecha.strftime("%Y-%m-%d"),
            "ventas_estimadas": estimacion,
            "cantidad_en_promocion": float(cantidades[fecha]),
        })

    return {
        "tienda": int(tienda),
        "familia": familia,
        "fecha_origen": origen.strftime("%Y-%m-%d"),
        "horizonte": int(horizonte),
        "epoca": int(estado["epoch"]),
        "pronosticos": predicciones,
    }
