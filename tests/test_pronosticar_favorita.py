"""Pruebas de pronóstico recursivo, causalidad y validaciones."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.datos_favorita import leer_datos_favorita
from src.evaluacion_favorita import cargar_best_favorita
from src.pronosticar_favorita import (
    PronosticoNoDisponible,
    pronosticar_favorita,
)
from test_evaluacion_favorita import checkpoint


class ModeloRecursivo:
    def __init__(self):
        self.entradas = []

    def __call__(self, entrada):
        self.entradas.append(entrada.detach().numpy().copy())
        return torch.tensor([[100.0 * len(self.entradas)]], dtype=entrada.dtype)


def _datos(checkpoint, fecha_origen="2017-08-15", horizonte=3):
    _, csv, tabla = checkpoint
    historial = leer_datos_favorita(csv)
    futuras = pd.date_range(fecha_origen, periods=horizonte + 1, freq="D")[1:]
    ventas_futuras = pd.DataFrame({
        "id": np.arange(len(tabla), len(tabla) + len(futuras)),
        "fecha": futuras,
        "tienda": 1,
        "familia": "BEAUTY",
        "ventas_objetivo": np.arange(9000.0, 9000.0 + len(futuras)),
        "cantidad_en_promocion": 0,
    })
    historial = pd.concat([historial, ventas_futuras], ignore_index=True)
    promociones = pd.DataFrame({
        "fecha": futuras,
        "tienda": 1,
        "familia": "BEAUTY",
        "cantidad_en_promocion": np.arange(1, len(futuras) + 1),
    })
    return historial, promociones


def _pronosticar(checkpoint, historial, promociones, *, horizonte=3):
    return pronosticar_favorita(
        checkpoint[0], historial, promociones, tienda=1, familia="BEAUTY",
        fecha_origen="2017-08-15", horizonte=horizonte,
    )


def test_recursiona_las_predicciones_y_no_usa_ventas_futuras(checkpoint, monkeypatch):
    ruta, _, _ = checkpoint
    historial, promociones = _datos(checkpoint)
    estado = cargar_best_favorita(ruta)
    modelo = ModeloRecursivo()
    monkeypatch.setattr("src.pronosticar_favorita.reconstruir_modelo_favorita", lambda _: modelo)
    resultado = _pronosticar(checkpoint, historial, promociones)

    assert [fila["ventas_estimadas"] for fila in resultado["pronosticos"]] == [100.0, 200.0, 300.0]
    assert [fila["fecha"] for fila in resultado["pronosticos"]] == [
        "2017-08-16", "2017-08-17", "2017-08-18",
    ]
    parametros = estado["preparador"].parametros_escalado
    medias, escalas = parametros["media"], parametros["escala"]
    entradas = np.stack([llamada[0] for llamada in modelo.entradas])
    valores = entradas[:, :4] * escalas + medias
    ventas = historial.set_index("fecha")["ventas_objetivo"]
    assert valores[0, :3] == pytest.approx([
        ventas[pd.Timestamp("2017-08-15")],
        ventas[pd.Timestamp("2017-08-09")],
        np.mean([ventas[pd.Timestamp(f"2017-08-{dia:02d}")] for dia in range(9, 16)]),
    ])
    assert valores[1, :3] == pytest.approx([
        100.0, ventas[pd.Timestamp("2017-08-10")],
        np.mean([ventas[pd.Timestamp(f"2017-08-{dia:02d}")] for dia in range(10, 16)] + [100.0]),
    ])
    assert valores[2, :3] == pytest.approx([
        200.0, ventas[pd.Timestamp("2017-08-11")],
        np.mean([ventas[pd.Timestamp(f"2017-08-{dia:02d}")] for dia in range(11, 16)] + [100.0, 200.0]),
    ])

    historial_alterado = historial.copy()
    posteriores = historial_alterado["fecha"] > pd.Timestamp("2017-08-15")
    historial_alterado.loc[posteriores, "ventas_objetivo"] = 1_000_000_000
    monkeypatch.setattr(
        "src.pronosticar_favorita.reconstruir_modelo_favorita",
        lambda _: ModeloRecursivo(),
    )
    repetido = _pronosticar(checkpoint, historial_alterado, promociones)
    assert repetido["pronosticos"] == resultado["pronosticos"]


def test_rechaza_huecos_de_historia_y_promociones_incompletas(checkpoint):
    historial, promociones = _datos(checkpoint)
    sin_historia = historial.loc[historial["fecha"] != pd.Timestamp("2017-08-12")]
    with pytest.raises(PronosticoNoDisponible, match="siete ventas diarias"):
        _pronosticar(checkpoint, sin_historia, promociones)

    with pytest.raises(PronosticoNoDisponible, match="Faltan promociones"):
        _pronosticar(checkpoint, historial, promociones.iloc[:-1])

    duplicadas = pd.concat([promociones, promociones.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicadas"):
        _pronosticar(checkpoint, historial, duplicadas)


@pytest.mark.parametrize("horizonte", [0, 8, 1.5, True])
def test_rechaza_horizonte_fuera_de_rango(checkpoint, horizonte):
    historial, promociones = _datos(checkpoint, horizonte=7)
    with pytest.raises(ValueError, match="entero entre 1 y 7"):
        _pronosticar(checkpoint, historial, promociones, horizonte=horizonte)


def test_rechaza_tienda_o_familia_ajenas_al_checkpoint(checkpoint):
    historial, promociones = _datos(checkpoint)
    with pytest.raises(PronosticoNoDisponible, match="no compatible"):
        pronosticar_favorita(
            checkpoint[0], historial, promociones, tienda=2, familia="BEAUTY",
            fecha_origen="2017-08-15", horizonte=3,
        )

