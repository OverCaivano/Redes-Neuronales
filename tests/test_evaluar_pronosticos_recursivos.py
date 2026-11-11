"""Pruebas del armado de ventanas y métricas del backtest."""

import numpy as np
import pandas as pd
import pytest

from src.evaluar_pronosticos_recursivos import (
    _evaluar_ventana,
    _metricas,
    _origenes_validacion,
    _resumen_filas,
    guardar_resultados,
)


class PronosticadorFalso:
    def __init__(self):
        self.llamadas = []

    def pronosticar(self, historial, promociones, *, tienda, familia, fecha_origen, horizonte):
        self.llamadas.append((historial.copy(), promociones.copy()))
        fechas = pd.date_range(fecha_origen, periods=horizonte + 1, freq="D")[1:]
        return {
            "pronosticos": [
                {"fecha": fecha.strftime("%Y-%m-%d"),
                 "ventas_estimadas": float(indice),
                 "cantidad_en_promocion": float(promociones.iloc[indice - 1]["cantidad_en_promocion"])}
                for indice, fecha in enumerate(fechas, start=1)
            ]
        }


def _serie():
    fechas = pd.date_range("2017-06-08", periods=14, freq="D")
    return pd.DataFrame({
        "fecha": fechas,
        "tienda": 1,
        "familia": "BEAUTY",
        "ventas_objetivo": np.arange(1.0, 15.0),
        "cantidad_en_promocion": np.arange(14),
    })


def test_metricas_y_resumen_por_horizonte():
    assert _metricas([1, 3], [2, 1]) == {
        "mae": 1.5, "rmse": pytest.approx(np.sqrt(2.5)), "n": 2,
    }
    filas = pd.DataFrame([
        {"horizonte": 1, "tienda": 1, "familia": "BEAUTY",
         "ventas_reales": 10.0, "ventas_estimadas": 12.0, "referencia_semanal": 14.0},
        {"horizonte": 1, "tienda": 2, "familia": "BEAUTY",
         "ventas_reales": 20.0, "ventas_estimadas": 18.0, "referencia_semanal": 16.0},
        {"horizonte": 2, "tienda": 1, "familia": "BEAUTY",
         "ventas_reales": 5.0, "ventas_estimadas": 5.0, "referencia_semanal": 10.0},
    ])
    _, general, por_horizonte, por_serie = _resumen_filas(filas)
    assert general["mlp"]["n"] == 3
    assert [fila["horizonte"] for fila in por_horizonte] == [1, 2]
    assert por_horizonte[0]["mlp"]["mae"] == 2.0
    assert por_horizonte[1]["mlp"]["rmse"] == 0.0
    assert len(por_serie) == 3


def test_origenes_solo_generan_ventanas_completas_dentro_de_validacion():
    origenes = _origenes_validacion(("2017-06-16", "2017-07-15"), 6)
    assert len(origenes) == 6
    assert origenes[0] == pd.Timestamp("2017-06-15")
    assert origenes[-1] == pd.Timestamp("2017-07-08")
    assert all(
        origen + pd.Timedelta(days=7) <= pd.Timestamp("2017-07-15")
        for origen in origenes
    )


def test_ventana_no_pasa_ventas_futuras_y_baseline_usa_solo_origen():
    serie = _serie()
    origen = pd.Timestamp("2017-06-14")
    pronosticador = PronosticadorFalso()
    registros, motivo = _evaluar_ventana(pronosticador, serie, "D", origen)
    assert motivo is None
    historial, promociones = pronosticador.llamadas[0]
    assert historial["fecha"].max() == origen
    assert "ventas_objetivo" not in promociones
    assert promociones["fecha"].min() == origen + pd.Timedelta(days=1)
    assert promociones["fecha"].max() == origen + pd.Timedelta(days=7)
    assert [fila["referencia_semanal"] for fila in registros] == list(range(1, 8))

    alterada = serie.copy()
    alterada.loc[alterada["fecha"] > origen, "ventas_objetivo"] += 10_000
    otro_pronosticador = PronosticadorFalso()
    otros_registros, motivo = _evaluar_ventana(otro_pronosticador, alterada, "D", origen)
    assert motivo is None
    assert otros_registros[0]["ventas_estimadas"] == registros[0]["ventas_estimadas"]
    assert otros_registros[0]["referencia_semanal"] == registros[0]["referencia_semanal"]
    assert otros_registros[0]["ventas_reales"] != registros[0]["ventas_reales"]
    pd.testing.assert_frame_equal(
        pronosticador.llamadas[0][0], otro_pronosticador.llamadas[0][0]
    )


def test_ventana_incompleta_se_omite_con_motivo_explicito():
    serie = _serie().iloc[:-1]
    registros, motivo = _evaluar_ventana(
        PronosticadorFalso(), serie, "D", pd.Timestamp("2017-06-14")
    )
    assert registros is None
    assert motivo == "faltan ventas reales para completar los siete días"


def test_guarda_tablas_aplanadas_en_directorio_nuevo(tmp_path, monkeypatch):
    import src.evaluar_pronosticos_recursivos as evaluador

    monkeypatch.setattr(evaluador, "RAIZ", tmp_path)
    comparacion = {
        "mlp": {"mae": 1.0, "rmse": 2.0, "n": 3},
        "referencia_semanal": {"mae": 2.0, "rmse": 3.0, "n": 3},
        "mejora_porcentual": {"mae": 50.0, "rmse": 33.3},
    }
    informe = {
        "por_horizonte": [{"horizonte": 1, **comparacion}],
        "por_tienda_familia_horizonte": [
            {"tienda": 1, "familia": "BEAUTY", "horizonte": 1, **comparacion}
        ],
    }
    detalle = pd.DataFrame([{"horizonte": 1, "ventas_reales": 1.0}])
    destino = guardar_resultados(
        informe, detalle, "runs/evaluaciones/pronostico_recursivo_prueba"
    )
    tabla = pd.read_csv(destino / "por_horizonte.csv")
    assert tabla.loc[0, "mlp_mae"] == 1.0
    assert tabla.loc[0, "referencia_semanal_rmse"] == 3.0
    assert tabla.loc[0, "mejora_mae_porcentual"] == 50.0
    with pytest.raises(FileExistsError):
        guardar_resultados(informe, detalle, destino)

