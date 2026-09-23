"""Causalidad y calendario con series pequenas construidas para estas pruebas."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.preprocesamiento_temporal import (
    VARIABLES_HISTORICAS,
    dividir_por_fecha,
    generar_variables_historicas,
)


@pytest.fixture
def serie():
    return crear_serie()


def crear_serie(inicio="2017-01-01", dias=12, tienda=1, familia="A"):
    return pd.DataFrame({
        "id": range(dias),
        "fecha": pd.date_range(inicio, periods=dias),
        "tienda": tienda,
        "familia": familia,
        "ventas_objetivo": [float(i) for i in range(1, dias + 1)],
        "cantidad_en_promocion": 2,
    })


def test_rezagos_promedio_conteos_y_metadatos(serie):
    original = serie.copy(deep=True)
    resultado = generar_variables_historicas(serie)
    primera = resultado.datos.iloc[0]
    assert primera["fecha"] == pd.Timestamp("2017-01-08")
    assert primera["ventas_hace_1_dia"] == 7
    assert primera["ventas_hace_7_dias"] == 1
    assert primera["promedio_ventas_7_dias"] == 4
    assert primera["ventas_objetivo"] == 8
    assert primera["tienda"] == 1
    assert primera["familia"] == "A"
    assert primera["cantidad_en_promocion"] == 2
    assert resultado.filas_entrada == 12
    assert resultado.filas_validas == 5
    assert resultado.filas_descartadas_historia == 7
    assert_frame_equal(serie, original)


def test_objetivo_no_interviene_en_sus_entradas(serie):
    base = generar_variables_historicas(serie).datos
    serie.loc[serie["fecha"].eq("2017-01-08"), "ventas_objetivo"] = 999999
    alterada = generar_variables_historicas(serie).datos
    assert_frame_equal(
        base.loc[base["fecha"].eq("2017-01-08"), VARIABLES_HISTORICAS],
        alterada.loc[alterada["fecha"].eq("2017-01-08"), VARIABLES_HISTORICAS],
    )


def test_futuro_no_modifica_pasado(serie):
    base = generar_variables_historicas(serie).datos
    serie.loc[serie["fecha"].gt("2017-01-09"), "ventas_objetivo"] = 999999
    alterada = generar_variables_historicas(serie).datos
    assert_frame_equal(
        base.loc[base["fecha"].le("2017-01-09")],
        alterada.loc[alterada["fecha"].le("2017-01-09")],
    )


def test_aislamiento_tienda_y_familia(serie):
    otra_tienda = crear_serie(tienda=2)
    otra_familia = crear_serie(familia="B")
    otra_tienda["ventas_objetivo"] *= 100
    otra_familia["ventas_objetivo"] *= 1000
    juntas = pd.concat([serie, otra_tienda, otra_familia], ignore_index=True)
    resultado = generar_variables_historicas(juntas).datos
    for tienda, familia, factor in [(1, "A", 1), (2, "A", 100), (1, "B", 1000)]:
        primera = resultado.loc[
            resultado["tienda"].eq(tienda) & resultado["familia"].eq(familia)
        ].iloc[0]
        assert primera[VARIABLES_HISTORICAS].tolist() == [7 * factor, factor, 4 * factor]


def test_hueco_navidad_no_es_fila_anterior_ni_cero():
    datos = crear_serie("2016-12-15", dias=25)
    datos = datos.loc[~datos["fecha"].eq("2016-12-25")]
    resultado = generar_variables_historicas(datos)
    fechas = resultado.datos["fecha"]
    assert not fechas.between("2016-12-25", "2017-01-01").any()
    recuperada = resultado.datos.loc[fechas.eq("2017-01-02")].iloc[0]
    assert recuperada["ventas_hace_1_dia"] == 18
    assert recuperada["ventas_hace_7_dias"] == 12
    assert recuperada["promedio_ventas_7_dias"] == 15
    assert resultado.filas_descartadas_historia == 14
    assert resultado.filas_entrada == 24


def test_ceros_son_historia_observada(serie):
    serie["ventas_objetivo"] = 0.0
    resultado = generar_variables_historicas(serie)
    assert resultado.filas_validas == 5
    assert resultado.datos[VARIABLES_HISTORICAS].eq(0).all().all()


def test_orden_original_no_afecta_resultado(serie):
    assert_frame_equal(
        generar_variables_historicas(serie).datos,
        generar_variables_historicas(serie.sample(frac=1, random_state=42)).datos,
    )


@pytest.mark.parametrize("dias", [0, 1, 7])
def test_historia_insuficiente_y_entrada_vacia(dias):
    resultado = generar_variables_historicas(crear_serie(dias=dias))
    assert resultado.datos.empty
    assert resultado.filas_descartadas_historia == dias
    assert set(VARIABLES_HISTORICAS).issubset(resultado.datos.columns)
    assert all(parte.empty for parte in dividir_por_fecha(resultado).values())


def test_duplicados_rechazados_en_preprocesamiento(serie):
    with pytest.raises(ValueError, match="Claves duplicadas"):
        generar_variables_historicas(pd.concat([serie, serie.iloc[[0]]]))


def test_particiones_limites_y_contexto_legitimo():
    datos = crear_serie("2017-06-01", dias=76)
    resultado = generar_variables_historicas(datos)
    partes = dividir_por_fecha(resultado)
    entrenamiento, validacion, prueba = (
        partes["entrenamiento"], partes["validacion"], partes["test"]
    )
    assert entrenamiento["fecha"].max() == pd.Timestamp("2017-06-15")
    assert validacion["fecha"].min() == pd.Timestamp("2017-06-16")
    assert validacion["fecha"].max() == pd.Timestamp("2017-07-15")
    assert prueba["fecha"].min() == pd.Timestamp("2017-07-16")
    assert prueba["fecha"].max() == pd.Timestamp("2017-08-15")
    assert sum(map(len, partes.values())) == resultado.filas_validas
    fechas = [set(parte["fecha"]) for parte in partes.values()]
    assert fechas[0].isdisjoint(fechas[1])
    assert fechas[0].isdisjoint(fechas[2])
    assert fechas[1].isdisjoint(fechas[2])
    assert validacion.iloc[0][VARIABLES_HISTORICAS].tolist() == [15, 9, 12]
    assert prueba.iloc[0][VARIABLES_HISTORICAS].tolist() == [45, 39, 42]


def test_division_exige_preprocesamiento(serie):
    with pytest.raises(TypeError, match="Primero"):
        dividir_por_fecha(serie)


def test_division_no_omite_fechas_fuera_del_periodo():
    resultado = generar_variables_historicas(crear_serie("2018-01-01"))
    with pytest.raises(ValueError, match="fuera del periodo"):
        dividir_por_fecha(resultado)
