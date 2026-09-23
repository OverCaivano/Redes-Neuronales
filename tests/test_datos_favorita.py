"""Pruebas con CSV diminutos temporales; no requieren el dataset real."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.datos_favorita import COLUMNAS_ORIGINALES, leer_datos_favorita


@pytest.fixture
def tabla():
    return pd.DataFrame({
        "id": [2, 1, 0],
        "date": ["2017-01-03", "2017-01-02", "2017-01-01"],
        "store_nbr": [1, 2, 1],
        "family": ["A", "B", "A"],
        "sales": [3.0, 2.0, 1.0],
        "onpromotion": [2, 0, 1],
    })


def guardar(tmp_path, tabla):
    ruta = tmp_path / "train.csv"
    tabla.to_csv(ruta, index=False)
    return ruta


@pytest.mark.parametrize("columna", list(COLUMNAS_ORIGINALES))
def test_esquema_obligatorio(tmp_path, tabla, columna):
    ruta = guardar(tmp_path, tabla.drop(columns=columna))
    with pytest.raises(ValueError, match="Faltan columnas obligatorias"):
        leer_datos_favorita(ruta)


def test_lectura_orden_tipos_y_original_intacto(tmp_path, tabla):
    tabla["columna_extra"] = "ignorar"
    ruta = guardar(tmp_path, tabla)
    original = ruta.read_bytes()
    datos = leer_datos_favorita(ruta, tamano_bloque=1)
    assert datos.columns.tolist() == list(COLUMNAS_ORIGINALES.values())
    assert datos["id"].tolist() == [0, 2, 1]
    assert pd.api.types.is_datetime64_any_dtype(datos["fecha"])
    assert datos["cantidad_en_promocion"].tolist() == [1, 2, 0]
    assert ruta.read_bytes() == original


def test_duplicados_entre_bloques(tmp_path, tabla):
    duplicada = tabla.iloc[[0]].assign(id=99)
    ruta = guardar(tmp_path, pd.concat([tabla, duplicada], ignore_index=True))
    with pytest.raises(ValueError, match="Claves duplicadas"):
        leer_datos_favorita(ruta, tamano_bloque=1)


@pytest.mark.parametrize(
    "tiendas,familias,ids",
    [(None, None, [0, 2, 1]), ([1], None, [0, 2]),
     (None, ["B"], [1]), ([1], ["B"], []), ([], None, [])],
)
def test_filtros(tmp_path, tabla, tiendas, familias, ids):
    datos = leer_datos_favorita(
        guardar(tmp_path, tabla), tiendas=tiendas, familias=familias, tamano_bloque=1
    )
    assert datos["id"].tolist() == ids


def test_fecha_invalida(tmp_path, tabla):
    tabla.loc[0, "date"] = "no-es-fecha"
    with pytest.raises(ValueError):
        leer_datos_favorita(guardar(tmp_path, tabla))


@pytest.mark.parametrize("valor", [float("nan"), float("inf")])
def test_ventas_observadas_invalidas(tmp_path, tabla, valor):
    tabla.loc[0, "sales"] = valor
    with pytest.raises(ValueError, match="faltantes|no finitos"):
        leer_datos_favorita(guardar(tmp_path, tabla))


def test_lectura_por_bloques_equivalente(tmp_path, tabla):
    ruta = guardar(tmp_path, tabla)
    assert_frame_equal(
        leer_datos_favorita(ruta, tamano_bloque=1),
        leer_datos_favorita(ruta, tamano_bloque=100),
    )
