"""Codificacion categorica con fixtures pequenas, sin usar los CSV reales."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from scipy.sparse import isspmatrix_csr
from sklearn.preprocessing import OneHotEncoder

from src.preprocesamiento_temporal import generar_variables_historicas, dividir_por_fecha
from src.transformaciones import CodificadorCategorias, codificar_particiones


@pytest.fixture
def entrenamiento():
    return pd.DataFrame({
        "fecha": pd.to_datetime(["2017-06-01", "2017-06-02", "2017-06-03"]),
        "tienda": [2, 1, 2],
        "familia": ["BEAUTY", "BABY CARE", "BABY CARE"],
        "ventas_objetivo": [10.0, 20.0, 30.0],
    })


def test_codificacion_dimensiones_nombres_y_dispersion(entrenamiento):
    original = entrenamiento.copy(deep=True)
    codificador = CodificadorCategorias(entrenamiento)
    matriz = codificador.transformar(entrenamiento)
    assert isspmatrix_csr(matriz)
    assert matriz.dtype == np.float32
    assert matriz.shape == (3, 4)
    assert matriz.nnz == 6
    assert codificador.nombres_columnas == (
        "tienda_1", "tienda_2", "familia_BABY CARE", "familia_BEAUTY"
    )
    np.testing.assert_array_equal(
        matriz.toarray(), [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 1, 0]]
    )
    assert_frame_equal(entrenamiento, original)


def test_orden_estable_ante_reordenamiento(entrenamiento):
    codificador = CodificadorCategorias(entrenamiento)
    otro = CodificadorCategorias(entrenamiento.iloc[::-1])
    assert otro.nombres_columnas == codificador.nombres_columnas
    reordenada = entrenamiento.iloc[::-1][["familia", "ventas_objetivo", "tienda", "fecha"]]
    np.testing.assert_array_equal(
        codificador.transformar(reordenada).toarray(),
        codificador.transformar(entrenamiento).toarray()[::-1],
    )


def test_mismo_codificador_sin_ajuste_en_validacion_test(entrenamiento, monkeypatch):
    particiones = {
        "entrenamiento": entrenamiento,
        "validacion": entrenamiento.iloc[[0]].assign(fecha=pd.Timestamp("2017-06-16")),
        "test": entrenamiento.iloc[[1]].assign(fecha=pd.Timestamp("2017-07-16")),
    }
    original_fit = OneHotEncoder.fit
    llamadas = []

    def registrar(self, datos, *args, **kwargs):
        llamadas.append(datos.copy())
        return original_fit(self, datos, *args, **kwargs)

    monkeypatch.setattr(OneHotEncoder, "fit", registrar)
    codificador, matrices = codificar_particiones(particiones)
    assert len(llamadas) == 1
    assert_frame_equal(llamadas[0], entrenamiento[["tienda", "familia"]])

    def prohibido(*args, **kwargs):
        pytest.fail("transformar no debe ajustar el codificador")

    monkeypatch.setattr(OneHotEncoder, "fit", prohibido)
    for nombre, datos in particiones.items():
        assert matrices[nombre].shape[1] == 4
        assert isspmatrix_csr(matrices[nombre])
        np.testing.assert_array_equal(
            matrices[nombre].toarray(), codificador.transformar(datos).toarray()
        )


@pytest.mark.parametrize("columna,valor", [("tienda", 99), ("familia", "FAMILIA NUEVA")])
def test_categoria_desconocida_error_sin_modificar_estado(entrenamiento, columna, valor):
    codificador = CodificadorCategorias(entrenamiento)
    antes = codificador.transformar(entrenamiento)
    nombres = codificador.nombres_columnas
    desconocidos = entrenamiento.iloc[[0]].assign(**{columna: valor})
    with pytest.raises(ValueError, match=f"Categorias desconocidas en {columna}"):
        codificador.transformar(desconocidos)
    assert nombres == codificador.nombres_columnas
    assert (antes != codificador.transformar(entrenamiento)).nnz == 0


@pytest.mark.parametrize("particion", ["validacion", "test"])
def test_no_aprende_categorias_exclusivas_de_otras_particiones(entrenamiento, particion):
    partes = {nombre: entrenamiento.copy() for nombre in ("entrenamiento", "validacion", "test")}
    partes[particion]["familia"] = "EXCLUSIVA"
    with pytest.raises(ValueError, match="Categorias desconocidas en familia"):
        codificar_particiones(partes)


@pytest.mark.parametrize("fecha", ["2017-06-16", "2017-07-16"])
def test_rechaza_ajuste_con_fechas_fuera_de_entrenamiento(entrenamiento, fecha):
    entrenamiento.loc[0, "fecha"] = pd.Timestamp(fecha)
    with pytest.raises(ValueError, match="exclusivamente fechas de entrenamiento"):
        CodificadorCategorias(entrenamiento)


def test_particion_vacia_y_entrenamiento_vacio(entrenamiento):
    codificador = CodificadorCategorias(entrenamiento)
    matriz = codificador.transformar(entrenamiento.iloc[:0])
    assert isspmatrix_csr(matriz)
    assert matriz.shape == (0, 4)
    with pytest.raises(ValueError, match="entrenamiento vacio"):
        CodificadorCategorias(entrenamiento.iloc[:0])


def test_esquema_y_faltantes(entrenamiento):
    with pytest.raises(ValueError, match="Faltan columnas categoricas"):
        CodificadorCategorias(entrenamiento.drop(columns="familia"))
    entrenamiento.loc[0, "familia"] = None
    with pytest.raises(ValueError, match="valores faltantes"):
        CodificadorCategorias(entrenamiento)


def test_compatibilidad_con_historia_ciclos_y_particiones():
    fechas = pd.date_range("2017-06-01", "2017-08-15")
    partes = []
    for tienda, familia in [(1, "BABY CARE"), (2, "BEAUTY")]:
        partes.append(pd.DataFrame({
            "id": range(len(fechas)), "fecha": fechas, "tienda": tienda,
            "familia": familia, "ventas_objetivo": 1.0, "cantidad_en_promocion": 0,
        }))
    temporal = generar_variables_historicas(pd.concat(partes, ignore_index=True))
    particiones = dividir_por_fecha(temporal)
    originales = {nombre: datos.copy(deep=True) for nombre, datos in particiones.items()}
    codificador, matrices = codificar_particiones(particiones)
    for nombre, datos in particiones.items():
        matriz = matrices[nombre]
        assert matriz.shape == (len(datos), 4)
        assert matriz.nnz == 2 * len(datos)
        assert_frame_equal(datos, originales[nombre])
        # La fila codificada conserva alineacion con objetivo y metadatos.
        primera = codificador.transformar(datos.iloc[[0]])
        assert (matriz[0] != primera).nnz == 0
