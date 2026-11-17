import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.datos_sinteticos import (
    CATEGORIAS,
    COLUMNAS_CATALOGO,
    COLUMNAS_INVENTARIO,
    COLUMNAS_VENTAS,
    FECHA_FIN,
    FECHA_INICIO,
    TIENDAS,
    generar_archivos_sinteticos,
    generar_catalogo_productos,
    generar_inventario_sintetico,
    generar_ventas_sinteticas,
    validar_catalogo_productos,
    validar_inventario,
    validar_ventas,
)


@pytest.fixture(scope="module")
def datos():
    catalogo = generar_catalogo_productos()
    ventas = generar_ventas_sinteticas()
    inventario = generar_inventario_sintetico(ventas, catalogo)
    return catalogo, ventas, inventario


def test_catalogo_cubre_los_productos_y_categorias_requeridos(datos):
    catalogo, _, _ = datos
    assert tuple(catalogo.columns) == COLUMNAS_CATALOGO
    assert len(catalogo) == 30
    assert catalogo["producto_id"].is_unique
    assert set(catalogo["categoria"]) == set(CATEGORIAS)
    assert catalogo["categoria"].value_counts().to_dict() == {
        categoria: 6 for categoria in CATEGORIAS
    }
    validar_catalogo_productos(catalogo)


def test_ventas_tienen_panel_completo_y_no_incluyen_fechas_futuras(datos):
    _, ventas, _ = datos
    dias = len(pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D"))
    assert tuple(ventas.columns) == COLUMNAS_VENTAS
    assert len(ventas) == dias * len(TIENDAS) * 30
    assert ventas["fecha"].min() == FECHA_INICIO.strftime("%Y-%m-%d")
    assert ventas["fecha"].max() == FECHA_FIN.strftime("%Y-%m-%d")
    assert not ventas.duplicated(["fecha", "tienda_id", "producto_id"]).any()
    assert ventas["promocion"].dtype == bool
    assert validar_ventas(ventas) == len(ventas)


def test_generadores_son_reproducibles(datos):
    catalogo, ventas, inventario = datos
    assert_frame_equal(catalogo, generar_catalogo_productos())
    assert_frame_equal(ventas, generar_ventas_sinteticas())
    assert_frame_equal(
        inventario,
        generar_inventario_sintetico(generar_ventas_sinteticas(), catalogo),
    )
    assert not ventas.equals(generar_ventas_sinteticas(semilla=43))


def test_promociones_afectan_precio_y_demanda(datos):
    _, ventas, _ = datos
    assert ventas.loc[ventas["promocion"], "precio"].mean() < ventas.loc[
        ~ventas["promocion"], "precio"
    ].mean()
    assert ventas.loc[ventas["promocion"], "unidades_vendidas"].mean() > ventas.loc[
        ~ventas["promocion"], "unidades_vendidas"
    ].mean()


def test_inventario_es_separado_y_valido(datos):
    catalogo, _, inventario = datos
    assert tuple(inventario.columns) == COLUMNAS_INVENTARIO
    assert len(inventario) == len(TIENDAS) * len(catalogo)
    assert not inventario.duplicated(["tienda_id", "producto_id"]).any()
    assert (inventario[["stock_actual", "stock_minimo"]] >= 0).all().all()
    assert (inventario["tiempo_reposicion"] > 0).all()
    assert validar_inventario(inventario, catalogo) == len(inventario)


@pytest.mark.parametrize(
    "cambio, mensaje",
    [
        (lambda tabla: tabla.drop(columns="precio"), "esquema"),
        (lambda tabla: pd.concat([tabla, tabla.iloc[[0]]], ignore_index=True), "claves duplicadas"),
        (lambda tabla: tabla.assign(unidades_vendidas=-1), "enteros finitos no negativos"),
        (lambda tabla: tabla.assign(producto_id="NO-EXISTE"), "identificadores desconocidos"),
        (lambda tabla: tabla.assign(fecha="2026-11-17"), "entre 2023-01-01 y 2026-11-16"),
        (lambda tabla: tabla.assign(fecha="no-es-fecha"), "fechas inválidas"),
        (lambda tabla: tabla.assign(precio=-1), "valores finitos positivos"),
        (lambda tabla: tabla.assign(promocion=1), "valores booleanos"),
        (lambda tabla: tabla.iloc[1:].copy(), "exactamente"),
    ],
)
def test_validador_de_ventas_rechaza_datos_invalidos(datos, cambio, mensaje):
    _, ventas, _ = datos
    with pytest.raises(ValueError, match=mensaje):
        validar_ventas(cambio(ventas.copy()))


@pytest.mark.parametrize(
    "columna, valor, mensaje",
    [
        ("producto_id", "BEB-001", "producto_id duplicados"),
        ("categoria", "otra", "cinco categorías"),
    ],
)
def test_validador_de_catalogo_rechaza_inconsistencias(datos, columna, valor, mensaje):
    catalogo, _, _ = datos
    invalido = catalogo.copy()
    invalido.loc[1, columna] = valor
    with pytest.raises(ValueError, match=mensaje):
        validar_catalogo_productos(invalido)


@pytest.mark.parametrize(
    "cambio, mensaje",
    [
        (lambda tabla: tabla.assign(stock_actual=-1), "enteros finitos no negativos"),
        (lambda tabla: tabla.assign(stock_minimo=True), "no booleanos"),
        (lambda tabla: tabla.assign(tiempo_reposicion=0), "debe ser positivo"),
        (lambda tabla: pd.concat([tabla, tabla.iloc[[0]]], ignore_index=True), "claves duplicadas"),
        (lambda tabla: tabla.assign(producto_id="NO-EXISTE"), "identificadores desconocidos"),
    ],
)
def test_validador_de_inventario_rechaza_datos_invalidos(datos, cambio, mensaje):
    catalogo, _, inventario = datos
    with pytest.raises(ValueError, match=mensaje):
        validar_inventario(cambio(inventario.copy()), catalogo)


def test_escritura_de_archivos_mantiene_esquemas_y_registros(tmp_path, datos):
    catalogo, ventas, inventario = datos
    archivos = generar_archivos_sinteticos(tmp_path)
    assert set(archivos) == {"ventas", "catalogo", "inventario"}
    assert tuple(pd.read_csv(archivos["ventas"]).columns) == COLUMNAS_VENTAS
    assert tuple(pd.read_csv(archivos["catalogo"]).columns) == COLUMNAS_CATALOGO
    assert tuple(pd.read_csv(archivos["inventario"]).columns) == COLUMNAS_INVENTARIO
    assert len(pd.read_csv(archivos["ventas"])) == len(ventas)
    assert len(pd.read_csv(archivos["catalogo"])) == len(catalogo)
    assert len(pd.read_csv(archivos["inventario"])) == len(inventario)
