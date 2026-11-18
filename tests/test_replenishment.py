from datetime import date, timedelta
import math

import pandas as pd
import pytest

from src.application.date_validation import (
    calcular_fecha_maxima_prediccion,
    parsear_fecha,
)
from src.application.replenishment_strategies import (
    ReposicionBasica,
    ReposicionConSeguridad,
)
from src.application.services.replenishment_service import ServicioReposicion
from src.datos_sinteticos import (
    COLUMNAS_INVENTARIO,
    generar_catalogo_productos,
)
from src.domain.exceptions import (
    EstrategiaReposicionNoEncontrada,
    ErrorInventario,
    InventarioNoEncontrado,
    SolicitudPrediccionInvalida,
)
from src.domain.models import Inventario, Prediccion, Producto
from src.infrastructure.repositories.csv_inventory_repository import (
    RepositorioInventarioCSV,
)


PRODUCTO = Producto("BEB-002", "Agua mineral 1.5 l", "bebidas")


def crear_inventario(
    *,
    tienda_id=2,
    stock_actual=4,
    stock_minimo=2,
    tiempo_reposicion=3,
):
    return Inventario(
        tienda_id=tienda_id,
        producto_id=PRODUCTO.producto_id,
        stock_actual=stock_actual,
        stock_minimo=stock_minimo,
        tiempo_reposicion=tiempo_reposicion,
    )


@pytest.mark.parametrize(
    "estrategia,demanda,stock,minimo,esperado",
    [
        (ReposicionBasica(), 6.31, 4, 2, 3),
        (ReposicionConSeguridad(), 6.31, 4, 2, 5),
        (ReposicionBasica(), 4, 10, 4, 0),
        (ReposicionConSeguridad(), 4, 10, 4, 0),
        (ReposicionBasica(), 2.01, 0, 0, 3),
        (ReposicionConSeguridad(), 0, 0, 2, 2),
    ],
)
def test_estrategias_redondean_hacia_arriba_y_no_recomiendan_negativos(
    estrategia, demanda, stock, minimo, esperado
):
    inventario = crear_inventario(
        stock_actual=stock,
        stock_minimo=minimo,
    )

    assert estrategia.calcular(demanda, inventario) == esperado


@pytest.mark.parametrize("estrategia", [ReposicionBasica(), ReposicionConSeguridad()])
@pytest.mark.parametrize("demanda", [-1, float("nan"), float("inf"), True, "4"])
def test_estrategias_rechazan_demanda_invalida(estrategia, demanda):
    with pytest.raises(ValueError, match="demanda acumulada"):
        estrategia.calcular(demanda, crear_inventario())


@pytest.mark.parametrize(
    "campo,valor",
    [
        ("tienda_id", 0),
        ("tienda_id", True),
        ("stock_actual", -1),
        ("stock_minimo", -1),
        ("tiempo_reposicion", 0),
        ("tiempo_reposicion", -1),
    ],
)
def test_entidad_inventario_rechaza_valores_invalidos(campo, valor):
    valores = {
        "tienda_id": 2,
        "producto_id": PRODUCTO.producto_id,
        "stock_actual": 4,
        "stock_minimo": 2,
        "tiempo_reposicion": 3,
    }
    valores[campo] = valor

    with pytest.raises(ValueError):
        Inventario(**valores)


class RepositorioInventarioFalso:
    def __init__(self, inventario):
        self.inventario = inventario
        self.consultas = []

    def buscar(self, *, tienda_id, producto_id):
        self.consultas.append((tienda_id, producto_id))
        if self.inventario is None:
            return None
        if (tienda_id, producto_id) == (
            self.inventario.tienda_id,
            self.inventario.producto_id,
        ):
            return self.inventario
        return None


class PronosticadorFalso:
    tiendas_disponibles = (1, 2, 3)
    fecha_corte = date(2026, 11, 16)

    def __init__(self, demanda_por_dia=1.01):
        self.demanda_por_dia = demanda_por_dia
        self.consultas = []

    def predecir(self, *, tienda_id, producto_id, fecha):
        self.consultas.append((tienda_id, producto_id, fecha))
        return Prediccion(
            tienda_id=tienda_id,
            producto=PRODUCTO,
            fecha=fecha,
            unidades_estimadas=self.demanda_por_dia,
        )

    def validar_fecha(self, fecha):
        dia = parsear_fecha(fecha)
        fecha_maxima = calcular_fecha_maxima_prediccion(self.fecha_corte)
        if dia > fecha_maxima:
            raise SolicitudPrediccionInvalida(
                f"La fecha excede el horizonte de demostración; fecha máxima: "
                f"{fecha_maxima.isoformat()}"
            )
        return dia


def crear_servicio(inventario, pronosticador):
    estrategias = (ReposicionBasica(), ReposicionConSeguridad())
    return ServicioReposicion(
        RepositorioInventarioFalso(inventario),
        pronosticador,
        {estrategia.codigo: estrategia for estrategia in estrategias},
    )


def test_servicio_predice_todos_los_dias_inclusive_y_no_modifica_stock():
    inventario = crear_inventario(
        stock_actual=2,
        stock_minimo=3,
        tiempo_reposicion=3,
    )
    pronosticador = PronosticadorFalso()
    servicio = crear_servicio(inventario, pronosticador)
    stock_original = inventario.stock_actual

    resultado = servicio.recomendar(
        tienda_id=2,
        producto_id="BEB-002",
        fecha="2026-11-25",
        estrategia="seguridad",
    )

    assert [consulta[2] for consulta in pronosticador.consultas] == [
        date(2026, 11, 25),
        date(2026, 11, 26),
        date(2026, 11, 27),
    ]
    assert len(resultado.predicciones) == inventario.tiempo_reposicion
    assert math.isclose(resultado.demanda_acumulada, 3.03)
    assert resultado.cantidad_recomendada == 5
    assert resultado.requiere_reposicion is True
    assert inventario.stock_actual == stock_original


@pytest.mark.parametrize(
    "tiempo_reposicion,fecha_maxima",
    [
        (1, "2027-02-14"),
        (4, "2027-02-11"),
        (7, "2027-02-08"),
    ],
)
def test_horizonte_de_reposicion_depende_del_plazo(
    tiempo_reposicion, fecha_maxima
):
    servicio = crear_servicio(
        crear_inventario(tiempo_reposicion=tiempo_reposicion),
        PronosticadorFalso(),
    )

    horizonte = servicio.obtener_horizonte_reposicion(
        tienda_id=2,
        producto_id="BEB-002",
    )

    assert horizonte == {
        "tiempo_reposicion": tiempo_reposicion,
        "fecha_maxima_reposicion": fecha_maxima,
    }


@pytest.mark.parametrize(
    "tiempo_reposicion,fecha_maxima",
    [
        (1, date(2027, 2, 14)),
        (4, date(2027, 2, 11)),
        (7, date(2027, 2, 8)),
    ],
)
def test_ultima_fecha_de_reposicion_predice_cobertura_inclusiva(
    tiempo_reposicion, fecha_maxima
):
    pronosticador = PronosticadorFalso()
    servicio = crear_servicio(
        crear_inventario(tiempo_reposicion=tiempo_reposicion),
        pronosticador,
    )

    resultado = servicio.recomendar(
        tienda_id=2,
        producto_id="BEB-002",
        fecha=fecha_maxima,
        estrategia="basica",
    )

    assert len(resultado.predicciones) == tiempo_reposicion
    assert pronosticador.consultas[0][2] == fecha_maxima
    assert pronosticador.consultas[-1][2] == date(2027, 2, 14)


@pytest.mark.parametrize(
    "tiempo_reposicion,fecha_maxima",
    [
        (1, date(2027, 2, 14)),
        (4, date(2027, 2, 11)),
        (7, date(2027, 2, 8)),
    ],
)
def test_dia_posterior_al_maximo_de_reposicion_se_rechaza(
    tiempo_reposicion, fecha_maxima
):
    pronosticador = PronosticadorFalso()
    servicio = crear_servicio(
        crear_inventario(tiempo_reposicion=tiempo_reposicion),
        pronosticador,
    )

    fecha_siguiente = fecha_maxima + timedelta(days=1)
    if fecha_siguiente > date(2027, 2, 14):
        mensaje = "fecha máxima: 2027-02-14"
    else:
        mensaje = f"última fecha válida para reponer: {fecha_maxima.isoformat()}"
    with pytest.raises(SolicitudPrediccionInvalida, match=mensaje):
        servicio.recomendar(
            tienda_id=2,
            producto_id="BEB-002",
            fecha=fecha_siguiente,
            estrategia="basica",
        )

    if fecha_siguiente > date(2027, 2, 14):
        assert pronosticador.consultas == []
    else:
        assert len(pronosticador.consultas) == 1
        assert pronosticador.consultas[0][2] == fecha_siguiente


def test_fecha_extremadamente_lejana_no_se_solicita_al_pronosticador():
    pronosticador = PronosticadorFalso()
    servicio = crear_servicio(crear_inventario(), pronosticador)

    with pytest.raises(SolicitudPrediccionInvalida, match="fecha máxima"):
        servicio.recomendar(
            tienda_id=2,
            producto_id="BEB-002",
            fecha="2263-01-01",
            estrategia="basica",
        )

    assert pronosticador.consultas == []


def test_servicio_indica_cuando_stock_alcanza_y_valida_estrategia():
    inventario = crear_inventario(
        stock_actual=100,
        tiempo_reposicion=2,
    )
    servicio = crear_servicio(inventario, PronosticadorFalso())

    resultado = servicio.recomendar(
        tienda_id=2,
        producto_id="BEB-002",
        fecha="2026-11-25",
        estrategia="basica",
    )

    assert resultado.cantidad_recomendada == 0
    assert resultado.requiere_reposicion is False
    assert "no es necesario reponer" in resultado.mensaje
    with pytest.raises(EstrategiaReposicionNoEncontrada):
        servicio.recomendar(
            tienda_id=2,
            producto_id="BEB-002",
            fecha="2026-11-25",
            estrategia="otra",
        )


def test_servicio_falla_si_no_hay_inventario():
    servicio = crear_servicio(None, PronosticadorFalso())

    with pytest.raises(InventarioNoEncontrado):
        servicio.recomendar(
            tienda_id=2,
            producto_id="BEB-002",
            fecha="2026-11-25",
            estrategia="basica",
        )


def crear_csv_inventario(ruta):
    catalogo = generar_catalogo_productos()
    filas = [
        {
            "tienda_id": tienda_id,
            "producto_id": producto_id,
            "stock_actual": 11,
            "stock_minimo": 5,
            "tiempo_reposicion": 4,
        }
        for tienda_id in (1, 2, 3)
        for producto_id in catalogo["producto_id"]
    ]
    pd.DataFrame(filas, columns=COLUMNAS_INVENTARIO).to_csv(ruta, index=False)


def test_repositorio_busca_por_tienda_y_producto_en_csv(tmp_path):
    ruta = tmp_path / "inventario.csv"
    crear_csv_inventario(ruta)
    repositorio = RepositorioInventarioCSV(ruta)

    inventario = repositorio.buscar(tienda_id=2, producto_id="BEB-002")

    assert inventario == Inventario(
        tienda_id=2,
        producto_id="BEB-002",
        stock_actual=11,
        stock_minimo=5,
        tiempo_reposicion=4,
    )
    assert repositorio.buscar(tienda_id=4, producto_id="BEB-002") is None


def test_repositorio_informa_csv_inexistente_o_invalido(tmp_path):
    with pytest.raises(ErrorInventario):
        RepositorioInventarioCSV(tmp_path / "no-existe.csv").buscar(
            tienda_id=2,
            producto_id="BEB-002",
        )

    ruta = tmp_path / "duplicado.csv"
    crear_csv_inventario(ruta)
    inventario = pd.read_csv(ruta)
    pd.concat([inventario, inventario.iloc[[0]]]).to_csv(ruta, index=False)
    with pytest.raises(ErrorInventario):
        RepositorioInventarioCSV(ruta).buscar(
            tienda_id=2,
            producto_id="BEB-002",
        )
