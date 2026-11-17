import ast
from datetime import date
from pathlib import Path

import pytest

from src.application.services.prediction_service import ServicioPrediccion
from src.domain.exceptions import (
    ProductoNoEncontrado,
    SolicitudPrediccionInvalida,
    TiendaNoEncontrada,
)
from src.domain.models import Prediccion, Producto


FECHA_CORTE = date(2026, 11, 16)
PRODUCTO = Producto("BEB-001", "Agua mineral 500 ml", "bebidas")


class RepositorioEnMemoria:
    def __init__(self, productos):
        self.productos = {producto.producto_id: producto for producto in productos}

    def buscar_por_id(self, producto_id):
        return self.productos.get(producto_id)

    def listar(self):
        return tuple(self.productos.values())


class MotorFalso:
    fecha_corte = FECHA_CORTE
    tiendas_disponibles = (1, 2, 3)

    def __init__(self, estimacion=12.5):
        self.estimacion = estimacion
        self.consultas = []

    def predecir(self, *, tienda_id, producto_id, fecha):
        self.consultas.append((tienda_id, producto_id, fecha))
        return self.estimacion


def crear_servicio():
    motor = MotorFalso()
    servicio = ServicioPrediccion(
        RepositorioEnMemoria([PRODUCTO]),
        motor,
        hoy=lambda: date(2026, 11, 17),
    )
    return servicio, motor


def test_servicio_inyecta_dependencias_y_devuelve_entidad_de_dominio():
    servicio, motor = crear_servicio()

    resultado = servicio.predecir(
        tienda_id=2,
        producto_id="BEB-001",
        fecha="2026-11-17",
    )

    assert resultado == Prediccion(
        tienda_id=2,
        producto=PRODUCTO,
        fecha=date(2026, 11, 17),
        unidades_estimadas=12.5,
    )
    assert motor.consultas == [(2, "BEB-001", date(2026, 11, 17))]


@pytest.mark.parametrize(
    "consulta, error",
    [
        ({"tienda_id": 99, "producto_id": "BEB-001", "fecha": "2026-11-17"}, TiendaNoEncontrada),
        ({"tienda_id": 1, "producto_id": "NO-EXISTE", "fecha": "2026-11-17"}, ProductoNoEncontrado),
        ({"tienda_id": True, "producto_id": "BEB-001", "fecha": "2026-11-17"}, SolicitudPrediccionInvalida),
        ({"tienda_id": 1, "producto_id": "BEB-001", "fecha": "2026-11-16"}, SolicitudPrediccionInvalida),
        ({"tienda_id": 1, "producto_id": "BEB-001", "fecha": "2026-02-30"}, SolicitudPrediccionInvalida),
    ],
)
def test_servicio_rechaza_consultas_invalidas_sin_invocar_motor(consulta, error):
    servicio, motor = crear_servicio()

    with pytest.raises(error):
        servicio.predecir(**consulta)

    assert motor.consultas == []


def test_opciones_y_fecha_minima_respetan_hoy_y_corte_del_dataset():
    servicio = ServicioPrediccion(
        RepositorioEnMemoria([PRODUCTO]),
        MotorFalso(),
        hoy=lambda: date(2026, 11, 20),
    )

    assert servicio.obtener_opciones() == {
        "tiendas": (1, 2, 3),
        "fecha_corte": "2026-11-16",
        "fecha_minima": "2026-11-20",
        "fecha_maxima": "2027-02-14",
    }
    with pytest.raises(SolicitudPrediccionInvalida, match="fecha mínima"):
        servicio.predecir(
            tienda_id=1,
            producto_id="BEB-001",
            fecha="2026-11-19",
        )


def test_servicio_acepta_primera_y_ultima_fecha_del_horizonte():
    servicio, motor = crear_servicio()

    primera = servicio.predecir(
        tienda_id=1,
        producto_id="BEB-001",
        fecha="2026-11-17",
    )
    ultima = servicio.predecir(
        tienda_id=1,
        producto_id="BEB-001",
        fecha="2027-02-14",
    )

    assert primera.fecha == date(2026, 11, 17)
    assert ultima.fecha == date(2027, 2, 14)
    assert [consulta[2] for consulta in motor.consultas] == [
        date(2026, 11, 17),
        date(2027, 2, 14),
    ]


@pytest.mark.parametrize("fecha", ["2027-02-15", "2263-01-01"])
def test_servicio_rechaza_fechas_posteriores_al_horizonte(fecha):
    servicio, motor = crear_servicio()

    with pytest.raises(SolicitudPrediccionInvalida, match="fecha máxima: 2027-02-14"):
        servicio.predecir(
            tienda_id=1,
            producto_id="BEB-001",
            fecha=fecha,
        )

    assert motor.consultas == []


def test_servicio_no_importa_framework_persistencia_ni_pytorch():
    ruta = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "application"
        / "services"
        / "prediction_service.py"
    )
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    modulos = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            modulos.update(alias.name.split(".")[0] for alias in nodo.names)
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            modulos.add(nodo.module.split(".")[0])

    assert modulos.isdisjoint({"flask", "csv", "pandas", "torch"})
    assert modulos.issubset({"datetime", "numbers", "src", "typing"})
