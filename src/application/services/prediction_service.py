"""Orquestación del caso de uso de predicción por producto."""

from datetime import date, timedelta
from numbers import Integral
from typing import Callable

from src.application.ports import MotorPrediccion, RepositorioProductos
from src.application.date_validation import (
    HORIZONTE_PREDICCION_DIAS,
    calcular_fecha_maxima_prediccion,
    parsear_fecha,
)
from src.domain.exceptions import (
    ProductoNoEncontrado,
    SolicitudPrediccionInvalida,
    TiendaNoEncontrada,
)
from src.domain.models import Prediccion


class ServicioPrediccion:
    def __init__(
        self,
        repositorio_productos: RepositorioProductos,
        motor_prediccion: MotorPrediccion,
        hoy: Callable[[], date] = date.today,
    ):
        self._repositorio_productos = repositorio_productos
        self._motor_prediccion = motor_prediccion
        self._hoy = hoy

    def listar_productos(self):
        return self._repositorio_productos.listar()

    @property
    def fecha_corte(self) -> date:
        return self._motor_prediccion.fecha_corte

    @property
    def tiendas_disponibles(self) -> tuple[int, ...]:
        return self._motor_prediccion.tiendas_disponibles

    def obtener_opciones(self):
        fecha_corte = self._motor_prediccion.fecha_corte
        return {
            "tiendas": self._motor_prediccion.tiendas_disponibles,
            "fecha_corte": fecha_corte.isoformat(),
            "fecha_minima": self._fecha_minima(fecha_corte).isoformat(),
            "fecha_maxima": calcular_fecha_maxima_prediccion(fecha_corte).isoformat(),
        }

    def validar_fecha(self, fecha: date | str) -> date:
        dia = self._parsear_fecha(fecha)
        fecha_corte = self._motor_prediccion.fecha_corte
        fecha_minima = self._fecha_minima(fecha_corte)
        if dia < fecha_minima:
            raise SolicitudPrediccionInvalida(
                "La fecha debe ser actual o futura y posterior al último día "
                f"observado ({fecha_corte.isoformat()}); fecha mínima: "
                f"{fecha_minima.isoformat()}"
            )
        fecha_maxima = calcular_fecha_maxima_prediccion(fecha_corte)
        if dia > fecha_maxima:
            raise SolicitudPrediccionInvalida(
                "La fecha excede el horizonte de demostración; fecha máxima: "
                f"{fecha_maxima.isoformat()} "
                f"({HORIZONTE_PREDICCION_DIAS} días posteriores al último día "
                "disponible)."
            )
        return dia

    def predecir(self, *, tienda_id, producto_id, fecha):
        tienda = self._validar_tienda(tienda_id)
        producto_id = self._validar_producto_id(producto_id)
        dia = self.validar_fecha(fecha)

        producto = self._repositorio_productos.buscar_por_id(producto_id)
        if producto is None:
            raise ProductoNoEncontrado(f"producto_id inexistente: {producto_id}")

        estimacion = self._motor_prediccion.predecir(
            tienda_id=tienda,
            producto_id=producto.producto_id,
            fecha=dia,
        )
        return Prediccion(
            tienda_id=tienda,
            producto=producto,
            fecha=dia,
            unidades_estimadas=float(estimacion),
        )

    def _fecha_minima(self, fecha_corte):
        return max(self._hoy(), fecha_corte + timedelta(days=1))

    def _validar_tienda(self, tienda_id):
        if isinstance(tienda_id, bool) or not isinstance(tienda_id, Integral):
            raise SolicitudPrediccionInvalida("tienda_id debe ser un entero")
        tienda = int(tienda_id)
        if tienda not in self._motor_prediccion.tiendas_disponibles:
            raise TiendaNoEncontrada(f"tienda_id inexistente: {tienda}")
        return tienda

    @staticmethod
    def _validar_producto_id(producto_id):
        if not isinstance(producto_id, str) or not producto_id.strip():
            raise SolicitudPrediccionInvalida(
                "producto_id debe ser texto no vacío"
            )
        return producto_id

    @staticmethod
    def _parsear_fecha(fecha):
        return parsear_fecha(fecha)
