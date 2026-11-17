"""Orquestación del caso de uso de predicción por producto."""

from datetime import date, datetime
from numbers import Integral

from src.application.ports import MotorPrediccion, RepositorioProductos
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
    ):
        self._repositorio_productos = repositorio_productos
        self._motor_prediccion = motor_prediccion

    def listar_productos(self):
        return self._repositorio_productos.listar()

    def predecir(self, *, tienda_id, producto_id, fecha):
        tienda = self._validar_tienda(tienda_id)
        producto_id = self._validar_producto_id(producto_id)
        dia = self._parsear_fecha(fecha)
        if dia <= self._motor_prediccion.fecha_corte:
            raise SolicitudPrediccionInvalida(
                "La fecha debe ser posterior al último día observado "
                f"({self._motor_prediccion.fecha_corte.isoformat()})"
            )

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
        if isinstance(fecha, datetime):
            raise SolicitudPrediccionInvalida(
                "fecha debe ser válida y tener el formato AAAA-MM-DD"
            )
        if isinstance(fecha, date):
            return fecha
        if not isinstance(fecha, str):
            raise SolicitudPrediccionInvalida(
                "fecha debe tener el formato AAAA-MM-DD"
            )
        try:
            dia = datetime.strptime(fecha, "%Y-%m-%d")
        except ValueError:
            raise SolicitudPrediccionInvalida(
                "fecha debe ser válida y tener el formato AAAA-MM-DD"
            ) from None
        if dia.strftime("%Y-%m-%d") != fecha:
            raise SolicitudPrediccionInvalida(
                "fecha debe ser válida y tener el formato AAAA-MM-DD"
            )
        return dia.date()
