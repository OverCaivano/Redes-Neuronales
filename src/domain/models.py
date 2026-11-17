"""Entidades inmutables del caso de uso de predicción."""

from dataclasses import dataclass
from datetime import date, datetime
import math
from numbers import Integral, Real


@dataclass(frozen=True)
class Producto:
    producto_id: str
    nombre: str
    categoria: str

    def __post_init__(self):
        for campo in ("producto_id", "nombre", "categoria"):
            valor = getattr(self, campo)
            if not isinstance(valor, str) or not valor.strip():
                raise ValueError(f"{campo} debe ser texto no vacío")


@dataclass(frozen=True)
class Prediccion:
    tienda_id: int
    producto: Producto
    fecha: date
    unidades_estimadas: float
    origen: str = "datos_sinteticos"

    def __post_init__(self):
        if (
            isinstance(self.tienda_id, bool)
            or not isinstance(self.tienda_id, Integral)
            or self.tienda_id <= 0
        ):
            raise ValueError("tienda_id debe ser un entero positivo")
        if isinstance(self.fecha, datetime) or not isinstance(self.fecha, date):
            raise ValueError("fecha debe ser una fecha sin hora")
        if (
            isinstance(self.unidades_estimadas, bool)
            or not isinstance(self.unidades_estimadas, Real)
            or not math.isfinite(self.unidades_estimadas)
            or self.unidades_estimadas < 0
        ):
            raise ValueError("unidades_estimadas debe ser un número finito no negativo")
        if self.origen != "datos_sinteticos":
            raise ValueError("El origen de esta predicción debe ser sintético")
