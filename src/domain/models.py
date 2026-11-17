"""Entidades inmutables del caso de uso de predicción."""

from dataclasses import dataclass
from datetime import date, datetime, timedelta
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


@dataclass(frozen=True)
class Inventario:
    tienda_id: int
    producto_id: str
    stock_actual: int
    stock_minimo: int
    tiempo_reposicion: int

    def __post_init__(self):
        for campo in (
            "tienda_id",
            "stock_actual",
            "stock_minimo",
            "tiempo_reposicion",
        ):
            valor = getattr(self, campo)
            if (
                isinstance(valor, bool)
                or not isinstance(valor, Integral)
                or valor < 0
            ):
                raise ValueError(f"{campo} debe ser un entero no negativo")
        if self.tienda_id == 0:
            raise ValueError("tienda_id debe ser positivo")
        if self.tiempo_reposicion == 0:
            raise ValueError("tiempo_reposicion debe ser positivo")
        if not isinstance(self.producto_id, str) or not self.producto_id.strip():
            raise ValueError("producto_id debe ser texto no vacío")


@dataclass(frozen=True)
class RecomendacionReposicion:
    tienda_id: int
    producto: Producto
    fecha_inicio: date
    inventario: Inventario
    predicciones: tuple[Prediccion, ...]
    demanda_acumulada: float
    estrategia_id: str
    estrategia_nombre: str
    cantidad_recomendada: int
    mensaje: str

    def __post_init__(self):
        if (
            isinstance(self.tienda_id, bool)
            or not isinstance(self.tienda_id, Integral)
            or self.tienda_id <= 0
        ):
            raise ValueError("tienda_id debe ser un entero positivo")
        if self.tienda_id != self.inventario.tienda_id:
            raise ValueError("La tienda no coincide con el inventario")
        if self.producto.producto_id != self.inventario.producto_id:
            raise ValueError("El producto no coincide con el inventario")
        if not isinstance(self.fecha_inicio, date) or isinstance(
            self.fecha_inicio, datetime
        ):
            raise ValueError("fecha_inicio debe ser una fecha sin hora")
        if len(self.predicciones) != self.inventario.tiempo_reposicion:
            raise ValueError("La cobertura debe incluir todo el plazo de reposición")
        if not self.predicciones:
            raise ValueError("La cobertura debe incluir al menos un día")
        for indice, prediccion in enumerate(self.predicciones):
            if (
                prediccion.tienda_id != self.tienda_id
                or prediccion.producto.producto_id != self.producto.producto_id
                or prediccion.fecha != self.fecha_inicio + timedelta(days=indice)
            ):
                raise ValueError("Las predicciones no cubren las fechas solicitadas")
        demanda_calculada = math.fsum(
            prediccion.unidades_estimadas for prediccion in self.predicciones
        )
        if (
            isinstance(self.demanda_acumulada, bool)
            or not isinstance(self.demanda_acumulada, Real)
            or not math.isfinite(self.demanda_acumulada)
            or not math.isclose(
                self.demanda_acumulada,
                demanda_calculada,
                rel_tol=1e-12,
                abs_tol=1e-9,
            )
        ):
            raise ValueError("demanda_acumulada no coincide con las predicciones")
        if (
            isinstance(self.cantidad_recomendada, bool)
            or not isinstance(self.cantidad_recomendada, Integral)
            or self.cantidad_recomendada < 0
        ):
            raise ValueError("cantidad_recomendada debe ser un entero no negativo")
        for campo in ("estrategia_id", "estrategia_nombre", "mensaje"):
            valor = getattr(self, campo)
            if not isinstance(valor, str) or not valor.strip():
                raise ValueError(f"{campo} debe ser texto no vacío")

    @property
    def demanda_fecha_seleccionada(self):
        return self.predicciones[0].unidades_estimadas

    @property
    def fecha_fin(self):
        return self.predicciones[-1].fecha

    @property
    def requiere_reposicion(self):
        return self.cantidad_recomendada > 0
