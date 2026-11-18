"""Contratos que aíslan los casos de uso de sus detalles técnicos."""

from datetime import date
from typing import Protocol

from src.domain.models import Inventario, Prediccion, Producto


class RepositorioProductos(Protocol):
    def buscar_por_id(self, producto_id: str) -> Producto | None:
        ...

    def listar(self) -> tuple[Producto, ...]:
        ...


class MotorPrediccion(Protocol):
    @property
    def fecha_corte(self) -> date:
        ...

    @property
    def tiendas_disponibles(self) -> tuple[int, ...]:
        ...

    def predecir(
        self, *, tienda_id: int, producto_id: str, fecha: date
    ) -> float:
        ...


class RepositorioInventario(Protocol):
    def buscar(self, *, tienda_id: int, producto_id: str) -> Inventario | None:
        ...


class PronosticadorVentas(Protocol):
    @property
    def fecha_corte(self) -> date:
        ...

    @property
    def tiendas_disponibles(self) -> tuple[int, ...]:
        ...

    def predecir(
        self, *, tienda_id: int, producto_id: str, fecha: date | str
    ) -> Prediccion:
        ...

    def validar_fecha(self, fecha: date | str) -> date:
        ...


class EstrategiaReposicion(Protocol):
    codigo: str
    nombre: str

    def calcular(
        self, demanda_acumulada: float, inventario: Inventario
    ) -> int:
        ...
