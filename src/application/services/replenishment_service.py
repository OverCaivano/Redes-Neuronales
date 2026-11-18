"""Caso de uso que acumula predicciones y calcula una recomendación."""

import math
from datetime import timedelta
from numbers import Integral
from typing import Mapping

from src.application.date_validation import calcular_fecha_maxima_reposicion
from src.application.ports import (
    EstrategiaReposicion,
    PronosticadorVentas,
    RepositorioInventario,
)
from src.domain.exceptions import (
    EstrategiaReposicionNoEncontrada,
    InventarioNoEncontrado,
    SolicitudPrediccionInvalida,
    TiendaNoEncontrada,
)
from src.domain.models import RecomendacionReposicion


class ServicioReposicion:
    def __init__(
        self,
        repositorio_inventario: RepositorioInventario,
        pronosticador: PronosticadorVentas,
        estrategias: Mapping[str, EstrategiaReposicion],
    ):
        if not estrategias or any(
            codigo != estrategia.codigo
            for codigo, estrategia in estrategias.items()
        ):
            raise ValueError("Cada estrategia debe estar registrada con su código")
        self._repositorio_inventario = repositorio_inventario
        self._pronosticador = pronosticador
        self._estrategias = dict(estrategias)

    def listar_estrategias(self):
        return tuple(
            {"codigo": codigo, "nombre": estrategia.nombre}
            for codigo, estrategia in self._estrategias.items()
        )

    def obtener_horizonte_reposicion(
        self, *, tienda_id: int, producto_id: str
    ) -> dict[str, int | str]:
        if isinstance(tienda_id, bool) or not isinstance(tienda_id, Integral):
            raise SolicitudPrediccionInvalida("tienda_id debe ser un entero")
        tienda = int(tienda_id)
        if tienda not in self._pronosticador.tiendas_disponibles:
            raise TiendaNoEncontrada(f"tienda_id inexistente: {tienda}")
        if not isinstance(producto_id, str) or not producto_id.strip():
            raise SolicitudPrediccionInvalida(
                "producto_id debe ser texto no vacío"
            )
        inventario = self._repositorio_inventario.buscar(
            tienda_id=tienda,
            producto_id=producto_id,
        )
        if inventario is None:
            raise InventarioNoEncontrado(
                "No existe inventario para la tienda y producto seleccionados"
            )
        return {
            "tiempo_reposicion": inventario.tiempo_reposicion,
            "fecha_maxima_reposicion": calcular_fecha_maxima_reposicion(
                self._pronosticador.fecha_corte,
                inventario.tiempo_reposicion,
            ).isoformat(),
        }

    def recomendar(self, *, tienda_id, producto_id, fecha, estrategia):
        if not isinstance(estrategia, str) or estrategia not in self._estrategias:
            raise EstrategiaReposicionNoEncontrada(
                f"Estrategia de reposición no admitida: {estrategia}"
            )
        fecha_inicio = self._pronosticador.validar_fecha(fecha)
        pronostico_inicial = self._pronosticador.predecir(
            tienda_id=tienda_id,
            producto_id=producto_id,
            fecha=fecha_inicio,
        )
        inventario = self._repositorio_inventario.buscar(
            tienda_id=pronostico_inicial.tienda_id,
            producto_id=pronostico_inicial.producto.producto_id,
        )
        if inventario is None:
            raise InventarioNoEncontrado(
                "No existe inventario para la tienda y producto seleccionados"
            )
        fecha_maxima_reposicion = calcular_fecha_maxima_reposicion(
            self._pronosticador.fecha_corte,
            inventario.tiempo_reposicion,
        )
        if fecha_inicio > fecha_maxima_reposicion:
            raise SolicitudPrediccionInvalida(
                "La fecha seleccionada excede el horizonte de reposición para "
                f"un plazo de {inventario.tiempo_reposicion} días; última fecha "
                f"válida para reponer: {fecha_maxima_reposicion.isoformat()}."
            )

        predicciones = [pronostico_inicial]
        for dia in range(1, inventario.tiempo_reposicion):
            predicciones.append(
                self._pronosticador.predecir(
                    tienda_id=pronostico_inicial.tienda_id,
                    producto_id=pronostico_inicial.producto.producto_id,
                    fecha=fecha_inicio + timedelta(days=dia),
                )
            )
        demanda_acumulada = math.fsum(
            prediccion.unidades_estimadas for prediccion in predicciones
        )
        estrategia_seleccionada = self._estrategias[estrategia]
        cantidad = estrategia_seleccionada.calcular(demanda_acumulada, inventario)
        if cantidad == 0:
            mensaje = (
                "El stock disponible cubre la demanda prevista según la "
                "estrategia seleccionada; no es necesario reponer."
            )
        else:
            mensaje = (
                f"Se recomienda reponer {cantidad} unidades. La recomendación "
                "no modifica el inventario ni realiza una compra."
            )
        return RecomendacionReposicion(
            tienda_id=pronostico_inicial.tienda_id,
            producto=pronostico_inicial.producto,
            fecha_inicio=fecha_inicio,
            inventario=inventario,
            predicciones=tuple(predicciones),
            demanda_acumulada=demanda_acumulada,
            estrategia_id=estrategia_seleccionada.codigo,
            estrategia_nombre=estrategia_seleccionada.nombre,
            cantidad_recomendada=cantidad,
            mensaje=mensaje,
        )
