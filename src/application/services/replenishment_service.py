"""Caso de uso que acumula predicciones y calcula una recomendación."""

import math
from datetime import timedelta
from typing import Mapping

from src.application.date_validation import parsear_fecha
from src.application.ports import (
    EstrategiaReposicion,
    PronosticadorVentas,
    RepositorioInventario,
)
from src.domain.exceptions import (
    EstrategiaReposicionNoEncontrada,
    InventarioNoEncontrado,
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

    def recomendar(self, *, tienda_id, producto_id, fecha, estrategia):
        if not isinstance(estrategia, str) or estrategia not in self._estrategias:
            raise EstrategiaReposicionNoEncontrada(
                f"Estrategia de reposición no admitida: {estrategia}"
            )
        fecha_inicio = parsear_fecha(fecha)
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
