"""Adaptador que oculta PyTorch y el formato del checkpoint sintético."""

from datetime import date
from pathlib import Path

from src.domain.exceptions import ErrorInferencia
from src.predecir_producto import PronosticadorProducto


class AdaptadorPrediccionPyTorch:
    def __init__(self, ruta_checkpoint):
        self._pronosticador = PronosticadorProducto(Path(ruta_checkpoint))

    @property
    def fecha_corte(self) -> date:
        return self._pronosticador.fecha_corte

    @property
    def tiendas_disponibles(self) -> tuple[int, ...]:
        return self._pronosticador.tiendas_disponibles

    def predecir(self, *, tienda_id: int, producto_id: str, fecha: date) -> float:
        try:
            resultado = self._pronosticador.predecir(
                tienda_id=tienda_id,
                producto_id=producto_id,
                fecha=fecha.isoformat(),
            )
        except (KeyError, RuntimeError, ValueError) as error:
            raise ErrorInferencia("No se pudo completar la inferencia") from error
        return resultado["unidades_estimadas"]
