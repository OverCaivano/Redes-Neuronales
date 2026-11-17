"""Adaptador que oculta PyTorch y el formato del checkpoint sintético."""

from datetime import date
from pathlib import Path
from threading import Lock

from src.domain.exceptions import ErrorInferencia
from src.predecir_producto import PronosticadorProducto


class AdaptadorPrediccionPyTorch:
    def __init__(self, ruta_checkpoint):
        self._ruta_checkpoint = Path(ruta_checkpoint)
        self._pronosticador = None
        self._lock = Lock()

    def _obtener_pronosticador(self):
        if self._pronosticador is None:
            with self._lock:
                if self._pronosticador is None:
                    try:
                        self._pronosticador = PronosticadorProducto(
                            self._ruta_checkpoint
                        )
                    except (
                        KeyError,
                        OSError,
                        RuntimeError,
                        TypeError,
                        ValueError,
                    ) as error:
                        raise ErrorInferencia(
                            "El checkpoint sintético no está disponible o es inválido"
                        ) from error
        return self._pronosticador

    @property
    def fecha_corte(self) -> date:
        return self._obtener_pronosticador().fecha_corte

    @property
    def tiendas_disponibles(self) -> tuple[int, ...]:
        return self._obtener_pronosticador().tiendas_disponibles

    def predecir(self, *, tienda_id: int, producto_id: str, fecha: date) -> float:
        try:
            resultado = self._obtener_pronosticador().predecir(
                tienda_id=tienda_id,
                producto_id=producto_id,
                fecha=fecha.isoformat(),
            )
        except ErrorInferencia:
            raise
        except (KeyError, RuntimeError, ValueError) as error:
            raise ErrorInferencia("No se pudo completar la inferencia") from error
        return resultado["unidades_estimadas"]
