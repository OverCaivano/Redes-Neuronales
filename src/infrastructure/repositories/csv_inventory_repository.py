"""Repositorio CSV del inventario sintético."""

from pathlib import Path

import pandas as pd

from src.datos_sinteticos import validar_inventario
from src.domain.exceptions import ErrorInventario
from src.domain.models import Inventario


class RepositorioInventarioCSV:
    def __init__(self, ruta_inventario):
        self._ruta_inventario = Path(ruta_inventario)
        self._inventario_cache = None

    def buscar(self, *, tienda_id, producto_id):
        return self._cargar_inventario().get((tienda_id, producto_id))

    def _cargar_inventario(self):
        if self._inventario_cache is None:
            try:
                inventario = pd.read_csv(
                    self._ruta_inventario,
                    dtype={"producto_id": str},
                )
                validar_inventario(inventario)
            except (OSError, ValueError, pd.errors.ParserError) as error:
                raise ErrorInventario(
                    "El inventario sintético no está disponible o es inválido"
                ) from error
            self._inventario_cache = {
                (int(fila.tienda_id), fila.producto_id): Inventario(
                    tienda_id=int(fila.tienda_id),
                    producto_id=fila.producto_id,
                    stock_actual=int(fila.stock_actual),
                    stock_minimo=int(fila.stock_minimo),
                    tiempo_reposicion=int(fila.tiempo_reposicion),
                )
                for fila in inventario.itertuples(index=False)
            }
        return self._inventario_cache
