"""Repositorio CSV del catálogo sintético de productos."""

from pathlib import Path

import pandas as pd

from src.datos_sinteticos import validar_catalogo_productos
from src.domain.exceptions import ErrorCatalogo
from src.domain.models import Producto


class RepositorioProductosCSV:
    def __init__(self, ruta_catalogo):
        self._ruta_catalogo = Path(ruta_catalogo)
        self._productos_cache = None

    def buscar_por_id(self, producto_id):
        return self._cargar_productos().get(producto_id)

    def listar(self):
        return tuple(self._cargar_productos().values())

    def _cargar_productos(self):
        if self._productos_cache is None:
            try:
                catalogo = pd.read_csv(
                    self._ruta_catalogo,
                    dtype={"producto_id": str, "nombre": str, "categoria": str},
                )
                validar_catalogo_productos(catalogo)
            except (OSError, ValueError, pd.errors.ParserError) as error:
                raise ErrorCatalogo(
                    "El catálogo sintético no está disponible o es inválido"
                ) from error
            self._productos_cache = {
                fila.producto_id: Producto(
                    producto_id=fila.producto_id,
                    nombre=fila.nombre,
                    categoria=fila.categoria,
                )
                for fila in catalogo.itertuples(index=False)
            }
        return self._productos_cache
