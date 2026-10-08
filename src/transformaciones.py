"""One-hot de tienda y familia, ajustado solo con entrenamiento.

Las matrices CSR contienen unicamente las categorias; objetivos, fechas y
variables numericas permanecen en los DataFrames originales, sin modificarlos.
Las filas conservan el orden de entrada. No se escala ni se entrena el MLP.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

from src.preprocesamiento_temporal import PERIODOS


COLUMNAS_CATEGORICAS = ["tienda", "familia"]


def _seleccionar_categorias(datos):
    faltantes = set(COLUMNAS_CATEGORICAS) - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas categoricas: {sorted(faltantes)}")
    categorias = datos.loc[:, COLUMNAS_CATEGORICAS]
    if categorias.isna().any().any():
        raise ValueError("tienda y familia no pueden contener valores faltantes")
    if not pd.api.types.is_numeric_dtype(categorias["tienda"]):
        raise ValueError("tienda debe contener identificadores numericos")
    if not np.isfinite(categorias["tienda"].to_numpy()).all():
        raise ValueError("tienda contiene identificadores no finitos")
    if not categorias["familia"].map(lambda valor: isinstance(valor, str)).all():
        raise ValueError("familia debe conservar sus nombres originales como texto")
    return categorias


class CodificadorCategorias:
    """Ajuste unico al construir; transformar nunca vuelve a ajustar.

    Requiere fecha datetime dentro del periodo de entrenamiento fijado.
    Orden: tiendas numericamente ascendentes, luego familias alfabeticamente
    ordenadas, conservando exactamente sus nombres (espacios y mayusculas).
    No elimina categorias de referencia: cada fila tiene dos unos.
    """

    def __init__(self, entrenamiento):
        if entrenamiento.empty:
            raise ValueError("No se puede ajustar con entrenamiento vacio")
        if "fecha" not in entrenamiento or not pd.api.types.is_datetime64_any_dtype(
            entrenamiento["fecha"]
        ):
            raise ValueError("El ajuste requiere fecha de tipo datetime")
        inicio, fin = PERIODOS["entrenamiento"]
        if not entrenamiento["fecha"].between(inicio, fin).all():
            raise ValueError("El ajuste admite exclusivamente fechas de entrenamiento")
        categorias = _seleccionar_categorias(entrenamiento)
        self._codificador = OneHotEncoder(
            handle_unknown="error", sparse_output=True, dtype=np.float32, drop=None
        )
        self._codificador.fit(categorias)

    @property
    def nombres_columnas(self):
        """Tupla inmutable con el orden de las columnas de salida."""
        return tuple(self._codificador.get_feature_names_out(COLUMNAS_CATEGORICAS))

    def transformar(self, datos):
        """Devuelve CSR, sin ajuste ni conversion a matriz densa.

        Las categorias desconocidas provocan un error que identifica la
        columna y sus valores. Una particion vacia conserva el ancho aprendido.
        """
        categorias = _seleccionar_categorias(datos)
        for columna, conocidas in zip(
            COLUMNAS_CATEGORICAS, self._codificador.categories_
        ):
            desconocidas = categorias.loc[
                ~categorias[columna].isin(conocidas), columna
            ].unique().tolist()
            if desconocidas:
                raise ValueError(
                    f"Categorias desconocidas en {columna}: {desconocidas}. "
                    "No estaban presentes en entrenamiento."
                )
        if categorias.empty:
            return csr_matrix((0, len(self.nombres_columnas)), dtype=np.float32)
        return self._codificador.transform(categorias)


def codificar_particiones(particiones):
    """Recibe dividir_por_fecha(...); devuelve codificador y matrices CSR.

    Solo particiones['entrenamiento'] participa del ajuste. Validacion y test
    se transforman con esa misma instancia. No concatena ni densifica datos.
    """
    nombres = ("entrenamiento", "validacion", "test")
    faltantes = set(nombres) - set(particiones)
    if faltantes:
        raise ValueError(f"Faltan particiones: {sorted(faltantes)}")
    codificador = CodificadorCategorias(particiones["entrenamiento"])
    matrices = {
        nombre: codificador.transformar(particiones[nombre]) for nombre in nombres
    }
    return codificador, matrices
