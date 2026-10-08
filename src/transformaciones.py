"""One-hot de tienda y familia, ajustado solo con entrenamiento.

CodificadorCategorias devuelve solo categorias; PreparadorEntradas devuelve
entradas completas. Objetivos y metadatos quedan en los DataFrames originales,
sin modificarlos.
Las filas conservan el orden de entrada. PreparadorEntradas agrega escalado
numerico y ciclos sin escalar; no se entrena el MLP.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack, vstack
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocesamiento_temporal import PERIODOS, VARIABLES_CICLICAS


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


COLUMNAS_NUMERICAS = [
    "ventas_hace_1_dia",
    "ventas_hace_7_dias",
    "promedio_ventas_7_dias",
    "cantidad_en_promocion",
]


def _validar_entradas_numericas(datos):
    columnas = COLUMNAS_NUMERICAS + VARIABLES_CICLICAS
    faltantes = set(columnas) - set(datos.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas de entrada: {sorted(faltantes)}")
    for columna in columnas:
        if not pd.api.types.is_numeric_dtype(datos[columna]):
            raise ValueError(f"{columna} debe ser numerica")
        if datos[columna].isna().any() or not np.isfinite(
            datos[columna].to_numpy()
        ).all():
            raise ValueError(f"{columna} contiene valores no finitos")


class PreparadorEntradas:
    """Ajusta categorias y StandardScaler exclusivamente con entrenamiento.

    Orden: COLUMNAS_NUMERICAS escaladas, VARIABLES_CICLICAS sin escalar,
    y one-hot en el orden de CodificadorCategorias. Objetivo y metadatos
    permanecen fuera de la matriz. La fila i corresponde a datos.iloc[i],
    incluso con indices repetidos o desordenados. No modifica datos.
    """

    def __init__(self, entrenamiento):
        _validar_entradas_numericas(entrenamiento)
        self.codificador = CodificadorCategorias(entrenamiento)
        self._escalador = StandardScaler()
        self._escalador.fit(entrenamiento[COLUMNAS_NUMERICAS])
        if not all(np.isfinite(parametro).all() for parametro in (
            self._escalador.mean_, self._escalador.var_, self._escalador.scale_
        )):
            raise ValueError("El ajuste del escalador produjo parametros no finitos")

    @property
    def nombres_columnas(self):
        return tuple(COLUMNAS_NUMERICAS + VARIABLES_CICLICAS) + self.codificador.nombres_columnas

    @property
    def parametros_escalado(self):
        """Copias para inspeccion: modificarlas no altera el escalador."""
        return {
            "media": self._escalador.mean_.copy(),
            "varianza": self._escalador.var_.copy(),
            "escala": self._escalador.scale_.copy(),
        }

    def transformar(self, datos, *, tamano_bloque=100_000):
        """CSR float32 por bloques; nunca densifica las columnas one-hot.

        Solo las cuatro numericas pasan por StandardScaler.transform.
        Se comprueba finitud antes y despues de convertir a float32.
        """
        if tamano_bloque <= 0:
            raise ValueError("tamano_bloque debe ser positivo")
        _validar_entradas_numericas(datos)
        _seleccionar_categorias(datos)
        bloques = []
        for inicio in range(0, len(datos), tamano_bloque):
            bloque = datos.iloc[inicio:inicio + tamano_bloque]
            numericas = self._escalador.transform(bloque[COLUMNAS_NUMERICAS])
            ciclicas = bloque[VARIABLES_CICLICAS].to_numpy(dtype=np.float64)
            valores = np.column_stack([numericas, ciclicas])
            if not np.isfinite(valores).all() or (
                np.abs(valores) > np.finfo(np.float32).max
            ).any():
                raise ValueError("Las entradas no son finitas o exceden float32")
            matriz = hstack(
                [csr_matrix(valores.astype(np.float32)),
                 self.codificador.transformar(bloque)],
                format="csr", dtype=np.float32,
            )
            if matriz.shape != (len(bloque), len(self.nombres_columnas)):
                raise ValueError("Dimensiones de entradas desalineadas")
            bloques.append(matriz)
        if not bloques:
            return csr_matrix((0, len(self.nombres_columnas)), dtype=np.float32)
        resultado = vstack(bloques, format="csr", dtype=np.float32)
        if resultado.shape[0] != len(datos) or not np.isfinite(resultado.data).all():
            raise ValueError("Entradas no finitas o filas desalineadas")
        return resultado


def preparar_particiones(particiones):
    """Devuelve preparador y entradas CSR, con un unico ajuste en entrenamiento.

    Reutiliza el mismo escalador y codificador para validacion y test.
    Los objetivos y metadatos siguen en las particiones, en el mismo orden.
    """
    nombres = ("entrenamiento", "validacion", "test")
    faltantes = set(nombres) - set(particiones)
    if faltantes:
        raise ValueError(f"Faltan particiones: {sorted(faltantes)}")
    preparador = PreparadorEntradas(particiones["entrenamiento"])
    matrices = {
        nombre: preparador.transformar(particiones[nombre]) for nombre in nombres
    }
    return preparador, matrices
