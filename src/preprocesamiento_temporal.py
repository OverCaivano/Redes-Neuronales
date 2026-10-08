"""Historia diaria causal para predicciones emitidas al cierre de t-1.

Primero generar_variables_historicas; despues dividir_por_fecha.
No se ajustan escaladores, codificadores ni modelos. Los datos recibidos y
los archivos originales permanecen intactos.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.datos_favorita import ORDEN, validar_datos


VARIABLES_HISTORICAS = [
    "ventas_hace_1_dia",
    "ventas_hace_7_dias",
    "promedio_ventas_7_dias",
]
VARIABLES_CICLICAS = [
    "dia_semana_seno",
    "dia_semana_coseno",
    "mes_seno",
    "mes_coseno",
]
PERIODOS = {
    "entrenamiento": ("2013-01-01", "2017-06-15"),
    "validacion": ("2017-06-16", "2017-07-15"),
    "test": ("2017-07-16", "2017-08-15"),
}


@dataclass(frozen=True)
class ResultadoTemporal:
    """Conteos referidos a filas originales, no a fechas insertadas."""
    datos: pd.DataFrame
    filas_entrada: int
    filas_descartadas_historia: int

    @property
    def filas_validas(self):
        return len(self.datos)


def generar_variables_historicas(datos):
    """Conserva solo objetivos con siete dias anteriores observados.

    Cada serie se reindexa diariamente en memoria. Los huecos quedan como NaN;
    rolling con min_periods=7 exige las siete observaciones. Las fechas creadas
    solo sirven como contexto: nunca se convierten en objetivos artificiales.

    Agrega ciclos desde la fecha objetivo: lunes=0 hasta domingo=6,
    enero=0 hasta diciembre=11. Usa seno/coseno de 2*pi*indice/periodo,
    con periodos 7 y 12 respectivamente, en [-1, 1]. No dependen de ventas
    ni requieren ajuste; se conservan tambien en resultados vacios.
    """
    validar_datos(datos)
    ordenados = datos.sort_values(ORDEN)
    partes = []
    for _, serie in ordenados.groupby(["tienda", "familia"], sort=False, observed=True):
        calendario = pd.date_range(serie["fecha"].min(), serie["fecha"].max(), freq="D")
        ventas = serie.set_index("fecha")["ventas_objetivo"].reindex(calendario)
        anteriores = ventas.shift(1)
        historia = pd.DataFrame({
            "ventas_hace_1_dia": anteriores,
            "ventas_hace_7_dias": ventas.shift(7),
            "promedio_ventas_7_dias": anteriores.rolling(7, min_periods=7).mean(),
        })
        historia = historia.reindex(serie["fecha"])
        validas = historia.notna().all(axis=1).to_numpy()
        parte = serie.loc[validas].copy()
        for columna in VARIABLES_HISTORICAS:
            parte[columna] = historia.loc[validas, columna].to_numpy()
        partes.append(parte)

    if partes:
        resultado = pd.concat(partes, ignore_index=True)
    else:
        resultado = ordenados.iloc[:0].copy()
        for columna in VARIABLES_HISTORICAS:
            resultado[columna] = pd.Series(dtype="float64")

    angulo_semana = 2 * np.pi * resultado["fecha"].dt.dayofweek / 7
    angulo_mes = 2 * np.pi * (resultado["fecha"].dt.month - 1) / 12
    resultado["dia_semana_seno"] = np.sin(angulo_semana)
    resultado["dia_semana_coseno"] = np.cos(angulo_semana)
    resultado["mes_seno"] = np.sin(angulo_mes)
    resultado["mes_coseno"] = np.cos(angulo_mes)

    return ResultadoTemporal(
        datos=resultado,
        filas_entrada=len(datos),
        filas_descartadas_historia=len(datos) - len(resultado),
    )


def dividir_por_fecha(resultado):
    """Divide el resultado ya preparado por fecha objetivo, inclusive.

    La historia previa a cada corte se conserva porque fue calculada antes.
    Devuelve copias independientes. Rechaza fechas fuera del periodo fijado
    para no excluir observaciones silenciosamente.
    """
    if not isinstance(resultado, ResultadoTemporal):
        raise TypeError("Primero ejecute generar_variables_historicas")
    datos = resultado.datos
    validar_datos(datos)
    if not set(VARIABLES_HISTORICAS).issubset(datos.columns):
        raise ValueError("Faltan variables historicas")
    if datos[VARIABLES_HISTORICAS].isna().any().any():
        raise ValueError("Hay ejemplos sin historia suficiente")
    if not datos["fecha"].between("2013-01-01", "2017-08-15").all():
        raise ValueError("Hay fechas objetivo fuera del periodo configurado")
    return {
        nombre: datos.loc[datos["fecha"].between(inicio, fin)].copy().reset_index(drop=True)
        for nombre, (inicio, fin) in PERIODOS.items()
    }
