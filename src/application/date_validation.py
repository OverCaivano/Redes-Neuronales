"""Validación compartida de fechas para los casos de uso de predicción."""

from datetime import date, datetime, timedelta

from src.domain.exceptions import SolicitudPrediccionInvalida


HORIZONTE_PREDICCION_DIAS = 90


def calcular_fecha_maxima_prediccion(fecha_corte: date) -> date:
    return fecha_corte + timedelta(days=HORIZONTE_PREDICCION_DIAS)


def calcular_fecha_maxima_reposicion(
    fecha_corte: date, tiempo_reposicion: int
) -> date:
    return calcular_fecha_maxima_prediccion(fecha_corte) - timedelta(
        days=tiempo_reposicion - 1
    )


def parsear_fecha(fecha):
    if isinstance(fecha, datetime):
        raise SolicitudPrediccionInvalida(
            "fecha debe ser válida y tener el formato AAAA-MM-DD"
        )
    if isinstance(fecha, date):
        return fecha
    if not isinstance(fecha, str):
        raise SolicitudPrediccionInvalida(
            "fecha debe tener el formato AAAA-MM-DD"
        )
    try:
        dia = datetime.strptime(fecha, "%Y-%m-%d")
    except ValueError:
        raise SolicitudPrediccionInvalida(
            "fecha debe ser válida y tener el formato AAAA-MM-DD"
        ) from None
    if dia.strftime("%Y-%m-%d") != fecha:
        raise SolicitudPrediccionInvalida(
            "fecha debe ser válida y tener el formato AAAA-MM-DD"
        )
    return dia.date()
