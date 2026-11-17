"""Validación compartida de fechas para los casos de uso de predicción."""

from datetime import date, datetime

from src.domain.exceptions import SolicitudPrediccionInvalida


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
