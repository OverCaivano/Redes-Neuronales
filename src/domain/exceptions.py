"""Excepciones de dominio y de aplicación para predicciones."""


class SolicitudPrediccionInvalida(ValueError):
    """La solicitud no cumple las reglas de predicción."""


class ProductoNoEncontrado(LookupError):
    """El catálogo no contiene el producto solicitado."""


class TiendaNoEncontrada(LookupError):
    """El modelo no tiene datos para la tienda solicitada."""


class ErrorInferencia(RuntimeError):
    """El adaptador no pudo completar la inferencia."""
