"""Excepciones de dominio y de aplicación para predicciones."""


class SolicitudPrediccionInvalida(ValueError):
    """La solicitud no cumple las reglas de predicción."""


class ProductoNoEncontrado(LookupError):
    """El catálogo no contiene el producto solicitado."""


class ErrorCatalogo(RuntimeError):
    """El catálogo sintético no está disponible o no es válido."""


class InventarioNoEncontrado(LookupError):
    """No existe una fila de inventario para la tienda y producto."""


class ErrorInventario(RuntimeError):
    """El inventario sintético no está disponible o no es válido."""


class EstrategiaReposicionNoEncontrada(ValueError):
    """La estrategia de reposición no está registrada."""


class TiendaNoEncontrada(LookupError):
    """El modelo no tiene datos para la tienda solicitada."""


class ErrorInferencia(RuntimeError):
    """El adaptador no pudo completar la inferencia."""
