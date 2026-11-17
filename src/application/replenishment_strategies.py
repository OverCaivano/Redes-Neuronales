"""Estrategias intercambiables para calcular cantidades de reposición."""

import math
from numbers import Real

from src.domain.models import Inventario


def _validar_demanda(demanda_acumulada):
    if (
        isinstance(demanda_acumulada, bool)
        or not isinstance(demanda_acumulada, Real)
        or not math.isfinite(demanda_acumulada)
        or demanda_acumulada < 0
    ):
        raise ValueError("La demanda acumulada debe ser finita y no negativa")


class ReposicionBasica:
    codigo = "basica"
    nombre = "Reposición básica"

    def calcular(self, demanda_acumulada: float, inventario: Inventario) -> int:
        _validar_demanda(demanda_acumulada)
        return math.ceil(max(0.0, demanda_acumulada - inventario.stock_actual))


class ReposicionConSeguridad:
    codigo = "seguridad"
    nombre = "Reposición con stock de seguridad"

    def calcular(self, demanda_acumulada: float, inventario: Inventario) -> int:
        _validar_demanda(demanda_acumulada)
        objetivo = demanda_acumulada + inventario.stock_minimo
        return math.ceil(max(0.0, objetivo - inventario.stock_actual))
