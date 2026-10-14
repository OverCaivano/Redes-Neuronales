"""Metricas de regresion acumuladas por observacion, en unidades originales."""

import math

import torch


def validar_predicciones(predicciones, objetivos):
    if (
        predicciones.ndim != 2 or objetivos.ndim != 2
        or predicciones.shape != objetivos.shape or predicciones.shape[1] != 1
        or predicciones.shape[0] == 0
    ):
        raise ValueError("Prediccion y objetivo deben tener la misma forma [B, 1]")
    if not torch.isfinite(predicciones).all() or not torch.isfinite(objetivos).all():
        raise ValueError("Predicciones u objetivos no finitos")


class MetricasRegresion:
    def __init__(self):
        self.cantidad = 0
        self.error_absoluto = 0.0
        self.error_cuadratico = 0.0

    def actualizar(self, predicciones, objetivos):
        validar_predicciones(predicciones, objetivos)
        error = predicciones.detach().double() - objetivos.detach().double()
        self.cantidad += len(error)
        self.error_absoluto += error.abs().sum().item()
        self.error_cuadratico += error.square().sum().item()
        if not math.isfinite(self.error_absoluto) or not math.isfinite(self.error_cuadratico):
            raise ValueError("Metricas acumuladas no finitas")

    def resultado(self):
        if self.cantidad == 0:
            raise ValueError("No se pueden calcular metricas sin observaciones")
        mse = self.error_cuadratico / self.cantidad
        return {
            "mse": mse, "mae": self.error_absoluto / self.cantidad,
            "rmse": math.sqrt(mse), "n": self.cantidad,
        }
