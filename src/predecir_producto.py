"""Inferencia futura para el checkpoint sintético de ventas por producto."""

import argparse
from datetime import date, datetime
from numbers import Integral
from pathlib import Path

import pandas as pd
import torch

from src.pipeline_productos import (
    DIRECTORIO_SALIDA,
    _cargar_checkpoint,
)


def validar_consulta_producto(checkpoint, tienda_id, producto_id, fecha):
    """Valida identificadores y exige una fecha posterior al corte observado."""
    if isinstance(tienda_id, bool) or not isinstance(tienda_id, Integral):
        raise ValueError("tienda_id debe ser un entero")
    if not isinstance(producto_id, str) or not producto_id.strip():
        raise ValueError("producto_id debe ser texto no vacío")
    if not isinstance(fecha, str):
        raise ValueError("fecha debe tener el formato AAAA-MM-DD")
    try:
        dia = datetime.strptime(fecha, "%Y-%m-%d")
        if dia.strftime("%Y-%m-%d") != fecha:
            raise ValueError()
    except ValueError:
        raise ValueError("fecha debe ser válida y tener el formato AAAA-MM-DD") from None
    fecha_corte = datetime.strptime(
        checkpoint["metadata"]["fecha_maxima_observada"], "%Y-%m-%d"
    )
    if dia.date() <= fecha_corte.date():
        raise ValueError(
            "fecha debe ser posterior al último día observado "
            f"({fecha_corte.strftime('%Y-%m-%d')})"
        )

    productos = {
        producto["producto_id"]: producto
        for producto in checkpoint["catalogo"]
    }
    tiendas = checkpoint["preparador"]["categorias"]["tienda_id"]
    if int(tienda_id) not in tiendas:
        raise ValueError(f"tienda_id inexistente: {tienda_id}")
    if producto_id not in productos:
        raise ValueError(f"producto_id inexistente: {producto_id}")
    return int(tienda_id), productos[producto_id], dia.date()


class PronosticadorProducto:
    """Carga una vez el checkpoint sintético y permite consultas repetidas."""

    def __init__(self, checkpoint_path):
        self._checkpoint, self._preparador, self._modelo = _cargar_checkpoint(
            checkpoint_path
        )
        self.fecha_corte = date.fromisoformat(
            self._checkpoint["metadata"]["fecha_maxima_observada"]
        )
        self.tiendas_disponibles = tuple(
            self._checkpoint["preparador"]["categorias"]["tienda_id"]
        )

    def predecir(self, *, tienda_id, producto_id, fecha):
        """Predice una fecha futura usando solo ID, catálogo y calendario."""
        tienda, producto, dia = validar_consulta_producto(
            self._checkpoint, tienda_id, producto_id, fecha
        )
        fila = pd.DataFrame([{
            "fecha": pd.Timestamp(dia),
            "tienda_id": tienda,
            "producto_id": producto["producto_id"],
            "categoria": producto["categoria"],
            "mes": dia.month,
        }])
        caracteristicas = self._preparador.transformar(fila)
        with torch.inference_mode():
            salida = torch.expm1(
                self._modelo(torch.from_numpy(caracteristicas))
            ).clamp_min(0)
        if salida.shape != (1, 1) or not torch.isfinite(salida).all():
            raise ValueError("El modelo produjo una predicción inválida")
        return {
            "tienda_id": tienda,
            "producto_id": producto["producto_id"],
            "nombre": producto["nombre"],
            "categoria": producto["categoria"],
            "fecha": dia.isoformat(),
            "unidades_estimadas": float(salida.item()),
            "datos_sinteticos": True,
            "supuesto": (
                "No se suministran precio ni promoción; la predicción no está "
                "condicionada por esas variables."
            ),
        }


def predecir_venta(checkpoint_path, *, tienda_id, producto_id, fecha):
    """Mantiene la API original para consultas individuales."""
    return PronosticadorProducto(checkpoint_path).predecir(
        tienda_id=tienda_id,
        producto_id=producto_id,
        fecha=fecha,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Predecir ventas sintéticas futuras por producto y tienda."
    )
    parser.add_argument("--checkpoint", type=Path, default=DIRECTORIO_SALIDA / "best_model.pt")
    parser.add_argument("--store-id", type=int, required=True)
    parser.add_argument("--product-id", required=True)
    parser.add_argument("--date", required=True, help="Fecha futura AAAA-MM-DD.")
    args = parser.parse_args()
    resultado = predecir_venta(
        args.checkpoint,
        tienda_id=args.store_id,
        producto_id=args.product_id,
        fecha=args.date,
    )
    print(
        f"{resultado['fecha']} | tienda {resultado['tienda_id']} | "
        f"{resultado['producto_id']} ({resultado['nombre']}) | "
        f"unidades estimadas: {resultado['unidades_estimadas']:.2f}"
    )
    print(
        "No se suministran precio ni promocion; la prediccion no esta "
        "condicionada por esas variables."
    )


if __name__ == "__main__":
    main()
