"""Consultas historicas con un modelo fijo, sin usar el objetivo ni futuro."""

import argparse
from datetime import datetime
from numbers import Integral
from pathlib import Path
import sys

import pandas as pd
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datos_favorita import leer_datos_favorita
from src.evaluacion_favorita import cargar_best_favorita, reconstruir_modelo_favorita
from src.preprocesamiento_temporal import generar_variables_historicas
from src.reanudacion import huella_dataset


def validar_consulta(tienda, familia, fecha):
    if isinstance(tienda, bool) or not isinstance(tienda, Integral) or tienda <= 0:
        raise ValueError("tienda debe ser un entero positivo")
    if not isinstance(familia, str) or not familia.strip():
        raise ValueError("familia debe ser su nombre original no vacio")
    try:
        dia = datetime.strptime(fecha, "%Y-%m-%d")
        if dia.strftime("%Y-%m-%d") != fecha:
            raise ValueError()
    except (TypeError, ValueError):
        raise ValueError("fecha debe ser una fecha valida en formato AAAA-MM-DD") from None
    return pd.Timestamp(dia)


def preparar_consulta(datos, *, tienda, familia, fecha):
    """Reutiliza el generador causal con una ventana de ocho fechas como maximo.

    El cero temporal del objetivo solo satisface el contrato del generador:
    nunca es una venta historica, una imputacion de huecos ni una entrada.
    Se elimina junto a id antes de transformar. No modifica datos recibidos.
    """
    dia = validar_consulta(tienda, familia, fecha)
    contexto = datos.loc[
        datos["tienda"].eq(tienda) & datos["familia"].eq(familia)
        & datos["fecha"].between(dia - pd.Timedelta(days=7), dia)
    ].copy()
    objetivo = contexto["fecha"].eq(dia)
    if objetivo.sum() != 1:
        raise ValueError("Debe existir una unica fila para tienda, familia y fecha solicitadas")
    # Borrar el objetivo ANTES del generador: ni ventas(t) ni ventas futuras llegan a las features.
    contexto.loc[objetivo, "ventas_objetivo"] = 0.0
    temporal = generar_variables_historicas(contexto).datos
    fila = temporal.loc[temporal["fecha"].eq(dia)].copy()
    if len(fila) != 1:
        raise ValueError("Historia insuficiente: se requieren siete dias calendario anteriores observados")
    return fila.drop(columns=["ventas_objetivo", "id"]).reset_index(drop=True)


def predecir_favorita(checkpoint, *, tienda, familia, fecha, data_path=None):
    """Consulta en CPU; solo admite el CSV original identificado por SHA-256."""
    validar_consulta(tienda, familia, fecha)
    estado = cargar_best_favorita(checkpoint)
    config = estado["config"]["data"]
    for clave, valor in (("tiendas", tienda), ("familias", familia)):
        if config.get(clave) is not None and valor not in config[clave]:
            raise ValueError(f"{clave}: {valor!r} no pertenece al subconjunto del checkpoint")
    ruta = Path(data_path or config["path"])
    if huella_dataset(ruta) != estado["dataset_sha256"]:
        raise ValueError("Dataset incompatible: SHA-256 diferente; usar el train.csv original")
    datos = leer_datos_favorita(ruta, tiendas=[tienda], familias=[familia])
    fila = preparar_consulta(datos, tienda=tienda, familia=familia, fecha=fecha)
    preparador = estado["preparador"]
    matriz = preparador.transformar(fila)  # Sin fit; rechaza categorias desconocidas.
    if (matriz.shape != (1, estado["input_dim"])
        or preparador.nombres_columnas != tuple(estado["nombres_columnas"])):
        raise ValueError("Dimension u orden de columnas incompatible con el checkpoint")
    modelo = reconstruir_modelo_favorita(estado)
    with torch.inference_mode():
        # Solo se densifica UNA fila, nunca el dataset.
        salida = modelo(torch.from_numpy(matriz.toarray()))
        if salida.shape != (1, 1) or not torch.isfinite(salida).all():
            raise ValueError("La prediccion debe ser finita y tener forma [1, 1]")
    resultado = {
        "tienda": int(tienda), "familia": familia, "fecha": fecha,
        "ventas_estimadas": salida.item(), "checkpoint": str(Path(checkpoint).resolve()),
        "epoca": estado["epoch"], "input_dim": estado["input_dim"],
        "cantidad_en_promocion": float(fila.iloc[0]["cantidad_en_promocion"]),
        "tipo_consulta": "retrospectiva_modelo_fijo", "device": "cpu",
    }
    print(f"Tienda {tienda} | familia {familia} | fecha {fecha}")
    print(f"Ventas estimadas por dia y familia: {resultado['ventas_estimadas']:.6f}")
    print(f"Checkpoint: {resultado['checkpoint']} (epoca {estado['epoch']})")
    print("Consulta retrospectiva; promociones del dia consideradas conocidas anticipadamente.")
    print("No representa stock disponible ni una recomendacion de compra.")
    if resultado["ventas_estimadas"] < 0:
        print("Salida negativa del MLP: se informa sin recorte; limitacion del modelo de regresion.")
    return resultado


def main():
    parser = argparse.ArgumentParser(description="Consultar ventas diarias Favorita sin entrenar")
    parser.add_argument("--checkpoint", required=True, help="best.pt propio y confiable")
    parser.add_argument("--tienda", type=int, required=True)
    parser.add_argument("--familia", required=True, help="Nombre original; entre comillas si tiene espacios")
    parser.add_argument("--fecha", required=True, help="AAAA-MM-DD, con fila y siete dias previos disponibles")
    parser.add_argument("--data-path", help="Mismo train.csv en otra ubicacion")
    args = parser.parse_args()
    try:
        predecir_favorita(args.checkpoint, tienda=args.tienda, familia=args.familia,
                         fecha=args.fecha, data_path=args.data_path)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
