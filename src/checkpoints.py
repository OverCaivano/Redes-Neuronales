"""Escritura atomica local de resultados y checkpoints propios."""

import os
from pathlib import Path
import tempfile

import torch


def guardar_atomico(ruta, escribir):
    """Reemplaza destino solo despues de completar la escritura temporal."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporal = tempfile.mkstemp(prefix=ruta.name + ".", suffix=".tmp", dir=ruta.parent)
    os.close(descriptor)
    temporal = Path(temporal)
    try:
        escribir(temporal)
        os.replace(temporal, ruta)
    finally:
        temporal.unlink(missing_ok=True)


def guardar_checkpoint(ruta, estado):
    guardar_atomico(ruta, lambda temporal: torch.save(estado, temporal))


def guardar_texto(ruta, texto):
    guardar_atomico(ruta, lambda temporal: temporal.write_text(texto, encoding="utf-8"))


def guardar_epoca(directorio, estado, *, mejora, metricas_csv, resumen_json, configuracion_yaml):
    """Prepara los cinco archivos antes de reemplazar; last.pt se publica ultimo.

    Ante una excepcion de escritura/reemplazo se restauran los archivos previos.
    No es una transaccion distribuida ni garantiza recuperacion ante un corte
    de energia durante varios renombrados; validar_reanudacion detecta desajustes.
    """
    import shutil

    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".epoca-", dir=directorio) as temporal:
        temporal = Path(temporal)
        nuevos = {
            "metricas.csv": metricas_csv, "resumen.json": resumen_json,
            "configuracion.yaml": configuracion_yaml,
        }
        for nombre, texto in nuevos.items():
            (temporal / nombre).write_text(texto, encoding="utf-8")
        nombres = list(nuevos)
        if mejora:
            torch.save(estado, temporal / "best.pt")
            nombres.append("best.pt")
        torch.save(estado, temporal / "last.pt")
        nombres.append("last.pt")
        for nombre in nombres:
            destino = directorio / nombre
            if destino.exists():
                shutil.copy2(destino, temporal / (nombre + ".anterior"))
        publicados = []
        try:
            for nombre in nombres:
                os.replace(temporal / nombre, directorio / nombre)
                publicados.append(nombre)
        except Exception:
            for nombre in reversed(publicados):
                respaldo = temporal / (nombre + ".anterior")
                if respaldo.exists():
                    os.replace(respaldo, directorio / nombre)
                else:
                    (directorio / nombre).unlink(missing_ok=True)
            raise
