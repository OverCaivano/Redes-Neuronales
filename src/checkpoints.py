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
