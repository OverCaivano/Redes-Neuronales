"""Adaptador CSR a PyTorch sin densificar las entradas completas.

No lee CSV ni ajusta transformaciones. La fila i de entradas debe corresponder
al objetivo i; los indices/etiquetas de pandas no se usan para alinear datos.
"""

from functools import partial
from numbers import Integral

import numpy as np
from scipy.sparse import isspmatrix_csr
import torch
from torch.utils.data import DataLoader, Dataset


class DatasetFavorita(Dataset):
    """Conserva X como CSR float32 y objetivos como arreglo float32 [N, 1].

    __getitem__ devuelve posiciones para collate_lote. La CSR se conserva por
    referencia para evitar una copia completa: el llamador no debe modificarla
    mientras se utiliza el Dataset. Los objetivos se copian sin sus etiquetas.
    """

    def __init__(self, entradas, objetivos):
        if not isspmatrix_csr(entradas):
            raise TypeError("entradas debe ser una matriz CSR de scipy")
        if entradas.ndim != 2 or min(entradas.shape) == 0:
            raise ValueError("entradas debe tener filas y columnas no vacias")
        if entradas.dtype != np.dtype("float32"):
            raise TypeError("entradas CSR debe tener dtype float32")
        if not np.isfinite(entradas.data).all():
            raise ValueError("entradas contiene valores no finitos")

        valores = np.asarray(objetivos)
        if valores.ndim == 1:
            valores = valores.reshape(-1, 1)
        if valores.ndim != 2 or valores.shape[1] != 1:
            raise ValueError("objetivos debe tener forma [N] o [N, 1]")
        if valores.shape[0] != entradas.shape[0]:
            raise ValueError("entradas y objetivos deben tener la misma cantidad de filas")
        if valores.dtype.kind not in "iuf":
            raise TypeError("objetivos debe contener valores numericos reales")
        if not np.isfinite(valores).all():
            raise ValueError("objetivos contiene valores no finitos")
        limite = np.finfo(np.float32).max
        if (valores > limite).any() or (valores < -limite).any():
            raise ValueError("objetivos excede el rango de float32")

        self.entradas = entradas
        self.objetivos = np.array(valores, dtype=np.float32, order="C", copy=True)
        self.cantidad_ejemplos, self.input_dim = entradas.shape

    def __len__(self):
        return self.cantidad_ejemplos

    def __getitem__(self, indice):
        if isinstance(indice, (bool, np.bool_)) or not isinstance(indice, Integral):
            raise TypeError("El indice debe ser un entero posicional")
        if not 0 <= indice < len(self):
            raise IndexError("Indice fuera del Dataset")
        return int(indice)


def collate_lote(indices, *, dataset):
    """Densifica solo la seleccion CSR del lote; devuelve X [B,D], y [B,1].

    Puede pasarse a DataLoader mediante partial(collate_lote, dataset=dataset).
    No utiliza las etiquetas originales de pandas ni reordena la seleccion.
    """
    if len(indices) == 0:
        raise ValueError("El lote no puede estar vacio")
    posiciones = np.asarray([dataset[indice] for indice in indices], dtype=np.int64)
    seleccion = dataset.entradas[posiciones]
    entradas = seleccion.toarray()
    objetivos = dataset.objetivos[posiciones]
    if not np.isfinite(entradas).all() or not np.isfinite(objetivos).all():
        raise ValueError("El lote contiene valores no finitos")
    return torch.from_numpy(entradas), torch.from_numpy(objetivos)


def crear_dataloader(
    dataset, *, batch_size=256, shuffle=False, semilla=42, num_workers=0
):
    """Construye lotes sin perder la ultima fraccion y con mezcla reproducible.

    Dos loaders nuevos con la misma semilla y datos producen la misma secuencia
    de epocas. El generador avanza entre epocas; no repite siempre la permutacion.
    num_workers=0 evita copias por procesos trabajadores en Windows.
    """
    if not isinstance(dataset, DatasetFavorita):
        raise TypeError("dataset debe ser DatasetFavorita")
    if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size <= 0:
        raise ValueError("batch_size debe ser un entero positivo")
    if isinstance(num_workers, bool) or not isinstance(num_workers, Integral) or num_workers < 0:
        raise ValueError("num_workers debe ser un entero no negativo")
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle debe ser booleano")
    if isinstance(semilla, bool) or not isinstance(semilla, Integral):
        raise TypeError("semilla debe ser un entero")
    generador = torch.Generator().manual_seed(int(semilla))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=shuffle,
        generator=generador,
        num_workers=int(num_workers),
        collate_fn=partial(collate_lote, dataset=dataset),
        drop_last=False,
    )
