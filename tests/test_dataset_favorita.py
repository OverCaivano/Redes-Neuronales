"""Lotes de fixtures pequenas: no entrenan ni leen el dataset real."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix
import torch

from src.dataset_favorita import DatasetFavorita, collate_lote, crear_dataloader
from src.models import MLP


@pytest.fixture
def datos():
    matriz = csr_matrix(np.arange(30, dtype=np.float32).reshape(10, 3))
    objetivos = pd.Series(np.arange(10) * 10, index=[8, 1, 8, 3, 2, 9, 5, 4, 0, 6])
    return matriz, objetivos


def test_dimensiones_tipos_y_ultimo_lote(datos):
    matriz, objetivos = datos
    dataset = DatasetFavorita(matriz, objetivos)
    assert dataset.entradas is matriz
    assert dataset.cantidad_ejemplos == len(dataset) == 10
    assert dataset.input_dim == matriz.shape[1]
    assert dataset.objetivos.dtype == np.float32
    loader = crear_dataloader(dataset, batch_size=4)
    assert loader.num_workers == 0
    lotes = list(loader)
    assert [len(x) for x, _ in lotes] == [4, 4, 2]
    for x, y in lotes:
        assert x.shape == (len(x), 3)
        assert y.shape == (len(x), 1)
        assert x.dtype == y.dtype == torch.float32
        assert torch.isfinite(x).all() and torch.isfinite(y).all()
    np.testing.assert_array_equal(torch.cat([x for x, _ in lotes]).numpy(), matriz.toarray())
    np.testing.assert_array_equal(torch.cat([y for _, y in lotes]).numpy().ravel(), objetivos.to_numpy())


def test_alineacion_posicional_con_repetidos_y_pandas(datos):
    matriz, objetivos = datos
    dataset = DatasetFavorita(matriz, objetivos)
    indices = [7, 0, 7, 2]
    x, y = collate_lote(indices, dataset=dataset)
    np.testing.assert_array_equal(x.numpy(), matriz[indices].toarray())
    np.testing.assert_array_equal(y.numpy().ravel(), objetivos.iloc[indices].to_numpy())
    objetivos.iloc[0] = -100
    assert dataset.objetivos[0, 0] == 0


def test_objetivos_columna_y_lote_unitario(datos):
    matriz, objetivos = datos
    dataset = DatasetFavorita(matriz, objetivos.to_frame())
    x, y = next(iter(crear_dataloader(dataset, batch_size=1)))
    assert x.shape == (1, 3)
    assert y.shape == (1, 1)


def test_shuffle_reproducible_y_alineado(datos):
    matriz, objetivos = datos
    dataset = DatasetFavorita(matriz, objetivos)
    uno = crear_dataloader(dataset, batch_size=3, shuffle=True, semilla=17)
    dos = crear_dataloader(dataset, batch_size=3, shuffle=True, semilla=17)
    for _ in range(2):
        lotes_uno, lotes_dos = list(uno), list(dos)
        for (x1, y1), (x2, y2) in zip(lotes_uno, lotes_dos):
            assert torch.equal(x1, x2)
            assert torch.equal(y1, y2)
            assert torch.equal(x1[:, 0] / 3 * 10, y1[:, 0])
        orden = torch.cat([y[:, 0] for _, y in lotes_uno]).tolist()
        assert sorted(orden) == objetivos.tolist()
        assert orden != objetivos.tolist()


def test_densificacion_limitada_al_lote(datos, monkeypatch):
    matriz, objetivos = datos
    toarray_original = csr_matrix.toarray
    tamanos = []

    def vigilar(self, *args, **kwargs):
        assert self is not matriz
        assert self.shape[0] <= 3
        tamanos.append(self.shape[0])
        return toarray_original(self, *args, **kwargs)

    monkeypatch.setattr(csr_matrix, "toarray", vigilar)
    dataset = DatasetFavorita(matriz, objetivos)
    assert tamanos == []
    list(crear_dataloader(dataset, batch_size=3))
    assert tamanos == [3, 3, 3, 1]


@pytest.mark.parametrize("caso", ["densa", "csc", "float64", "compleja", "sin_filas", "sin_columnas", "nan", "inf"])
def test_entradas_invalidas(caso):
    x = csr_matrix(np.ones((2, 3), dtype=np.float32))
    if caso == "densa":
        x = np.ones((2, 3), dtype=np.float32)
    elif caso == "csc":
        x = x.tocsc()
    elif caso == "float64":
        x = x.astype(np.float64)
    elif caso == "compleja":
        x = x.astype(np.complex64)
    elif caso == "sin_filas":
        x = csr_matrix((0, 3), dtype=np.float32)
    elif caso == "sin_columnas":
        x = csr_matrix((2, 0), dtype=np.float32)
    else:
        x.data[0] = np.nan if caso == "nan" else np.inf
    with pytest.raises((ValueError, TypeError)):
        DatasetFavorita(x, np.ones(x.shape[0], dtype=np.float32))


@pytest.mark.parametrize("y", [
    [1], [[1, 2], [3, 4]], 3, np.ones((2, 1, 1)),
    [np.nan, 1], [np.inf, 1], [-np.inf, 1], [1e100, 0],
    ["1", "2"], [1j, 2j], [True, False],
])
def test_objetivos_invalidos(y):
    with pytest.raises((ValueError, TypeError)):
        DatasetFavorita(csr_matrix(np.ones((2, 3), dtype=np.float32)), y)


@pytest.mark.parametrize("opciones", [
    {"batch_size": 0}, {"batch_size": 1.5}, {"batch_size": True},
    {"num_workers": -1}, {"num_workers": 1.5}, {"shuffle": "si"}, {"semilla": 1.5},
])
def test_configuracion_invalida(datos, opciones):
    with pytest.raises((ValueError, TypeError)):
        crear_dataloader(DatasetFavorita(*datos), **opciones)


@pytest.mark.parametrize("indices", [[], [-1], [10], [1.5], [True]])
def test_indices_invalidos(datos, indices):
    with pytest.raises((ValueError, TypeError, IndexError)):
        collate_lote(indices, dataset=DatasetFavorita(*datos))


def test_entradas_cero_son_validas():
    dataset = DatasetFavorita(csr_matrix((2, 3), dtype=np.float32), [0, 0])
    x, y = next(iter(crear_dataloader(dataset, batch_size=1)))
    assert torch.count_nonzero(x) == torch.count_nonzero(y) == 0


def test_forward_mlp_sin_entrenamiento(datos):
    dataset = DatasetFavorita(*datos)
    modelo = MLP(input_dim=dataset.input_dim, hidden_dim=8, output_dim=1)
    modelo.eval()
    antes = {nombre: valor.clone() for nombre, valor in modelo.state_dict().items()}
    with torch.no_grad():
        for x, y in crear_dataloader(dataset, batch_size=4):
            salida = modelo(x)
            assert salida.shape == y.shape == (len(x), 1)
            assert salida.dtype == torch.float32
            assert torch.isfinite(salida).all()
    for nombre, valor in modelo.state_dict().items():
        assert torch.equal(valor, antes[nombre])
    assert all(parametro.grad is None for parametro in modelo.parameters())
