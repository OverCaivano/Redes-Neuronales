"""Integracion con CSV temporales pequenos; sin entrenamiento real."""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, isspmatrix_csr
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from src.datos_favorita import leer_datos_favorita
from src.preprocesamiento_temporal import generar_variables_historicas, dividir_por_fecha
from src.transformaciones import COLUMNAS_NUMERICAS
from src.loaders_favorita import construir_loaders_favorita
from src.data_loader import build_dataloaders
from src.models import MLP


@pytest.fixture
def csv_realista(tmp_path):
    fechas = pd.date_range("2017-06-01", "2017-08-15")
    partes = []
    for tienda, familia in [(1, "AUTOMOTIVE"), (2, "BEAUTY")]:
        partes.append(pd.DataFrame({
            "date": fechas.strftime("%Y-%m-%d"), "store_nbr": tienda,
            "family": familia, "sales": np.arange(len(fechas), dtype=float) + tienda * 100,
            "onpromotion": np.arange(len(fechas)) % 3,
        }))
    tabla = pd.concat(partes, ignore_index=True)
    tabla.insert(0, "id", range(len(tabla)))
    ruta = tmp_path / "train.csv"
    tabla.sample(frac=1, random_state=42).to_csv(ruta, index=False)
    return ruta


def test_integridad_fechas_alineacion_y_forward(csv_realista):
    original = csv_realista.read_bytes()
    resultado = construir_loaders_favorita(csv_realista, batch_size=7)
    assert resultado.filas_entrada == 152
    assert resultado.filas_descartadas_historia == 14
    assert resultado.filas_por_particion == {"entrenamiento": 16, "validacion": 60, "test": 62}
    assert resultado.input_dim == len(resultado.nombres_columnas) == 12
    esperado = dividir_por_fecha(generar_variables_historicas(leer_datos_favorita(csv_realista)))
    rangos = []
    for nombre, loader in resultado.loaders.items():
        dataset = loader.dataset
        meta = resultado.metadatos[nombre]
        assert isspmatrix_csr(dataset.entradas)
        assert dataset.entradas.dtype == np.float32
        assert dataset.objetivos.dtype == np.float32
        assert dataset.objetivos.shape == (len(meta), 1)
        assert np.isfinite(dataset.entradas.data).all()
        assert np.isfinite(dataset.objetivos).all()
        assert meta["id"].tolist() == esperado[nombre]["id"].tolist()
        assert meta["fecha"].tolist() == esperado[nombre]["fecha"].tolist()
        assert meta["tienda"].tolist() == esperado[nombre]["tienda"].tolist()
        assert meta["familia"].tolist() == esperado[nombre]["familia"].tolist()
        np.testing.assert_array_equal(
            dataset.objetivos[:, 0], esperado[nombre]["ventas_objetivo"].to_numpy(dtype=np.float32)
        )
        assert (dataset.entradas != resultado.preparador.transformar(esperado[nombre])).nnz == 0
        assert loader.num_workers == 0 and not loader.drop_last
        rangos.append(set(meta["fecha"]))
        lotes = list(loader)
        assert sum(len(x) for x, _ in lotes) == len(dataset)
        if nombre != "entrenamiento":
            assert isinstance(loader.sampler, SequentialSampler)
            np.testing.assert_array_equal(
                torch.cat([y for _, y in lotes]).numpy(), dataset.objetivos
            )
            assert len(lotes[-1][0]) == len(dataset) % 7
        else:
            assert isinstance(loader.sampler, RandomSampler)
        modelo = MLP(input_dim=resultado.input_dim).eval()
        with torch.no_grad():
            x, y = lotes[0]
            assert x.dtype == y.dtype == torch.float32
            assert modelo(x).shape == y.shape
    assert rangos[0].isdisjoint(rangos[1])
    assert rangos[1].isdisjoint(rangos[2])
    assert rangos[0].isdisjoint(rangos[2])
    assert csv_realista.read_bytes() == original


def test_ajuste_exclusivo_entrenamiento(csv_realista, monkeypatch):
    esperadas = dividir_por_fecha(generar_variables_historicas(leer_datos_favorita(csv_realista)))
    llamadas = {}
    for clase in (OneHotEncoder, StandardScaler):
        fit_original = clase.fit

        def registrar(self, datos, *args, _fit=fit_original, _clase=clase, **kwargs):
            llamadas.setdefault(_clase, []).append(datos.copy())
            return _fit(self, datos, *args, **kwargs)

        monkeypatch.setattr(clase, "fit", registrar)
    resultado = construir_loaders_favorita(csv_realista)
    assert len(llamadas[OneHotEncoder]) == len(llamadas[StandardScaler]) == 1
    pd.testing.assert_frame_equal(
        llamadas[StandardScaler][0], esperadas["entrenamiento"][COLUMNAS_NUMERICAS]
    )
    pd.testing.assert_frame_equal(
        llamadas[OneHotEncoder][0], esperadas["entrenamiento"][["tienda", "familia"]]
    )
    parametros = resultado.preparador.parametros_escalado
    for loader in resultado.loaders.values():
        next(iter(loader))
    for nombre, valor in parametros.items():
        np.testing.assert_array_equal(valor, resultado.preparador.parametros_escalado[nombre])


def test_reproducibilidad_y_filtros(csv_realista):
    uno = construir_loaders_favorita(csv_realista, tiendas=[2], familias=["BEAUTY"], batch_size=3, semilla=19)
    dos = construir_loaders_favorita(csv_realista, tiendas=[2], familias=["BEAUTY"], batch_size=3, semilla=19)
    assert uno.input_dim == 10
    assert uno.filas_por_particion == {"entrenamiento": 8, "validacion": 30, "test": 31}
    for nombre in uno.loaders:
        for (x1, y1), (x2, y2) in zip(uno.loaders[nombre], dos.loaders[nombre]):
            assert torch.equal(x1, x2) and torch.equal(y1, y2)
        assert set(uno.metadatos[nombre]["tienda"]) == {2}
        assert set(uno.metadatos[nombre]["familia"]) == {"BEAUTY"}


@pytest.mark.parametrize("nombre", ["entrenamiento", "validacion", "test"])
def test_particiones_vacias_error_claro(csv_realista, nombre):
    tabla = pd.read_csv(csv_realista)
    if nombre == "entrenamiento":
        tabla = tabla.loc[tabla.date.ge("2017-06-16")]
    elif nombre == "validacion":
        tabla = tabla.loc[~tabla.date.between("2017-06-16", "2017-07-15")]
    else:
        tabla = tabla.loc[tabla.date.le("2017-07-15")]
    tabla.to_csv(csv_realista, index=False)
    with pytest.raises(ValueError, match=f"particion {nombre} esta vacia"):
        construir_loaders_favorita(csv_realista)


def test_construccion_sin_densificar(csv_realista, monkeypatch):
    def prohibido(*args, **kwargs):
        pytest.fail("La construccion no debe densificar CSR")
    monkeypatch.setattr(csr_matrix, "toarray", prohibido)
    monkeypatch.setattr(csr_matrix, "todense", prohibido)
    resultado = construir_loaders_favorita(csv_realista)
    assert len(resultado.loaders) == 3


def test_familia_exclusiva_de_test_no_se_aprende(csv_realista):
    tabla = pd.read_csv(csv_realista)
    tabla.loc[tabla.date.ge("2017-07-16"), "family"] = "DESCONOCIDA"
    tabla.to_csv(csv_realista, index=False)
    with pytest.raises(ValueError, match="Categorias desconocidas en familia"):
        construir_loaders_favorita(csv_realista)


def test_historia_validacion_usa_dias_de_entrenamiento(csv_realista):
    resultado = construir_loaders_favorita(csv_realista)
    meta = resultado.metadatos["validacion"]
    posicion = np.flatnonzero(meta["fecha"].eq("2017-06-16") & meta["tienda"].eq(1))[0]
    escaladas = resultado.loaders["validacion"].dataset.entradas[posicion, :3].toarray()[0]
    parametros = resultado.preparador.parametros_escalado
    reconstruidas = escaladas * parametros["escala"][:3] + parametros["media"][:3]
    np.testing.assert_allclose(reconstruidas, [114, 108, 111], atol=1e-5)


def test_pipeline_sintetico_permanece_compatible():
    config = {
        "seed": 42, "model": {"input_dim": 10},
        "training": {"batch_size": 8}, "validation": {"split": 0.2},
        "data": {"source": "synthetic", "n_samples": 40},
    }
    train, validacion = build_dataloaders(config)
    assert len(train.dataset) == 32 and len(validacion.dataset) == 8
    x, y = next(iter(train))
    assert x.shape == (8, 10) and y.shape == (8, 1)
