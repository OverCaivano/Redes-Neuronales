"""Consultas con fixtures pequenas; ningun entrenamiento en estas pruebas."""

import importlib

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from test_evaluacion_favorita import checkpoint
from src.datos_favorita import leer_datos_favorita
from src.predecir_favorita import preparar_consulta, predecir_favorita
from src.preprocesamiento_temporal import generar_variables_historicas
from src.reanudacion import huella_dataset
from src.models import MLP

modulo = importlib.import_module("src.predecir_favorita")
CONSULTA = dict(tienda=1, familia="BEAUTY", fecha="2017-07-16")


def test_prediccion_sin_fit_sin_optimizador_y_originales_intactos(checkpoint, monkeypatch):
    ruta, csv, _ = checkpoint
    antes = huella_dataset(ruta), huella_dataset(csv)
    def prohibido(*a, **kw):
        pytest.fail("No ajustar ni entrenar")
    monkeypatch.setattr(OneHotEncoder, "fit", prohibido)
    monkeypatch.setattr(StandardScaler, "fit", prohibido)
    monkeypatch.setattr(StandardScaler, "partial_fit", prohibido)
    monkeypatch.setattr(torch.optim.Adam, "__init__", prohibido)
    original = MLP.forward
    def forward(self, x):
        assert x.shape == (1, 10) and x.dtype == torch.float32
        assert not self.training and not torch.is_grad_enabled()
        return original(self, x)
    monkeypatch.setattr(MLP, "forward", forward)
    resultado = predecir_favorita(ruta, **CONSULTA)
    assert resultado["ventas_estimadas"] == 1.
    assert resultado["fecha"] == CONSULTA["fecha"]
    assert antes == (huella_dataset(ruta), huella_dataset(csv))


def test_causalidad_y_equivalencia_pipeline(checkpoint):
    ruta, csv, _ = checkpoint
    datos = leer_datos_favorita(csv)
    original = datos.copy(deep=True)
    fila = preparar_consulta(datos, **CONSULTA)
    esperado = generar_variables_historicas(datos).datos
    esperado = esperado.loc[esperado.fecha.eq(CONSULTA["fecha"])].drop(columns=["id", "ventas_objetivo"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(fila, esperado)
    assert "ventas_objetivo" not in fila
    assert fila.ventas_hace_1_dia.iloc[0] == 44.
    assert fila.ventas_hace_7_dias.iloc[0] == 38.
    assert fila.promedio_ventas_7_dias.iloc[0] == 41.
    alterados = datos.copy()
    alterados.loc[alterados.fecha.ge(CONSULTA["fecha"]), "ventas_objetivo"] = np.nan
    alterados.loc[alterados.fecha.gt(CONSULTA["fecha"]), "cantidad_en_promocion"] = 999999
    otra = preparar_consulta(alterados, **CONSULTA)
    pd.testing.assert_frame_equal(fila, otra)
    estado = torch.load(ruta, weights_only=False)
    x1 = estado["preparador"].transformar(fila).toarray()
    x2 = estado["preparador"].transformar(otra).toarray()
    np.testing.assert_array_equal(x1, x2)
    modelo = modulo.reconstruir_modelo_favorita(estado)
    with torch.inference_mode():
        assert torch.equal(modelo(torch.from_numpy(x1)), modelo(torch.from_numpy(x2)))
    pd.testing.assert_frame_equal(datos, original)


def test_no_toma_otra_serie(checkpoint):
    _, csv, _ = checkpoint
    datos = leer_datos_favorita(csv)
    extra = datos.copy()
    extra["tienda"] = 2
    extra["ventas_objetivo"] = 99999.
    combinado = pd.concat([extra, datos]).sample(frac=1, random_state=42)
    pd.testing.assert_frame_equal(preparar_consulta(datos, **CONSULTA),
                                  preparar_consulta(combinado, **CONSULTA))


@pytest.mark.parametrize("fecha", ["2017-06-01", "2017-06-07"])
def test_historia_inicial_insuficiente(checkpoint, fecha):
    datos = leer_datos_favorita(checkpoint[1])
    with pytest.raises(ValueError, match="siete dias"):
        preparar_consulta(datos, **dict(CONSULTA, fecha=fecha))


def test_hueco_no_es_cero_ni_fila_anterior(checkpoint):
    datos = leer_datos_favorita(checkpoint[1])
    datos = datos.loc[datos.fecha.ne("2017-07-12")]
    with pytest.raises(ValueError, match="siete dias"):
        preparar_consulta(datos, **CONSULTA)


@pytest.mark.parametrize("fecha", ["2017-08-16", "2030-01-01"])
def test_fecha_sin_fila_no_inventa_promocion(checkpoint, fecha):
    with pytest.raises(ValueError, match="unica fila"):
        predecir_favorita(checkpoint[0], **dict(CONSULTA, fecha=fecha))


@pytest.mark.parametrize("cambio,patron", [
    ({"tienda": 2}, "subconjunto"), ({"familia": "OTRA"}, "subconjunto"),
    ({"tienda": 0}, "entero"), ({"tienda": 1.2}, "entero"),
    ({"familia": ""}, "no vacio"), ({"fecha": "2017-02-30"}, "AAAA-MM-DD"),
    ({"fecha": "2017-7-16"}, "AAAA-MM-DD"),
])
def test_entradas_invalidas(checkpoint, cambio, patron):
    with pytest.raises(ValueError, match=patron):
        predecir_favorita(checkpoint[0], **dict(CONSULTA, **cambio))


def test_dataset_distinto_rechazado(checkpoint):
    ruta, csv, _ = checkpoint
    csv.write_text(csv.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA-256"):
        predecir_favorita(ruta, **CONSULTA)


@pytest.mark.parametrize("salida", [torch.zeros(1), torch.tensor([[float("nan")]])])
def test_salida_invalida(checkpoint, monkeypatch, salida):
    monkeypatch.setattr(MLP, "forward", lambda self, x: salida)
    with pytest.raises(ValueError, match="finita"):
        predecir_favorita(checkpoint[0], **CONSULTA)


def test_cli_y_error_comprensible(checkpoint, monkeypatch, capsys):
    monkeypatch.setattr("sys.argv", ["predecir", "--checkpoint", str(checkpoint[0]),
                                     "--tienda", "1", "--familia", "BEAUTY", "--fecha", "2017-07-16"])
    modulo.main()
    assert "No representa stock" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["predecir", "--checkpoint", str(checkpoint[0]),
                                     "--tienda", "2", "--familia", "BEAUTY", "--fecha", "2017-07-16"])
    with pytest.raises(SystemExit) as error:
        modulo.main()
    assert error.value.code == 2
    assert "subconjunto" in capsys.readouterr().err
