from datetime import date

import pytest

from src.domain.exceptions import ErrorInferencia
from src.infrastructure.ml import pytorch_prediction_adapter


FECHA_CORTE = date(2026, 11, 16)


def test_adaptador_pytorch_encapsula_y_reutiliza_el_pronosticador(
    monkeypatch, tmp_path
):
    instancias = []

    class PronosticadorFalso:
        fecha_corte = FECHA_CORTE
        tiendas_disponibles = (1, 2, 3)

        def __init__(self, ruta):
            self.ruta = ruta
            self.consultas = []
            instancias.append(self)

        def predecir(self, **consulta):
            self.consultas.append(consulta)
            return {"unidades_estimadas": 7.25}

    monkeypatch.setattr(
        pytorch_prediction_adapter,
        "PronosticadorProducto",
        PronosticadorFalso,
    )
    adaptador = pytorch_prediction_adapter.AdaptadorPrediccionPyTorch(
        tmp_path / "checkpoint.pt"
    )

    assert adaptador.fecha_corte == FECHA_CORTE
    assert adaptador.tiendas_disponibles == (1, 2, 3)
    assert adaptador.predecir(
        tienda_id=1,
        producto_id="BEB-001",
        fecha=date(2026, 11, 17),
    ) == 7.25
    assert len(instancias) == 1
    assert instancias[0].ruta == tmp_path / "checkpoint.pt"
    assert instancias[0].consultas == [{
        "tienda_id": 1,
        "producto_id": "BEB-001",
        "fecha": "2026-11-17",
    }]


def test_adaptador_expone_fallos_de_inferencia_con_error_de_aplicacion(monkeypatch):
    class PronosticadorFalso:
        fecha_corte = FECHA_CORTE
        tiendas_disponibles = (1, 2, 3)

        def __init__(self, ruta):
            self.ruta = ruta

        def predecir(self, **consulta):
            assert consulta["producto_id"] == "BEB-001"
            raise RuntimeError("detalle interno")

    monkeypatch.setattr(
        pytorch_prediction_adapter,
        "PronosticadorProducto",
        PronosticadorFalso,
    )
    adaptador = pytorch_prediction_adapter.AdaptadorPrediccionPyTorch("modelo.pt")

    with pytest.raises(ErrorInferencia) as error:
        adaptador.predecir(
            tienda_id=1,
            producto_id="BEB-001",
            fecha=date(2026, 11, 17),
        )
    assert isinstance(error.value.__cause__, RuntimeError)
