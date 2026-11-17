import json

import pandas as pd
import pytest

from src.datos_sinteticos import generar_archivos_sinteticos
from src.pipeline_productos import (
    FIN_ENTRENAMIENTO,
    FIN_VALIDACION,
    PreparadorCaracteristicas,
    cargar_dataset,
    entrenar_pipeline,
    evaluar_checkpoint,
    separar_por_fecha,
)
from src.predecir_producto import predecir_venta


@pytest.fixture(scope="module")
def datos(tmp_path_factory):
    directorio = tmp_path_factory.mktemp("datos_sinteticos")
    archivos = generar_archivos_sinteticos(directorio)
    return archivos["ventas"], archivos["catalogo"]


@pytest.fixture(scope="module")
def corrida(datos, tmp_path_factory):
    salida = tmp_path_factory.mktemp("modelo_producto")
    return entrenar_pipeline(
        *datos,
        salida,
        epocas=2,
        tamano_lote=8192,
    )


def test_particiones_cronologicas_y_disjuntas(datos):
    ventas, catalogo = cargar_dataset(*datos)
    partes = separar_por_fecha(ventas)
    assert partes["entrenamiento"]["fecha"].max() == FIN_ENTRENAMIENTO
    assert partes["validacion"]["fecha"].min() == FIN_ENTRENAMIENTO + pd.Timedelta(days=1)
    assert partes["validacion"]["fecha"].max() == FIN_VALIDACION
    assert partes["prueba"]["fecha"].min() == FIN_VALIDACION + pd.Timedelta(days=1)
    assert partes["prueba"]["fecha"].max() == pd.Timestamp("2026-11-16")
    assert sum(map(len, partes.values())) == len(ventas)
    assert len(catalogo) == 30


def test_transformador_ajusta_vocabulario_con_entrenamiento():
    entrenamiento = pd.DataFrame({
        "fecha": pd.to_datetime(["2025-01-01", "2025-01-02"]),
        "producto_id": ["BEB-001", "BEB-001"],
        "tienda_id": [1, 1],
        "categoria": ["bebidas", "bebidas"],
        "mes": [1, 1],
    })
    preparador = PreparadorCaracteristicas().ajustar(entrenamiento)
    assert preparador.categorias["producto_id"] == ["BEB-001"]
    desconocido = entrenamiento.iloc[[0]].assign(producto_id="ALI-001")
    with pytest.raises(ValueError, match="Categorías desconocidas en producto_id"):
        preparador.transformar(desconocido)


def test_entrenamiento_guarda_checkpoint_y_metricas_solo_sinteticas(corrida):
    reporte = corrida["reporte"]
    checkpoint = corrida["checkpoint"]
    assert checkpoint.is_file()
    assert corrida["metricas_path"].is_file()
    assert reporte["metadata"]["datos_sinteticos"] is True
    assert reporte["metadata"]["origen"] == "dataset_sintetico"
    assert reporte["config"]["epoca_seleccionada"] in (1, 2)
    assert reporte["config"]["arquitectura"]["input_dim"] == len(reporte["features"])
    assert reporte["filas_por_particion"] == {
        "entrenamiento": 98640,
        "validacion": 16290,
        "prueba": 12510,
    }
    persistido = json.loads(corrida["metricas_path"].read_text(encoding="utf-8"))
    assert persistido["metricas"]["prueba"] == reporte["metricas"]["prueba"]


def test_evaluacion_reutiliza_checkpoint_y_devuelve_mae_rmse(corrida, datos):
    metricas = evaluar_checkpoint(corrida["checkpoint"], *datos)
    assert metricas == corrida["reporte"]["metricas"]["prueba"]
    assert metricas["modelo"]["n"] == 12510
    assert metricas["baseline_media_producto_tienda"]["n"] == 12510
    for comparacion in metricas.values():
        assert comparacion["mae"] >= 0
        assert comparacion["rmse"] >= 0


def test_checkpoint_rechaza_dataset_modificado(corrida, datos, tmp_path):
    ventas, catalogo = datos
    copia = tmp_path / "ventas.csv"
    copia.write_bytes(ventas.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="huellas del checkpoint"):
        evaluar_checkpoint(corrida["checkpoint"], copia, catalogo)


def test_checkpoint_admite_inferencia_futura_sintetica(corrida):
    resultado = predecir_venta(
        corrida["checkpoint"],
        tienda_id=1,
        producto_id="BEB-001",
        fecha="2026-11-17",
    )
    assert resultado["fecha"] == "2026-11-17"
    assert resultado["tienda_id"] == 1
    assert resultado["producto_id"] == "BEB-001"
    assert resultado["datos_sinteticos"] is True
    assert resultado["unidades_estimadas"] >= 0
    assert "No se suministran precio ni promoción" in resultado["supuesto"]


@pytest.mark.parametrize(
    "consulta, mensaje",
    [
        ({"tienda_id": 99, "producto_id": "BEB-001", "fecha": "2026-11-17"}, "tienda_id inexistente"),
        ({"tienda_id": 1, "producto_id": "XXX-999", "fecha": "2026-11-17"}, "producto_id inexistente"),
        ({"tienda_id": True, "producto_id": "BEB-001", "fecha": "2026-11-17"}, "entero"),
        ({"tienda_id": 1, "producto_id": "BEB-001", "fecha": "2026-02-30"}, "fecha debe ser válida"),
        ({"tienda_id": 1, "producto_id": "BEB-001", "fecha": "2026-11-16"}, "posterior al último día"),
        ({"tienda_id": 1, "producto_id": "BEB-001", "fecha": "2025-01-01"}, "posterior al último día"),
    ],
)
def test_inferencia_rechaza_consultas_invalidas(corrida, consulta, mensaje):
    with pytest.raises(ValueError, match=mensaje):
        predecir_venta(corrida["checkpoint"], **consulta)


def test_no_sobrescribe_corrida_existente(corrida, datos):
    with pytest.raises(FileExistsError, match="La corrida ya contiene artefactos"):
        entrenar_pipeline(*datos, corrida["checkpoint"].parent, epocas=1)
