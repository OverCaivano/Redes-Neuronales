"""API e interfaz con checkpoint pequeño sin entrenamiento ni datasets reales."""

import json
import shutil
from pathlib import Path

import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from test_evaluacion_favorita import checkpoint
from src.reanudacion import huella_dataset
from src.web_app import ServicioPiloto, crear_app


@pytest.fixture
def servicio(checkpoint, tmp_path):
    ruta, csv, _ = checkpoint
    raiz = tmp_path / "proyecto"
    destino = raiz / "runs/favorita_subconjunto_reanudado_4"
    destino.mkdir(parents=True)
    (raiz / "data/raw").mkdir(parents=True)
    shutil.copyfile(ruta, destino / "best.pt")
    shutil.copyfile(csv, raiz / "data/raw/train.csv")
    resumen = {"epocas_completadas": 2, "mejor_epoca": 2, "mejor_val_mae": 1., "mejor_val_rmse": 2.,
               "input_dim": 10, "filas_por_particion": {"entrenamiento": 8, "validacion": 30, "test": 31},
               "ruta_privada": "C:/secret/modelo.pt"}
    (destino / "resumen.json").write_text(json.dumps(resumen))
    (destino / "metricas.csv").write_text("epoca,train_mse,val_mse,val_mae,val_rmse\n1,4,9,2,3\n2,3,4,1,2\n")
    evaluacion = raiz / "runs/evaluaciones/favorita_piloto_test"
    evaluacion.mkdir(parents=True)
    (evaluacion / "evaluacion.json").write_text(json.dumps({
        "checkpoint_sha256": huella_dataset(ruta), "checkpoint": "C:/secret/modelo.pt",
        "mlp": {"mae": 1., "rmse": 2., "n": 31}, "baseline_semanal": {"mae": 2., "rmse": 3., "n": 31},
        "fecha_min": "2017-07-16", "fecha_max": "2017-08-15"}))
    return ServicioPiloto(raiz)


@pytest.fixture
def cliente(servicio):
    app = crear_app(servicio)
    app.config["TESTING"] = True
    return app.test_client()


@pytest.mark.parametrize("ruta", ["/", "/inicio", "/predicciones", "/evaluacion", "/historial", "/acerca"])
def test_navegacion(cliente, ruta):
    respuesta = cliente.get(ruta)
    assert respuesta.status_code == 200
    texto = respuesta.get_data(as_text=True)
    assert "StockPredict IA" in texto
    for enlace in ("/inicio", "/predicciones", "/evaluacion", "/historial", "/acerca"):
        assert f'href="{enlace}"' in texto
    assert 'href="/static/favicon.svg"' in texto
    assert 'id="theme-toggle"' in texto
    assert 'src="/static/tema.js"></script>' in texto
    assert texto.index('src="/static/tema.js"></script>') < texto.index('href="/static/estilos.css"')
    assert 'src="/static/app.js"' in texto
    assert "C:/secret" not in texto
    for ficticio in ("Elena Ruiz", "98.4", "50/50", "Math.random", "p-value", "AdamW", "RUN-2024", "95%"):
        assert ficticio not in texto


def test_opciones_y_prediccion_real_sin_fit(cliente, servicio, monkeypatch):
    antes = huella_dataset(servicio.csv), huella_dataset(servicio.checkpoint)
    def prohibido(*a, **k):
        pytest.fail("La web no debe ajustar transformaciones")
    monkeypatch.setattr(OneHotEncoder, "fit", prohibido)
    monkeypatch.setattr(StandardScaler, "fit", prohibido)
    opciones = cliente.get("/api/opciones")
    assert opciones.status_code == 200
    serie = opciones.json["series"][0]
    assert serie["tienda"] == 1 and serie["familia"] == "BEAUTY"
    assert "2017-06-01" not in serie["fechas"]  # Sin historia suficiente.
    assert "2017-07-16" in serie["fechas"]
    respuesta = cliente.post("/api/predicciones", json={"tienda": 1, "familia": "BEAUTY", "fecha": "2017-07-16"})
    assert respuesta.status_code == 200
    assert respuesta.json["ventas_estimadas"] == 1.
    assert respuesta.json["modelo"] == "Piloto local · best.pt"
    assert "checkpoint" not in respuesta.json
    assert str(servicio.raiz) not in respuesta.get_data(as_text=True)
    assert antes == (huella_dataset(servicio.csv), huella_dataset(servicio.checkpoint))


@pytest.mark.parametrize("datos,estado", [
    ({}, 400), ([], 400), ({"tienda": 1}, 400),
    ({"tienda": True, "familia": "BEAUTY", "fecha": "2017-07-16"}, 400),
    ({"tienda": 1.5, "familia": "BEAUTY", "fecha": "2017-07-16"}, 400),
    ({"tienda": 1, "familia": "", "fecha": "2017-07-16"}, 400),
    ({"tienda": 1, "familia": "BEAUTY", "fecha": "2017-02-31"}, 400),
    ({"tienda": 2, "familia": "BEAUTY", "fecha": "2017-07-16"}, 422),
    ({"tienda": 1, "familia": "OTRA", "fecha": "2017-07-16"}, 422),
    ({"tienda": 1, "familia": "BEAUTY", "fecha": "2030-01-01"}, 422),
    ({"tienda": 1, "familia": "BEAUTY", "fecha": "2017-06-01"}, 422),
    ({"tienda": 1, "familia": "BEAUTY", "fecha": "2017-07-16", "checkpoint": "otro.pt"}, 400),
])
def test_validaciones(cliente, datos, estado):
    respuesta = cliente.post("/api/predicciones", json=datos)
    assert respuesta.status_code == estado
    assert "error" in respuesta.json


def test_json_origen_y_limites(cliente):
    assert cliente.post("/api/predicciones", data="hola").status_code == 415
    assert cliente.post("/api/predicciones", data="{", content_type="application/json").status_code == 400
    assert cliente.post("/api/predicciones", json={}, headers={"Origin": "https://externo.example"}).status_code == 403
    assert cliente.post("/api/predicciones", data="x" * 5000, content_type="application/json").status_code == 413
    assert cliente.get("/api/resumen", headers={"Host": "externo.example"}).status_code == 400


@pytest.mark.parametrize("ruta", ["/data/raw/train.csv", "/runs/favorita_subconjunto_reanudado_4/best.pt", "/static/../../data/raw/train.csv", "/disenos/stitch/inicio.html", "/desconocida"])
def test_no_expone_archivos(cliente, ruta):
    assert cliente.get(ruta).status_code == 404


def test_resumen_sanitizado_y_metricas_json(cliente):
    respuesta = cliente.get("/api/resumen")
    assert respuesta.json["evaluacion"]["mlp"]["mae"] == 1.
    assert respuesta.json["evaluacion"]["mejora"] == pytest.approx(100 / 3)
    assert "secret" not in respuesta.get_data(as_text=True)
    assert "No verificado" in respuesta.json["drive"]
    assert len(respuesta.json["corridas"]) == 1


def test_artefactos_ausentes_y_errores_no_filtran_rutas(tmp_path, cliente, servicio, monkeypatch):
    c = crear_app(ServicioPiloto(tmp_path)).test_client()
    assert c.get("/api/opciones").status_code == 503
    assert c.get("/evaluacion").status_code == 200
    assert c.get("/api/resumen").json["evaluacion"] is None
    def error(**kw):
        raise ValueError("C:/secret/modelo.pt")
    monkeypatch.setattr(servicio, "predecir", error)
    respuesta = cliente.post("/api/predicciones", json={"tienda": 1, "familia": "BEAUTY", "fecha": "2017-07-16"})
    assert respuesta.status_code == 503
    assert "secret" not in respuesta.get_data(as_text=True)


def test_evaluacion_otro_checkpoint_no_se_muestra(servicio):
    datos = json.loads(servicio.evaluacion.read_text())
    datos["checkpoint_sha256"] = "diferente"
    servicio.evaluacion.write_text(json.dumps(datos))
    assert servicio.resumen()["evaluacion"] is None


def test_estaticos_y_cabeceras(cliente):
    for ruta in ("/static/estilos.css", "/static/app.js", "/static/tema.js", "/static/favicon.svg"):
        respuesta = cliente.get(ruta)
        assert respuesta.status_code == 200
        assert respuesta.headers["X-Content-Type-Options"] == "nosniff"
        assert "frame-ancestors 'none'" in respuesta.headers["Content-Security-Policy"]
    js = cliente.get("/static/app.js").get_data(as_text=True)
    assert "Math.random" not in js
    assert 'fetch("/api/predicciones"' in js
    assert "aria-busy" in js
    assert "var(--chart-grid)" in js
    tema = cliente.get("/static/tema.js").get_data(as_text=True)
    assert '"stockpredict-theme"' in tema
    assert "prefers-color-scheme: dark" in tema
    assert 'localStorage.setItem(claveTema, tema)' in tema
    css = cliente.get("/static/estilos.css").get_data(as_text=True)
    assert "[data-theme=dark]" in css
    assert "--page-bg:#0f172a" in css
    assert "--surface-card:#1e293b" in css


def test_copias_stitch_identificadas():
    raiz = Path(__file__).resolve().parents[1]
    registro = json.loads((raiz / "disenos/stitch/procedencia.json").read_text())
    assert len(registro) == 10
    for entrada in registro:
        assert huella_dataset(raiz / "disenos/stitch" / entrada["copia"]) == entrada["sha256"]
