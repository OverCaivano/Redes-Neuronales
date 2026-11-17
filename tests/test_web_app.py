from datetime import date
import math

import pytest

from src.application.services.prediction_service import ServicioPrediccion
from src.application.replenishment_strategies import (
    ReposicionBasica,
    ReposicionConSeguridad,
)
from src.application.services.replenishment_service import ServicioReposicion
from src.composition import RUTA_CATALOGO, RUTA_CHECKPOINT, RUTA_INVENTARIO
from src.domain.models import Inventario, Producto
from src.infrastructure.ml.pytorch_prediction_adapter import (
    AdaptadorPrediccionPyTorch,
)
from src.infrastructure.repositories.csv_product_repository import (
    RepositorioProductosCSV,
)
from src.predecir_producto import predecir_venta
from src.web_app import PAGINAS, crear_app


FECHA_CORTE = date(2026, 11, 16)
FECHA_ACTUAL = date(2026, 11, 17)


class RepositorioEnMemoria:
    def __init__(self):
        self.productos = (
            Producto("BEB-001", "Agua mineral 500 ml", "bebidas"),
            Producto("LAC-001", "Leche entera 1 l", "lacteos"),
        )

    def buscar_por_id(self, producto_id):
        return next(
            (producto for producto in self.productos if producto.producto_id == producto_id),
            None,
        )

    def listar(self):
        return self.productos


class MotorFalso:
    fecha_corte = FECHA_CORTE
    tiendas_disponibles = (1, 2, 3)

    def __init__(self):
        self.consultas = []

    def predecir(self, *, tienda_id, producto_id, fecha):
        self.consultas.append((tienda_id, producto_id, fecha))
        return 12.75


class InventarioEnMemoria:
    def __init__(self):
        self.registros = {
            (tienda_id, producto_id): Inventario(
                tienda_id=tienda_id,
                producto_id=producto_id,
                stock_actual=20,
                stock_minimo=4,
                tiempo_reposicion=3,
            )
            for tienda_id in (1, 2, 3)
            for producto_id in ("BEB-001", "LAC-001")
        }

    def buscar(self, *, tienda_id, producto_id):
        return self.registros.get((tienda_id, producto_id))


@pytest.fixture
def cliente():
    servicio = ServicioPrediccion(
        RepositorioEnMemoria(),
        MotorFalso(),
        hoy=lambda: FECHA_ACTUAL,
    )
    estrategias = (ReposicionBasica(), ReposicionConSeguridad())
    servicio_reposicion = ServicioReposicion(
        InventarioEnMemoria(),
        servicio,
        {estrategia.codigo: estrategia for estrategia in estrategias},
    )
    app = crear_app(
        servicio_prediccion=servicio,
        servicio_reposicion=servicio_reposicion,
    )
    app.config["TESTING"] = True
    return app.test_client()


def test_paginas_y_navegacion_no_muestran_contenido_heredado(cliente):
    for pagina in PAGINAS:
        respuesta = cliente.get("/" if pagina == "inicio" else f"/{pagina}")
        assert respuesta.status_code == 200
        texto = respuesta.get_data(as_text=True)
        assert "StockPredict IA" in texto
        assert "datos sintéticos" in texto.lower()
        assert "Favorita" not in texto
        for enlace in PAGINAS:
            assert f'href="/{enlace}"' in texto
    for pagina in ("evaluacion", "historial"):
        respuesta = cliente.get(f"/{pagina}")
        assert "Vista temporalmente deshabilitada" in respuesta.get_data(as_text=True)


def test_opciones_catalogo_y_prediccion_por_la_nueva_api(cliente):
    opciones = cliente.get("/api/synthetic-v2/opciones")
    assert opciones.status_code == 200
    assert opciones.json == {
        "tiendas": [1, 2, 3],
        "fecha_corte": "2026-11-16",
        "fecha_minima": "2026-11-17",
        "datos_sinteticos": True,
    }

    catalogo = cliente.get("/api/synthetic-v2/productos")
    assert catalogo.status_code == 200
    assert len(catalogo.json["productos"]) == 2
    assert catalogo.json["productos"][0] == {
        "producto_id": "BEB-001",
        "nombre": "Agua mineral 500 ml",
        "categoria": "bebidas",
    }

    resultado = cliente.post(
        "/api/synthetic-v2/predicciones",
        json={
            "tienda_id": 1,
            "producto_id": "BEB-001",
            "fecha": "2026-11-17",
        },
    )
    assert resultado.status_code == 200
    assert resultado.json["producto"]["producto_id"] == "BEB-001"
    assert resultado.json["fecha"] == "2026-11-17"
    assert resultado.json["unidades_estimadas"] == 12.75
    assert resultado.json["datos_sinteticos"] is True


def test_endpoint_recomendacion_de_reposicion_y_contrato_json(cliente):
    estrategias = cliente.get("/api/synthetic-v2/reposicion/estrategias")
    assert estrategias.status_code == 200
    assert estrategias.json["estrategias"] == [
        {"codigo": "basica", "nombre": "Reposición básica"},
        {"codigo": "seguridad", "nombre": "Reposición con stock de seguridad"},
    ]

    respuesta = cliente.post(
        "/api/synthetic-v2/reposiciones",
        json={
            "tienda_id": 2,
            "producto_id": "BEB-001",
            "fecha": "2026-11-25",
            "estrategia": "seguridad",
        },
    )

    assert respuesta.status_code == 200
    assert respuesta.json["datos_sinteticos"] is True
    assert respuesta.json["tienda_id"] == 2
    assert respuesta.json["producto"]["producto_id"] == "BEB-001"
    assert respuesta.json["demanda_fecha_seleccionada"] == 12.75
    assert respuesta.json["inventario"] == {
        "stock_actual": 20,
        "stock_minimo": 4,
        "tiempo_reposicion": 3,
    }
    assert respuesta.json["periodo_cobertura"] == {
        "desde": "2026-11-25",
        "hasta": "2026-11-27",
        "dias": 3,
    }
    assert respuesta.json["demanda_acumulada"] == 38.25
    assert [item["fecha"] for item in respuesta.json["demanda_diaria"]] == [
        "2026-11-25",
        "2026-11-26",
        "2026-11-27",
    ]
    assert respuesta.json["estrategia"]["codigo"] == "seguridad"
    assert respuesta.json["cantidad_recomendada"] == 23
    assert respuesta.json["requiere_reposicion"] is True


@pytest.mark.parametrize(
    "datos,estado",
    [
        (
            {
                "tienda_id": 2,
                "producto_id": "BEB-001",
                "fecha": "2026-11-25",
                "estrategia": "desconocida",
            },
            400,
        ),
        (
            {
                "tienda_id": 2,
                "producto_id": "BEB-001",
                "fecha": "2026-11-25",
            },
            400,
        ),
    ],
)
def test_endpoint_reposicion_rechaza_estrategias_y_payload_invalidos(
    cliente, datos, estado
):
    respuesta = cliente.post("/api/synthetic-v2/reposiciones", json=datos)

    assert respuesta.status_code == estado
    assert "error" in respuesta.json


@pytest.mark.parametrize(
    "datos,estado",
    [
        ({"tienda_id": 99, "producto_id": "BEB-001", "fecha": "2026-11-17"}, 404),
        ({"tienda_id": 1, "producto_id": "NO-EXISTE", "fecha": "2026-11-17"}, 404),
        ({"tienda_id": True, "producto_id": "BEB-001", "fecha": "2026-11-17"}, 400),
        ({"tienda_id": 1, "producto_id": "BEB-001", "fecha": "2026-02-30"}, 400),
        ({"tienda_id": 1, "producto_id": "BEB-001", "fecha": "2026-11-16"}, 400),
        ({"tienda_id": 1, "producto_id": "BEB-001", "fecha": "2026-11-16T00:00:00"}, 400),
        ({"tienda_id": 1, "producto_id": "BEB-001", "fecha": "2026-11-17", "modelo": "viejo"}, 400),
    ],
)
def test_validaciones_de_prediccion(cliente, datos, estado):
    respuesta = cliente.post("/api/synthetic-v2/predicciones", json=datos)
    assert respuesta.status_code == estado
    assert "error" in respuesta.json


def test_origen_y_formato_json(cliente):
    assert cliente.post(
        "/api/synthetic-v2/predicciones",
        data="{}",
        content_type="text/plain",
    ).status_code == 415
    assert cliente.post(
        "/api/synthetic-v2/predicciones",
        data="{",
        content_type="application/json",
    ).status_code == 400
    assert cliente.post(
        "/api/synthetic-v2/predicciones",
        json={},
        headers={"Origin": "https://externo.example"},
    ).status_code == 403


def test_rutas_anteriores_quedan_inactivas(cliente):
    for ruta in (
        "/api/opciones",
        "/api/resumen",
        "/api/predicciones",
        "/api/pronosticos",
    ):
        respuesta = cliente.get(ruta) if ruta != "/api/predicciones" else cliente.post(ruta, json={})
        assert respuesta.status_code == 404


def test_checkpoint_invalido_no_impide_iniciar_y_responde_503(tmp_path):
    app = crear_app(
        ruta_catalogo=tmp_path / "catalogo.csv",
        ruta_checkpoint=tmp_path / "checkpoint.pt",
    )
    app.config["TESTING"] = True

    respuesta = app.test_client().get("/api/synthetic-v2/opciones")

    assert respuesta.status_code == 503
    assert respuesta.json == {"error": "El modelo sintético no está disponible."}
    assert str(tmp_path) not in respuesta.get_data(as_text=True)
    prediccion = app.test_client().post(
        "/api/synthetic-v2/predicciones",
        json={
            "tienda_id": 1,
            "producto_id": "BEB-001",
            "fecha": "2026-11-17",
        },
    )
    assert prediccion.status_code == 503
    assert prediccion.json == {
        "error": "El checkpoint sintético no está disponible o es inválido"
    }


def test_catalogo_invalido_responde_503_sin_ocultar_el_inicio(tmp_path):
    app = crear_app(
        ruta_catalogo=tmp_path / "catalogo-inexistente.csv",
        ruta_checkpoint=tmp_path / "checkpoint-no-consultado.pt",
    )
    app.config["TESTING"] = True
    cliente_local = app.test_client()

    assert cliente_local.get("/").status_code == 200
    respuesta = cliente_local.get("/api/synthetic-v2/productos")
    assert respuesta.status_code == 503
    assert respuesta.json == {"error": "El catálogo sintético no está disponible."}
    assert str(tmp_path) not in respuesta.get_data(as_text=True)


def test_frontend_usa_catalogo_y_api_sinteticos(cliente):
    pagina = cliente.get("/predicciones").get_data(as_text=True)
    assert 'id="store-select"' in pagina
    assert 'id="product-select"' in pagina
    assert 'id="date-input"' in pagina
    assert 'id="strategy-select"' in pagina
    assert 'id="model-select"' not in pagina
    assert 'id="forecast-form"' not in pagina
    assert 'id="forecast-origin"' not in pagina
    assert "familia" not in pagina.lower()
    assert "Favorita" not in pagina

    javascript = cliente.get("/static/app.js").get_data(as_text=True)
    assert 'fetch("/api/synthetic-v2/opciones")' in javascript
    assert 'fetch("/api/synthetic-v2/productos")' in javascript
    assert '"/api/synthetic-v2/reposicion/estrategias"' in javascript
    assert '"/api/synthetic-v2/reposiciones"' in javascript
    assert '"/api/synthetic-v2/predicciones"' in javascript
    assert "producto_id: producto.value" in javascript
    assert "solicitud.estrategia = estrategia.value" in javascript
    assert "/api/pronosticos" not in javascript
    assert "/api/opciones?" not in javascript

    presentacion = cliente.get("/static/presentacion.js").get_data(as_text=True)
    assert "BEAUTY" not in presentacion
    assert "nombreFamilia" not in presentacion


def test_activos_y_cabeceras(cliente):
    for ruta in (
        "/static/estilos.css",
        "/static/app.js",
        "/static/presentacion.js",
        "/static/tema.js",
        "/static/favicon.svg",
    ):
        respuesta = cliente.get(ruta)
        assert respuesta.status_code == 200
        assert respuesta.headers["X-Content-Type-Options"] == "nosniff"
        assert "frame-ancestors 'none'" in respuesta.headers["Content-Security-Policy"]


def test_prediccion_web_usa_el_checkpoint_real_y_no_un_valor_fijo():
    if (
        not RUTA_CATALOGO.is_file()
        or not RUTA_CHECKPOINT.is_file()
        or not RUTA_INVENTARIO.is_file()
    ):
        pytest.skip("Se requieren los datos y el checkpoint sintéticos locales.")
    servicio = ServicioPrediccion(
        RepositorioProductosCSV(RUTA_CATALOGO),
        AdaptadorPrediccionPyTorch(RUTA_CHECKPOINT),
        hoy=lambda: FECHA_ACTUAL,
    )
    app = crear_app(servicio_prediccion=servicio)
    app.config["TESTING"] = True
    cliente_real = app.test_client()
    opciones = cliente_real.get("/api/synthetic-v2/opciones")
    assert opciones.status_code == 200
    catalogo = cliente_real.get("/api/synthetic-v2/productos")
    assert catalogo.status_code == 200
    producto = next(p for p in catalogo.json["productos"] if p["producto_id"] == "BEB-002")
    fecha = "2026-11-25"
    consulta = {"tienda_id": 2, "producto_id": producto["producto_id"], "fecha": fecha}

    respuesta = cliente_real.post("/api/synthetic-v2/predicciones", json=consulta)

    assert respuesta.status_code == 200
    prediccion_directa = predecir_venta(
        RUTA_CHECKPOINT,
        tienda_id=consulta["tienda_id"],
        producto_id=consulta["producto_id"],
        fecha=fecha,
    )
    assert respuesta.json["unidades_estimadas"] == prediccion_directa["unidades_estimadas"]
    assert isinstance(respuesta.json["unidades_estimadas"], float)
    assert respuesta.json["datos_sinteticos"] is True

    recomendacion = cliente_real.post(
        "/api/synthetic-v2/reposiciones",
        json={
            "tienda_id": 2,
            "producto_id": "BEB-002",
            "fecha": "2026-11-25",
            "estrategia": "basica",
        },
    )
    assert recomendacion.status_code == 200
    datos = recomendacion.get_json()
    assert datos["producto"]["producto_id"] == "BEB-002"
    assert datos["tienda_id"] == 2
    assert datos["periodo_cobertura"]["desde"] == "2026-11-25"
    assert datos["periodo_cobertura"]["dias"] == datos["inventario"]["tiempo_reposicion"]
    assert len(datos["demanda_diaria"]) == datos["inventario"]["tiempo_reposicion"]
    assert isinstance(datos["demanda_acumulada"], float)
    assert isinstance(datos["cantidad_recomendada"], int)
    pronosticos_esperados = [
        servicio.predecir(
            tienda_id=2,
            producto_id="BEB-002",
            fecha=item["fecha"],
        ).unidades_estimadas
        for item in datos["demanda_diaria"]
    ]
    assert [
        item["unidades_estimadas"] for item in datos["demanda_diaria"]
    ] == pronosticos_esperados
    assert datos["demanda_fecha_seleccionada"] == pronosticos_esperados[0]
    assert math.isclose(datos["demanda_acumulada"], math.fsum(pronosticos_esperados))
    assert datos["cantidad_recomendada"] == math.ceil(
        max(
            0,
            datos["demanda_acumulada"] - datos["inventario"]["stock_actual"],
        )
    )

    recomendacion_seguridad = cliente_real.post(
        "/api/synthetic-v2/reposiciones",
        json={
            "tienda_id": 1,
            "producto_id": "BEB-001",
            "fecha": "2026-11-25",
            "estrategia": "seguridad",
        },
    )
    assert recomendacion_seguridad.status_code == 200
    datos_seguridad = recomendacion_seguridad.get_json()
    cantidad_esperada = math.ceil(
        max(
            0,
            datos_seguridad["demanda_acumulada"]
            + datos_seguridad["inventario"]["stock_minimo"]
            - datos_seguridad["inventario"]["stock_actual"],
        )
    )
    assert datos_seguridad["cantidad_recomendada"] == cantidad_esperada
    assert cantidad_esperada > 0
