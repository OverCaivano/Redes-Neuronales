from datetime import date

from flask import Flask

from src.application.services.prediction_service import ServicioPrediccion
from src.domain.models import Producto
from src.interfaces.web.controllers.prediction_controller import (
    crear_blueprint_prediccion,
)


class RepositorioEnMemoria:
    def buscar_por_id(self, producto_id):
        if producto_id == "BEB-001":
            return Producto("BEB-001", "Agua mineral 500 ml", "bebidas")
        return None

    def listar(self):
        return (Producto("BEB-001", "Agua mineral 500 ml", "bebidas"),)


class MotorFalso:
    fecha_corte = date(2026, 11, 16)
    tiendas_disponibles = (1, 2, 3)

    def predecir(self, *, tienda_id, producto_id, fecha):
        return 12.5


def test_blueprint_expone_endpoints_sin_montarse_en_la_app_heredada():
    servicio = ServicioPrediccion(
        RepositorioEnMemoria(),
        MotorFalso(),
        hoy=lambda: date(2026, 11, 17),
    )
    app = Flask("test_blueprint_independiente")
    app.register_blueprint(crear_blueprint_prediccion(servicio))
    cliente = app.test_client()

    respuesta = cliente.get("/api/synthetic-v2/productos")
    assert respuesta.status_code == 200
    assert len(respuesta.get_json()["productos"]) == 1

    respuesta = cliente.post(
        "/api/synthetic-v2/predicciones",
        json={
            "tienda_id": 1,
            "producto_id": "BEB-001",
            "fecha": "2026-11-17",
        },
    )
    assert respuesta.status_code == 200
    assert respuesta.get_json()["datos_sinteticos"] is True
    assert respuesta.get_json()["unidades_estimadas"] == 12.5

    respuesta = cliente.post(
        "/api/synthetic-v2/predicciones",
        json={"tienda_id": 1, "producto_id": "NO-EXISTE", "fecha": "2026-11-17"},
    )
    assert respuesta.status_code == 404

    respuesta = cliente.post(
        "/api/synthetic-v2/predicciones",
        data="{",
        content_type="application/json",
    )
    assert respuesta.status_code == 400
