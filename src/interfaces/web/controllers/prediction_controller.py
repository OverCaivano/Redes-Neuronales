"""Blueprint HTTP independiente para el caso de uso sintético."""

from flask import Blueprint, current_app, jsonify, request

from src.domain.exceptions import (
    ErrorCatalogo,
    ErrorInferencia,
    ProductoNoEncontrado,
    SolicitudPrediccionInvalida,
    TiendaNoEncontrada,
)


def crear_blueprint_prediccion(servicio):
    blueprint = Blueprint(
        "prediccion_sintetica_v2",
        __name__,
        url_prefix="/api/synthetic-v2",
    )

    @blueprint.get("/opciones")
    def obtener_opciones():
        try:
            opciones = servicio.obtener_opciones()
        except ErrorInferencia:
            current_app.logger.exception("No se pudo cargar el checkpoint sintético")
            return jsonify(error="El modelo sintético no está disponible."), 503
        return jsonify(**opciones, datos_sinteticos=True)

    @blueprint.get("/productos")
    def listar_productos():
        try:
            productos = servicio.listar_productos()
        except ErrorCatalogo:
            current_app.logger.exception("No se pudo cargar el catálogo sintético")
            return jsonify(error="El catálogo sintético no está disponible."), 503
        return jsonify(productos=[
            {
                "producto_id": producto.producto_id,
                "nombre": producto.nombre,
                "categoria": producto.categoria,
            }
            for producto in productos
        ])

    @blueprint.post("/predicciones")
    def predecir():
        origen = request.headers.get("Origin")
        if origen and origen != request.host_url.rstrip("/"):
            return jsonify(error="Origen de solicitud no permitido."), 403
        if not request.is_json:
            return jsonify(error="El cuerpo debe ser JSON"), 415
        datos = request.get_json(silent=True)
        if not isinstance(datos, dict):
            return jsonify(error="El cuerpo debe ser un objeto JSON"), 400
        campos_requeridos = {"tienda_id", "producto_id", "fecha"}
        if set(datos) != campos_requeridos:
            return jsonify(
                error="Se requieren exactamente tienda_id, producto_id y fecha"
            ), 400
        try:
            resultado = servicio.predecir(
                tienda_id=datos["tienda_id"],
                producto_id=datos["producto_id"],
                fecha=datos["fecha"],
            )
        except SolicitudPrediccionInvalida as error:
            return jsonify(error=str(error)), 400
        except (ProductoNoEncontrado, TiendaNoEncontrada) as error:
            return jsonify(error=str(error)), 404
        except ErrorCatalogo:
            current_app.logger.exception("No se pudo consultar el catálogo sintético")
            return jsonify(error="El catálogo sintético no está disponible."), 503
        except ErrorInferencia as error:
            current_app.logger.exception("Falló la inferencia del modelo sintético")
            return jsonify(error=str(error)), 503
        return jsonify({
            "tienda_id": resultado.tienda_id,
            "producto": {
                "producto_id": resultado.producto.producto_id,
                "nombre": resultado.producto.nombre,
                "categoria": resultado.producto.categoria,
            },
            "fecha": resultado.fecha.isoformat(),
            "unidades_estimadas": resultado.unidades_estimadas,
            "datos_sinteticos": True,
        })

    return blueprint
