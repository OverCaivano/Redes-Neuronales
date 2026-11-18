"""Blueprint HTTP para recomendar reposición sin modificar existencias."""

from flask import Blueprint, current_app, jsonify, request

from src.domain.exceptions import (
    ErrorInferencia,
    ErrorInventario,
    EstrategiaReposicionNoEncontrada,
    InventarioNoEncontrado,
    ProductoNoEncontrado,
    SolicitudPrediccionInvalida,
    TiendaNoEncontrada,
)


def crear_blueprint_reposicion(servicio):
    blueprint = Blueprint(
        "reposicion_sintetica",
        __name__,
        url_prefix="/api/synthetic-v2",
    )

    @blueprint.get("/reposicion/horizonte")
    def obtener_horizonte():
        if set(request.args) != {"tienda_id", "producto_id"}:
            return jsonify(
                error="Se requieren exactamente tienda_id y producto_id"
            ), 400
        try:
            tienda_id = int(request.args["tienda_id"])
        except ValueError:
            return jsonify(error="tienda_id debe ser un entero"), 400
        try:
            horizonte = servicio.obtener_horizonte_reposicion(
                tienda_id=tienda_id,
                producto_id=request.args["producto_id"],
            )
        except SolicitudPrediccionInvalida as error:
            return jsonify(error=str(error)), 400
        except (TiendaNoEncontrada, InventarioNoEncontrado) as error:
            return jsonify(error=str(error)), 404
        except ErrorInventario:
            current_app.logger.exception("No se pudo consultar el inventario sintético")
            return jsonify(error="El inventario sintético no está disponible."), 503
        return jsonify(horizonte)

    @blueprint.get("/reposicion/estrategias")
    def listar_estrategias():
        return jsonify(estrategias=servicio.listar_estrategias())

    @blueprint.post("/reposiciones")
    def recomendar():
        origen = request.headers.get("Origin")
        if origen and origen != request.host_url.rstrip("/"):
            return jsonify(error="Origen de solicitud no permitido."), 403
        if not request.is_json:
            return jsonify(error="El cuerpo debe ser JSON"), 415
        datos = request.get_json(silent=True)
        campos_requeridos = {
            "tienda_id",
            "producto_id",
            "fecha",
            "estrategia",
        }
        if not isinstance(datos, dict) or set(datos) != campos_requeridos:
            return jsonify(
                error=(
                    "Se requieren exactamente tienda_id, producto_id, fecha "
                    "y estrategia"
                )
            ), 400
        try:
            resultado = servicio.recomendar(
                tienda_id=datos["tienda_id"],
                producto_id=datos["producto_id"],
                fecha=datos["fecha"],
                estrategia=datos["estrategia"],
            )
        except (SolicitudPrediccionInvalida, EstrategiaReposicionNoEncontrada) as error:
            return jsonify(error=str(error)), 400
        except (
            ProductoNoEncontrado,
            TiendaNoEncontrada,
            InventarioNoEncontrado,
        ) as error:
            return jsonify(error=str(error)), 404
        except (ErrorInferencia, ErrorInventario) as error:
            current_app.logger.exception("No se pudo calcular la recomendación sintética")
            return jsonify(error=str(error)), 503
        return jsonify({
            "tienda_id": resultado.tienda_id,
            "producto": {
                "producto_id": resultado.producto.producto_id,
                "nombre": resultado.producto.nombre,
                "categoria": resultado.producto.categoria,
            },
            "fecha": resultado.fecha_inicio.isoformat(),
            "demanda_fecha_seleccionada": resultado.demanda_fecha_seleccionada,
            "periodo_cobertura": {
                "desde": resultado.fecha_inicio.isoformat(),
                "hasta": resultado.fecha_fin.isoformat(),
                "dias": resultado.inventario.tiempo_reposicion,
            },
            "demanda_diaria": [
                {
                    "fecha": prediccion.fecha.isoformat(),
                    "unidades_estimadas": prediccion.unidades_estimadas,
                }
                for prediccion in resultado.predicciones
            ],
            "demanda_acumulada": resultado.demanda_acumulada,
            "inventario": {
                "stock_actual": resultado.inventario.stock_actual,
                "stock_minimo": resultado.inventario.stock_minimo,
                "tiempo_reposicion": resultado.inventario.tiempo_reposicion,
            },
            "estrategia": {
                "codigo": resultado.estrategia_id,
                "nombre": resultado.estrategia_nombre,
            },
            "cantidad_recomendada": resultado.cantidad_recomendada,
            "requiere_reposicion": resultado.requiere_reposicion,
            "mensaje": resultado.mensaje,
            "datos_sinteticos": True,
        })

    return blueprint
