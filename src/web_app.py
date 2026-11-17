"""Aplicación web StockPredict IA para predicciones sintéticas por producto."""

import argparse
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from src.composition import (
    RUTA_CATALOGO,
    RUTA_CHECKPOINT,
    crear_servicio_prediccion,
)
from src.interfaces.web.controllers.prediction_controller import (
    crear_blueprint_prediccion,
)


RAIZ = Path(__file__).resolve().parents[1]
PAGINAS = {
    "inicio": "Inicio",
    "predicciones": "Predicciones",
    "evaluacion": "Evaluación",
    "historial": "Historial",
    "acerca": "Acerca del proyecto",
}


def crear_app(
    servicio_prediccion=None,
    *,
    ruta_catalogo=RUTA_CATALOGO,
    ruta_checkpoint=RUTA_CHECKPOINT,
):
    app = Flask(
        __name__,
        template_folder=str(RAIZ / "web/templates"),
        static_folder=str(RAIZ / "web/static"),
    )
    app.config.update(
        MAX_CONTENT_LENGTH=4096,
        TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"],
    )
    if servicio_prediccion is None:
        servicio_prediccion = crear_servicio_prediccion(
            ruta_catalogo,
            ruta_checkpoint,
        )
    app.extensions["servicio_prediccion"] = servicio_prediccion
    app.register_blueprint(crear_blueprint_prediccion(servicio_prediccion))

    @app.after_request
    def cabeceras(respuesta):
        respuesta.headers["X-Content-Type-Options"] = "nosniff"
        respuesta.headers["Referrer-Policy"] = "no-referrer"
        respuesta.headers["Cache-Control"] = "no-store"
        respuesta.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; "
            "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; "
            "form-action 'self'"
        )
        return respuesta

    @app.get("/")
    @app.get("/<pagina>")
    def pagina_web(pagina="inicio"):
        if pagina not in PAGINAS:
            return render_template(
                "error.html",
                titulo="Página no encontrada",
                pagina="",
                paginas=PAGINAS,
            ), 404
        return render_template(
            pagina + ".html",
            pagina=pagina,
            titulo=PAGINAS[pagina],
            paginas=PAGINAS,
        )

    @app.errorhandler(HTTPException)
    def error_http(error):
        if request.path.startswith("/api/"):
            return jsonify(error="Solicitud no admitida."), error.code
        return render_template(
            "error.html",
            titulo="Solicitud no admitida",
            pagina="",
            paginas=PAGINAS,
        ), error.code

    return app


def main():
    parser = argparse.ArgumentParser(
        description="StockPredict IA, aplicación local de ventas sintéticas"
    )
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    crear_app().run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
