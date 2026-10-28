"""StockPredict IA: interfaz y API local, sin entrenar ni servir artefactos privados."""

import argparse
import csv
import json
import math
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from src.datos_favorita import leer_datos_favorita
from src.evaluacion_favorita import cargar_best_favorita
from src.predecir_favorita import predecir_favorita, validar_consulta
from src.preprocesamiento_temporal import generar_variables_historicas
from src.reanudacion import huella_dataset

RAIZ = Path(__file__).resolve().parents[1]
PAGINAS = {
    "inicio": "Inicio", "predicciones": "Predicciones",
    "evaluacion": "Evaluación del modelo", "historial": "Historial de entrenamientos",
    "acerca": "Acerca del proyecto",
}


def numero(valor):
    if isinstance(valor, bool) or not isinstance(valor, (float, int)) or not math.isfinite(valor):
        raise ValueError("Métrica inválida")
    return valor


class ServicioPiloto:
    """Rutas fijadas por el servidor; ninguna ruta proviene de una solicitud HTTP."""
    def __init__(self, raiz=RAIZ):
        self.raiz = Path(raiz)
        self.checkpoint = self.raiz / "runs/favorita_subconjunto_reanudado_4/best.pt"
        self.csv = self.raiz / "data/raw/train.csv"
        self.evaluacion = self.raiz / "runs/evaluaciones/favorita_piloto_test/evaluacion.json"
        self._opciones = None
        self._firma = None
        self._lock = Lock()

    def opciones(self):
        with self._lock:
            firma = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in (self.checkpoint, self.csv))
            if self._opciones is None or self._firma != firma:
                estado = cargar_best_favorita(self.checkpoint)
                if huella_dataset(self.csv) != estado["dataset_sha256"]:
                    raise ValueError("Dataset incompatible")
                cfg = estado["config"]["data"]
                datos = leer_datos_favorita(self.csv, tiendas=cfg.get("tiendas"), familias=cfg.get("familias"))
                validos = generar_variables_historicas(datos).datos
                series = [{"tienda": int(t), "familia": str(f),
                           "fechas": grupo.fecha.dt.strftime("%Y-%m-%d").tolist()}
                          for (t, f), grupo in validos.groupby(["tienda", "familia"], observed=True, sort=True)]
                if not series:
                    raise ValueError("No hay consultas compatibles")
                self._opciones = {"series": series, "epoca": estado["epoch"],
                                  "input_dim": estado["input_dim"], "modelo": "Piloto local · MLP"}
                self._firma = firma
            return self._opciones

    def predecir(self, tienda, familia, fecha):
        opciones = self.opciones()
        if not any(s["tienda"] == tienda and s["familia"] == familia and fecha in s["fechas"]
                   for s in opciones["series"]):
            raise LookupError("La combinación o fecha no tiene historia compatible con el piloto.")
        # La función CLI/API existente vuelve a verificar huella y ejecuta un único forward.
        with self._lock:
            resultado = predecir_favorita(self.checkpoint, tienda=tienda, familia=familia,
                                         fecha=fecha, data_path=self.csv)
        return {k: resultado[k] for k in ("tienda", "familia", "fecha", "ventas_estimadas", "epoca")}

    def resumen(self):
        corridas, advertencias = [], []
        for carpeta, etiqueta in (("favorita_subconjunto", "Piloto inicial"),
                                  ("favorita_subconjunto_reanudado_4", "Piloto reanudado")):
            directorio = self.raiz / "runs" / carpeta
            try:
                datos = json.loads((directorio / "resumen.json").read_text(encoding="utf-8"))
                with (directorio / "metricas.csv").open(encoding="utf-8", newline="") as archivo:
                    historial = [{k: numero(float(f[k])) for k in
                                  ("epoca", "train_mse", "val_mse", "val_mae", "val_rmse")}
                                 for f in csv.DictReader(archivo)]
                corridas.append({"nombre": etiqueta, "epocas": numero(datos["epocas_completadas"]),
                                 "mejor_epoca": numero(datos["mejor_epoca"]),
                                 "mae": numero(datos["mejor_val_mae"]), "rmse": numero(datos["mejor_val_rmse"]),
                                 "input_dim": numero(datos["input_dim"]), "historial": historial,
                                 "filas": {k: numero(datos["filas_por_particion"][k]) for k in
                                           ("entrenamiento", "validacion", "test")}})
            except (OSError, ValueError, KeyError, TypeError):
                advertencias.append(f"Resultados no disponibles: {etiqueta}.")
        evaluacion = None
        try:
            datos = json.loads(self.evaluacion.read_text(encoding="utf-8"))
            if datos["checkpoint_sha256"] != huella_dataset(self.checkpoint):
                raise ValueError("Evaluación de otro checkpoint")
            mlp = {k: numero(datos["mlp"][k]) for k in ("mae", "rmse", "n")}
            baseline = {k: numero(datos["baseline_semanal"][k]) for k in ("mae", "rmse", "n")}
            if mlp["n"] != baseline["n"] or mlp["n"] <= 0 or baseline["rmse"] <= 0:
                raise ValueError("Conteos o métricas incompatibles")
            evaluacion = {"mlp": mlp, "baseline": baseline,
                          "mejora": 100 * (1 - mlp["rmse"] / baseline["rmse"]),
                          "inicio": datos["fecha_min"], "fin": datos["fecha_max"]}
        except (OSError, ValueError, KeyError, TypeError):
            advertencias.append("Evaluación local no disponible o incompatible con el checkpoint.")
        return {"corridas": corridas, "evaluacion": evaluacion, "advertencias": advertencias,
                "drive": "No verificado: esta aplicación no accede a Google Drive."}


def crear_app(servicio=None):
    app = Flask(__name__, template_folder=str(RAIZ / "web/templates"), static_folder=str(RAIZ / "web/static"))
    app.config.update(MAX_CONTENT_LENGTH=4096, TRUSTED_HOSTS=["localhost", "127.0.0.1", "[::1]"])
    servicio = servicio if servicio is not None else ServicioPiloto()
    app.extensions["piloto"] = servicio

    @app.after_request
    def cabeceras(respuesta):
        respuesta.headers["X-Content-Type-Options"] = "nosniff"
        respuesta.headers["Referrer-Policy"] = "no-referrer"
        respuesta.headers["Cache-Control"] = "no-store"
        respuesta.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
            "font-src 'self' https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
            "frame-ancestors 'none'; base-uri 'self'; form-action 'self'")
        return respuesta

    @app.get("/")
    @app.get("/<pagina>")
    def pagina_web(pagina="inicio"):
        if pagina not in PAGINAS:
            return render_template("error.html", titulo="Página no encontrada", pagina="", paginas=PAGINAS), 404
        return render_template(pagina + ".html", pagina=pagina, titulo=PAGINAS[pagina],
                               paginas=PAGINAS, resumen=servicio.resumen())

    @app.get("/api/opciones")
    def opciones():
        try:
            return jsonify(servicio.opciones())
        except Exception:
            app.logger.exception("No se pudo preparar el piloto local")
            return jsonify(error="Piloto no disponible. Revisá los archivos locales y el entorno en la consola."), 503

    @app.get("/api/resumen")
    def resumen():
        return jsonify(servicio.resumen())

    @app.post("/api/predicciones")
    def prediccion():
        origen = request.headers.get("Origin")
        if origen and origen != request.host_url.rstrip("/"):
            return jsonify(error="Origen de solicitud no permitido."), 403
        if not request.is_json:
            return jsonify(error="Enviar JSON con tienda, familia y fecha."), 415
        datos = request.get_json(silent=True)
        if not isinstance(datos, dict) or set(datos) != {"tienda", "familia", "fecha"}:
            return jsonify(error="Se requieren únicamente tienda, familia y fecha."), 400
        try:
            validar_consulta(**datos)
        except ValueError as error:
            return jsonify(error=str(error)), 400
        try:
            resultado = servicio.predecir(**datos)
            return jsonify(**resultado, modelo="Piloto local · best.pt", aviso=
                           "Ventas estimadas por día y familia; no representan stock ni recomendaciones de compra.")
        except LookupError:
            return jsonify(error="Tienda, familia o fecha sin historia compatible con el piloto."), 422
        except Exception:
            app.logger.exception("Error de inferencia local")
            return jsonify(error="No se pudo generar la predicción. Revisá el checkpoint y el dataset en la consola."), 503

    @app.errorhandler(HTTPException)
    def error_http(error):
        if request.path.startswith("/api/"):
            return jsonify(error="Solicitud no admitida."), error.code
        return render_template("error.html", titulo="Solicitud no admitida", pagina="", paginas=PAGINAS), error.code

    return app


def main():
    parser = argparse.ArgumentParser(description="StockPredict IA, aplicación exclusivamente local")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()
    crear_app().run(host="127.0.0.1", port=args.port, debug=False)


if __name__ == "__main__":
    main()
