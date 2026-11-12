"""StockPredict IA: interfaz y API local, sin entrenar ni servir artefactos privados."""

import argparse
import csv
import json
import math
from datetime import date, timedelta
from pathlib import Path
from threading import Lock

import yaml
from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from src.datos_favorita import leer_datos_favorita
from src.evaluacion_favorita import cargar_best_favorita
from src.predecir_favorita import predecir_favorita, validar_consulta
from src.preprocesamiento_temporal import generar_variables_historicas
from src.pronosticar_favorita import leer_promociones_favorita, pronosticar_favorita
from src.reanudacion import huella_dataset
from src.sincronizacion import ARCHIVOS as ARCHIVOS_CORRIDA

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


COLUMNAS_METRICAS_EPOCA = (
    "epoca", "train_mse", "train_mae", "train_rmse", "val_mse", "val_mae", "val_rmse",
    "baseline_mae", "baseline_rmse", "n_train", "n_validacion", "learning_rate", "segundos_epoca",
)
COLUMNAS_ENTERAS_EPOCA = {"epoca", "n_train", "n_validacion"}


def leer_metricas_epoca(ruta, epocas_esperadas):
    with Path(ruta).open(encoding="utf-8", newline="") as archivo:
        lector = csv.DictReader(archivo)
        if not lector.fieldnames or not set(COLUMNAS_METRICAS_EPOCA).issubset(lector.fieldnames):
            raise ValueError("CSV de métricas incompleto")
        historial = []
        for fila in lector:
            if not fila or all(valor in (None, "") for valor in fila.values()):
                continue
            valores = {}
            for campo in COLUMNAS_METRICAS_EPOCA:
                valor = numero(float(fila[campo]))
                if campo in COLUMNAS_ENTERAS_EPOCA:
                    if not valor.is_integer():
                        raise ValueError("Conteo o época no entero")
                    valor = int(valor)
                valores[campo] = valor
            historial.append(valores)
    if len(historial) != epocas_esperadas or [fila["epoca"] for fila in historial] != list(
        range(1, epocas_esperadas + 1)
    ):
        raise ValueError("El historial no contiene todas las épocas consecutivas")
    return historial


def verificar_manifiesto_historial(directorio):
    ruta_manifiesto = Path(directorio) / "manifiesto.json"
    if (Path(directorio) / "COMPLETO").read_text(encoding="utf-8").strip() != huella_dataset(
        ruta_manifiesto
    ):
        raise ValueError("Marca del manifiesto incorrecta")
    manifiesto = json.loads(ruta_manifiesto.read_text(encoding="utf-8"))
    if (
        manifiesto["version"] != 1
        or set(manifiesto["archivos"]) != set(ARCHIVOS_CORRIDA)
        or not all(clave in manifiesto for clave in ("identidad", "entorno", "codigo_sha256"))
    ):
        raise ValueError("Manifiesto del entrenamiento incompleto")
    for nombre in ("best.pt", "configuracion.yaml", "resumen.json"):
        archivo = Path(directorio) / nombre
        registrado = manifiesto["archivos"][nombre]
        if archivo.stat().st_size != registrado["bytes"] or huella_dataset(archivo) != registrado["sha256"]:
            raise ValueError(f"Artefacto del entrenamiento incompatible: {nombre}")
    return manifiesto


class ServicioPiloto:
    """Rutas fijadas por el servidor; ninguna ruta proviene de una solicitud HTTP."""
    def __init__(self, raiz=RAIZ):
        self.raiz = Path(raiz)
        self.checkpoints = {
            "piloto": self.raiz / "runs/favorita_subconjunto_reanudado_4/best.pt",
            "completo": self.raiz / "runs/favorita_completa_20/best.pt",
        }
        self.checkpoint = self.checkpoints["piloto"]
        self.csv = self.raiz / "data/raw/train.csv"
        self.promociones = self.raiz / "data/raw/test.csv"
        self.evaluacion = self.raiz / "runs/evaluaciones/favorita_piloto_test/evaluacion.json"
        self.evaluacion_completa = self.raiz / "runs/evaluacion_completa_20/evaluacion.json"
        self.backtest_completo = (
            self.raiz / "runs/evaluaciones/pronostico_recursivo_validacion_20261111_final/informe.json"
        )
        self._opciones = {}
        self._firma = {}
        self._lock = Lock()

    def modelos(self):
        return [
            {"id": "piloto", "nombre": "Piloto · 13 características"},
            {"id": "completo", "nombre": "Completo · 95 características"},
        ]

    def opciones(self, modelo="piloto"):
        if modelo not in self.checkpoints:
            raise ValueError("Modelo no admitido")
        with self._lock:
            checkpoint = self.checkpoints[modelo]
            firma = tuple((p.stat().st_size, p.stat().st_mtime_ns) for p in (checkpoint, self.csv))
            if self._opciones.get(modelo) is None or self._firma.get(modelo) != firma:
                estado = cargar_best_favorita(checkpoint)
                if huella_dataset(self.csv) != estado["dataset_sha256"]:
                    raise ValueError("Dataset incompatible")
                cfg = estado["config"]["data"]
                if modelo == "piloto":
                    datos = leer_datos_favorita(
                        self.csv, tiendas=cfg.get("tiendas"), familias=cfg.get("familias")
                    )
                    validos = generar_variables_historicas(datos).datos
                    series = [
                        {"tienda": int(t), "familia": str(f),
                         "fechas": grupo.fecha.dt.strftime("%Y-%m-%d").tolist()}
                        for (t, f), grupo in validos.groupby(
                            ["tienda", "familia"], observed=True, sort=True
                        )
                    ]
                else:
                    categorias = estado["preparador"].codificador.categorias
                    inicio = estado["periodos"]["entrenamiento"][0]
                    fecha_min = (date.fromisoformat(inicio) + timedelta(days=7)).isoformat()
                    fecha_max = estado["periodos"]["test"][1]
                    series = [
                        {"tienda": int(tienda), "familia": str(familia),
                         "fecha_min": fecha_min, "fecha_max": fecha_max}
                        for tienda in categorias["tienda"]
                        for familia in categorias["familia"]
                    ]
                if not series:
                    raise ValueError("No hay consultas compatibles")
                self._opciones[modelo] = {
                    "series": series, "epoca": estado["epoch"], "input_dim": estado["input_dim"],
                    "modelo": "Piloto local · MLP" if modelo == "piloto" else "Modelo completo · MLP",
                    "modelo_id": modelo, "modelos": self.modelos(),
                }
                if modelo == "completo":
                    self._opciones[modelo]["fecha_max_historica"] = fecha_max
                self._firma[modelo] = firma
            return self._opciones[modelo]

    def predecir(self, tienda, familia, fecha, modelo="piloto"):
        opciones = self.opciones(modelo)
        if modelo == "piloto":
            if not any(
                s["tienda"] == tienda and s["familia"] == familia and fecha in s["fechas"]
                for s in opciones["series"]
            ):
                raise LookupError("La combinación o fecha no tiene historia compatible con el piloto.")
        elif not any(
            s["tienda"] == tienda and s["familia"] == familia for s in opciones["series"]
        ):
            raise LookupError("Tienda o familia no compatible con el modelo completo.")
        # La función CLI/API existente vuelve a verificar huella y ejecuta un único forward.
        with self._lock:
            try:
                resultado = predecir_favorita(
                    self.checkpoints[modelo], tienda=tienda, familia=familia,
                    fecha=fecha, data_path=self.csv
                )
            except ValueError as error:
                if "unica fila" in str(error) or "Historia insuficiente" in str(error):
                    raise LookupError(
                        "La fecha no tiene una fila válida y siete días previos observados."
                    ) from None
                raise
        return {k: resultado[k] for k in ("tienda", "familia", "fecha", "ventas_estimadas", "epoca")}

    def pronosticar(self, tienda, familia, fecha_origen, horizonte=7):
        opciones = self.opciones("completo")
        if not any(
            s["tienda"] == tienda and s["familia"] == familia for s in opciones["series"]
        ):
            raise LookupError("Tienda o familia no compatible con el modelo completo.")
        with self._lock:
            estado = cargar_best_favorita(self.checkpoints["completo"])
            if huella_dataset(self.csv) != estado["dataset_sha256"]:
                raise ValueError("Dataset incompatible")
            historial = leer_datos_favorita(
                self.csv, tiendas=[tienda], familias=[familia]
            )
            promociones = leer_promociones_favorita(self.promociones)
            return pronosticar_favorita(
                self.checkpoints["completo"], historial, promociones,
                tienda=tienda, familia=familia, fecha_origen=fecha_origen,
                horizonte=horizonte,
            )

    def resumen(self):
        corridas, advertencias = [], []
        for carpeta, etiqueta in (("favorita_subconjunto", "Piloto inicial"),
                                  ("favorita_subconjunto_reanudado_4", "Piloto reanudado")):
            directorio = self.raiz / "runs" / carpeta
            try:
                datos = json.loads((directorio / "resumen.json").read_text(encoding="utf-8"))
                historial = leer_metricas_epoca(
                    directorio / "metricas.csv", int(datos["epocas_completadas"])
                )
                corridas.append({"nombre": etiqueta, "epocas": numero(datos["epocas_completadas"]),
                                 "mejor_epoca": numero(datos["mejor_epoca"]),
                                 "mae": numero(datos["mejor_val_mae"]), "rmse": numero(datos["mejor_val_rmse"]),
                                 "input_dim": numero(datos["input_dim"]), "historial": historial,
                                 "filas": {k: numero(datos["filas_por_particion"][k]) for k in
                                           ("entrenamiento", "validacion", "test")}})
            except (OSError, ValueError, KeyError, TypeError):
                advertencias.append(f"Resultados no disponibles: {etiqueta}.")
        entrenamiento_completo = None
        directorio = self.raiz / "runs/favorita_completa_20"
        try:
            datos = json.loads((directorio / "resumen.json").read_text(encoding="utf-8"))
            configuracion = yaml.safe_load(
                (directorio / "configuracion.yaml").read_text(encoding="utf-8")
            )
            manifiesto = verificar_manifiesto_historial(directorio)
            modelo = configuracion["model"]
            entrenamiento = configuracion["training"]
            epocas = int(datos["epocas_completadas"])
            if (
                manifiesto["epoca"] != epocas
                or epocas != entrenamiento["epochs"]
                or datos["input_dim"] != modelo["input_dim"]
                or datos["mejor_epoca"] > epocas
                or datos["filas_por_particion"]["test"] <= 0
            ):
                raise ValueError("Resumen, configuración y manifiesto incompatibles")
            historial = []
            historial_verificado = False
            try:
                ruta_metricas = directorio / "metricas.csv"
                registrado = manifiesto["archivos"]["metricas.csv"]
                if (
                    ruta_metricas.stat().st_size != registrado["bytes"]
                    or huella_dataset(ruta_metricas) != registrado["sha256"]
                ):
                    raise ValueError("CSV de métricas incompatible con su manifiesto")
                historial = leer_metricas_epoca(ruta_metricas, epocas)
                historial_verificado = True
            except (OSError, ValueError, KeyError, TypeError):
                advertencias.append("Métricas por época del modelo completo ausentes o incompletas.")
            entrenamiento_completo = {
                "nombre": "Modelo completo",
                "checkpoint": "runs/favorita_completa_20/best.pt",
                "epocas": epocas,
                "mejor_epoca": int(datos["mejor_epoca"]),
                "input_dim": int(modelo["input_dim"]),
                "hidden_dim": int(modelo["hidden_dim"]),
                "output_dim": int(modelo["output_dim"]),
                "semilla": int(configuracion["seed"]),
                "device": configuracion["device"],
                "batch_size": int(entrenamiento["batch_size"]),
                "learning_rate": numero(entrenamiento["learning_rate"]),
                "num_workers": int(entrenamiento["num_workers"]),
                "optimizador": entrenamiento.get("optimizer", configuracion.get("optimizer")),
                "perdida": entrenamiento.get("loss", configuracion.get("loss")),
                "mejor_val_mae": numero(datos["mejor_val_mae"]),
                "mejor_val_rmse": numero(datos["mejor_val_rmse"]),
                "filas": {k: numero(datos["filas_por_particion"][k]) for k in
                          ("entrenamiento", "validacion", "test")},
                "periodos": datos["periodos"],
                "historial": historial,
                "historial_verificado": historial_verificado,
            }
        except (OSError, ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError):
            advertencias.append("Historial del modelo completo no disponible o incompatible con sus artefactos.")
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
                          "inicio": datos["fecha_min"], "fin": datos["fecha_max"],
                          "epoca": numero(datos["epoca_seleccionada_por_validacion"]),
                          "checkpoint": "runs/favorita_subconjunto_reanudado_4/best.pt",
                          "particion": datos["particion"], "protocolo": datos["protocolo"],
                          "tiendas": datos["tiendas"], "familias": datos["familias"]}
        except (OSError, ValueError, KeyError, TypeError):
            advertencias.append("Evaluación local no disponible o incompatible con el checkpoint.")

        evaluacion_completa = None
        resumen_completo = None
        sha_checkpoint_completo = None
        try:
            resumen_completo = json.loads(
                (self.raiz / "runs/favorita_completa_20/resumen.json").read_text(encoding="utf-8")
            )
            sha_checkpoint_completo = huella_dataset(self.checkpoints["completo"])
            datos = json.loads(self.evaluacion_completa.read_text(encoding="utf-8"))
            filas_test = datos["filas_por_particion"]["test"]
            columnas = datos["nombres_columnas"]
            mlp = {k: numero(datos["mlp"][k]) for k in ("mae", "rmse", "n")}
            baseline = {k: numero(datos["baseline_semanal"][k]) for k in ("mae", "rmse", "n")}
            epoca = numero(datos["epoca_seleccionada_por_validacion"])
            input_dim = numero(datos["input_dim"])
            if (
                datos["checkpoint_sha256"] != sha_checkpoint_completo
                or datos["dataset_sha256"] != resumen_completo["dataset_sha256"]
                or datos["particion"] != "test_interno"
                or datos["protocolo"] != "diario_a_un_paso_historia_observada_hasta_t-1"
                or epoca != resumen_completo["mejor_epoca"]
                or input_dim != resumen_completo["input_dim"]
                or not isinstance(columnas, list)
                or len(columnas) != input_dim
                or any(not isinstance(nombre, str) for nombre in columnas)
                or filas_test != resumen_completo["filas_por_particion"]["test"]
                or mlp["n"] != baseline["n"]
                or mlp["n"] != filas_test
                or baseline["mae"] <= 0
                or baseline["rmse"] <= 0
            ):
                raise ValueError("Evaluación completa incompatible con checkpoint o partición")
            evaluacion_completa = {
                "mlp": mlp,
                "baseline": baseline,
                "mejora_mae": 100 * (1 - mlp["mae"] / baseline["mae"]),
                "mejora_rmse": 100 * (1 - mlp["rmse"] / baseline["rmse"]),
                "inicio": datos["fecha_min"],
                "fin": datos["fecha_max"],
                "epoca": epoca,
                "input_dim": input_dim,
                "checkpoint": "runs/favorita_completa_20/best.pt",
                "particion": datos["particion"],
                "protocolo": datos["protocolo"],
                "tiendas": sum(nombre.startswith("tienda_") for nombre in columnas),
                "familias": sum(nombre.startswith("familia_") for nombre in columnas),
            }
        except (OSError, ValueError, KeyError, TypeError):
            advertencias.append("Evaluación diaria del modelo completo no disponible o incompatible.")

        backtest_completo = None
        try:
            if resumen_completo is None or sha_checkpoint_completo is None:
                raise ValueError("Checkpoint completo no disponible")
            datos = json.loads(self.backtest_completo.read_text(encoding="utf-8"))
            general = datos["general"]
            mlp = {k: numero(general["mlp"][k]) for k in ("mae", "rmse", "n")}
            baseline = {k: numero(general["referencia_semanal"][k]) for k in ("mae", "rmse", "n")}
            filas_horizonte = []
            for fila in datos["por_horizonte"]:
                horizonte = fila["horizonte"]
                metrica_mlp = {k: numero(fila["mlp"][k]) for k in ("mae", "rmse", "n")}
                metrica_baseline = {
                    k: numero(fila["referencia_semanal"][k]) for k in ("mae", "rmse", "n")
                }
                if (
                    metrica_mlp["n"] != metrica_baseline["n"]
                    or metrica_baseline["mae"] <= 0
                    or metrica_baseline["rmse"] <= 0
                    or metrica_mlp["mae"] < 0
                    or metrica_mlp["rmse"] < 0
                ):
                    raise ValueError("Métricas incompatibles por horizonte")
                filas_horizonte.append({
                    "horizonte": horizonte,
                    "mlp": metrica_mlp,
                    "baseline": metrica_baseline,
                    "mejora_mae": 100 * (1 - metrica_mlp["mae"] / metrica_baseline["mae"]),
                    "mejora_rmse": 100 * (1 - metrica_mlp["rmse"] / metrica_baseline["rmse"]),
                })
            origenes = datos["origenes_solicitados"]
            ventanas = datos["ventanas_completas"]
            pronosticos = datos["predicciones_evaluadas"]
            if (
                datos["checkpoint"]["sha256"] != sha_checkpoint_completo
                or datos["dataset_sha256"] != resumen_completo["dataset_sha256"]
                or datos["checkpoint"]["epoca"] != resumen_completo["mejor_epoca"]
                or datos["checkpoint"]["input_dim"] != resumen_completo["input_dim"]
                or datos["particion"] != "validacion"
                or datos["protocolo"] != "origen_movil_validacion_pronostico_recursivo_7_dias"
                or datos["test_preservado"] is not True
                or datos["horizonte_maximo"] != 7
                or ventanas != datos["ventanas_con_resultados"]
                or ventanas != datos["ventanas_completas"]
                or ventanas != len(origenes)
                or len(filas_horizonte) != 7
                or [fila["horizonte"] for fila in filas_horizonte] != list(range(1, 8))
                or mlp["n"] != baseline["n"]
                or mlp["n"] != pronosticos
                or sum(fila["mlp"]["n"] for fila in filas_horizonte) != pronosticos
                or mlp["mae"] < 0
                or mlp["rmse"] < 0
                or baseline["mae"] < 0
                or baseline["rmse"] < 0
                or baseline["mae"] <= 0
                or baseline["rmse"] <= 0
            ):
                raise ValueError("Backtest incompatible con checkpoint o alcance")
            backtest_completo = {
                "mlp": mlp,
                "baseline": baseline,
                "mejora_mae": 100 * (1 - mlp["mae"] / baseline["mae"]),
                "mejora_rmse": 100 * (1 - mlp["rmse"] / baseline["rmse"]),
                "inicio": datos["periodo_validacion"][0],
                "fin": datos["periodo_validacion"][1],
                "origen_inicio": origenes[0],
                "origen_fin": origenes[-1],
                "ventanas": ventanas,
                "series": datos["series_evaluadas"],
                "predicciones": pronosticos,
                "epoca": datos["checkpoint"]["epoca"],
                "input_dim": datos["checkpoint"]["input_dim"],
                "horizontes": filas_horizonte,
                "supuesto_promociones": datos["promociones"]["supuesto"],
            }
        except (OSError, ValueError, KeyError, TypeError, IndexError):
            advertencias.append("Backtest recursivo del modelo completo no disponible o incompatible.")
        return {
            "corridas": corridas,
            "entrenamiento_completo": entrenamiento_completo,
            "evaluacion": evaluacion,
            "evaluacion_completa": evaluacion_completa,
            "backtest_completo": backtest_completo,
            "advertencias": advertencias,
            "drive": "No verificado: esta aplicación no accede a Google Drive.",
        }


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
        modelo = request.args.get("modelo", "piloto")
        if modelo not in ("piloto", "completo"):
            return jsonify(error="Modelo no admitido."), 400
        try:
            resultado = servicio.opciones(modelo)
            return jsonify(**resultado)
        except Exception:
            app.logger.exception("No se pudo preparar el piloto local")
            return jsonify(error="Modelo no disponible. Revisá los artefactos locales y el entorno en la consola."), 503

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
        if not isinstance(datos, dict) or set(datos) not in (
            {"tienda", "familia", "fecha"}, {"tienda", "familia", "fecha", "modelo"}
        ):
            return jsonify(error="Se requieren tienda, familia y fecha; modelo es opcional."), 400
        modelo = datos.get("modelo", "piloto")
        if modelo not in ("piloto", "completo"):
            return jsonify(error="Modelo no admitido."), 400
        try:
            validar_consulta(tienda=datos["tienda"], familia=datos["familia"], fecha=datos["fecha"])
        except ValueError as error:
            return jsonify(error=str(error)), 400
        try:
            resultado = servicio.predecir(
                tienda=datos["tienda"], familia=datos["familia"], fecha=datos["fecha"], modelo=modelo
            )
            etiqueta = "Piloto local · best.pt" if modelo == "piloto" else "Completo · best.pt"
            return jsonify(**resultado, modelo=etiqueta, aviso=
                           "Ventas estimadas por día y familia; no representan stock ni recomendaciones de compra.")
        except LookupError as error:
            return jsonify(error=str(error)), 422
        except Exception:
            app.logger.exception("Error de inferencia local")
            return jsonify(error="No se pudo generar la predicción. Revisá el checkpoint y el dataset en la consola."), 503

    @app.post("/api/pronosticos")
    def pronostico():
        origen = request.headers.get("Origin")
        if origen and origen != request.host_url.rstrip("/"):
            return jsonify(error="Origen de solicitud no permitido."), 403
        if not request.is_json:
            return jsonify(error="Enviar JSON con tienda, familia, fecha_origen y horizonte."), 415
        datos = request.get_json(silent=True)
        if not isinstance(datos, dict) or set(datos) != {
            "tienda", "familia", "fecha_origen", "horizonte"
        }:
            return jsonify(error="Se requieren tienda, familia, fecha_origen y horizonte."), 400
        try:
            validar_consulta(
                tienda=datos["tienda"], familia=datos["familia"], fecha=datos["fecha_origen"]
            )
            horizonte = datos["horizonte"]
            if isinstance(horizonte, bool) or not isinstance(horizonte, int) or not 1 <= horizonte <= 7:
                raise ValueError("horizonte debe ser un entero entre 1 y 7")
        except ValueError as error:
            return jsonify(error=str(error)), 400
        try:
            resultado = servicio.pronosticar(
                tienda=datos["tienda"], familia=datos["familia"],
                fecha_origen=datos["fecha_origen"], horizonte=horizonte,
            )
            return jsonify(
                **resultado,
                modelo="Completo · best.pt",
                aviso="Demostración con datos históricos de 2017; no es una predicción actual de 2026.",
            )
        except LookupError as error:
            return jsonify(error=str(error)), 422
        except Exception:
            app.logger.exception("Error de pronóstico recursivo local")
            return jsonify(
                error="No se pudo generar el pronóstico. Revisá el checkpoint, el historial y las promociones en la consola."
            ), 503

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
