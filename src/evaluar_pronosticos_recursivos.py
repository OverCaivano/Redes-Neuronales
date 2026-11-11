"""Backtest cronologico y representativo de pronosticos Favorita a siete dias."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from src.datos_favorita import leer_datos_favorita
from src.pronosticar_favorita import PronosticadorFavorita
from src.reanudacion import huella_dataset


RAIZ = Path(__file__).resolve().parents[1]
RUTA_CHECKPOINT = RAIZ / "runs/favorita_completa_20/best.pt"
RUTA_DATOS = RAIZ / "data/raw/train.csv"
RUTA_TIENDAS = RAIZ / "data/raw/stores.csv"
FAMILIAS_MUESTRA = ("AUTOMOTIVE", "BEAUTY", "GROCERY I", "PRODUCE")
HORIZONTE = 7


def _origenes_validacion(periodo, cantidad):
    if isinstance(cantidad, bool) or not isinstance(cantidad, int) or cantidad < 1:
        raise ValueError("La cantidad de ventanas debe ser un entero positivo")
    inicio, fin = (pd.Timestamp(fecha) for fecha in periodo)
    primer_origen = inicio - pd.Timedelta(days=1)
    ultimo_origen = fin - pd.Timedelta(days=HORIZONTE)
    dias = (ultimo_origen - primer_origen).days
    if dias < 0:
        raise ValueError("El período no alcanza para una ventana completa de siete días")
    cantidad = min(cantidad, dias + 1)
    desplazamientos = np.rint(np.linspace(0, dias, num=cantidad)).astype(int)
    return [primer_origen + pd.Timedelta(days=int(dia)) for dia in desplazamientos]


def _seleccionar_tiendas(ruta):
    columnas = ["store_nbr", "type", "city", "state", "cluster"]
    tiendas = pd.read_csv(ruta, usecols=columnas).sort_values("store_nbr")
    if tiendas["store_nbr"].duplicated().any() or tiendas[columnas].isna().any().any():
        raise ValueError("stores.csv contiene claves duplicadas o metadatos faltantes")
    muestra = [
        grupo.iloc[len(grupo) // 2]
        for _, grupo in tiendas.groupby("type", sort=True, observed=True)
    ]
    if not muestra:
        raise ValueError("stores.csv no contiene tiendas")
    return pd.DataFrame(muestra).sort_values("type").reset_index(drop=True)


def _resumen_tiendas(tiendas):
    return [
        {
            "tienda": int(fila["store_nbr"]),
            "tipo": str(fila["type"]),
            "ciudad": str(fila["city"]),
            "provincia": str(fila["state"]),
            "cluster": int(fila["cluster"]),
        }
        for _, fila in tiendas.iterrows()
    ]


def _metricas(reales, estimadas):
    reales = np.asarray(reales, dtype=np.float64)
    estimadas = np.asarray(estimadas, dtype=np.float64)
    if reales.ndim != 1 or reales.shape != estimadas.shape or reales.size == 0:
        raise ValueError("Las métricas requieren vectores no vacíos alineados")
    if not np.isfinite(reales).all() or not np.isfinite(estimadas).all():
        raise ValueError("No se pueden resumir ventas o pronósticos no finitos")
    errores = estimadas - reales
    return {
        "mae": float(np.mean(np.abs(errores))),
        "rmse": float(np.sqrt(np.mean(np.square(errores)))),
        "n": int(reales.size),
    }


def _comparacion(filas):
    mlp = _metricas(filas["ventas_reales"], filas["ventas_estimadas"])
    semanal = _metricas(filas["ventas_reales"], filas["referencia_semanal"])
    mejoras = {}
    for metrica in ("mae", "rmse"):
        base = semanal[metrica]
        mejoras[metrica] = (
            float(100 * (base - mlp[metrica]) / base) if base else None
        )
    return {"mlp": mlp, "referencia_semanal": semanal, "mejora_porcentual": mejoras}


def _evaluar_ventana(pronosticador, serie, tienda_tipo, origen):
    fechas_objetivo = pd.date_range(
        origen + pd.Timedelta(days=1), periods=HORIZONTE, freq="D"
    )
    conocidos = serie.loc[serie["fecha"].le(origen)].copy()
    futuros = serie.loc[serie["fecha"].isin(fechas_objetivo)].sort_values("fecha")
    if not futuros["fecha"].reset_index(drop=True).equals(
        pd.Series(fechas_objetivo, name="fecha")
    ):
        return None, "faltan ventas reales para completar los siete días"

    historial = conocidos.loc[:, ["fecha", "tienda", "familia", "ventas_objetivo"]]
    promociones = futuros.loc[
        :, ["fecha", "tienda", "familia", "cantidad_en_promocion"]
    ].copy()
    if not promociones["fecha"].reset_index(drop=True).equals(
        pd.Series(fechas_objetivo, name="fecha")
    ):
        return None, "faltan promociones para completar los siete días"

    resultado = pronosticador.pronosticar(
        historial,
        promociones,
        tienda=int(serie["tienda"].iloc[0]),
        familia=str(serie["familia"].iloc[0]),
        fecha_origen=origen.strftime("%Y-%m-%d"),
        horizonte=HORIZONTE,
    )
    estimaciones = resultado["pronosticos"]
    if len(estimaciones) != HORIZONTE:
        raise ValueError("El pronosticador no devolvió siete resultados")

    ventas_conocidas = conocidos.set_index("fecha")["ventas_objetivo"]
    reales = futuros.set_index("fecha")["ventas_objetivo"]
    registros = []
    for horizonte, (fecha, estimacion) in enumerate(zip(fechas_objetivo, estimaciones), start=1):
        if estimacion["fecha"] != fecha.strftime("%Y-%m-%d"):
            raise ValueError("Las fechas del pronóstico no coinciden con la ventana")
        fecha_referencia = fecha - pd.Timedelta(days=7)
        if fecha_referencia not in ventas_conocidas.index:
            raise ValueError("La referencia semanal no está disponible en el origen")
        registros.append({
            "fecha_origen": origen.strftime("%Y-%m-%d"),
            "horizonte": horizonte,
            "fecha": fecha.strftime("%Y-%m-%d"),
            "tienda": int(serie["tienda"].iloc[0]),
            "tipo_tienda": tienda_tipo,
            "familia": str(serie["familia"].iloc[0]),
            "ventas_reales": float(reales.loc[fecha]),
            "ventas_estimadas": float(estimacion["ventas_estimadas"]),
            "referencia_semanal": float(ventas_conocidas.loc[fecha_referencia]),
            "cantidad_en_promocion": float(estimacion["cantidad_en_promocion"]),
        })
    return registros, None


def _resumen_filas(filas):
    detalle = pd.DataFrame(filas)
    general = _comparacion(detalle)
    por_horizonte = [
        {"horizonte": int(horizonte), **_comparacion(grupo)}
        for horizonte, grupo in detalle.groupby("horizonte", sort=True)
    ]
    por_serie_horizonte = [
        {"tienda": int(tienda), "familia": familia, "horizonte": int(horizonte),
         **_comparacion(grupo)}
        for (tienda, familia, horizonte), grupo in detalle.groupby(
            ["tienda", "familia", "horizonte"], sort=True
        )
    ]
    return detalle, general, por_horizonte, por_serie_horizonte


def _tabla_comparaciones(resumenes, identificadores):
    filas = []
    for resumen in resumenes:
        fila = {nombre: resumen[nombre] for nombre in identificadores}
        for prefijo, clave in (
            ("mlp", "mlp"),
            ("referencia_semanal", "referencia_semanal"),
        ):
            fila.update({
                f"{prefijo}_{metrica}": valor
                for metrica, valor in resumen[clave].items()
            })
        fila.update({
            f"mejora_{metrica}_porcentual": valor
            for metrica, valor in resumen["mejora_porcentual"].items()
        })
        filas.append(fila)
    return pd.DataFrame(filas)


def evaluar_pronosticos_recursivos(
    checkpoint=RUTA_CHECKPOINT,
    data_path=RUTA_DATOS,
    stores_path=RUTA_TIENDAS,
    *,
    ventanas=6,
):
    """Evalúa seis orígenes en validación, cinco tipos de tienda y cuatro familias."""
    inicio_ejecucion = perf_counter()
    pronosticador = PronosticadorFavorita(checkpoint)
    estado = pronosticador.estado
    digest = huella_dataset(data_path)
    if digest != estado["dataset_sha256"]:
        raise ValueError("Dataset incompatible con el checkpoint: SHA-256 diferente")
    if not set(FAMILIAS_MUESTRA).issubset(pronosticador.categorias["familia"]):
        raise ValueError("Las familias de la muestra no son compatibles con el checkpoint")

    tiendas = _seleccionar_tiendas(stores_path)
    if not set(tiendas["store_nbr"]).issubset(pronosticador.categorias["tienda"]):
        raise ValueError("Las tiendas seleccionadas no son compatibles con el checkpoint")
    ids_tienda = tiendas["store_nbr"].astype(int).tolist()
    datos = leer_datos_favorita(
        data_path, tiendas=ids_tienda, familias=FAMILIAS_MUESTRA
    )
    series = {
        (int(tienda), str(familia)): grupo.sort_values("fecha").reset_index(drop=True)
        for (tienda, familia), grupo in datos.groupby(
            ["tienda", "familia"], sort=True, observed=True
        )
    }
    origenes = _origenes_validacion(estado["periodos"]["validacion"], ventanas)
    tipos = dict(zip(tiendas["store_nbr"].astype(int), tiendas["type"]))
    filas, omitidas = [], []
    pares = [
        (int(tienda), familia)
        for tienda in ids_tienda
        for familia in FAMILIAS_MUESTRA
    ]
    for origen in origenes:
        for tienda, familia in pares:
            serie = series.get((tienda, familia))
            if serie is None:
                omitidas.append({
                    "fecha_origen": origen.strftime("%Y-%m-%d"),
                    "tienda": tienda, "familia": familia,
                    "motivo": "la serie no existe en el historial",
                })
                continue
            registros, motivo = _evaluar_ventana(
                pronosticador, serie, tipos[tienda], origen
            )
            if motivo:
                omitidas.append({
                    "fecha_origen": origen.strftime("%Y-%m-%d"),
                    "tienda": tienda, "familia": familia, "motivo": motivo,
                })
            else:
                filas.extend(registros)
    if not filas:
        raise ValueError("No hay ventanas completas de siete días para evaluar")

    detalle, general, por_horizonte, por_serie_horizonte = _resumen_filas(filas)
    conteos_origen = (
        detalle.drop_duplicates(["fecha_origen", "tienda", "familia"])
        .groupby("fecha_origen")
        .size()
    )
    ventanas_completas = int((conteos_origen == len(pares)).sum())
    informe = {
        "protocolo": "origen_movil_validacion_pronostico_recursivo_7_dias",
        "particion": "validacion",
        "periodo_validacion": list(estado["periodos"]["validacion"]),
        "origenes_solicitados": [fecha.strftime("%Y-%m-%d") for fecha in origenes],
        "ventanas_con_resultados": int(detalle["fecha_origen"].nunique()),
        "ventanas_completas": ventanas_completas,
        "series_solicitadas": len(pares),
        "series_evaluadas": int(detalle[["tienda", "familia"]].drop_duplicates().shape[0]),
        "predicciones_evaluadas": len(detalle),
        "horizonte_maximo": HORIZONTE,
        "checkpoint": {
            "archivo": Path(checkpoint).name,
            "sha256": huella_dataset(checkpoint),
            "epoca": int(estado["epoch"]),
            "input_dim": int(estado["input_dim"]),
        },
        "dataset_sha256": digest,
        "muestra_tiendas": _resumen_tiendas(tiendas),
        "familias_muestra": list(FAMILIAS_MUESTRA),
        "promociones": {
            "fuente": "cantidad_en_promocion de train.csv para los días posteriores al origen",
            "supuesto": (
                "Se asume que el calendario de promociones de los siete días era conocido "
                "en la fecha de origen; train.csv no registra cuándo se publicó."
            ),
            "ventas_futuras_usadas_como_entrada": False,
        },
        "referencia_semanal": (
            "Para cada horizonte h, ventas reales del día objetivo menos siete días; "
            "h=1..7 y todos esos valores estaban disponibles en el origen."
        ),
        "test_preservado": True,
        "general": general,
        "por_horizonte": por_horizonte,
        "por_tienda_familia_horizonte": por_serie_horizonte,
        "ventanas_omitidas": omitidas,
        "segundos_total": perf_counter() - inicio_ejecucion,
    }
    return informe, detalle


def guardar_resultados(informe, detalle, output_dir):
    destino = Path(output_dir)
    if not destino.is_absolute():
        destino = RAIZ / destino
    runs = (RAIZ / "runs").resolve()
    destino = destino.resolve()
    if destino == runs or not destino.is_relative_to(runs):
        raise ValueError("Los resultados deben guardarse en una carpeta nueva dentro de runs/")
    destino.mkdir(parents=True, exist_ok=False)
    (destino / "informe.json").write_text(
        json.dumps(informe, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    detalle.to_csv(destino / "detalle_pronosticos.csv", index=False, encoding="utf-8")
    _tabla_comparaciones(informe["por_horizonte"], ("horizonte",)).to_csv(
        destino / "por_horizonte.csv", index=False, encoding="utf-8"
    )
    _tabla_comparaciones(
        informe["por_tienda_familia_horizonte"],
        ("tienda", "familia", "horizonte"),
    ).to_csv(
        destino / "por_tienda_familia_horizonte.csv", index=False, encoding="utf-8"
    )
    return destino


def main():
    parser = argparse.ArgumentParser(
        description="Backtest recursivo de siete días usando solo la partición de validación"
    )
    parser.add_argument("--checkpoint", default=str(RUTA_CHECKPOINT))
    parser.add_argument("--data-path", default=str(RUTA_DATOS))
    parser.add_argument("--stores-path", default=str(RUTA_TIENDAS))
    parser.add_argument("--ventanas", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        help="Carpeta nueva bajo runs/; por defecto se crea una con fecha y hora",
    )
    args = parser.parse_args()

    salida = args.output_dir or str(
        RAIZ / "runs/evaluaciones"
        / f"pronostico_recursivo_validacion_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    informe, detalle = evaluar_pronosticos_recursivos(
        args.checkpoint, args.data_path, args.stores_path, ventanas=args.ventanas
    )
    destino = guardar_resultados(informe, detalle, salida)
    print(f"Partición: validación {informe['periodo_validacion']}")
    print(
        f"Ventanas completas: {informe['ventanas_completas']}; "
        f"series: {informe['series_evaluadas']}; "
        f"pronósticos: {informe['predicciones_evaluadas']}"
    )
    for fila in informe["por_horizonte"]:
        print(
            f"Día {fila['horizonte']}: "
            f"MLP MAE={fila['mlp']['mae']:.6f} RMSE={fila['mlp']['rmse']:.6f}; "
            f"semanal MAE={fila['referencia_semanal']['mae']:.6f} "
            f"RMSE={fila['referencia_semanal']['rmse']:.6f}"
        )
    print(f"Tiempo total: {informe['segundos_total']:.3f} s")
    print(f"Resultados: {destino}")


if __name__ == "__main__":
    main()
