"""Generación reproducible de ventas sintéticas por producto y tienda."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


FECHA_INICIO = pd.Timestamp("2023-01-01")
FECHA_FIN = pd.Timestamp("2026-11-16")
SEMILLA_PREDETERMINADA = 42
TIENDAS = (1, 2, 3)
CATEGORIAS = ("bebidas", "alimentos", "limpieza", "higiene", "lacteos")

COLUMNAS_VENTAS = (
    "fecha", "tienda_id", "producto_id", "unidades_vendidas", "precio", "promocion",
)
COLUMNAS_CATALOGO = ("producto_id", "nombre", "categoria")
COLUMNAS_INVENTARIO = (
    "tienda_id", "producto_id", "stock_actual", "stock_minimo", "tiempo_reposicion",
)

# ID, nombre, categoría, demanda media diaria y precio de lista.
_PRODUCTOS = (
    ("BEB-001", "Agua mineral 500 ml", "bebidas", 18, 1.20),
    ("BEB-002", "Agua mineral 1.5 l", "bebidas", 12, 2.10),
    ("BEB-003", "Jugo de naranja 1 l", "bebidas", 8, 3.40),
    ("BEB-004", "Gaseosa cola 1.5 l", "bebidas", 11, 3.10),
    ("BEB-005", "Té frío 500 ml", "bebidas", 6, 2.50),
    ("BEB-006", "Café molido 250 g", "bebidas", 5, 6.80),
    ("ALI-001", "Arroz 1 kg", "alimentos", 13, 2.30),
    ("ALI-002", "Pasta seca 500 g", "alimentos", 11, 1.80),
    ("ALI-003", "Harina 1 kg", "alimentos", 8, 1.50),
    ("ALI-004", "Aceite vegetal 900 ml", "alimentos", 7, 4.20),
    ("ALI-005", "Galletas 200 g", "alimentos", 10, 2.10),
    ("ALI-006", "Lentejas 500 g", "alimentos", 6, 2.60),
    ("LIM-001", "Detergente para ropa 1 l", "limpieza", 5, 5.80),
    ("LIM-002", "Lavavajillas 750 ml", "limpieza", 6, 3.20),
    ("LIM-003", "Limpiador multiuso 900 ml", "limpieza", 4, 3.70),
    ("LIM-004", "Lavandina 1 l", "limpieza", 7, 1.60),
    ("LIM-005", "Esponjas pack x3", "limpieza", 5, 1.90),
    ("LIM-006", "Bolsas de residuos pack x10", "limpieza", 4, 2.40),
    ("HIG-001", "Jabón de tocador 125 g", "higiene", 9, 1.30),
    ("HIG-002", "Shampoo 400 ml", "higiene", 5, 5.20),
    ("HIG-003", "Pasta dental 90 g", "higiene", 6, 2.80),
    ("HIG-004", "Papel higiénico pack x4", "higiene", 8, 4.10),
    ("HIG-005", "Desodorante 150 ml", "higiene", 4, 4.70),
    ("HIG-006", "Toallas femeninas pack x8", "higiene", 4, 3.60),
    ("LAC-001", "Leche entera 1 l", "lacteos", 19, 1.70),
    ("LAC-002", "Yogur natural 190 g", "lacteos", 12, 1.10),
    ("LAC-003", "Queso cremoso 300 g", "lacteos", 6, 4.80),
    ("LAC-004", "Manteca 200 g", "lacteos", 5, 3.20),
    ("LAC-005", "Crema de leche 200 ml", "lacteos", 5, 2.60),
    ("LAC-006", "Queso rallado 40 g", "lacteos", 4, 1.90),
)

_FACTORES_SEMANALES = {
    "bebidas": (0.90, 0.92, 0.95, 1.00, 1.08, 1.22, 1.28),
    "alimentos": (1.00, 1.00, 1.00, 1.00, 1.05, 1.15, 1.12),
    "limpieza": (0.95, 0.95, 0.98, 1.00, 1.05, 1.16, 1.12),
    "higiene": (1.00, 1.00, 1.00, 1.00, 1.02, 1.08, 1.05),
    "lacteos": (1.02, 1.00, 1.00, 1.00, 1.04, 1.10, 1.08),
}

# Factores mensuales con estacionalidad de verano/invierno y fin de año.
_FACTORES_MENSUALES = {
    "bebidas": (1.22, 1.18, 1.12, 1.04, 0.96, 0.88, 0.84, 0.88, 0.96, 1.02, 1.10, 1.18),
    "alimentos": (1.04, 1.02, 1.00, 0.98, 1.00, 1.02, 1.02, 1.00, 1.00, 1.02, 1.06, 1.16),
    "limpieza": (0.98, 1.08, 1.04, 0.98, 0.96, 0.98, 1.00, 1.00, 1.02, 1.04, 1.02, 1.06),
    "higiene": (1.02, 1.00, 0.98, 1.00, 1.02, 1.04, 1.04, 1.02, 1.00, 1.00, 1.02, 1.08),
    "lacteos": (0.96, 0.98, 1.00, 1.04, 1.08, 1.12, 1.14, 1.12, 1.08, 1.04, 1.00, 0.98),
}

_FACTORES_TIENDAS = {1: 0.88, 2: 1.00, 3: 1.16}
_FACTORES_PRECIO_TIENDA = {1: 1.00, 2: 1.03, 3: 0.98}
_PROBABILIDAD_PROMOCION_MENSUAL = (
    0.12, 0.11, 0.10, 0.09, 0.09, 0.10,
    0.10, 0.09, 0.10, 0.11, 0.13, 0.16,
)


def generar_catalogo_productos():
    """Devuelve el catálogo sintético, con 30 productos y cinco categorías."""
    catalogo = pd.DataFrame(_PRODUCTOS, columns=(
        "producto_id", "nombre", "categoria", "_demanda_base", "_precio_base",
    ))
    return catalogo.loc[:, COLUMNAS_CATALOGO].copy()


def _validar_esquema(datos, columnas, nombre):
    if not isinstance(datos, pd.DataFrame):
        raise TypeError(f"{nombre} debe ser un DataFrame")
    if datos.columns.has_duplicates or set(datos.columns) != set(columnas):
        raise ValueError(f"El esquema de {nombre} debe contener exactamente: {list(columnas)}")


def _validar_enteros_no_negativos(datos, columnas, nombre):
    for columna in columnas:
        if datos[columna].map(lambda valor: isinstance(valor, (bool, np.bool_))).any():
            raise ValueError(f"{nombre}.{columna} debe contener enteros, no booleanos")
        valores = pd.to_numeric(datos[columna], errors="coerce")
        arreglo = valores.to_numpy(dtype=np.float64)
        if (
            valores.isna().any()
            or not np.isfinite(arreglo).all()
            or (arreglo < 0).any()
            or not np.equal(arreglo, np.floor(arreglo)).all()
        ):
            raise ValueError(f"{nombre}.{columna} debe contener enteros finitos no negativos")


def validar_catalogo_productos(catalogo):
    """Valida esquema, identificadores y categorías del catálogo esperado."""
    _validar_esquema(catalogo, COLUMNAS_CATALOGO, "catálogo")
    if catalogo.isna().any().any():
        raise ValueError("El catálogo no puede tener valores faltantes")
    for columna in ("producto_id", "nombre", "categoria"):
        if not catalogo[columna].map(lambda valor: isinstance(valor, str) and bool(valor.strip())).all():
            raise ValueError(f"catálogo.{columna} debe contener texto no vacío")
    if catalogo["producto_id"].duplicated().any():
        raise ValueError("El catálogo contiene producto_id duplicados")
    if catalogo["nombre"].duplicated().any():
        raise ValueError("El catálogo contiene nombres duplicados")
    if len(catalogo) != len(_PRODUCTOS) or set(catalogo["categoria"]) != set(CATEGORIAS):
        raise ValueError("El catálogo debe tener 30 productos en las cinco categorías definidas")
    cantidades = catalogo["categoria"].value_counts()
    if any(cantidades.get(categoria, 0) != 6 for categoria in CATEGORIAS):
        raise ValueError("El catálogo debe tener seis productos por categoría")


def _validar_fechas_ventas(fechas):
    try:
        fechas = pd.to_datetime(fechas, errors="raise")
    except (TypeError, ValueError, OverflowError):
        raise ValueError("ventas.fecha contiene fechas inválidas") from None
    if fechas.isna().any() or fechas.dt.tz is not None:
        raise ValueError("ventas.fecha debe contener días válidos sin zona horaria")
    if not fechas.eq(fechas.dt.normalize()).all():
        raise ValueError("ventas.fecha debe representar días completos")
    if not fechas.between(FECHA_INICIO, FECHA_FIN).all():
        raise ValueError("ventas.fecha debe estar entre 2023-01-01 y 2026-11-16")
    fechas_unicas = pd.DatetimeIndex(fechas.unique()).sort_values()
    esperadas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    if not fechas_unicas.equals(esperadas):
        raise ValueError("ventas debe incluir cada día del período sintético")
    return fechas


def validar_ventas(ventas, catalogo=None):
    """Valida esquema, periodo, panel diario, claves e importes de ventas."""
    _validar_esquema(ventas, COLUMNAS_VENTAS, "ventas")
    catalogo = generar_catalogo_productos() if catalogo is None else catalogo
    validar_catalogo_productos(catalogo)
    if ventas.isna().any().any():
        raise ValueError("ventas no puede tener valores faltantes")

    _validar_fechas_ventas(ventas["fecha"])
    _validar_enteros_no_negativos(ventas, ("tienda_id", "unidades_vendidas"), "ventas")
    if not set(ventas["tienda_id"].unique()).issubset(TIENDAS):
        raise ValueError("ventas.tienda_id contiene identificadores desconocidos")
    productos_validos = set(catalogo["producto_id"])
    if not ventas["producto_id"].map(
        lambda valor: isinstance(valor, str) and valor in productos_validos
    ).all():
        raise ValueError("ventas.producto_id contiene identificadores desconocidos")
    if ventas.duplicated(["fecha", "tienda_id", "producto_id"]).any():
        raise ValueError("ventas contiene claves duplicadas por fecha, tienda_id y producto_id")

    precios = pd.to_numeric(ventas["precio"], errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(precios).all() or (precios <= 0).any():
        raise ValueError("ventas.precio debe contener valores finitos positivos")
    if not ventas["promocion"].map(
        lambda valor: isinstance(valor, (bool, np.bool_))
    ).all():
        raise ValueError("ventas.promocion debe contener valores booleanos")

    filas_esperadas = len(pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")) * len(TIENDAS) * len(catalogo)
    if len(ventas) != filas_esperadas:
        raise ValueError(f"ventas debe contener exactamente {filas_esperadas} filas")
    return len(ventas)


def validar_inventario(inventario, catalogo=None):
    """Valida inventario por tienda-producto, separado de las ventas."""
    _validar_esquema(inventario, COLUMNAS_INVENTARIO, "inventario")
    catalogo = generar_catalogo_productos() if catalogo is None else catalogo
    validar_catalogo_productos(catalogo)
    if inventario.isna().any().any():
        raise ValueError("inventario no puede tener valores faltantes")
    _validar_enteros_no_negativos(
        inventario, ("tienda_id", "stock_actual", "stock_minimo", "tiempo_reposicion"),
        "inventario",
    )
    if (inventario["tiempo_reposicion"] == 0).any():
        raise ValueError("inventario.tiempo_reposicion debe ser positivo")
    if not set(inventario["tienda_id"].unique()).issubset(TIENDAS):
        raise ValueError("inventario.tienda_id contiene identificadores desconocidos")
    if not set(inventario["producto_id"]).issubset(set(catalogo["producto_id"])):
        raise ValueError("inventario.producto_id contiene identificadores desconocidos")
    if inventario.duplicated(["tienda_id", "producto_id"]).any():
        raise ValueError("inventario contiene claves duplicadas por tienda_id y producto_id")
    filas_esperadas = len(TIENDAS) * len(catalogo)
    if len(inventario) != filas_esperadas:
        raise ValueError(f"inventario debe contener exactamente {filas_esperadas} filas")
    return len(inventario)


def generar_ventas_sinteticas(semilla=SEMILLA_PREDETERMINADA):
    """Genera un panel diario completo sin observaciones posteriores al corte."""
    if isinstance(semilla, bool) or not isinstance(semilla, (int, np.integer)):
        raise TypeError("semilla debe ser un entero")
    generador = np.random.default_rng(int(semilla))
    catalogo = pd.DataFrame(_PRODUCTOS, columns=(
        "producto_id", "nombre", "categoria", "demanda_base", "precio_base",
    ))
    fechas = pd.date_range(FECHA_INICIO, FECHA_FIN, freq="D")
    ventas = pd.MultiIndex.from_product(
        [fechas, TIENDAS, catalogo["producto_id"]],
        names=["fecha", "tienda_id", "producto_id"],
    ).to_frame(index=False)

    categorias = catalogo["categoria"].to_numpy()
    indices_producto = ventas["producto_id"].map(
        {producto_id: indice for indice, producto_id in enumerate(catalogo["producto_id"])}
    ).to_numpy()
    indices_categoria = pd.Index(CATEGORIAS).get_indexer(categorias)[indices_producto]
    dias_semana = ventas["fecha"].dt.dayofweek.to_numpy()
    meses = ventas["fecha"].dt.month.to_numpy() - 1
    tiendas = ventas["tienda_id"].to_numpy()

    base_demanda = catalogo["demanda_base"].to_numpy(dtype=np.float64)[indices_producto]
    factores_semana = np.asarray(
        [_FACTORES_SEMANALES[categoria] for categoria in CATEGORIAS], dtype=np.float64
    )
    factores_mes = np.asarray(
        [_FACTORES_MENSUALES[categoria] for categoria in CATEGORIAS], dtype=np.float64
    )
    probabilidad_promocion = np.asarray(_PROBABILIDAD_PROMOCION_MENSUAL)[meses]
    promociones = generador.random(len(ventas)) < probabilidad_promocion

    demanda_media = (
        base_demanda
        * np.asarray([_FACTORES_TIENDAS[tienda] for tienda in tiendas])
        * factores_semana[indices_categoria, dias_semana]
        * factores_mes[indices_categoria, meses]
        * np.where(promociones, 1.35, 1.0)
    )
    unidades = generador.poisson(demanda_media)

    precios_base = catalogo["precio_base"].to_numpy(dtype=np.float64)[indices_producto]
    precios = (
        precios_base
        * np.asarray([_FACTORES_PRECIO_TIENDA[tienda] for tienda in tiendas])
        * np.where(promociones, 0.85, 1.0)
        * generador.uniform(0.98, 1.02, len(ventas))
    )
    resultado = pd.DataFrame({
        "fecha": ventas["fecha"].dt.strftime("%Y-%m-%d"),
        "tienda_id": tiendas.astype(np.int16),
        "producto_id": ventas["producto_id"],
        "unidades_vendidas": unidades.astype(np.int32),
        "precio": np.round(precios, 2),
        "promocion": promociones,
    }, columns=COLUMNAS_VENTAS)
    validar_ventas(resultado, generar_catalogo_productos())
    return resultado


def generar_inventario_sintetico(ventas, catalogo=None, semilla=SEMILLA_PREDETERMINADA):
    """Genera un snapshot de inventario independiente para cada tienda-producto."""
    if isinstance(semilla, bool) or not isinstance(semilla, (int, np.integer)):
        raise TypeError("semilla debe ser un entero")
    catalogo = generar_catalogo_productos() if catalogo is None else catalogo
    validar_ventas(ventas, catalogo)

    pares = pd.MultiIndex.from_product(
        [TIENDAS, catalogo["producto_id"]],
        names=["tienda_id", "producto_id"],
    ).to_frame(index=False)
    fecha_corte = FECHA_FIN.strftime("%Y-%m-%d")
    ultimos_dias = ventas.loc[
        pd.to_datetime(ventas["fecha"]).between(
            FECHA_FIN - pd.Timedelta(days=27), FECHA_FIN
        )
    ]
    promedios = ultimos_dias.groupby(
        ["tienda_id", "producto_id"], sort=True
    )["unidades_vendidas"].mean()
    media_diaria = np.asarray([
        promedios.loc[(tienda, producto)]
        for tienda, producto in zip(pares["tienda_id"], pares["producto_id"])
    ])

    generador = np.random.default_rng(int(semilla) + 1)
    reposicion = generador.integers(1, 8, size=len(pares))
    stock_minimo = np.ceil(media_diaria * reposicion * 0.75).astype(np.int32)
    stock_actual = generador.poisson(media_diaria * (reposicion + 2)).astype(np.int32)
    resultado = pd.DataFrame({
        "tienda_id": pares["tienda_id"].astype(np.int16),
        "producto_id": pares["producto_id"],
        "stock_actual": stock_actual,
        "stock_minimo": stock_minimo,
        "tiempo_reposicion": reposicion.astype(np.int16),
    }, columns=COLUMNAS_INVENTARIO)
    validar_inventario(resultado, catalogo)
    resultado.attrs["fecha_corte"] = fecha_corte
    return resultado


def generar_archivos_sinteticos(
    directorio=None, *, semilla=SEMILLA_PREDETERMINADA
):
    """Valida y escribe ventas, catálogo e inventario en una carpeta separada."""
    destino = Path(directorio) if directorio is not None else (
        Path(__file__).resolve().parents[1] / "data" / "synthetic"
    )
    catalogo = generar_catalogo_productos()
    ventas = generar_ventas_sinteticas(semilla)
    inventario = generar_inventario_sintetico(ventas, catalogo, semilla)
    validar_catalogo_productos(catalogo)
    validar_ventas(ventas, catalogo)
    validar_inventario(inventario, catalogo)
    destino.mkdir(parents=True, exist_ok=True)

    archivos = {
        "ventas": destino / "ventas_sinteticas.csv",
        "catalogo": destino / "catalogo_productos_sinteticos.csv",
        "inventario": destino / "inventario_sintetico.csv",
    }
    ventas.to_csv(archivos["ventas"], index=False, lineterminator="\n")
    catalogo.to_csv(archivos["catalogo"], index=False, lineterminator="\n")
    inventario.to_csv(archivos["inventario"], index=False, lineterminator="\n")
    return archivos


def main():
    parser = argparse.ArgumentParser(
        description="Genera datos sintéticos diarios por producto y tienda."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directorio de salida (por defecto: data/synthetic).",
    )
    parser.add_argument(
        "--seed", type=int, default=SEMILLA_PREDETERMINADA,
        help=f"Semilla reproducible (por defecto: {SEMILLA_PREDETERMINADA}).",
    )
    args = parser.parse_args()
    archivos = generar_archivos_sinteticos(args.output_dir, semilla=args.seed)
    print("Datos sinteticos generados; no contienen ventas reales ni futuras.")
    for nombre, ruta in archivos.items():
        tabla = pd.read_csv(ruta)
        print(f"{nombre}: {ruta} ({len(tabla)} registros)")


if __name__ == "__main__":
    main()
