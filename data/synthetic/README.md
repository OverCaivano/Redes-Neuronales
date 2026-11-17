# Datos sintéticos por producto

Las ventas, el catálogo y el inventario de esta carpeta son datos generados
artificialmente y no representan transacciones ni existencias reales. El panel
diario contiene 3 tiendas, 30 productos y fechas entre `2023-01-01` y
`2026-11-16` inclusive.

Desde la raíz del proyecto, regenerar los tres CSV con la semilla fija:

```powershell
.\venv\Scripts\python.exe -m src.datos_sinteticos --output-dir data\synthetic --seed 42
```

El generador produce:

- `ventas_sinteticas.csv`: fecha, tienda, producto, unidades vendidas, precio y
  promoción (127.440 registros).
- `catalogo_productos_sinteticos.csv`: identificador, nombre y categoría de
  cada producto.
- `inventario_sintetico.csv`: instantánea sintética con tienda, producto,
  stock actual, stock mínimo y tiempo de reposición.

El pipeline cronológico usa entrenamiento `2023-01-01`–`2025-12-31`,
validación `2026-01-01`–`2026-06-30` y prueba `2026-07-01`–`2026-11-16`.
Entrena y evalúa explícitamente con:

```powershell
.\venv\Scripts\python.exe -m src.pipeline_productos train
.\venv\Scripts\python.exe -m src.pipeline_productos evaluate
```

El entrenamiento inicial guarda el checkpoint y `metricas.json` en
`runs\synthetic_product_sales_v1`. El pipeline no sobrescribe una corrida
existente; para repetir entrenamiento, elegí un directorio de salida nuevo y
configurá la aplicación para apuntar a sus artefactos.

La aplicación Flask realiza inferencia con el checkpoint existente; no entrena
al iniciarse. Las métricas del reporte y las recomendaciones de inventario son
demostrativas y no deben utilizarse como decisiones comerciales.
