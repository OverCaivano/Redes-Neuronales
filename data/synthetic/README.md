# Datos sintéticos por producto

Los CSV de esta carpeta son **sintéticos**, no son ventas reales y no usan
datasets, checkpoints ni código de Favorita. El generador produce un panel
diario de 3 tiendas y 30 productos entre `2023-01-01` y `2026-11-16`
(ambos inclusive). No genera ventas después de esa fecha.

Desde la raíz del repositorio, regenerarlos con la semilla fija predeterminada:

```powershell
.\venv\Scripts\python.exe -m src.datos_sinteticos
```

Para guardar en otra carpeta o repetir con otra semilla explícita:

```powershell
.\venv\Scripts\python.exe -m src.datos_sinteticos --output-dir data\synthetic --seed 42
```

Archivos generados:

- `ventas_sinteticas.csv`: `fecha`, `tienda_id`, `producto_id`,
  `unidades_vendidas`, `precio`, `promocion`.
- `catalogo_productos_sinteticos.csv`: identificador, nombre y categoría de
  cada producto.
- `inventario_sintetico.csv`: una instantánea sintética al `2026-11-16` con
  `tienda_id`, `producto_id`, `stock_actual`, `stock_minimo` y
  `tiempo_reposicion` en días.

El panel de ventas tiene una fila por fecha, tienda y producto: 127.440 filas.

## Pipeline de predicción

El pipeline del bloque 2 usa este esquema y separa los periodos por fecha:

- Entrenamiento: `2023-01-01` a `2025-12-31`.
- Validación: `2026-01-01` a `2026-06-30`.
- Prueba: `2026-07-01` a `2026-11-16`.

Entrenar y medir el MLP:

```powershell
.\venv\Scripts\python.exe -m src.pipeline_productos train
```

Evaluar de nuevo el checkpoint contra el mismo dataset:

```powershell
.\venv\Scripts\python.exe -m src.pipeline_productos evaluate
```

Consultar una fecha posterior al último día observado:

```powershell
.\venv\Scripts\python.exe -m src.predecir_producto --store-id 1 --product-id BEB-001 --date 2026-11-17
```

El modelo es un MLP `54 → 64 → 1`, entrenado con MSE sobre
`log1p(unidades_vendidas)`. Usa IDs de producto, tienda y categoría codificados
con vocabularios ajustados en entrenamiento, el mes y ciclos de día de semana
y año. No usa unidades vendidas del día objetivo, rezagos ni datos posteriores
al corte; tampoco usa precio o promoción como entrada. Por eso la predicción
futura no requiere ventas del día anterior y representa una estimación media
que no está condicionada por esos factores.

Los artefactos se guardan en `runs/synthetic_product_sales_v1/`, separado de
Favorita. Cada corrida requiere una carpeta de salida nueva para evitar
sobrescribir checkpoints previos. Los datos y las métricas son sintéticos y
solo sirven para validar el pipeline; no demuestran precisión en ventas reales.

## Aplicación web (Bloques 3–5)

La lógica nueva está separada en dominio, contratos y servicios de aplicación,
repositorio CSV, adaptador PyTorch y controlador Flask. El Blueprint se crea
desde `src.composition.crear_blueprint_prediccion` y está registrado en
`src/web_app.py`. La página `/predicciones` consulta opciones y productos del
flujo sintético y envía la tienda, el producto y la fecha futura al Blueprint.
El servicio recibe el repositorio y el motor por inyección; no depende de
Flask, CSV ni PyTorch. Las rutas web anteriores de Favorita ya no están
registradas; sus módulos se conservan para la limpieza final.

## Recomendación de reposición (Bloque 5)

La pantalla permite elegir `Reposición básica` o `Reposición con stock de
seguridad`. El servicio pronostica cada día de los próximos N días desde la
fecha seleccionada, incluyendo esa fecha, donde N es `tiempo_reposicion`.
La estrategia básica cubre la demanda prevista menos el stock disponible; la
estrategia con seguridad agrega `stock_minimo` como colchón. Las cantidades se
redondean hacia arriba y nunca son negativas. El sistema solo muestra una
recomendación: no modifica el CSV, no registra movimientos y no efectúa compras.

El endpoint `POST /api/synthetic-v2/reposiciones` recibe
`tienda_id`, `producto_id`, `fecha` y `estrategia`. Tanto las ventas como el
inventario son sintéticos y no deben interpretarse como datos operativos reales.
