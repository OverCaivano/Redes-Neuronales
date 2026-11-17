# StockPredict IA

Prototipo académico para predecir ventas futuras de **productos individuales**
por tienda con Python, PyTorch y Flask. La aplicación web activa utiliza el
dataset sintético del proyecto; sus resultados no representan ventas reales ni
demuestran precisión comercial.

## Aplicación web

Desde la raíz del repositorio, con las dependencias instaladas:

```powershell
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-web.txt -c requirements-colab.txt torchvision==0.28.0
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m src.web_app
```

Abrir `http://127.0.0.1:5000`. La aplicación escucha solo en localhost y no
entrena ni sobrescribe modelos al iniciarse.

Requisitos locales del flujo web:

- `data/synthetic/catalogo_productos_sinteticos.csv`
- `runs/synthetic_product_sales_v1/best_model.pt`

Si falta el catálogo, regenerar los datos sintéticos:

```powershell
.\venv\Scripts\python.exe -m src.datos_sinteticos --output-dir data\synthetic --seed 42
```

El checkpoint vive en `runs/`, que no se versiona. Si no está disponible, Flask
puede iniciar, pero las opciones y predicciones responden `503`; no se usa un
checkpoint alternativo ni se carga un modelo antiguo. El checkpoint existente
se puede evaluar con:

```powershell
.\venv\Scripts\python.exe -m src.pipeline_productos evaluate
```

Entrenar es una acción explícita. Cada nueva corrida necesita un directorio de
salida que no contenga artefactos, por ejemplo `runs/synthetic_product_sales_v2`.

### Interfaz y endpoints

- `/` y `/inicio`: alcance del modelo sintético.
- `/predicciones`: selección de tienda, producto, fecha actual/futura y
  estrategia de reposición sintética.
- `/acerca`: datos, arquitectura y limitaciones.
- `/evaluacion` y `/historial`: páginas informativas que indican claramente
  que las vistas anteriores se deshabilitaron; no presentan métricas históricas.
- `GET /api/synthetic-v2/opciones`: tiendas y fechas admitidas.
- `GET /api/synthetic-v2/productos`: catálogo sintético.
- `POST /api/synthetic-v2/predicciones`: recibe `tienda_id`, `producto_id` y
  `fecha` (`AAAA-MM-DD`), y devuelve las unidades estimadas.
- `GET /api/synthetic-v2/reposicion/estrategias`: estrategias disponibles.
- `POST /api/synthetic-v2/reposiciones`: recibe tienda, producto, fecha y
  estrategia (`basica` o `seguridad`). Predice cada día del plazo de reposición,
  incluyendo la fecha inicial, y devuelve la demanda acumulada, el inventario
  sintético y la cantidad recomendada. No modifica existencias ni realiza
  compras.

El último día observado es `2026-11-16`. Las consultas deben ser actuales o
futuras y posteriores a ese corte. No se requieren ventas del día anterior;
precio y promoción no forman parte de las entradas de inferencia.

Las rutas web/API anteriores (`/api/opciones`, `/api/resumen`,
`/api/predicciones` y `/api/pronosticos`) ya no están registradas. Los módulos,
pruebas y artefactos Favorita se conservan temporalmente para la limpieza final,
pero la aplicación Flask no los importa ni los sirve.

## Datos y modelo

Los detalles de regeneración, esquema y limitaciones están en
[data/synthetic/README.md](data/synthetic/README.md). El generador produce:

- 127.440 observaciones diarias de ventas entre `2023-01-01` y `2026-11-16`.
- 3 tiendas, 30 productos individuales y cinco categorías.
- Catálogo e inventario sintéticos en archivos separados.

El pipeline cronológico de entrenamiento/validación/prueba y la inferencia CLI
están en `src/pipeline_productos.py` y `src/predecir_producto.py`. La inferencia
web usa el checkpoint `runs/synthetic_product_sales_v1/best_model.pt`.

## Arquitectura

```text
Flask Blueprints → Controllers
    ├── ServicioPrediccion
    │   ├── Puerto de catálogo → repositorio CSV sintético
    │   └── Puerto de predicción → adaptador PyTorch/checkpoint
    └── ServicioReposicion
        ├── ServicioPrediccion (inyectado)
        ├── Puerto de inventario → repositorio CSV sintético
        └── Strategy (básica / stock de seguridad)
```

`src/composition.py` construye las implementaciones y `src/web_app.py` registra
los Blueprints. Los servicios dependen de contratos inyectados; no acceden
directamente a Flask, CSV ni PyTorch. Las pantallas de evaluación e historial
del modelo antiguo están deshabilitadas hasta que haya una vista vinculada al
pipeline sintético.

## Pruebas

```powershell
.\venv\Scripts\python.exe -m pytest tests\test_web_app.py tests\test_replenishment.py tests\test_servicio_prediccion.py tests\test_repositorio_productos.py tests\test_adaptador_prediccion.py tests\test_controlador_prediccion.py tests\test_datos_sinteticos.py tests\test_pipeline_productos.py -q
node --test tests\test_presentacion.js tests\test_app.js
```

Las pruebas Flask validan la predicción individual, la recomendación y sus
estrategias. La prueba de extremo a extremo contrasta las fechas de cobertura
con el checkpoint local; se omite si faltan el checkpoint, el catálogo o el
inventario sintéticos.

## Limitaciones

Las ventas y existencias son sintéticas y no representan condiciones reales.
La reposición es una recomendación matemática sencilla, no una compra ni un
movimiento de stock. No incluye intervalos de confianza y las predicciones no
están condicionadas por precio o promoción. Estos resultados no demuestran
precisión comercial.
