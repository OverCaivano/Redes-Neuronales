# StockPredict IA

Prototipo académico en Python, PyTorch y Flask para pronosticar unidades
vendidas por producto individual y tienda. El dataset de ventas, el catálogo y
el inventario son sintéticos: no representan actividad comercial real.

## Inicio rápido en Windows / PowerShell

Desde la raíz del repositorio:

```powershell
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m src.datos_sinteticos --output-dir data\synthetic --seed 42
.\venv\Scripts\python.exe -m src.pipeline_productos train
.\venv\Scripts\python.exe -m src.pipeline_productos evaluate
.\venv\Scripts\python.exe -m src.web_app
```

Abrir `http://127.0.0.1:5000`. La aplicación no entrena ni reemplaza artefactos
al iniciarse. El comando de entrenamiento inicial crea un checkpoint y
`metricas.json` en `runs\synthetic_product_sales_v1`; el pipeline rechaza
sobrescribir una corrida existente. Si ya hay artefactos, conservá la corrida y
usá un directorio nuevo al entrenar. Para correr la evaluación de nuevo contra
los datos y checkpoint guardados:

```powershell
.\venv\Scripts\python.exe -m src.pipeline_productos evaluate
```

Si falta el dataset sintético, el generador produce 127.440 filas de ventas
diarias del `2023-01-01` al `2026-11-16`, un catálogo de 30 productos y un
inventario sintético separado. Los CSV generados y los artefactos de
entrenamiento están excluidos de Git. Flask requiere localmente
`catalogo_productos_sinteticos.csv`, `inventario_sintetico.csv` y
`runs\synthetic_product_sales_v1\best_model.pt`; sin esos artefactos informa
que el modelo o los datos no están disponibles, sin cargar otro modelo.

El entrenamiento predeterminado ejecuta 12 épocas. En la verificación limpia
con Python 3.13 y CPU tardó aproximadamente 16,5 segundos; el tiempo varía
según el equipo.

## Interfaz de una sola pantalla

`/` reúne la selección de tienda, producto, fecha futura y estrategia; muestra
la predicción del MLP y la recomendación de reposición. Las antiguas páginas
informativas no se registran como rutas. Las métricas se consultan desde el
comando de evaluación, fuera de Flask.

## API

- `GET /api/synthetic-v2/opciones` y `GET /api/synthetic-v2/productos`.
- `POST /api/synthetic-v2/predicciones`: recibe `tienda_id`, `producto_id` y
  fecha ISO `AAAA-MM-DD`.
- `GET /api/synthetic-v2/reposicion/estrategias` y
  `POST /api/synthetic-v2/reposiciones`: recomendación con estrategia básica o
  stock de seguridad; no modifica existencias ni efectúa compras.

El período de prueba termina el `2026-11-16`; la predicción web acepta fechas
posteriores. Las métricas se leen de
`runs\synthetic_product_sales_v1\metricas.json`, el reporte guardado por el
pipeline. Consultarlas y volver a evaluar el checkpoint:

```powershell
.\venv\Scripts\python.exe -m src.pipeline_productos evaluate
```

El comando informa MAE/RMSE del MLP y el baseline en el test sintético; no
entrena ni reemplaza el checkpoint.

## Arquitectura

```text
Flask / MVC
  Controllers → Application Services → Ports
                                  ↑
                      CSV Repositories / PyTorch Adapter
```

- **Dominio:** productos, predicciones, inventario y reglas.
- **Aplicación:** servicios independientes de Flask, archivos CSV y PyTorch.
- **Infraestructura:** repositorios CSV y adaptador del modelo PyTorch.
- **Interfaz:** Blueprints y controladores Flask que traducen HTTP al caso de
  uso y renderizan las vistas.
- **Composición:** `src/composition.py` construye dependencias; los servicios
  reciben sus contratos por inyección.

Los principios SOLID aparecen en la separación por responsabilidad (SRP), las
estrategias reemplazables para extender la reposición (OCP/LSP), contratos
pequeños por capacidad (ISP) e inversión de dependencias mediante Protocol e
inyección (DIP). Repository aísla CSV/JSON; Adapter integra PyTorch detrás del
puerto de predicción; Strategy selecciona el cálculo básico o con stock de
seguridad. Singleton, Builder y Chain of Responsibility no aportan una
necesidad concreta al flujo actual y no se implementan.

## Pruebas

```powershell
.\venv\Scripts\python.exe -m pytest -q
node --test tests\test_presentacion.js tests\test_app.js
git diff --check
```

La suite incluye generación y validación de datos, pipeline, predicción,
reposición, la ruta principal de Flask y frontend. Las pruebas de predicción
real requieren los CSV y el checkpoint locales. Las pruebas JavaScript
requieren Node.js; se verificaron con la versión 22.20.0. Node.js no es
necesario para ejecutar Flask.

## Límites

Las ventas e inventario son sintéticos. MAE/RMSE describen únicamente la prueba
cronológica sintética y no prueban desempeño con ventas reales. La salida de
reposiciones es una recomendación demostrativa: no es una compra, movimiento
de stock ni asesoramiento operativo.
