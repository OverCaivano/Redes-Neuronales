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

## Versiones verificadas

El proyecto requiere Python 3.13. `requirements.txt` fija PyTorch 2.15.0,
NumPy 2.5.5, pandas 3.1.1 y pytest 9.2.0; `requirements-web.txt` fija Flask
3.1.3. Estas versiones corresponden a la instalación limpia de Windows/Python
3.13 verificada para el proyecto. El entorno local previo tenía PyTorch
2.13.0+cpu, NumPy 2.5.2, pandas 3.0.5 y pytest 9.1.1; se conserva sin cambios.
Los requisitos fijan dependencias directas, no todas las transitivas.

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

El último día disponible es `2026-11-16`. La predicción web admite fechas desde
el día siguiente (o la fecha actual, si es posterior) hasta 90 días después del
corte del dataset. Ese máximo es un límite de alcance para la demostración, no
una garantía de precisión.

```powershell
.\venv\Scripts\python.exe -m src.pipeline_productos evaluate
```

Durante el entrenamiento, el pipeline guarda el checkpoint y un reporte
`runs\synthetic_product_sales_v1\metricas.json`, que conserva configuración,
métricas e historial de validación por época. El comando `evaluate` no lee ese
JSON: carga el checkpoint y los CSV, verifica que los datos coincidan con las
huellas guardadas en el checkpoint y vuelve a calcular MAE/RMSE sobre el período
de prueba. Informa los resultados del MLP y del baseline de media histórica por
tienda y producto; no entrena ni modifica el checkpoint o el reporte JSON.

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

En MVC, los Blueprints/controladores reciben las solicitudes, las plantillas
Jinja presentan los datos y el dominio junto con los servicios de aplicación
contiene el modelo y los casos de uso. SRP se refleja en la separación entre
esas responsabilidades; OCP/LSP, en las estrategias de reposición
intercambiables; ISP, en los contratos pequeños de `src/application/ports.py`;
y DIP, en servicios que dependen de esos Protocol, ensamblados con
implementaciones concretas en `src/composition.py`. Los dos Repository
concretos son `RepositorioProductosCSV` y `RepositorioInventarioCSV`; aíslan
la lectura y validación de los CSV de catálogo e inventario. No existe un
repositorio JSON: el pipeline escribe `metricas.json`. Adapter envuelve la
inferencia PyTorch detrás del puerto de predicción, y Strategy selecciona el
cálculo básico o con stock de seguridad. Singleton, Builder y Chain of
Responsibility no responden a una necesidad del flujo actual.

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
de stock ni asesoramiento operativo. El inventario mostrado es un snapshot
sintético con fecha de corte `2026-11-16`; no se descuentan automáticamente las
ventas posteriores a ese corte, por lo que el stock no se proyecta a la fecha
futura consultada. El baseline es la media histórica de unidades por tienda y
producto calculada con entrenamiento. La comparación MLP/baseline usa solo el
test sintético: no demuestra desempeño con ventas reales.
