# Comandos de StockPredict IA

Comandos para PowerShell desde la raíz del repositorio. Los datos y artefactos
de modelo son sintéticos y no se versionan.

## Instalación e inicio desde cero

Requisito recomendado: Python 3.13 de 64 bits.

```powershell
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\venv\Scripts\python.exe -m pip check
.\venv\Scripts\python.exe -m src.datos_sinteticos --output-dir data\synthetic --seed 42
.\venv\Scripts\python.exe -m src.pipeline_productos train
.\venv\Scripts\python.exe -m src.pipeline_productos evaluate
.\venv\Scripts\python.exe -m src.web_app
```

Abrir `http://127.0.0.1:5000` y detener Flask con `Ctrl+C`. El servidor queda
ligado a localhost; no usarlo como servicio de producción.

## Dataset

El generador produce ventas sintéticas diarias de 3 tiendas y 30 productos
desde `2023-01-01` hasta `2026-11-16`, además de catálogo e inventario separados.
Con la semilla 42 se obtienen 127.440 registros de ventas. Los CSV generados
están en `data\synthetic` y son ignorados por Git.

```powershell
.\venv\Scripts\python.exe -m src.datos_sinteticos --output-dir data\synthetic --seed 42
```

## Entrenamiento y evaluación

El primer entrenamiento guarda `best_model.pt` y `metricas.json` en
`runs\synthetic_product_sales_v1`. El pipeline no sobrescribe artefactos
existentes; para otro entrenamiento se requiere una carpeta de salida nueva.

```powershell
.\venv\Scripts\python.exe -m src.pipeline_productos train
.\venv\Scripts\python.exe -m src.pipeline_productos train --output-dir runs\synthetic_product_sales_v2 --epochs 12 --batch-size 2048 --seed 42
.\venv\Scripts\python.exe -m src.pipeline_productos evaluate
```

El entrenamiento guarda el checkpoint y `metricas.json`, que contiene el
reporte de esa corrida, incluidas las métricas y el historial de validación por
época. `evaluate` no lee ese JSON: carga el checkpoint y los CSV, verifica sus
huellas y vuelve a calcular MAE/RMSE del MLP y del baseline en el test. No
entrena ni sobrescribe el checkpoint o el reporte.

La aplicación usa el checkpoint configurado en `src/composition.py`.
Flask también necesita el catálogo y el inventario generados en
`data\synthetic`. Si se entrena en otra carpeta, actualizar allí las rutas
antes de iniciar la web. El entrenamiento es explícito; nunca se ejecuta al
abrir Flask. El entrenamiento predeterminado ejecutó 12 épocas y tardó
aproximadamente 16,5 segundos en la verificación limpia con Python 3.13 y CPU;
el tiempo varía según el equipo.

## Pruebas y verificaciones

```powershell
.\venv\Scripts\python.exe -m pytest -q
node --test tests\test_presentacion.js tests\test_app.js
git diff --check
```

Node.js se requiere solamente para ejecutar las pruebas JavaScript; se
verificaron con Node.js 22.20.0. No es un requisito para iniciar Flask.

## Arquitectura actual

- **MVC:** rutas/controladores Flask reciben las solicitudes; plantillas
  presentan datos; los servicios contienen los casos de uso.
- **Controller–Service–Repository:** los controladores no leen CSV, JSON ni
  checkpoints. Los servicios trabajan con contratos y reciben dependencias por
  inyección.
- **Repository:** `RepositorioProductosCSV` y `RepositorioInventarioCSV`
  aíslan los CSV sintéticos de catálogo e inventario. No hay un repositorio
  JSON; `metricas.json` es escrito por el pipeline.
- **Adapter:** el adaptador PyTorch implementa el puerto de predicción.
- **Strategy:** `ReposicionBasica` y `ReposicionConSeguridad` comparten contrato
  y son intercambiables por el servicio de reposición.
- **SOLID:** SRP por separación de responsabilidades; OCP/LSP en estrategias;
  ISP en contratos pequeños; DIP en servicios desacoplados de implementaciones.
- No se agregaron Singleton, Builder ni Chain of Responsibility porque no
  resuelven una necesidad del flujo actual.

## Notas de datos y resultados

- Entrenamiento: `2023-01-01`–`2025-12-31`.
- Validación: `2026-01-01`–`2026-06-30`.
- Prueba: `2026-07-01`–`2026-11-16`.
- La predicción web permite hasta 90 días posteriores al último día del
  dataset; es un límite de alcance de la demostración, no una garantía de
  precisión.
- El inventario es un snapshot sintético al `2026-11-16`; no descuenta ventas
  posteriores ni se proyecta a la fecha consultada.
- `evaluate` muestra MAE/RMSE del MLP y del baseline de media histórica por
  tienda-producto, recalculados sobre el test sintético. El JSON guarda el
  reporte generado durante el entrenamiento; ninguna de estas métricas prueba
  desempeño con ventas reales.
- Las recomendaciones no modifican existencias ni realizan compras. Ningún
  dato mostrado representa ventas o inventario reales.
