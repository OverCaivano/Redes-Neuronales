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
- **Repository:** los repositorios aíslan los CSV sintéticos.
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
- El comando `.\venv\Scripts\python.exe -m src.pipeline_productos evaluate`
  informa MAE/RMSE del MLP y del baseline sobre el test sintético, sin
  reentrenar.
- `metricas.json` conserva las métricas de evaluación y el historial de épocas.
- Las pruebas y reportes corresponden exclusivamente a datos sintéticos.
- Las recomendaciones no modifican existencias ni realizan compras. Ningún
  dato mostrado representa ventas o inventario reales.
