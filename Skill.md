# Flujos de desarrollo

## Validar datos sintéticos

Ejecutar el generador y las pruebas de esquema:

```powershell
.\venv\Scripts\python.exe -m src.datos_sinteticos --output-dir data\synthetic --seed 42
.\venv\Scripts\python.exe -m pytest tests\test_datos_sinteticos.py -q
```

## Entrenar o evaluar el MLP

El entrenamiento es una acción explícita. Usar una ruta de salida nueva para
cada corrida; el pipeline conserva y rechaza sobrescribir los artefactos
existentes.

```powershell
.\venv\Scripts\python.exe -m src.pipeline_productos train
.\venv\Scripts\python.exe -m src.pipeline_productos evaluate
```

## Validar aplicación y frontend

```powershell
.\venv\Scripts\python.exe -m pytest -q
node --test tests\test_presentacion.js tests\test_app.js
```

La aplicación Flask debe leer solo el catálogo, inventario, checkpoint y
reporte configurados para el pipeline sintético.
