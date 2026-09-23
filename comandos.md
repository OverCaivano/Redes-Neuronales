# Comandos de Ejecución Autónoma

Este archivo almacena la memoria de comandos que el Agente IA debe emplear para gestionar el proyecto.

## 🐍 Entorno y Dependencias
```bash
# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # En Linux/Mac
# venv\Scripts\activate   # En Windows

# Instalar dependencias
pip install -r requirements.txt
```

## Entrenamiento y evaluación

```bash
# Ejecutar entrenamiento del modelo principal
python src/train.py --config configs/default.yaml

# Ejecutar evaluación/test del modelo
python src/evaluate.py --model_path models/best_model.pt
```

## Pruebas y calidad de código

```bash
# Correr tests unitarios
pytest tests/

# Formateo e inspección de código
black src/
flake8 src/
```

## Bloque 1B: lectura e historia temporal de Favorita

- `src/datos_favorita.py`: `leer_datos_favorita(ruta, tiendas=None, familias=None)` lee y valida el subconjunto seleccionado sin modificar el CSV.
- `src/preprocesamiento_temporal.py`: ejecutar `generar_variables_historicas(datos)` antes de `dividir_por_fecha(resultado)`. El resultado incluye `datos`, `filas_entrada`, `filas_validas` y `filas_descartadas_historia`.
- Se requieren siete dias calendario previos observados. Los huecos no se imputan. Las promociones del dia objetivo se suponen planificadas y conocidas al cierre del dia anterior.
- Particiones por fecha objetivo: entrenamiento hasta 2017-06-15; validacion 2017-06-16 a 2017-07-15; test interno 2017-07-16 a 2017-08-15, desde 2013-01-01.
- No incluye codificacion, escalado ni conexion al entrenamiento. Las pruebas utilizan fixtures pequenas.
- Validacion completa en PowerShell: `.\venv\Scripts\python.exe -m pytest -q`.
