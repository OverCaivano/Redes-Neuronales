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
- No incluye one-hot, escalado ni conexion al entrenamiento. Las pruebas utilizan fixtures pequenas.
- Validacion completa en PowerShell: `.\venv\Scripts\python.exe -m pytest -q`.


## Bloque 1C: variables temporales ciclicas

- El resultado de `generar_variables_historicas(datos)` agrega `dia_semana_seno`, `dia_semana_coseno`, `mes_seno` y `mes_coseno`, calculadas desde la fecha objetivo.
- Convencion: lunes=0 a domingo=6; enero=0 a diciembre=11. Angulo `2*pi*indice/periodo`, con periodos 7 y 12. Seno y coseno quedan en [-1, 1] y no requieren ajuste.
- Se conservan los rezagos, los conteos de descartes y las particiones existentes. El comando de tests sigue siendo el mismo.

## Bloque 1C: codificacion categorica one-hot

- `src/transformaciones.py`: `codificador, matrices = codificar_particiones(dividir_por_fecha(resultado))` ajusta exclusivamente con entrenamiento y transforma las tres particiones con la misma instancia.
- `CodificadorCategorias(entrenamiento)` verifica las fechas de entrenamiento y se ajusta una sola vez. `transformar(datos)` no reajusta; devuelve CSR float32, sin densificar, con dos unos por fila.
- `codificador.nombres_columnas` fija el orden: tiendas ascendentes y luego familias alfabeticas, conservando los nombres originales. Las categorias desconocidas producen un error claro.
- Las matrices contienen solo tienda/familia. Conservan el orden de filas; fechas, objetivo y variables numericas siguen en las particiones originales. No incluye escalado ni conexion al MLP.
- Tests: `.\venv\Scripts\python.exe -m pytest -q`.


## Bloque 1C: escalado y entradas completas

- `preparador, matrices = preparar_particiones(particiones)` en `src/transformaciones.py` ajusta one-hot y StandardScaler solo con entrenamiento; validacion/test reutilizan esas instancias sin reajuste.
- Orden de `preparador.nombres_columnas`: ventas_hace_1_dia, ventas_hace_7_dias, promedio_ventas_7_dias, cantidad_en_promocion (escaladas); dia_semana_seno, dia_semana_coseno, mes_seno, mes_coseno (sin escalar); tiendas y familias one-hot (sin escalar).
- `PreparadorEntradas.transformar(datos, tamano_bloque=100_000)` produce CSR float32 por bloques. Solo el bloque numerico/ciclico es denso temporalmente; nunca se densifica one-hot. Comprueba finitud y conserva la posicion de cada fila, incluso con indices repetidos.
- Objetivos y metadatos permanecen intactos en las particiones. No hay log1p ni conexion al entrenamiento. `parametros_escalado` devuelve copias de media, varianza y escala.
- El API anterior `codificar_particiones` sigue devolviendo solo categorias. Tests: `.\venv\Scripts\python.exe -m pytest -q`.

## Bloque 1D: adaptador CSR a lotes de PyTorch

- `src/dataset_favorita.py`: `dataset = DatasetFavorita(X, y)` exige X CSR float32 no vacia y objetivos numericos finitos [N] o [N, 1]; conserva X sin copia completa y copia y como float32 [N, 1]. No modificar X mientras se usa el Dataset.
- `loader = crear_dataloader(dataset, batch_size=256, shuffle=False, semilla=42, num_workers=0)` conserva el ultimo lote incompleto y usa un generador reproducible independiente.
- `collate_lote(indices, dataset=dataset)` selecciona las mismas posiciones en X e y y densifica solo esa seleccion; devuelve tensores float32 [B, D] y [B, 1]. Ignora etiquetas pandas: X e y deben llegar en el mismo orden.
- `dataset.input_dim` se obtiene de X.shape[1]. El test de compatibilidad realiza un forward del MLP con no_grad, sin optimizador ni cambios de pesos. No se conecta entrenamiento ni se crean checkpoints nuevos del adaptador.
- Pruebas completas: `.\venv\Scripts\python.exe -m pytest -q`. Revision: `git diff --check` y `git status`.

## Bloque 1D-B: construccion de los tres loaders Favorita

- `from src.loaders_favorita import construir_loaders_favorita`.
- `resultado = construir_loaders_favorita(ruta="data/raw/train.csv", tiendas=[1, 2], familias=["AUTOMOTIVE", "BABY CARE", "BEAUTY"], batch_size=256, semilla=42, num_workers=0)`. Omitir filtros utiliza todas las series y conserva toda su historia.
- `resultado.loaders` contiene entrenamiento, validacion y test; solo entrenamiento mezcla filas. `preparador`, `nombres_columnas`, `input_dim`, `metadatos`, `filas_entrada`, `filas_descartadas_historia` y `filas_por_particion` permiten inspeccionar la preparacion.
- Se genera historia antes de dividir por las fechas fijadas; se ajusta una unica vez con entrenamiento. Particiones vacias o categorias desconocidas producen errores. CSR float32 y objetivos float32 [N,1] se validan sin densificacion completa.
- Los metadatos id/fecha/tienda/familia estan alineados por posicion del Dataset; validacion/test mantienen ese orden. No asociar secuencialmente metadatos con lotes mezclados de entrenamiento. Se liberan tablas intermedias y se conserva familia como categoria en metadatos.
- No entrena, no escribe datasets ni checkpoints. Tests: `.\venv\Scripts\python.exe -m pytest -q`. Revision: `git diff --check` y `git status`.

## Bloque 1E-A: entrenamiento piloto Favorita

- Configuracion: `configs/favorita.yaml` conserva 10 epocas, CPU, semilla 42, batch 256, Adam lr 0.001 y el subconjunto de dos tiendas/tres familias.
- Piloto limitado a dos epocas: `.\venv\Scripts\python.exe -m src.train --config configs/favorita.yaml --epochs 2`. La configuracion efectiva guardada refleja dos epocas; el YAML original no cambia.
- `src/metricas.py` acumula MAE/MSE/RMSE por muestra. Validacion no calcula gradientes y compara con el rezago semanal reconstruido a unidades originales. Test no se evalua ni selecciona modelos.
- `src/checkpoints.py` escribe archivos temporales y reemplaza cada destino al completar la escritura. `runs/favorita_subconjunto/` contiene metricas.csv, resumen.json, configuracion.yaml, last.pt y best.pt; best se selecciona por RMSE de validacion.
- Los checkpoints incluyen pesos, optimizador, epoca, configuracion, dimension, columnas, preparador ajustado, historial y estados aleatorios. La carga de estos artefactos propios con objetos sklearn requiere `torch.load(..., weights_only=False)`. No se ha implementado reanudacion ni evaluacion de test; evaluate.py sigue siendo el evaluador sintetico.
- Una carpeta de salida no vacia se rechaza para no sobrescribir corridas. Para otra corrida usar una ruta distinta. Los errores numericos detienen la corrida antes de guardar la epoca fallida; queda el ultimo checkpoint valido.
- No hay log1p ni recorte de predicciones; objetivos y metricas estan en unidades originales. El MSE de entrenamiento corresponde a predicciones durante las actualizaciones de la epoca.
- Verificar: `.\venv\Scripts\python.exe -m pytest -q`, `git diff --check` y `git status`. No agregar runs/ a Git.

## Bloque 1E-B: reanudacion entre epocas

- `.\venv\Scripts\python.exe -m src.train --resume runs/favorita_subconjunto/last.pt --epochs 4` continua hasta cuatro epocas TOTALES. Si se omite --config se recupera la configuracion del checkpoint. Un --config explicito se valida: solo se permite cambiar el total, la ubicacion del mismo CSV (verificada por SHA-256) y la carpeta de la copia.
- Solo last.pt es un punto de reanudacion. La carpeta de resultados se toma de su ubicacion, por lo que se puede trabajar sobre una copia completa sin tocar el original. Si el total ya fue alcanzado, no se entrena ni se escriben archivos.
- `src/reanudacion.py` verifica configuracion, columnas, particiones, historial, best.pt y huella del dataset. `construir_loaders_favorita(..., preparador=...)` reutiliza las transformaciones guardadas sin fit.
- Se restauran pesos, Adam, mejor epoca/RMSE, historial y RNG de PyTorch y shuffle; formato v2 agrega Python, NumPy y generadores de loaders. Comparacion exacta comprobada en CPU con el mismo entorno. No se garantiza equivalencia bit a bit entre versiones, hardware o CPU/GPU; cambios de dispositivo se rechazan en esta version.
- Compatibilidad legacy v1: no existe huella historica del CSV. Se comprueban periodos, conteos y metricas de validacion, se advierte esta limitacion y se registra SHA-256 al guardar v2. Eso no demuestra que el CSV sea identico al historico. No hay fit adicional.
- Guardado de epoca: prepara todos los archivos temporalmente, publica last.pt al final y revierte reemplazos ante excepciones de E/S. Un corte de energia durante multiples reemplazos no es una transaccion atomica; inconsistencias se rechazan y requieren restaurar una copia completa.
- Respaldar la carpeta completa antes de reanudar pilotos importantes, dentro de runs/ (ignorado por Git). No se reanuda a mitad de epoca y no se evalua test.
