# Comandos de Ejecución Autónoma

Este archivo almacena la memoria de comandos que el Agente IA debe emplear para gestionar el proyecto.


## Instalación rápida desde cero (Windows / PowerShell)

> **Clonar NO alcanza para realizar predicciones.** GitHub incluye la aplicación, pero no los CSV ni los checkpoints Favorita. Antes de la demostración, solicitá al autor los artefactos indicados abajo. No hay un enlace público de descarga verificado de estos modelos en el repositorio.

**Requisitos:** Git y Python **3.13 de 64 bits**, con el lanzador `py` disponible. Los dos checkpoints locales fueron comprobados con Python 3.13, PyTorch 2.13.0 y scikit-learn 1.9.0. El cargador rechaza versiones incompatibles; Python 3.10 o 3.12 no sirven para estos artefactos.

### 1. Clonar e instalar

Abrí PowerShell en la carpeta donde quieras guardar el proyecto:

```powershell
git clone https://github.com/OverCaivano/Redes-Neuronales.git
cd Redes-Neuronales
git switch main
py -3.13 --version
py -3.13 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-web.txt -c requirements-colab.txt torchvision==0.28.0
.\venv\Scripts\python.exe -m pip check
```

No hace falta activar el entorno: todos los comandos utilizan directamente su intérprete. `requirements-web.txt` ya incluye `requirements.txt`. Se usa `requirements-colab.txt` como restricciones de versiones también en Windows porque fija las bibliotecas de los checkpoints; torchvision 0.28.0 coincide con el entorno local comprobado. No instalar luego versiones diferentes con otro comando sin restricciones. No se necesita GPU, CUDA, Node ni recompilar CSS para iniciar la web.

**Límite de esta verificación:** se comprobó la carga de los modelos en el entorno local existente; no se reinstalaron dependencias ni se ensayó una instalación limpia en otra computadora. Si pip no consigue una versión compatible, detenerse y consultar al autor; no quitar los pins para forzar la carga.

### 2. Obtener los archivos externos

- Dataset: obtener los CSV originales de la competencia **Store Sales - Time Series Forecasting** de Corporación Favorita en Kaggle, con los permisos que exija la plataforma. Para esta web se necesitan `train.csv` y, para pronósticos recursivos, `test.csv`. También se pueden recibir del autor por un medio permitido. **No editar ni volver a guardar los CSV:** `train.csv` debe coincidir por SHA-256 con el utilizado al entrenar.
- Modelos y resultados: solicitar al autor una copia de las carpetas enumeradas en la siguiente sección, tomada del proyecto local o de su respaldo verificado en Drive. No existe descarga automática desde Flask ni acceso automático a Drive. No cambiar el nombre `best.pt`, ni usar `models/best_model.pt` como sustituto.
- Extraer/copiar dentro de la raíz `Redes-Neuronales`, conservando exactamente la estructura `data/raw/` y `runs/`. Si un archivo comprimido contiene ambas carpetas, no dejar un nivel extra como `Redes-Neuronales/paquete/runs/`.

Comprobar los cuatro archivos mínimos para **ambos modelos y el pronóstico recursivo**:

```powershell
@(
  'data/raw/train.csv',
  'data/raw/test.csv',
  'runs/favorita_subconjunto_reanudado_4/best.pt',
  'runs/favorita_completa_20/best.pt'
) | ForEach-Object {
  if (-not (Test-Path -LiteralPath $_ -PathType Leaf)) { throw "Falta el archivo: $_" }
}
```

Este control comprueba existencia, no compatibilidad. La aplicación verifica después el checkpoint, las transformaciones y la huella del dataset. Cargar únicamente checkpoints propios o recibidos del autor: contienen objetos Python serializados.

### 3. Iniciar, usar y detener

Desde la raíz del repositorio:

```powershell
.\venv\Scripts\python.exe -m src.web_app
```

Dejar esa terminal abierta y abrir **http://127.0.0.1:5000** en el navegador de la misma computadora. En Predicciones, probar tienda **1**, familia **BEAUTY** (mostrada en español), fecha **2017-07-16** con el piloto o el modelo completo. Para el pronóstico recursivo usar origen **2017-08-15** y horizonte **7**; requiere el modelo completo y las promociones de `test.csv`.

Detener con **Ctrl+C** en la terminal. No es necesario entrenar ningún modelo para esta instalación. El servidor escucha solo en localhost, sin depurador. Las fuentes de Google son opcionales: sin Internet se usan fuentes del sistema.

## Archivos necesarios para ejecutar las predicciones

Estado comprobado en la rama local `main` el **12/11/2026**. Rutas relativas a la raíz del repositorio:

| Archivo o carpeta | ¿Incluido en Git? | Para qué se necesita |
|---|---|---|
| `src/`, `web/templates/`, `web/static/` | Sí | Aplicación, API, lógica predictiva y estilos compilados. |
| `requirements.txt`, `requirements-web.txt`, `requirements-colab.txt`, `configs/` | Sí | Dependencias y configuraciones de referencia. |
| `notebooks/favorita_colab.ipynb` | Sí | Flujo de entrenamiento en Colab; no es necesario para usar los modelos entregados. |
| `data/raw/train.csv` | **No; ignorado** | Consultas históricas e historial para ambos modelos. Localmente: 121.800.373 bytes. |
| `data/raw/test.csv` | **No; ignorado** | Promociones futuras para pronósticos recursivos; no contiene ventas objetivo. Localmente: 1.022.269 bytes. |
| `runs/favorita_subconjunto_reanudado_4/best.pt` | **No; ignorado** | Predicción piloto: tiendas 1 y 2, familias AUTOMOTIVE, BABY CARE y BEAUTY, 13 entradas. |
| `runs/favorita_completa_20/best.pt` | **No; ignorado** | Predicción completa y recursiva: 54 tiendas, 33 familias, 95 entradas; corrida de 20 épocas. |
| `models/best_model.pt` | Sí | Checkpoint del pipeline sintético anterior; **no reemplaza** los modelos Favorita. |

Para que **Evaluación e Historial** también muestren toda la información, pedir estas carpetas completas (todas existen localmente, pero **ninguna está versionada**):

- `runs/favorita_completa_20/`: `best.pt`, `last.pt`, `configuracion.yaml`, `metricas.csv`, `resumen.json`, `manifiesto.json` y `COMPLETO`. El historial verifica manifiesto, hashes y coherencia; no editar estos archivos. `last.pt` se conserva en el paquete para reanudación, pero no es necesario para inferencia.
- `runs/favorita_subconjunto/` y `runs/favorita_subconjunto_reanudado_4/`: en cada una, `best.pt`, `last.pt`, `configuracion.yaml`, `metricas.csv` y `resumen.json`. Para mostrar el historial piloto se leen `metricas.csv` y `resumen.json`.
- `runs/evaluacion_completa_20/evaluacion.json`: evaluación diaria del modelo completo; requiere también su `resumen.json` y checkpoint.
- `runs/evaluaciones/favorita_piloto_test/evaluacion.json`: evaluación diaria piloto.
- `runs/evaluaciones/pronostico_recursivo_validacion_20261111_final/informe.json`: backtest recursivo de validación. Los CSV acompañantes son útiles para auditar, pero la página lee el JSON.

**Mínimo piloto:** `train.csv` y el `best.pt` del piloto. **Todas las predicciones:** los cuatro archivos de la comprobación anterior. **Demostración completa con métricas e historial:** además, los artefactos de esta lista. Los demás CSV de Favorita no son necesarios para iniciar la web ni consultar los modelos; `stores.csv` sí participa en el comando separado de backtest por tipos de tienda.

No basta con copiar pesos de otra corrida: cada checkpoint contiene su preparador y debe corresponder a los resultados archivados. No modificar los paths de Colab dentro del checkpoint/configuración archivada: la web utiliza las rutas locales fijas de `src/web_app.py`. Para consultas CLI del modelo completo, indicar `--data-path data/raw/train.csv` si el checkpoint guarda una ruta `/content/...`.

Los YAML de `configs/` son recetas para nuevas corridas, no contienen pesos: `configs/favorita_completa.yaml` configura 10 épocas y otra salida; no genera por sí solo el paquete archivado `favorita_completa_20`. El notebook permite entrenar y guardar artefactos por separado, pero no es un paso de instalación rápida ni garantiza reproducir los mismos resultados. Para la entrega al profesor, usar la corrida ya verificada del autor.

## Funcionalidades actuales y alcance

- **Piloto y modelo completo:** ambos están disponibles en el código; solo funcionan si sus archivos externos están presentes. El completo ya existe en esta computadora, pero un clon no lo recibe.
- **Consulta histórica:** predice una fecha existente con siete días previos observados, sin usar la venta real del día objetivo. Se reutiliza el preparador guardado, sin reajuste.
- **Pronóstico recursivo:** de 1 a 7 días con el modelo completo. Usa sus propias estimaciones como historia para los días siguientes y exige promociones conocidas para cada fecha. La demostración desde 2017-08-15 pronostica 2017-08-16 a 2017-08-22; no predice ventas actuales de 2026.
- **Evaluación:** separa el test diario reservado, el backtest recursivo de validación y el piloto. Sus métricas no son intercambiables ni se recalculan al abrir la página.
- **Historial:** muestra épocas y métricas de archivos locales; no consulta ni sincroniza Drive automáticamente. Si faltan archivos muestra advertencias, no resultados ficticios.
- La interfaz presenta unidades enteras y familias traducidas; la API conserva decimales e identificadores originales. No se calcula stock, recomendaciones de compra ni intervalos de confianza.

## Solución de errores frecuentes

| Problema | Qué hacer |
|---|---|
| `py` o Python no encontrado | Instalar Python 3.13 de 64 bits con el lanzador de Windows, cerrar y abrir PowerShell y comprobar `py -3.13 --version`. Si se usa `python` en lugar de `py`, verificar primero `python --version` y crear el entorno con ese Python 3.13. |
| No existe `venv\Scripts\python.exe` | Estar dentro de `Redes-Neuronales` y ejecutar `py -3.13 -m venv venv`. No hace falta activar scripts ni cambiar la política de ejecución de PowerShell. |
| `ModuleNotFoundError` o dependencias faltantes | Ejecutar el comando de instalación de la sección rápida usando el intérprete del venv, no `pip` global; luego `python.exe -m pip check` con la misma ruta completa. |
| Versiones incompatibles / pip no encuentra una versión | Confirmar Python 3.13 y las restricciones indicadas. Pedir al autor el entorno compatible; no editar el checkpoint ni saltarse la validación. Las versiones en requirements generales no están todas fijadas. |
| Puerto 5000 ocupado | Ejecutar `.\venv\Scripts\python.exe -m src.web_app --port 5001` y abrir **http://127.0.0.1:5001**. |
| Dataset faltante o SHA-256 diferente | Colocar el `train.csv` original en `data/raw/`, sin convertirlo con Excel ni modificar saltos de línea. Para pronósticos, colocar también `test.csv`. Solicitar al autor la copia coincidente si el hash no coincide. |
| «Modelo no disponible» o checkpoint faltante | Revisar los dos paths `runs/.../best.pt` indicados arriba y los mensajes de la terminal. Un clon no descarga esos archivos. No usar el checkpoint sintético. |
| Evaluación o historial ausente | Copiar los JSON, CSV, configuración y manifiesto de la misma corrida a sus rutas exactas; no regenerar valores manualmente. |
| Fecha sin historia compatible | Elegir una fecha permitida; los huecos de calendario no se rellenan con ceros. Para la demostración recursiva usar el origen 2017-08-15. |


## Referencia de desarrollo (no necesaria para instalar la web)

Los comandos siguientes incluyen entrenamiento y backtests: no ejecutarlos como parte del arranque para el profesor.

## Entrenamiento y evaluación

```bash
# Entrenar el boilerplate sintético, no el modelo Favorita de la web
python src/train.py --config configs/default.yaml

# Ejecutar evaluación/test del modelo
python src/evaluate.py --model_path models/best_model.pt
```

## Pruebas y calidad de código

```bash
# Correr tests unitarios
pytest tests/

# Probar traducciones y formato visual de la interfaz
node --test tests/test_presentacion.js

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


## Bloque 1E-C: Google Colab y Drive

- Abrir https://colab.research.google.com/ y usar Archivo > Abrir notebook > GitHub, repositorio `https://github.com/OverCaivano/Redes-Neuronales.git`, archivo `notebooks/favorita_colab.ipynb`. Publicar primero estos cambios por el flujo habitual; este bloque no hace commit ni push. Tambien se puede subir el notebook, pero el codigo clonado debe contener este bloque.
- Completar URL/revision, rutas del CSV y resultados bajo `/content/drive/`, total de epocas, batch, semilla y dispositivo. Ejecutar celdas en orden; `drive.mount` solicita autorizacion. El CSV se copia y verifica en `/content`, sin modificar Drive. No subirlo a GitHub.
- `configs/favorita_completa.yaml` es independiente del piloto y no tiene filtros. Dimension automatica, particiones y MLP existentes. `requirements-colab.txt` fija las versiones observadas; se instala en un venv aislado del kernel. No se cambia el entorno local. Si no existen ruedas para el runtime, detenerse y seleccionar un entorno compatible; no actualizar sklearn silenciosamente al reanudar.
- Corrida nueva: `MODO="nuevo"`, carpeta Drive vacia y `EJECUTAR_ENTRENAMIENTO=True` despues de revisar parametros. El notebook invoca `python -m src.train --config <config_local> --epochs <total> --sync-dir <carpeta_drive>`.
- Guardado: primero la transaccion local existente, despues `src/sincronizacion.py` copia cada epoca a una generacion nueva `epoca_000001_<uuid>`. Conserva last.pt, best.pt, metricas.csv, resumen.json y configuracion.yaml; agrega manifiesto con SHA-256/tamanos, entorno y huella de codigo, y publica COMPLETO al final. Verifica por relectura. No pisa ni elimina generaciones anteriores. Un unico escritor por carpeta; no lanzar sesiones concurrentes.
- Reanudar: misma revision (guardar el commit impreso), entorno, semilla, batch y dispositivo; `MODO="reanudar"` y numero TOTAL de epocas. Restaura a una carpeta local nueva la ultima copia integra, avisa si omite una corrupta, valida dataset/configuracion y usa `--resume <local>/last.pt`. Rechaza checkpoints del subconjunto. No reajusta transformaciones. No se garantiza identidad numerica entre hardware o CPU/GPU.
- Ante fallo de Drive el entrenamiento se detiene conservando el ultimo checkpoint local. El notebook documenta como reintentar `sincronizar_epoca(local, remoto)` antes de perder la VM, sin entrenar. Si se perdio la VM se recupera la ultima generacion integra; el trabajo de una epoca incompleta se pierde. El montaje de Drive no garantiza una transaccion distribuida ni persistencia remota instantanea. Revisar cuota/espacio: se conservan todas las generaciones.
- CPU es el valor inicial; se detecta CUDA y se rechaza GPU no disponible. No se supone que GPU sea mas rapida. Se muestran ejemplos, dispositivo, epoca, MAE/RMSE, mejor epoca y rutas. No se evalua test ni se ejecuta entrenamiento completo durante la implementacion.
- Pruebas locales usan carpetas temporales que simulan Drive: `.\venv\Scripts\python.exe -m pytest -q`; luego `git diff --check` y `git status`. Dataset y runs/ siguen ignorados. Solo cargar checkpoints propios: SHA-256 detecta corrupcion, no autentica origen.


## Evaluacion del test interno Favorita

Evaluar exclusivamente un `best.pt` propio y confiable, elegido previamente por RMSE de validacion:

```powershell
.\venv\Scripts\python.exe -m src.evaluate --source favorita --model_path runs/favorita_subconjunto_reanudado_4/best.pt --output-dir runs/evaluaciones/favorita_piloto_test
```

- No entrena ni reajusta transformaciones. Recupera arquitectura, filtros, columnas y preparador del checkpoint. Usa CPU, matrices CSR y lotes; recorre solo test. El constructor existente reconstruye las tres particiones en memoria, sin recorrer entrenamiento/validacion para calcular metricas.
- Test interno: 2017-07-16 a 2017-08-15, extraido del mismo `train.csv`. NO usar `test.csv` de Kaggle. El SHA-256 del archivo debe coincidir. `--data-path <ruta/train.csv>` permite cambiar solamente la ubicacion del mismo archivo.
- Evaluacion diaria a un paso: cada fecha usa historia observada hasta t-1, incluso dias anteriores del test. No es un pronostico de todo el mes emitido desde una unica fecha. Las promociones del dia objetivo se consideran conocidas anticipadamente.
- MAE y RMSE se acumulan por muestra, en unidades originales; el baseline semanal se reconstruye desde el rezago escalado y usa las mismas filas. No se usa test para elegir epocas, hiperparametros ni nuevas decisiones de ajuste.
- Se requiere formato v2 con SHA-256, periodos, versiones y preparador compatibles. Los checkpoints antiguos sin esa informacion se rechazan. La carga contiene objetos Python; solo archivos propios/confiables.
- `--output-dir` es opcional: sin el argumento solo imprime. Si se utiliza, debe ser una carpeta NUEVA dentro de `runs/`, fuera de la corrida original. Guarda `evaluacion.json` con metricas, conteos, periodo, huellas y duracion. No sobrescribe resultados anteriores ni cambia `resumen.json` o checkpoints de entrenamiento. Para repetir, omitir salida o elegir otra carpeta.
- El evaluador anterior se conserva: `.\venv\Scripts\python.exe -m src.evaluate --model_path models/best_model.pt` devuelve MSE de validacion sintetica; tambien admite `--source synthetic`.
- Validacion: `.\venv\Scripts\python.exe -m pytest -q`, `git diff --check`, `git status`.


## Bloque B: consultas de ventas estimadas Favorita

```powershell
.\venv\Scripts\python.exe -m src.predecir_favorita --checkpoint runs/favorita_subconjunto_reanudado_4/best.pt --tienda 1 --familia BEAUTY --fecha 2017-07-16
```

- Consulta retrospectiva de una fila real, con modelo fijo. No entrena, no reajusta transformaciones, no escribe resultados ni consulta la venta real del objetivo para generar entradas. No es stock disponible ni recomendacion de compra.
- Se exige `best.pt` propio/confiable y compatible, las categorias originales y el mismo CSV identificado por SHA-256. `--data-path <ruta/train.csv>` cambia solo su ubicacion. Familias con espacios deben ir entre comillas. No se permiten filtros/categorias ajenos al entrenamiento.
- Debe existir la fila de la fecha consultada para obtener la promocion del dia (asumida planificada y conocida anticipadamente), y los siete dias calendario previos con ventas observadas. Si falta historia o fecha, se rechaza; no se inventan ventas ni promociones futuras.
- `src/predecir_favorita.py` reutiliza lector, generador temporal, preparador y carga de checkpoint. Antes de generar variables selecciona solo t-7..t y oculta sales(t). Usa un marcador interno cero para cumplir el contrato del generador; NO representa ventas ni imputa huecos, no se usa como rezago y se elimina antes de transformar. Fechas posteriores no participan. Solo se densifica una fila.
- Se muestra la salida original del MLP sin redondear a enteros ni recortar negativos. Una salida negativa se informa como limitacion del modelo. Son ventas agregadas diarias por familia/tienda.
- Es una consulta historica con pesos ya ajustados: consultar fechas de entrenamiento no constituye un backtest fuera de muestra ni demuestra que el modelo existiera en esa fecha. No usar consultas del test para seleccionar modelos o hiperparametros.
- No cambia el comando de evaluacion sintetica ni Favorita. Pruebas: `.\venv\Scripts\python.exe -m pytest -q`; revision: `git diff --check` y `git status`.


## Bloque 3: backtest recursivo Favorita

```powershell
.\venv\Scripts\python.exe -m src.evaluar_pronosticos_recursivos
```

- Sin entrenamiento. Usa seis origenes de validacion, horizonte maximo de siete dias, cinco tiendas (una tienda central por tipo, ordenadas por `store_nbr`) y cuatro familias; el test se reserva para una evaluacion final.
- Reutiliza `PronosticadorFavorita` y el checkpoint completo. Mide MAE/RMSE por horizonte, general y por tienda/familia; compara con `ventas(t-7)`.
- Las promociones futuras se leen de `train.csv` y se asumen conocidas al origen, aunque el dataset no registra su fecha de publicacion. Las ventas futuras solo se usan como etiquetas de evaluacion, nunca como entradas.
- Escribe JSON y CSV en una carpeta nueva con marca de tiempo dentro de `runs/evaluaciones/`; no sobrescribe resultados. Por defecto son 20 series, seis ventanas y hasta 840 pronosticos. Revisar el informe para ventanas omitidas y limitaciones de muestra.
- En la ejecución de referencia (840 pronósticos), el modelo obtuvo MAE 301,54 y RMSE 681,90 frente a MAE 288,23 y RMSE 750,25 del baseline semanal: resultados mixtos, no evidencia de precisión generalizable.


## Bloque C: StockPredict IA — aplicación web local

Con el entorno y los artefactos externos ya preparados según «Instalación rápida desde cero», desde la raíz del repositorio:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-web.txt -c requirements-colab.txt torchvision==0.28.0
.\venv\Scripts\python.exe -m src.web_app
```

Abrir **http://127.0.0.1:5000**. Detener con Ctrl+C. Si el puerto está ocupado: `.\venv\Scripts\python.exe -m src.web_app --port 5001`.

Para demostrar: generar una consulta histórica con el piloto y otra con el modelo completo en **Predicciones**; después generar siete días con origen `2017-08-15`. Las predicciones son estimaciones, no stock ni recomendaciones de compra.

- Cinco páginas: Inicio, Predicciones, Evaluación, Historial y Acerca. HTML adaptado de Stitch mediante plantillas Jinja compartidas; JavaScript nativo. Flask escucha únicamente en localhost, sin depurador. Es un servidor para uso local, no un despliegue público.
- Diseños originales copiados sin alteraciones a `disenos/stitch/`: Descargas/1 → predicciones, /2 → inicio, /3 → evaluacion, /4 → acerca, /5 → historial. Cada pantalla conserva HTML y DESIGN con nombres descriptivos; `procedencia.json` registra hashes. Estas referencias contienen el contenido ficticio original; no se sirven por HTTP. Las plantillas adaptadas están en `web/templates/`.
- La página Predicciones permite elegir el piloto (`runs/favorita_subconjunto_reanudado_4/best.pt`, 13 entradas) o el modelo completo (`runs/favorita_completa_20/best.pt`, 95 entradas); ambos usan `data/raw/train.csv`. El piloto continúa seleccionado por defecto. No hay selector de rutas ni carga de checkpoints mediante HTTP.
- `GET /api/opciones?modelo=piloto|completo` publica solo las categorías del checkpoint y fechas/rangos históricos para la interfaz, no ventas originales ni CSV. `POST /api/predicciones` recibe `tienda`, `familia`, `fecha` y el campo opcional `modelo` (`piloto` por defecto); reutiliza `predecir_favorita`, comprueba el SHA-256 del dataset y las siete fechas previas observadas. No entrena ni usa la venta objetivo como entrada. El servidor rechaza fechas sin fila o historia suficiente; en el formulario completo los límites de fecha son un rango y la validez exacta se confirma en el servidor.
- La seccion Pronostico futuro usa exclusivamente el modelo completo, hasta siete dias desde la ultima fecha historica. `src/pronosticar_favorita.py` recibe historia y promociones por separado, reutiliza transformaciones guardadas y realimenta solo sus predicciones. La web lee promociones de `data/raw/test.csv`; no usa ventas futuras. La API `POST /api/pronosticos` recibe `tienda`, `familia`, `fecha_origen` y `horizonte` (1-7). El ejemplo disponible es de 2017, no es una prediccion para 2026.
- Ejemplo real del piloto: tienda 1, familia BEAUTY, fecha 2017-07-16. El botón de ejemplo solo completa el formulario; la predicción se solicita al backend. Sin resultados simulados, intervalos de confianza ni recomendaciones de compra. El ejemplo funciona también con el modelo completo cuando su checkpoint está presente.
- Evaluación lee `runs/evaluaciones/favorita_piloto_test/evaluacion.json` y verifica que corresponda al checkpoint. Historial lee los resúmenes y métricas del piloto inicial y su continuación. Si falta un artefacto se informa; no se reemplaza por métricas inventadas. La aplicación no consulta Drive y usa checkpoints locales.
- API de lectura pública: `GET /api/resumen`, sin rutas privadas, pesos ni configuraciones completas. Los errores HTTP no muestran trazas. Dataset y resultados continúan ignorados por Git.
- Los estilos compilados están incluidos en `web/static/estilos.css`: no se necesita Node para iniciar. Fuentes Google opcionales; sin Internet se usan fuentes del sistema. No se utiliza el script CDN de Tailwind en la aplicación.
- El selector claro/oscuro está disponible en las cinco pantallas. La preferencia `stockpredict-theme` se conserva en `localStorage`; sin preferencia usa `prefers-color-scheme`. El script `web/static/tema.js` aplica el tema antes de cargar estilos.
- El adaptador `web/adaptar_stitch.py` es anterior a las integraciones actuales de selección de modelos, tema y pronóstico. No lo ejecutes para regenerar plantillas en esta versión: las sobrescribiría y perderías esas integraciones. Los estilos ya están compilados; no se necesita Node para iniciar.
- Pruebas: `.\venv\Scripts\python.exe -m pytest -q`, `git diff --check`, `git status`. Las pruebas API usan checkpoints pequeños en carpetas temporales.

Ejemplo de consulta HTTP en PowerShell con el servidor iniciado:

```powershell
$cuerpo = @{ tienda = 1; familia = "BEAUTY"; fecha = "2017-07-16" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/predicciones -ContentType "application/json" -Body $cuerpo
```

Pronostico recursivo de siete dias:

```powershell
$cuerpo = @{ tienda = 1; familia = "BEAUTY"; fecha_origen = "2017-08-15"; horizonte = 7 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/pronosticos -ContentType "application/json" -Body $cuerpo
```
