# 🧠 Sistema Inteligente de Predicción de Demanda y Gestión de Stock mediante Redes Neuronales

Este repositorio contiene un prototipo académico de predicción de ventas con Favorita, desarrollado con Python, PyTorch y Flask. Incluye modelos piloto y completo, consultas históricas y un pronóstico recursivo de hasta siete días para una demostración con datos de 2017.

La aplicación estima ventas por tienda y familia; no conoce el inventario actual y no genera recomendaciones de reposición. El soporte de datos sintéticos sigue disponible para las pruebas del pipeline original.


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


---

## 🎯 Objetivo del proyecto

El objetivo de largo plazo es:

> **Desarrollar una red neuronal capaz de predecir la demanda futura de productos utilizando información histórica de ventas y variables temporales, con el objetivo de generar recomendaciones de reposición de stock.**

La aplicación actual todavía no sirve como herramienta de gestión de inventario: solo estima ventas a partir del historial disponible.

A partir del análisis de datos históricos, la red neuronal buscará aprender patrones relacionados con:

* ventas anteriores;
* comportamiento de cada producto;
* días de la semana;
* meses;
* temporadas;
* feriados;
* promociones;
* precios;
* stock disponible;
* tendencias históricas de demanda.

El modelo Favorita actual utiliza rezagos y promedio de ventas, promociones, variables temporales cíclicas y codificaciones de tienda/familia. El resto de las variables de esta lista es trabajo futuro.

---

## 💡 Problema a resolver

Una gestión ineficiente del inventario puede producir principalmente dos situaciones:

### Falta de stock

Cuando la demanda supera las unidades disponibles, el comercio puede perder ventas y clientes.

### Exceso de stock

Comprar cantidades muy superiores a la demanda genera costos innecesarios de almacenamiento y puede producir pérdidas, especialmente en productos perecederos.

La estimación de ventas puede ser un insumo de ese análisis, pero el prototipo no integra existencias ni anticipa faltantes.

Para recomendar una reposición harían falta, entre otros datos, inventario y pedidos actuales, que no forman parte del modelo ni de la aplicación.

---

## 🧠 Tipo de problema de Machine Learning

El problema principal será abordado como una tarea de **regresión orientada a la predicción de demanda utilizando información temporal**.

La salida de la red neuronal será un valor numérico que represente la demanda estimada de un producto para un determinado período futuro.

Flujo conceptual:

```text
Historial de ventas
        +
Variables temporales
        +
Información del producto
        ↓
Preprocesamiento
        ↓
Red neuronal
        ↓
Predicción de demanda
        ↓
Stock disponible
        ↓
Recomendación de reposición
```

---

## 📊 Dataset

### Estado actual

La aplicación utiliza los archivos locales de Corporación Favorita: `data/raw/train.csv` para el historial de ventas y `data/raw/test.csv` únicamente como calendario de promociones futuras de la demostración. Las pruebas del pipeline original también conservan datos sintéticos.

Los datos disponibles terminan en 2017. La demostración no es un pronóstico actual de 2026 y no se puede adaptar a esa fecha sin ventas recientes y promociones correspondientes.

Las particiones de Favorita se separan cronológicamente por fecha objetivo:

```text
TRAIN
  ↓
VALIDATION
  ↓
TEST
```

El modelo completo no recibe ventas objetivo ni ventas futuras como características.

---

## 📥 Variables previstas

### Variables de entrada

El modelo completo usa 95 entradas: cuatro variables de ventas/promoción, cuatro variables temporales cíclicas y codificaciones one-hot para tiendas y familias. No incluye stock, precio ni feriados.

Variables que podrían evaluarse más adelante, pero no son entradas del checkpoint actual:

* producto o categoría;
* ventas históricas;
* día de la semana;
* día del mes;
* semana del año;
* mes;
* temporada;
* precio;
* promociones;
* feriados;
* stock disponible;
* comportamiento reciente de ventas.

### Variable objetivo

La aplicación estima:

> **Ventas diarias registradas por tienda y familia.**

Las consultas históricas estiman un día; el pronóstico recursivo completo genera hasta siete días consecutivos.

---

## 🏗️ Arquitectura de la red neuronal

### Modelos actuales

El proyecto conserva el MLP base para pruebas y cuenta con dos checkpoints Favorita locales: el piloto, de 13 entradas, y el completo, de 95 entradas. La aplicación usa los pesos existentes; no entrena al consultar.

```text
Datos
  ↓
DataLoader
  ↓
MLP
  ↓
Entrenamiento
  ↓
Validación
  ↓
Evaluación
  ↓
Checkpoint
```

Conceptualmente, el modelo contiene:

```text
Variables de entrada
        ↓
Capa totalmente conectada
        ↓
ReLU
        ↓
Capa/s oculta/s
        ↓
Capa de salida
        ↓
Predicción numérica
```

Ambos checkpoints son MLP; el completo se usa también en el pronóstico recursivo de hasta siete días.

### Arquitectura temporal prevista

Como posible trabajo futuro, se podría evaluar una red:

**LSTM — Long Short-Term Memory**

Las redes LSTM están diseñadas para trabajar con información secuencial y permiten considerar el comportamiento histórico de una serie.

La estrategia prevista será:

```text
MLP
 ↓
Entrenamiento
 ↓
Evaluación
 ↓
LSTM
 ↓
Entrenamiento
 ↓
Evaluación
 ↓
Comparación
 ↓
Selección del modelo
```

De esta forma será posible determinar experimentalmente qué arquitectura obtiene mejores resultados.

---

## ⚙️ Proceso de entrenamiento

El framework principal utilizado es **PyTorch**.

Los hiperparámetros del entrenamiento son configurables mediante archivos YAML.

Entre ellos:

* número de épocas;
* tamaño del batch;
* tasa de aprendizaje;
* dimensiones de entrada;
* arquitectura del modelo.

Flujo general:

```text
Dataset
   ↓
Preprocesamiento
   ↓
Train / Validation / Test
   ↓
Entrenamiento
   ↓
Validación
   ↓
Guardado del mejor modelo
   ↓
Evaluación final
```

El pipeline base puede almacenar el mejor modelo en:

```text
models/best_model.pt
```

---

## 📈 Evaluación del modelo

Debido a que el problema consiste principalmente en predecir cantidades numéricas, se utilizarán métricas correspondientes a problemas de regresión.

Para regresión se utilizan métricas como:

* **MAE** — Mean Absolute Error.
* **MSE** — Mean Squared Error.
* **RMSE** — Root Mean Squared Error.
* **R²** — Coeficiente de determinación.

Las métricas disponibles incluyen MAE y RMSE. La página Evaluación muestra la evaluación diaria a un paso del piloto; el backtest recursivo del modelo completo es una evaluación separada y limitada a una muestra de validación.

Si los resultados obtenidos no son satisfactorios, se podrán modificar:

* arquitectura;
* hiperparámetros;
* variables de entrada;
* procesamiento de datos;
* cantidad de épocas;
* batch size;
* learning rate.

Posteriormente se realizará un nuevo entrenamiento y se compararán los resultados.

---

## 📦 Recomendaciones de stock (no implementadas)

La aplicación no dispone de datos de inventario ni calcula cantidades para reponer. Combinar ventas estimadas con existencias, pedidos y plazos de entrega queda fuera del alcance actual.

---

## 🗄️ Base de datos

El proyecto incorporará una base de datos para almacenar información relacionada con:

* productos;
* ventas;
* stock;
* predicciones;
* resultados reales;
* feedback.

Inicialmente se plantea utilizar **SQLite**, debido a su facilidad de integración con Python y a que no requiere un servidor de base de datos independiente.

Una posible estructura conceptual será:

```text
PRODUCTOS
    │
    ├── VENTAS
    │
    └── STOCK
          │
          ▼
     RED NEURONAL
          │
          ▼
     PREDICCIONES
          │
          ▼
   RESULTADOS REALES
          │
          ▼
       FEEDBACK
```

La estructura definitiva será diseñada durante la implementación.

---

## 🔄 Feedback Loop (trabajo futuro)

Se contempla implementar un mecanismo de feedback que permita comparar las predicciones realizadas por la red con los resultados observados posteriormente.

Ejemplo hipotético con valores ilustrativos, no corresponde a una salida del modelo:

```text
Predicción:
70 unidades

Venta real:
82 unidades

Error:
12 unidades
```

Esta información podrá almacenarse para analizar errores y generar nuevos datos útiles para futuros entrenamientos.

Flujo previsto:

```text
Datos históricos
      ↓
Entrenamiento
      ↓
Predicción
      ↓
Resultado real
      ↓
Cálculo del error
      ↓
Almacenamiento
      ↓
Nuevos datos
      ↓
Reentrenamiento
```

---

## 💬 Interfaz conversacional

Como evolución del proyecto se plantea incorporar una interfaz conversacional.

Su función será permitir que el usuario consulte fácilmente la información producida por el sistema.

Por ejemplo:

```text
¿Qué productos debería reponer esta semana?

¿Cuánto se espera vender del producto X?

¿Qué productos tienen riesgo de quedarse sin stock?

¿Por qué se recomienda aumentar el stock de este producto?
```

La interfaz conversacional **no reemplazará a la red neuronal**.

La arquitectura conceptual será:

```text
                  Usuario
                     ↓
          Interfaz conversacional
                     ↓
          Sistema inteligente
               ↙           ↘
      Red neuronal      Base de datos
               ↘           ↙
               Predicciones
                     ↓
              Recomendaciones
```

La red neuronal continuará siendo responsable de generar las predicciones de demanda.

---

## 🤖 Integración con Agentes de IA

Uno de los objetivos del proyecto es mantener un entorno preparado para trabajar con agentes de Inteligencia Artificial durante el desarrollo.

El agente podrá utilizar el contexto documentado en el repositorio para colaborar en tareas como:

* análisis del proyecto;
* ejecución de tests;
* entrenamiento;
* evaluación;
* detección de errores;
* documentación;
* optimización futura.

El repositorio incluye:

* `AGENT.md` — especificación y contexto del agente.
* `memoria.md` — memoria persistente con decisiones relevantes.
* `comandos.md` — comandos útiles para ejecución.
* `Skill.md` — definición de habilidades y responsabilidades.

El objetivo es proporcionar un entorno reproducible en el cual un agente pueda ejecutar tareas de manera autónoma y eficiente, sin depender exclusivamente de la ejecución manual desde un IDE.

---

## ☁️ Entorno de ejecución

El entorno previsto para la ejecución y entrenamiento del modelo definitivo será **Google Colab**.

Esto permitirá ejecutar experimentos en un entorno reproducible y disponer de recursos externos de cómputo cuando sean necesarios.

Flujo previsto:

```text
GitHub
   ↓
Google Colab
   ↓
Instalación de dependencias
   ↓
Carga del dataset
   ↓
Entrenamiento
   ↓
Evaluación
   ↓
Modelo entrenado
```

Los notebooks utilizados para experimentación se almacenarán dentro de:

```text
notebooks/
```

---

## 🧪 Testing

El proyecto incluye pruebas automatizadas mediante **pytest**.

Los tests permiten comprobar componentes del pipeline de manera independiente.

Durante el desarrollo se ampliarán las pruebas para cubrir:

* carga de datos;
* preprocesamiento;
* arquitectura;
* entrenamiento;
* predicción;
* base de datos;
* recomendaciones.

---

## 📁 Estructura del proyecto

```text
.
├── configs/                       # Configuraciones Favorita y pipeline base
├── data/raw/                      # CSV locales originales, no servidos por Flask
├── runs/                           # Checkpoints y resultados locales
├── src/web_app.py                 # Aplicación y API Flask
├── src/predecir_favorita.py       # Predicción histórica
├── src/pronosticar_favorita.py    # Pronóstico recursivo
├── web/templates/                 # Cinco páginas Jinja
├── web/static/                    # CSS y JavaScript locales
├── disenos/stitch/                # Diseños originales de referencia
├── tests/                          # Pruebas automatizadas
├── README.md
└── comandos.md
```

---

## 🛠️ Requisitos previos

* Python 3.13 de 64 bits para los checkpoints Favorita entregados
* PyTorch
* Git
* pip

---

## 🚀 Repetir la demostración local (entorno y artefactos ya preparados)

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-web.txt -c requirements-colab.txt torchvision==0.28.0
.\venv\Scripts\python.exe -m src.web_app
```

Abrí **http://127.0.0.1:5000**. En **Predicciones**, generá una consulta con el piloto y luego con el modelo completo; en el formulario recursivo usá el origen 2017-08-15 y horizonte de siete días. El pronóstico es una demostración histórica de 2017, no de 2026. Detené Flask con Ctrl+C.

El pipeline sintético original sigue disponible con `.\venv\Scripts\python.exe src\train.py --config configs\default.yaml`; ese comando no es necesario para demostrar la aplicación.

---

## 📄 Configuración de datos

El pipeline genérico conserva una configuración sintética para pruebas. La integración Favorita utiliza módulos y configuraciones propios.

Para utilizar un archivo CSV:

1. Cambiar `data.source` a `csv`.
2. Colocar el dataset dentro de:

```text
data/raw/
```

3. Configurar en `configs/default.yaml`:

```text
data.path
data.target_column
input_dim
```

La configuración anterior corresponde al pipeline genérico; la predicción Favorita usa `configs/favorita.yaml` y `configs/favorita_completa.yaml`.

---

## 📌 Estado actual y trabajo futuro

### Implementado

* Lectura y preparación temporal del dataset Favorita, con separación cronológica.
* Checkpoint piloto de 13 entradas y checkpoint completo de 95 entradas.
* Evaluación diaria a un paso del piloto y backtest recursivo de siete días en una muestra de validación.
* Aplicación Flask local con cinco páginas, temas claro/oscuro, consultas históricas y pronóstico recursivo con el modelo completo.
* Suite automatizada del pipeline y la aplicación.

### Pendiente

* Datos recientes y calendario de promociones actualizados para pronosticar años como 2026.
* Datos de inventario, pedidos y plazos; las recomendaciones de reposición no están implementadas.
* Evaluación recursiva más amplia e independiente en el test reservado.
* Incorporación y evaluación de variables externas, base de datos, feedback y otras arquitecturas.

El backtest disponible es un diagnóstico de desarrollo, no demuestra precisión generalizable. Ver Bloque 3 para su alcance y supuestos.

---

## 🛠️ Tecnologías utilizadas

* **Python**
* **PyTorch**
* **NumPy**
* **Pandas**
* **PyYAML**
* **pytest**
* **Git / GitHub**

### Tecnologías aún no integradas a la aplicación

* **SQLite**
* **Google Colab** (hay notebooks de entrenamiento, pero la aplicación no depende de Drive)

---

## 📚 Estado del desarrollo

**Prototipo local funcional para demostración con datos históricos de 2017.** La aplicación no debe presentarse como una herramienta de reposición ni como un pronóstico actual de 2026.

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

Ejecutar desde la raíz del repositorio, sin entrenamiento:

```powershell
.\venv\Scripts\python.exe -m src.evaluar_pronosticos_recursivos
```

- Evalua solo validacion con seis origenes moviles, siete dias por origen, cinco tiendas (una tienda central por tipo, ordenadas por `store_nbr`) y cuatro familias, para una muestra de 20 series. El test no se usa y queda reservado para una evaluacion final.
- Reutiliza `PronosticadorFavorita` del módulo `src/pronosticar_favorita.py`, carga el checkpoint una vez y excluye las ventas posteriores al origen de las entradas. La referencia semanal es `ventas(t-7)`, disponible para los siete horizontes.
- Las promociones se extraen de `train.csv` para los días posteriores al origen y solo se pasan como calendario; las ventas reales quedan separadas para puntuar. Se asume que esas promociones ya eran conocidas en cada origen: el CSV no indica cuándo se publicaron.
- Guarda un informe JSON, predicciones detalladas y tablas de MAE/RMSE por horizonte y tienda/familia en una carpeta nueva con marca de tiempo bajo `runs/evaluaciones/`. No sobrescribe resultados existentes. MAE/RMSE generales agrupan todos los horizontes; por horizonte hay 120 observaciones si no se omiten ventanas.
- Es una muestra de desarrollo, no una estimación definitiva de generalización. Deben revisarse las ventanas omitidas y no elegir modelos con el conjunto de test.
- En la ejecución de referencia (840 pronósticos), el modelo obtuvo MAE 301,54 y RMSE 681,90 frente a MAE 288,23 y RMSE 750,25 de la referencia semanal: mejora el RMSE y empeora el MAE. El resultado es mixto y limitado a esa muestra.


## Bloque C: StockPredict IA — aplicación web local

Con el entorno y los artefactos externos ya preparados según «Instalación rápida desde cero», desde la raíz del repositorio:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements-web.txt -c requirements-colab.txt torchvision==0.28.0
.\venv\Scripts\python.exe -m src.web_app
```

Abrir **http://127.0.0.1:5000**. Detener con Ctrl+C. Si el puerto está ocupado: `.\venv\Scripts\python.exe -m src.web_app --port 5001`.

Para demostrar: en **Predicciones**, usar «Usar caso del piloto» y generar la consulta; repetir seleccionando **Completo**. En la misma página, generar el pronóstico de siete días con origen 2017-08-15. **Evaluación** distingue la métrica diaria del backtest recursivo; **Historial** muestra las corridas del piloto.

En la interfaz, las ventas estimadas se presentan como unidades enteras, con mínimo visual de cero; los cálculos recursivos, las métricas y las respuestas de las API conservan sus decimales originales. Los nombres de las 33 familias se traducen solo para mostrarlos; los valores enviados a las API permanecen en inglés. Por ejemplo, `PLAYERS AND ELECTRONICS` se presenta como «Reproductores y electrónica».

- Cinco páginas: Inicio, Predicciones, Evaluación, Historial y Acerca. HTML adaptado de Stitch mediante plantillas Jinja compartidas; JavaScript nativo. Flask escucha únicamente en localhost, sin depurador. Es un servidor para uso local, no un despliegue público.
- Diseños originales copiados sin alteraciones a `disenos/stitch/`: Descargas/1 → predicciones, /2 → inicio, /3 → evaluacion, /4 → acerca, /5 → historial. Cada pantalla conserva HTML y DESIGN con nombres descriptivos; `procedencia.json` registra hashes. Estas referencias contienen el contenido ficticio original; no se sirven por HTTP. Las plantillas adaptadas están en `web/templates/`.
- La página Predicciones permite elegir el piloto (`runs/favorita_subconjunto_reanudado_4/best.pt`, 13 entradas) o el modelo completo (`runs/favorita_completa_20/best.pt`, 95 entradas); ambos usan `data/raw/train.csv`. El piloto continúa seleccionado por defecto. No hay selector de rutas ni carga de checkpoints mediante HTTP.
- `GET /api/opciones?modelo=piloto|completo` publica solo las categorías del checkpoint y fechas/rangos históricos para la interfaz, no ventas originales ni CSV. `POST /api/predicciones` recibe `tienda`, `familia`, `fecha` y el campo opcional `modelo` (`piloto` por defecto); reutiliza `predecir_favorita`, comprueba el SHA-256 del dataset y las siete fechas previas observadas. No entrena ni usa la venta objetivo como entrada. El servidor rechaza fechas sin fila o historia suficiente; en el formulario completo los límites de fecha son un rango y la validez exacta se confirma en el servidor.
- La sección **Pronóstico futuro** usa exclusivamente el modelo completo, hasta siete días desde la última fecha histórica disponible. `src/pronosticar_favorita.py` acepta historia y promociones como DataFrames separados, usa el preparador guardado sin reajustarlo y alimenta los rezagos futuros solo con sus propias predicciones. La demostración web toma promociones de `data/raw/test.csv`, que no contiene ventas reales; se rechazan huecos de historia o promociones. `POST /api/pronosticos` recibe `tienda`, `familia`, `fecha_origen` y `horizonte` (1–7). Los datos actuales de esta demostración son de 2017, no un pronóstico para 2026.
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

Ejemplo de pronóstico de siete días:

```powershell
$cuerpo = @{ tienda = 1; familia = "BEAUTY"; fecha_origen = "2017-08-15"; horizonte = 7 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/pronosticos -ContentType "application/json" -Body $cuerpo
```
