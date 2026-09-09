# 🧠 Sistema Inteligente de Predicción de Demanda y Gestión de Stock mediante Redes Neuronales

Este repositorio contiene el desarrollo de un sistema inteligente basado en redes neuronales orientado a la **predicción de demanda de productos y asistencia en la gestión de stock**.

El proyecto parte de un boilerplate funcional para entrenamiento y evaluación de modelos de redes neuronales desarrollado en **Python y PyTorch**, incorporando además testing automatizado, configuración reproducible y documentación destinada a la utilización de agentes de Inteligencia Artificial durante el desarrollo.

Actualmente el proyecto se encuentra en desarrollo. El boilerplate y el pipeline inicial están implementados, mientras que las siguientes etapas estarán orientadas a integrar datos reales de ventas y adaptar la red neuronal al problema específico de predicción de demanda.

---

## 🎯 Objetivo del proyecto

El objetivo principal es:

> **Desarrollar una red neuronal capaz de predecir la demanda futura de productos utilizando información histórica de ventas y variables temporales, con el objetivo de generar recomendaciones de reposición de stock.**

El sistema busca servir como herramienta de asistencia para supermercados, comercios y otros negocios que necesiten administrar inventarios.

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

La disponibilidad definitiva de estas variables dependerá del dataset seleccionado.

---

## 💡 Problema a resolver

Una gestión ineficiente del inventario puede producir principalmente dos situaciones:

### Falta de stock

Cuando la demanda supera las unidades disponibles, el comercio puede perder ventas y clientes.

### Exceso de stock

Comprar cantidades muy superiores a la demanda genera costos innecesarios de almacenamiento y puede producir pérdidas, especialmente en productos perecederos.

El sistema buscará anticipar estas situaciones mediante predicciones de demanda.

Por ejemplo:

```text
Producto: Gaseosa 2.25 L

Stock actual:
35 unidades

Demanda estimada próximos 7 días:
83 unidades

Recomendación:
Reponer aproximadamente 48 unidades.
```

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

Actualmente el boilerplate utiliza **datos sintéticos** para garantizar que el pipeline de entrenamiento y evaluación pueda ejecutarse independientemente de un dataset externo.

También existe soporte inicial para cargar información desde archivos CSV.

### Próxima etapa

Se seleccionará un **dataset real de ventas minoristas** adecuado para entrenar la red neuronal.

Se priorizarán datasets que incluyan:

* fecha;
* producto;
* cantidad vendida;
* historial de ventas;
* precio;
* promociones;
* feriados u otras variables temporales.

Los datos serán posteriormente limpiados, transformados y separados en conjuntos de:

```text
TRAIN
  ↓
VALIDATION
  ↓
TEST
```

Es importante mantener la separación temporal de los datos para evitar utilizar información futura durante el entrenamiento.

---

## 📥 Variables previstas

### Variables de entrada

Inicialmente se consideran:

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

La selección definitiva se realizará después de analizar el dataset.

### Variable objetivo

La variable principal a predecir será:

> **Cantidad de unidades demandadas o vendidas durante un período futuro determinado.**

Inicialmente se evaluarán predicciones diarias y/o para períodos cortos de varios días.

---

## 🏗️ Arquitectura de la red neuronal

### Modelo base actual

El boilerplate implementa actualmente una **red neuronal multicapa (MLP)** utilizando PyTorch.

Esta arquitectura permite comprobar el funcionamiento completo del pipeline:

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

La arquitectura actual funciona como **modelo base** y será adaptada al dataset real.

### Arquitectura temporal prevista

Debido a que las ventas poseen dependencia temporal, posteriormente se evaluará implementar una red:

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

El mejor modelo obtenido durante el entrenamiento puede almacenarse en:

```text
models/best_model.pt
```

---

## 📈 Evaluación del modelo

Debido a que el problema consiste principalmente en predecir cantidades numéricas, se utilizarán métricas correspondientes a problemas de regresión.

Entre las métricas previstas:

* **MAE** — Mean Absolute Error.
* **MSE** — Mean Squared Error.
* **RMSE** — Root Mean Squared Error.
* **R²** — Coeficiente de determinación.

Estas métricas permitirán comparar las predicciones de la red neuronal con los valores reales.

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

## 📦 Sistema de recomendaciones de stock

Las predicciones generadas por la red neuronal serán utilizadas posteriormente por un módulo de recomendaciones.

Ejemplo:

```text
Demanda estimada:
100 unidades

Stock disponible:
60 unidades

Diferencia:
40 unidades

Recomendación:
Reponer aproximadamente 40 unidades.
```

De esta manera, la predicción numérica se transforma en información útil para la toma de decisiones.

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

## 🔄 Feedback Loop

Se contempla implementar un mecanismo de feedback que permita comparar las predicciones realizadas por la red con los resultados observados posteriormente.

Ejemplo:

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
├── models/               # Pesos y modelos entrenados (.pt, .h5, .onnx)
├── data/                 # Datasets de entrenamiento, validación y prueba
│   └── raw/              # Dataset original sin modificar
├── configs/              # Configuraciones YAML del pipeline
├── notebooks/            # Experimentos y Google Colab
├── src/                  # Código fuente principal
│   ├── data_loader.py    # Carga de datos sintéticos o CSV
│   ├── evaluate.py       # Evaluación de checkpoints
│   ├── models/           # Arquitecturas reutilizables
│   └── train.py          # Entrenamiento
├── tests/                # Pruebas automatizadas
├── AGENT.md              # Contexto e instrucciones del agente IA
├── memoria.md            # Memoria persistente del proyecto
├── comandos.md           # Registro de comandos
├── Skill.md              # Habilidades del agente
├── requirements.txt      # Dependencias de Python
└── README.md             # Documentación principal
```

La estructura será ampliada progresivamente a medida que se incorporen la base de datos, el dataset definitivo y nuevos componentes.

---

## 🛠️ Requisitos previos

* Python 3.10+
* PyTorch
* Git
* pip

---

## 🚀 Inicio rápido

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

Ejecutar entrenamiento:

```bash
python src/train.py --config configs/default.yaml
```

Evaluar un modelo entrenado:

```bash
python src/evaluate.py --model_path models/best_model.pt
```

Ejecutar los tests:

```bash
python -m pytest
```

---

## 📄 Configuración de datos

Actualmente la configuración utiliza datos sintéticos para permitir que el pipeline funcione desde el primer día.

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

Esta funcionalidad será adaptada al dataset real seleccionado para predicción de demanda.

---

## 📌 Estado actual del proyecto

### ✅ Completado

* Boilerplate inicial.
* Estructura organizada del repositorio.
* Pipeline base de entrenamiento.
* Modelo MLP inicial.
* Datos sintéticos para pruebas.
* Soporte inicial para CSV.
* Configuración mediante YAML.
* Guardado de checkpoints.
* Tests iniciales.
* Integración inicial para agentes IA.
* Definición del problema.
* Definición del objetivo de la red neuronal.
* Diseño conceptual de la solución.

### 🚧 En desarrollo

* Selección del dataset real.
* Análisis exploratorio de datos.
* Definición definitiva de variables.
* Adaptación del DataLoader.
* Adaptación del modelo MLP.
* Implementación de métricas.
* Base de datos.
* Sistema de recomendaciones.
* Feedback loop.
* Notebook para Google Colab.
* Actualización de documentación del agente.

### 🔮 Evolución prevista

* Arquitectura LSTM.
* Comparación entre modelos.
* Interfaz conversacional.
* Visualización de predicciones.
* Optimización del modelo.
* Automatización parcial del reentrenamiento.

---

## 🗺️ Roadmap

```text
Boilerplate                         ✅
        ↓
Definición del problema             ✅
        ↓
Selección del dataset               🚧
        ↓
Preprocesamiento
        ↓
Modelo base MLP
        ↓
Entrenamiento
        ↓
Evaluación
        ↓
Base de datos
        ↓
Predicción de demanda
        ↓
Recomendaciones de stock
        ↓
Feedback loop
        ↓
Modelo temporal LSTM
        ↓
Interfaz conversacional
        ↓
Evaluación y optimización final
```

---

## 🛠️ Tecnologías utilizadas

* **Python**
* **PyTorch**
* **NumPy**
* **Pandas**
* **PyYAML**
* **pytest**
* **Git / GitHub**

### Tecnologías previstas

* **SQLite**
* **Google Colab**

---

## 📚 Estado del desarrollo

**Proyecto en desarrollo activo.**

La primera etapa correspondiente al boilerplate se encuentra funcional.

Actualmente el proyecto se encuentra en la fase de **definición del problema y selección del dataset real**. El siguiente objetivo será integrar datos históricos de ventas y comenzar la adaptación del pipeline existente para la predicción de demanda.