# Boilerplate - Proyecto de Redes Neuronales

Este repositorio contiene la plantilla base (boilerplate) orientada al entrenamiento, evaluación y despliegue de modelos de redes neuronales e Inteligencia Artificial, integrando flujos de trabajo autónomos mediante agentes IA.

## 🎯 Objetivos del Proyecto
1. **Estandarizar el entorno de desarrollo** para la creación y experimentación con redes neuronales.
2. **Integración con Agentes Autónomos** para la ejecución programada de tareas, testing y optimización.
3. **Control de versiones y reproducibilidad** del pipeline de ML/DL.

## 📁 Estructura del Proyecto
.
├── models/             # Guardado de pesos y arquitecturas (.pt, .h5, .onnx)
├── data/               # Datasets de entrenamiento, validación y prueba
│   └── raw/             # Dataset original, sin modificar
├── configs/             # Configuraciones YAML del pipeline
├── notebooks/          # Experimentos en Jupyter Notebooks
├── src/                # Código fuente principal
│   ├── data_loader.py   # Dataset sintético o CSV
│   ├── evaluate.py      # Evaluación de checkpoints
│   ├── models/          # Arquitecturas reutilizables
│   └── train.py         # Entrenamiento
├── tests/               # Pruebas automatizadas
├── AGENT.md            # Especificación y contexto del Agente IA
├── memoria.md          # Memoria persistente de contexto e interacciones
├── comandos.md         # Registro de comandos de ejecución para el agente
├── Skill.md            # Definición de habilidades del agente
└── README.md           # Documentación principal

## 🛠️ Requisitos Previos
- Python 3.10+
- PyTorch / TensorFlow
- Git

## Inicio rápido

```bash
pip install -r requirements.txt
python src/train.py --config configs/default.yaml
python src/evaluate.py --model_path models/best_model.pt
pytest tests/
```

La configuración usa datos sintéticos para que el pipeline funcione desde el primer día. Para usar un CSV, cambia `data.source` a `csv`, coloca el archivo en `data/raw/` y ajusta `data.path`, `data.target_column` e `input_dim` en `configs/default.yaml`.
