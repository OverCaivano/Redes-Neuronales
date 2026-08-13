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

🚀 Entrenamiento y Evaluación
# Ejecutar entrenamiento del modelo principal
python src/train.py --config configs/default.yaml

# Ejecutar evaluación/test del modelo
python src/evaluate.py --model_path models/best_model.pt

🧪 Pruebas y Calidad de Código
# Correr tests unitarios
pytest tests/

# Formateo e inspección de código
black src/
flake8 src/
