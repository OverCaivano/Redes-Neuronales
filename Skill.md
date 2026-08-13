Define las capacidades específicas que el agente sabe usar dentro de este entorno.

```markdown
# Habilidades del Agente (Skill Set)

Este documento detalla las capacidades operativas del Agente IA y el procedimiento para ejecutarlas.

## 🧰 Habilidades Principales

### 1. Data Preprocessing (`skill-data-prep`)
- **Descripción:** Carga, limpia y normaliza los conjuntos de datos en `data/`.
- **Uso:** Modificar los parámetros en `src/data_loader.py` y validar distribuciones.

### 2. Model Training & Tuning (`skill-model-train`)
- **Descripción:** Entrena modelos de redes neuronales y optimiza hiperparámetros.
- **Uso:** Invocar mediante `python src/train.py` especificando el tipo de arquitectura.

### 3. Automatic Documentation (`skill-auto-doc`)
- **Descripción:** Actualiza `memoria.md` con las métricas finales de cada corrida y registra cambios en `comandos.md`.
- **Uso:** El agente ejecuta esta habilidad al finalizar exitosamente un pipeline.
