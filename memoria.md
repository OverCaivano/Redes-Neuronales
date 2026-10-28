# Memoria Persistente del Agente IA

## 🧠 Estado Actual del Proyecto
- **Fase actual:** Base funcional del pipeline de regresión.
- **Entorno:** PyTorch, PyYAML y loaders para datos sintéticos o CSV.

## 📊 Registro de Experimentos e Interacciones
- **[Fecha Inicial]:** Creación de la estructura del repositorio y especificación de documentos de control (`AGENT.md`, `Skill.md`, `comandos.md`).
- **[2026-08-20]:** Se conectaron configuración, carga de datos, MLP, entrenamiento, evaluación y pruebas automatizadas.

## 📌 Notas para Próximas Tareas
- Reemplazar los datos sintéticos por el dataset del proyecto en `data/raw/`.
- Ajustar la arquitectura y la función de pérdida según el problema elegido.
- Añadir métricas específicas y validación cruzada cuando se defina el caso de uso.

- **[2026-10-14]:** Bloque 1E-C: notebook Colab para Favorita completo, entorno aislado y sincronizacion verificada por epoca a Drive; recuperacion de generaciones integras. No se ejecuto entrenamiento completo.

- **[2026-10-28]:** Evaluador explicito Favorita sobre test interno, reutilizando best.pt v2 y transformaciones sin fit. Evaluador sintetico conservado. Resultados de evaluacion separados bajo runs/evaluaciones/.

- **[2026-10-28]:** Bloque B: consultas CLI de ventas diarias Favorita por tienda/familia/fecha, con checkpoint fijo y siete dias previos observados. No usa la venta objetivo, no reajusta ni entrena.

- **[2026-10-28]:** Bloque C: StockPredict IA local con Flask, plantillas adaptadas de los cinco HTML de Stitch y API que reutiliza las predicciones Favorita. Métricas e historial leídos de artefactos locales; sin acceso ni suposiciones sobre Drive.
