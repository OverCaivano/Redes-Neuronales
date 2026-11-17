# Estado técnico de StockPredict IA

## Alcance

Prototipo académico para predecir ventas futuras de productos individuales
por tienda. Ventas, catálogo e inventario son sintéticos y reproducibles.
El panel cubre 3 tiendas, 30 productos y 127.440 filas entre 2023-01-01 y
2026-11-16.

## Componentes vigentes

- `src/datos_sinteticos.py`: generación y validación de los tres CSV.
- `src/pipeline_productos.py`: MLP PyTorch, partición cronológica,
  entrenamiento, evaluación y reporte por época.
- `src/application/`: contratos y servicios independientes de Flask/CSV.
- `src/infrastructure/`: repositorios CSV/JSON y adaptador PyTorch.
- `src/interfaces/web/controllers/`: Blueprints HTTP.
- `src/composition.py`: ensamblado de dependencias y rutas de artefactos.
- `src/web_app.py` y `web/`: aplicación Flask y presentación.

El pipeline conserva MAE/RMSE del MLP y el baseline, junto con métricas por
época, en `runs/synthetic_product_sales_v1/metricas.json`. La evaluación se
consulta por CLI; la web solo sirve la pantalla de predicción y reposición. La
web no entrena al iniciar.

Las predicciones de demostración están limitadas a 90 días posteriores al
último día disponible del dataset (`2026-11-16`). Este límite define el alcance
de la aplicación y no garantiza la precisión de las estimaciones. El inventario
es un snapshot sintético con fecha de corte `2026-11-16`; las ventas posteriores
a ese corte no se descuentan automáticamente y el stock no se proyecta hasta la
fecha futura consultada.

## Arquitectura y patrones

MVC y capas Controller–Service–Repository; inversión de dependencias con
Protocol e inyección; Repository para fuentes CSV/JSON; Adapter para la
inferencia PyTorch; Strategy para las reglas de reposición básica y con stock
de seguridad. No se usan Singleton, Builder ni Chain of Responsibility porque
no son necesarios para los casos de uso actuales.

## Reproducibilidad

Consultar `README.md`, `comandos.md` y `data/synthetic/README.md` para
instalación, generación, entrenamiento, evaluación y pruebas. Los CSV y los
artefactos de entrenamiento generados están ignorados por Git. No reemplazar
una corrida existente.
