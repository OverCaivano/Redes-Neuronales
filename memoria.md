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
- `src/infrastructure/`: repositorios CSV para catálogo e inventario, y
  adaptador PyTorch.
- `src/interfaces/web/controllers/`: Blueprints HTTP.
- `src/composition.py`: ensamblado de dependencias y rutas de artefactos.
- `src/web_app.py` y `web/`: aplicación Flask y presentación.

Al entrenar, el pipeline guarda configuración, métricas e historial de
validación por época en `runs/synthetic_product_sales_v1/metricas.json`, junto
con el checkpoint. El comando `python -m src.pipeline_productos evaluate` no
lee ese JSON: carga el checkpoint y los CSV, comprueba las huellas de los datos
y vuelve a calcular MAE/RMSE del MLP y del baseline sobre el período de prueba.
El baseline es la media histórica por tienda y producto, calculada con datos
de entrenamiento. La evaluación por CLI no entrena ni modifica esos artefactos;
la web solo sirve la pantalla de predicción y reposición y tampoco entrena al
iniciar.

Las predicciones de demostración están limitadas a 90 días posteriores al
último día disponible del dataset (`2026-11-16`). Este límite define el alcance
de la aplicación y no garantiza la precisión de las estimaciones. El inventario
es un snapshot sintético con fecha de corte `2026-11-16`; las ventas posteriores
a ese corte no se descuentan automáticamente y el stock no se proyecta hasta la
fecha futura consultada. El horizonte de reposición es más corto cuando el
producto tiene un plazo de entrega mayor: como el plazo incluye la fecha
seleccionada, su último inicio válido es la fecha máxima de predicción menos
`tiempo_reposicion - 1` días. Así, los plazos de 1, 4 y 7 días permiten iniciar,
respectivamente, hasta el máximo general, 3 días antes y 6 días antes. La
validación ocurre en el servicio de aplicación; la interfaz consulta ese límite
por tienda y producto y conserva el máximo general para predicciones
individuales.

## Arquitectura y patrones

MVC: los controladores Flask reciben las solicitudes, las plantillas presentan
las vistas y dominio/servicios contienen las entidades y los casos de uso.
Las implementaciones Repository existentes son `RepositorioProductosCSV` y
`RepositorioInventarioCSV`; ambos aíslan CSV sintéticos. No existe un
repositorio JSON: `metricas.json` lo escribe el pipeline de entrenamiento.
Adapter integra la inferencia PyTorch detrás del puerto de predicción; Strategy
permite intercambiar las reglas básica y con stock de seguridad. SRP se apoya
en la separación de responsabilidades; OCP/LSP, en estrategias sustituibles;
ISP, en Protocol pequeños; y DIP, en servicios que dependen de Protocol y en
la composición que inyecta implementaciones. No se usan Singleton, Builder ni
Chain of Responsibility porque no son necesarios para estos casos de uso.

Las métricas comparan el MLP con el baseline sobre el test cronológico
sintético. Describen únicamente esos datos y no garantizan desempeño con ventas
reales.

## Reproducibilidad

Consultar `README.md`, `comandos.md` y `data/synthetic/README.md` para
instalación, generación, entrenamiento, evaluación y pruebas. Los CSV y los
artefactos de entrenamiento generados están ignorados por Git. No reemplazar
una corrida existente.
