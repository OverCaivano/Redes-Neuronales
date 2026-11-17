# Datos sintéticos por producto

Los CSV de esta carpeta son **sintéticos**, no son ventas reales y no usan
datasets, checkpoints ni código de Favorita. El generador produce un panel
diario de 3 tiendas y 30 productos entre `2023-01-01` y `2026-11-16`
(ambos inclusive). No genera ventas después de esa fecha.

Desde la raíz del repositorio, regenerarlos con la semilla fija predeterminada:

```powershell
.\venv\Scripts\python.exe -m src.datos_sinteticos
```

Para guardar en otra carpeta o repetir con otra semilla explícita:

```powershell
.\venv\Scripts\python.exe -m src.datos_sinteticos --output-dir data\synthetic --seed 42
```

Archivos generados:

- `ventas_sinteticas.csv`: `fecha`, `tienda_id`, `producto_id`,
  `unidades_vendidas`, `precio`, `promocion`.
- `catalogo_productos_sinteticos.csv`: identificador, nombre y categoría de
  cada producto.
- `inventario_sintetico.csv`: una instantánea sintética al `2026-11-16` con
  `tienda_id`, `producto_id`, `stock_actual`, `stock_minimo` y
  `tiempo_reposicion` en días.

El panel de ventas tiene una fila por fecha, tienda y producto: 127.440 filas.
