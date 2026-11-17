# Guía de trabajo del repositorio

StockPredict IA es un prototipo Python/PyTorch/Flask de pronósticos por
producto y tienda. El dataset, el catálogo y el inventario son sintéticos.

## Reglas del proyecto

- Antes de modificar, revisar el código y las pruebas relacionadas.
- Mantener dominio, servicios, repositorios, adaptadores y controladores
  separados; inyectar dependencias mediante los contratos existentes.
- No entrenar ni reemplazar checkpoints al iniciar la aplicación web.
- No interpretar datos o resultados sintéticos como ventas o existencias reales.
- No sobrescribir una carpeta de corrida existente.
- Actualizar `README.md` y `comandos.md` cuando cambien los comandos o la
  arquitectura.
- Ejecutar pruebas específicas del cambio y `git diff --check`.

Los comandos vigentes están en [comandos.md](./comandos.md).
