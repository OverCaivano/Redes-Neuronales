"""Raíz de composición para ensamblar las implementaciones del Bloque 3."""

from pathlib import Path

from src.application.services.prediction_service import ServicioPrediccion
from src.infrastructure.ml.pytorch_prediction_adapter import (
    AdaptadorPrediccionPyTorch,
)
from src.infrastructure.repositories.csv_product_repository import (
    RepositorioProductosCSV,
)


RUTA_CATALOGO = Path("data/synthetic/catalogo_productos_sinteticos.csv")
RUTA_CHECKPOINT = Path("runs/synthetic_product_sales_v1/best_model.pt")
RUTA_INVENTARIO = Path("data/synthetic/inventario_sintetico.csv")


def crear_servicio_prediccion(
    ruta_catalogo=RUTA_CATALOGO,
    ruta_checkpoint=RUTA_CHECKPOINT,
):
    repositorio = RepositorioProductosCSV(ruta_catalogo)
    motor = AdaptadorPrediccionPyTorch(ruta_checkpoint)
    return ServicioPrediccion(repositorio, motor)


def crear_blueprint_prediccion(
    ruta_catalogo=RUTA_CATALOGO,
    ruta_checkpoint=RUTA_CHECKPOINT,
):
    from src.interfaces.web.controllers.prediction_controller import (
        crear_blueprint_prediccion as crear_blueprint,
    )

    return crear_blueprint(
        crear_servicio_prediccion(ruta_catalogo, ruta_checkpoint)
    )


def crear_servicio_reposicion(
    servicio_prediccion,
    ruta_inventario=RUTA_INVENTARIO,
):
    from src.application.replenishment_strategies import (
        ReposicionBasica,
        ReposicionConSeguridad,
    )
    from src.application.services.replenishment_service import ServicioReposicion
    from src.infrastructure.repositories.csv_inventory_repository import (
        RepositorioInventarioCSV,
    )

    estrategias = (ReposicionBasica(), ReposicionConSeguridad())
    return ServicioReposicion(
        RepositorioInventarioCSV(ruta_inventario),
        servicio_prediccion,
        {estrategia.codigo: estrategia for estrategia in estrategias},
    )
