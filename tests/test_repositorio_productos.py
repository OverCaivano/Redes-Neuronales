from src.datos_sinteticos import generar_catalogo_productos
from src.domain.models import Producto
from src.infrastructure.repositories.csv_product_repository import (
    RepositorioProductosCSV,
)
import pytest


def test_repositorio_csv_lee_catalogo_sintetico_y_valida_esquema(tmp_path):
    ruta = tmp_path / "catalogo.csv"
    generar_catalogo_productos().to_csv(ruta, index=False)
    repositorio = RepositorioProductosCSV(ruta)

    assert repositorio.buscar_por_id("BEB-001") == Producto(
        "BEB-001", "Agua mineral 500 ml", "bebidas"
    )
    assert len(repositorio.listar()) == 30
    assert repositorio.buscar_por_id("DESCONOCIDO") is None

    catalogo = generar_catalogo_productos()
    catalogo.loc[1, "producto_id"] = "BEB-001"
    catalogo.to_csv(ruta, index=False)
    with pytest.raises(ValueError, match="duplicados"):
        RepositorioProductosCSV(ruta).listar()
