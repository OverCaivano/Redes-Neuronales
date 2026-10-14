"""Drive simulado en tmp_path. Solo fixtures pequenas; no usa CSV reales."""

import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.train import train
from src.reanudacion import cargar_last
from src.sincronizacion import (
    ARCHIVOS, copiar_csv_local, huella, restaurar_ultima_copia,
    sincronizar_epoca, ultima_copia_integra, verificar_copia,
)

modulo = importlib.import_module("src.sincronizacion")


@pytest.fixture
def corrida(tmp_path):
    fechas = pd.date_range("2017-06-01", "2017-08-15")
    csv = tmp_path / "datos.csv"
    pd.DataFrame({
        "id": range(len(fechas)), "date": fechas, "store_nbr": 1,
        "family": "BEAUTY", "sales": np.arange(len(fechas)) % 7,
        "onpromotion": 0,
    }).to_csv(csv, index=False)
    config = {
        "seed": 42, "device": "cpu", "model": {"hidden_dim": 8, "output_dim": 1},
        "training": {"epochs": 2, "batch_size": 4, "learning_rate": .001, "num_workers": 0},
        "data": {"source": "favorita", "path": str(csv)},
        "output": {"directory": str(tmp_path / "local")},
    }
    remoto = tmp_path / "drive"
    train(config, sync_dir=remoto)
    return config, Path(config["output"]["directory"]), remoto


def test_copia_por_epoca_y_cinco_archivos(corrida):
    config, local, remoto = corrida
    generaciones = sorted(remoto.glob("epoca_*"))
    assert len(generaciones) == 2
    for epoca, ruta in enumerate(generaciones, 1):
        datos = verificar_copia(ruta)
        assert datos["epoca"] == epoca
        assert cargar_last(ruta / "last.pt")["epoch"] == epoca
        assert set(datos["archivos"]) == set(ARCHIVOS)
        assert len(pd.read_csv(ruta / "metricas.csv")) == epoca
    ruta, datos = ultima_copia_integra(remoto)
    assert datos["epoca"] == 2
    assert sincronizar_epoca(local, remoto) == ruta  # Reintento idempotente.
    assert len(list(remoto.iterdir())) == 2
    assert set(p.name for p in local.iterdir()) == set(ARCHIVOS)


def test_restaurar_y_reanudar_sin_fit(corrida, tmp_path, monkeypatch):
    config, local, remoto = corrida
    nueva = tmp_path / "restaurada"
    last = restaurar_ultima_copia(remoto, nueva)
    for nombre in ARCHIVOS:
        assert huella(nueva / nombre) == huella(local / nombre)
    anterior = cargar_last(last)
    def prohibido(*args, **kwargs):
        pytest.fail("No debe reajustar al recuperar Drive")
    monkeypatch.setattr(OneHotEncoder, "fit", prohibido)
    monkeypatch.setattr(StandardScaler, "fit", prohibido)
    config["training"]["epochs"] = 3
    train(config, resume=last, sync_dir=remoto)
    final = cargar_last(last)
    assert final["historial"][:2] == anterior["historial"]
    assert final["epoch"] == ultima_copia_integra(remoto)[1]["epoca"] == 3
    assert cargar_last(local / "last.pt")["epoch"] == 2


@pytest.mark.parametrize("archivo", [*ARCHIVOS, "manifiesto.json", "COMPLETO"])
def test_corrupcion_recupera_generacion_previa(corrida, archivo):
    _, _, remoto = corrida
    ultima, _ = ultima_copia_integra(remoto)
    (ultima / archivo).write_bytes(b"copia truncada")
    with pytest.warns(UserWarning, match="no recuperable"):
        _, datos = ultima_copia_integra(remoto)
    assert datos["epoca"] == 1


def test_fallo_transferencia_preserva_copia_y_estado_local(corrida, monkeypatch):
    config, local, remoto = corrida
    previa, _ = ultima_copia_integra(remoto)
    hashes = {p.name: huella(p) for p in previa.iterdir()}
    original = modulo.shutil.copyfile
    def fallo(origen, destino, *args, **kwargs):
        if Path(destino).parent.parent == remoto and Path(destino).name == "best.pt":
            Path(destino).write_bytes(b"parcial")
            raise OSError("Drive desconectado")
        return original(origen, destino, *args, **kwargs)
    monkeypatch.setattr(modulo.shutil, "copyfile", fallo)
    config["training"]["epochs"] = 4
    with pytest.raises(OSError, match="desconectado"):
        train(config, resume=local / "last.pt", sync_dir=remoto)
    assert cargar_last(local / "last.pt")["epoch"] == 3  # Nunca empieza la cuarta.
    with pytest.warns(UserWarning):
        assert ultima_copia_integra(remoto)[1]["epoca"] == 2
    assert hashes == {p.name: huella(p) for p in previa.iterdir()}


def test_restauracion_no_pisa_y_rechaza_entorno(corrida, tmp_path, monkeypatch):
    _, local, remoto = corrida
    with pytest.raises(FileExistsError):
        restaurar_ultima_copia(remoto, local)
    monkeypatch.setattr(modulo, "entorno_actual", lambda: {})
    with pytest.raises(ValueError, match="Entorno incompatible"):
        restaurar_ultima_copia(remoto, tmp_path / "otra")
    assert not (tmp_path / "otra").exists()


def test_codigo_distinto_rechazado(corrida, tmp_path, monkeypatch):
    _, _, remoto = corrida
    monkeypatch.setattr(modulo, "huella_codigo", lambda: "otra revision")
    with pytest.raises(ValueError, match="Codigo incompatible"):
        restaurar_ultima_copia(remoto, tmp_path / "otra")


def test_destino_incompatible_rechazado(corrida, tmp_path):
    config, _, remoto = corrida
    config = copy.deepcopy(config)
    config["seed"] = 5
    config["output"]["directory"] = str(tmp_path / "otra")
    with pytest.raises(ValueError, match="incompatible"):
        train(config, sync_dir=remoto)
    assert len(list(remoto.iterdir())) == 2


def test_sin_generaciones(tmp_path):
    with pytest.raises(ValueError, match="No hay copias"):
        ultima_copia_integra(tmp_path)


def test_csv_copia_verificada_y_original_intacto(tmp_path):
    original, copia = tmp_path / "raw.csv", tmp_path / "local.csv"
    original.write_bytes(b"id,sales\n1,10\n")
    digest = huella(original)
    assert copiar_csv_local(original, copia) == digest
    assert huella(original) == huella(copia) == digest
    assert copiar_csv_local(original, copia) == digest
    copia.write_bytes(b"diferente")
    with pytest.raises(ValueError, match="diferente"):
        copiar_csv_local(original, copia)
    with pytest.raises(ValueError, match="diferentes"):
        copiar_csv_local(original, original)


def test_corrupcion_durante_copia_csv_no_publica(tmp_path, monkeypatch):
    original, copia = tmp_path / "raw.csv", tmp_path / "local.csv"
    original.write_bytes(b"contenido")
    monkeypatch.setattr(modulo.shutil, "copyfile", lambda a, b: Path(b).write_bytes(b"roto"))
    with pytest.raises(ValueError, match="corrupta"):
        copiar_csv_local(original, copia)
    assert not copia.exists()
    assert original.read_bytes() == b"contenido"


def test_cli_propaga_sync_dir(monkeypatch):
    entrenamiento = importlib.import_module("src.train")
    recibido = {}
    monkeypatch.setattr(entrenamiento, "train", lambda config, **kwargs: recibido.update(kwargs))
    monkeypatch.setattr("sys.argv", ["train", "--config", "configs/favorita_completa.yaml", "--sync-dir", "drive"])
    entrenamiento.main()
    assert recibido == {"resume": None, "sync_dir": "drive"}


def test_notebook_valido_y_config_completa():
    import ast
    import yaml
    raiz = Path(__file__).resolve().parents[1]
    notebook = json.loads((raiz / "notebooks/favorita_colab.ipynb").read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    fuentes = []
    for celda in notebook["cells"]:
        fuente = "".join(celda["source"])
        fuentes.append(fuente)
        if celda["cell_type"] == "code":
            ast.parse(fuente)
            assert celda["outputs"] == [] and celda["execution_count"] is None
    texto = "\n".join(fuentes)
    assert texto.index('drive.mount(') < texto.index('copiar_csv_local(p[')
    assert texto.index('copiar_csv_local(p[') < texto.index('subprocess.run(comando')
    assert '"--resume"' in texto and '"--sync-dir"' in texto
    assert 'EJECUTAR_ENTRENAMIENTO = False' in texto
    assert 'C:\\Users' not in texto and 'ghp_' not in texto
    config = yaml.safe_load((raiz / "configs/favorita_completa.yaml").read_text())
    assert config["data"]["tiendas"] is None and config["data"]["familias"] is None
    assert "input_dim" not in config["model"]


@pytest.mark.parametrize("cambio", [None, "seed", "dataset", "subconjunto"])
def test_preparacion_real_del_notebook_reanudar(corrida, tmp_path, cambio):
    """Ejecuta el script de preparacion del notebook en el interprete de tests.

    No entrena: prueba el cableado de rutas/config y sus validaciones reales.
    """
    import ast
    import subprocess
    import sys
    config, local, remoto = corrida
    if cambio == "subconjunto":
        config["data"]["tiendas"] = [1]
        config["output"]["directory"] = str(tmp_path / "piloto")
        remoto = tmp_path / "drive_piloto"
        train(config, sync_dir=remoto)
    raiz = Path(__file__).resolve().parents[1]
    notebook = json.loads((raiz / "notebooks/favorita_colab.ipynb").read_text())
    script = None
    for celda in notebook["cells"]:
        if celda["cell_type"] == "code":
            for nodo in ast.parse("".join(celda["source"])).body:
                if isinstance(nodo, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "preparar" for t in nodo.targets):
                    script = ast.literal_eval(nodo.value)
    assert script is not None
    if cambio == "dataset":
        with Path(config["data"]["path"]).open("a") as archivo:
            archivo.write("\n")  # Mismos datos logicos, distintos bytes: rechazo estricto.
    destino = tmp_path / "config_colab.yaml"
    parametros = dict(modo="reanudar", csv_drive=config["data"]["path"], remoto=str(remoto),
                      csv_local=str(tmp_path / "copiado.csv"), local=str(tmp_path / "recuperada"),
                      config=str(destino), epochs=3, batch=4, device="cpu",
                      seed=1 if cambio == "seed" else 42)
    resultado = subprocess.run([sys.executable, "-B", "-c", script], input=json.dumps(parametros),
                              text=True, capture_output=True, cwd=raiz)
    if cambio is None:
        assert resultado.returncode == 0, resultado.stderr
        assert destino.exists()
        assert "Epoca recuperada: 2" in resultado.stdout
    else:
        assert resultado.returncode != 0
        assert not destino.exists()
        assert {"seed": "Configuracion incompatible", "dataset": "SHA-256 diferente",
                "subconjunto": "checkpoints del subconjunto"}[cambio] in resultado.stderr
