"""Copias inmutables por epoca para Drive montado; sin depender de rename remoto.

Un unico escritor por carpeta. SHA-256 detecta corrupcion, no autentica archivos.
Solo restaurar checkpoints propios. Nunca elimina generaciones anteriores.
"""

import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import tempfile
import uuid
import warnings

ARCHIVOS = ("last.pt", "best.pt", "metricas.csv", "resumen.json", "configuracion.yaml")
PAQUETES = ("torch", "numpy", "pandas", "scipy", "scikit-learn", "PyYAML")


def huella(ruta):
    resultado = hashlib.sha256()
    with Path(ruta).open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b""):
            resultado.update(bloque)
    return resultado.hexdigest()


def entorno_actual():
    # El sufijo +cpu/+cu depende de la plataforma; se conserva la version base.
    return {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "paquetes": {p: importlib.metadata.version(p).split("+")[0] for p in PAQUETES},
    }


def huella_codigo():
    raiz = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for ruta in sorted(raiz.rglob("*.py")):
        digest.update(ruta.relative_to(raiz).as_posix().encode())
        digest.update(b"\0")
        digest.update(ruta.read_bytes().replace(b"\r\n", b"\n"))
        digest.update(b"\0")
    return digest.hexdigest()


def copiar_csv_local(origen, destino):
    """Copia por streaming, verifica y publica localmente; origen inmutable."""
    origen, destino = Path(origen), Path(destino)
    if origen.resolve() == destino.resolve():
        raise ValueError("Origen y destino deben ser diferentes")
    digest = huella(origen)
    if destino.exists():
        if huella(destino) != digest:
            raise ValueError("El CSV local existente es diferente; use otra ruta")
        return digest
    destino.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destino.parent) as temporal:
        copia = Path(temporal) / "train.csv"
        shutil.copyfile(origen, copia)
        if huella(copia) != digest:
            raise ValueError("Copia del CSV incompleta o corrupta")
        copia.replace(destino)
    return digest


def verificar_copia(directorio):
    """Comprueba marca, manifiesto y TODOS los archivos, sin deserializar pickle."""
    directorio = Path(directorio)
    try:
        manifiesto = directorio / "manifiesto.json"
        if (directorio / "COMPLETO").read_text(encoding="utf-8").strip() != huella(manifiesto):
            raise ValueError("Marca de integridad incorrecta")
        datos = json.loads(manifiesto.read_text(encoding="utf-8"))
        if datos["version"] != 1 or set(datos["archivos"]) != set(ARCHIVOS):
            raise ValueError("Manifiesto incompatible")
        if not isinstance(datos["epoca"], int) or datos["epoca"] < 1:
            raise ValueError("Epoca invalida")
        for nombre, info in datos["archivos"].items():
            ruta = directorio / nombre
            if ruta.stat().st_size != info["bytes"] or huella(ruta) != info["sha256"]:
                raise ValueError(f"Archivo corrupto: {nombre}")
        for clave in ("identidad", "entorno", "codigo_sha256"):
            if clave not in datos:
                raise ValueError(f"Manifiesto incompleto: {clave}")
        return datos
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError(f"Copia incompleta o ilegible: {directorio}") from error


def ultima_copia_integra(raiz):
    """Ignora transferencias incompletas/corruptas y avisa; no borra nada."""
    candidatas = []
    for ruta in sorted(Path(raiz).glob("epoca_*")):
        try:
            datos = verificar_copia(ruta)
            candidatas.append((datos["epoca"], ruta.name, ruta, datos))
        except ValueError as error:
            warnings.warn(f"Se omite copia no recuperable {ruta.name}: {error}", UserWarning)
    if not candidatas:
        raise ValueError("No hay copias integras recuperables")
    identidades = {item[3]["identidad"] for item in candidatas}
    if len(identidades) != 1:
        raise ValueError("La carpeta mezcla corridas incompatibles")
    _, _, ruta, datos = max(candidatas, key=lambda item: item[:2])
    return ruta, datos


def validar_entorno(datos):
    if datos["entorno"] != entorno_actual():
        raise ValueError("Entorno incompatible: reinstale las versiones del manifiesto")
    if datos["codigo_sha256"] != huella_codigo():
        raise ValueError("Codigo incompatible: utilice la misma revision del repositorio")


def sincronizar_epoca(local, remoto):
    """Se llama sincronicamente despues del guardado local de cada epoca.

    Un fallo detiene el entrenamiento; conserva el ultimo estado local y todas
    las copias previas. La marca COMPLETO se escribe al final y se relee.
    """
    from src.reanudacion import cargar_last, validar_reanudacion, _config_comparable

    local, remoto = Path(local), Path(remoto)
    if local.resolve() == remoto.resolve() or local.resolve() in remoto.resolve().parents:
        raise ValueError("La carpeta remota debe estar fuera de los resultados locales")
    estado = cargar_last(local / "last.pt")
    if estado.get("format_version") != 2:
        raise ValueError("La sincronizacion requiere checkpoint v2 con huella del dataset")
    validar_reanudacion(local / "last.pt", estado, estado["config"], estado["dataset_sha256"])
    identidad = hashlib.sha256(json.dumps({
        "config": _config_comparable(estado["config"]),
        "dataset": estado["dataset_sha256"], "columnas": estado["nombres_columnas"],
        "codigo": huella_codigo(), "entorno": entorno_actual(),
    }, sort_keys=True).encode()).hexdigest()
    # Revisar generaciones existentes ANTES de crear una nueva; nunca mezclar runs.
    for previa in remoto.glob("epoca_*"):
        try:
            registrada = verificar_copia(previa)
        except ValueError:
            continue
        if registrada["identidad"] != identidad:
            raise ValueError("Destino contiene una corrida incompatible")
        if registrada["epoca"] == estado["epoch"]:
            if all(huella(local / nombre) == registrada["archivos"][nombre]["sha256"] for nombre in ARCHIVOS):
                return previa
            raise ValueError("Destino contiene otra version de la misma epoca")
        if registrada["epoca"] > estado["epoch"]:
            raise ValueError("Destino contiene una epoca posterior; restaure la ultima copia")
    remoto.mkdir(parents=True, exist_ok=True)
    destino = remoto / f"epoca_{estado['epoch']:06d}_{uuid.uuid4().hex}"
    destino.mkdir()
    archivos = {}
    for nombre in ARCHIVOS:
        origen = local / nombre
        archivos[nombre] = {"bytes": origen.stat().st_size, "sha256": huella(origen)}
        shutil.copyfile(origen, destino / nombre)
        if huella(destino / nombre) != archivos[nombre]["sha256"]:
            raise ValueError(f"Fallo de integridad al sincronizar {nombre}")
    datos = {
        "version": 1, "epoca": estado["epoch"], "identidad": identidad,
        "archivos": archivos, "entorno": entorno_actual(), "codigo_sha256": huella_codigo(),
    }
    manifiesto = destino / "manifiesto.json"
    manifiesto.write_text(json.dumps(datos, indent=2, sort_keys=True), encoding="utf-8")
    (destino / "COMPLETO").write_text(huella(manifiesto), encoding="utf-8")
    verificar_copia(destino)
    return destino


def restaurar_ultima_copia(remoto, local):
    """Restaura cinco archivos a una carpeta NUEVA; nunca pisa resultados locales."""
    origen, datos = ultima_copia_integra(remoto)
    validar_entorno(datos)  # Antes de cargar cualquier objeto serializado.
    local = Path(local)
    if local.exists():
        raise FileExistsError(f"El destino local ya existe: {local}")
    local.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=local.parent) as temporal:
        temporal = Path(temporal)
        for nombre in (*ARCHIVOS, "manifiesto.json", "COMPLETO"):
            shutil.copyfile(origen / nombre, temporal / nombre)
        verificar_copia(temporal)
        from src.reanudacion import cargar_last, validar_reanudacion
        estado = cargar_last(temporal / "last.pt")
        validar_reanudacion(temporal / "last.pt", estado, estado["config"], estado["dataset_sha256"])
        # Publicar solo los cinco artefactos; el manifiesto describe la copia remota.
        salida = temporal / "restaurada"
        salida.mkdir()
        for nombre in ARCHIVOS:
            (temporal / nombre).replace(salida / nombre)
        salida.replace(local)
    return local / "last.pt"
