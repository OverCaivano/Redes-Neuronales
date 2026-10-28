"""Adaptacion reproducible de los HTML exportados; ejecutar desde la raiz.

Solo herramienta de desarrollo: requiere beautifulsoup4. Conserva los originales
bajo disenos/stitch y reutiliza sus contenedores, formularios, tablas y tema.
"""
import copy
import json
from pathlib import Path
from bs4 import BeautifulSoup

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "web/templates"


def leer(nombre):
    return BeautifulSoup((RAIZ / "disenos/stitch" / (nombre + ".html")).read_text(encoding="utf-8"), "html.parser")


def contenido(nodo, html):
    nuevo = copy.deepcopy(nodo)
    nuevo.clear()
    nuevo.append(BeautifulSoup(html, "html.parser"))
    return str(nuevo)


def bloques(sopa):
    return sopa.main.find("div", recursive=False).find_all(recursive=False)


def titulo(sopa, subtitulo):
    original = sopa.find("h1")
    return '<div class="mb-space-lg"><p class="text-primary text-label-sm mb-2 uppercase tracking-wider">StockPredict IA · piloto local</p>' + str(original) + '<p class="text-on-surface-variant mt-2">' + subtitulo + '</p></div>'


def guardar(nombre, contenido_html):
    (DESTINO / (nombre + ".html")).write_text(
        '{% extends "base.html" %}{% block contenido %}' + contenido_html + '{% endblock %}\n', encoding="utf-8")


sopa = leer("predicciones")
config = json.loads(sopa.find("script", id="tailwind-config").string.split("=", 1)[1].rstrip(";"))
config["content"] = ["./web/templates/**/*.html", "./web/static/app.js"]
config["theme"]["extend"]["borderRadius"]["xl"] = "0.5rem"
config["theme"]["extend"]["fontFamily"] = {k: v + ["Arial", "sans-serif"] for k, v in config["theme"]["extend"]["fontFamily"].items()}
(RAIZ / "web/tailwind.config.cjs").write_text("module.exports = " + json.dumps(config) + ";\n", encoding="utf-8")

# Marco compartido extraido del HTML de Stitch: sidebar, navegacion, cabecera.
aside = copy.deepcopy(sopa.body.aside)
for imagen in aside.find_all("img"):
    imagen.replace_with(BeautifulSoup('<span class="marca" aria-hidden="true">SP</span>', "html.parser"))
nav = aside.find("nav")
nav.clear()
nav.append(BeautifulSoup("""{% for clave, nombre in paginas.items() %}<a href="/{{ clave }}" class="nav-link {% if pagina == clave %}activo{% endif %}" aria-current="{{ 'page' if pagina == clave else 'false' }}"><span class="nav-icon" aria-hidden="true">{{ {'inicio':'▦','predicciones':'↗','evaluacion':'▥','historial':'◷','acerca':'ⓘ'}[clave] }}</span>{{ nombre }}</a>{% endfor %}""", "html.parser"))
for nodo in aside.find_all(string=True):
    if "PyTorch 2.2" in str(nodo): nodo.replace_with("PyTorch / MLP")
    elif str(nodo).strip() == "Online": nodo.replace_with("Piloto local")
for elemento in aside.select(".animate-ping"):
    elemento.decompose()
for elemento in aside.select(".material-symbols-outlined"):
    elemento.string = "◇"
    elemento["class"] = ["text-primary"]
header = copy.deepcopy(sopa.body.find("header"))
header.clear()
header.append(BeautifulSoup('<button id="menu-toggle" type="button" aria-controls="navegacion" aria-expanded="false">☰ <span>Menú</span></button><span class="text-label-md">Favorita · consultas retrospectivas</span><span class="badge">Piloto local / CPU</span>', "html.parser"))
aside["id"] = "navegacion"
base = '<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>{{ titulo }} · StockPredict IA</title>'
base += '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin><link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@600;700&display=swap" rel="stylesheet">'
base += '<link rel="stylesheet" href="/static/estilos.css"><script src="/static/app.js" defer></script></head><body class="bg-surface font-body-md text-on-surface"><a class="skip" href="#contenido">Saltar al contenido</a>'
base += str(aside) + '<div class="pl-72 app-body">' + str(header) + '<main id="contenido" class="relative min-h-screen bg-surface w-full px-space-xl py-space-xl"><div class="flex flex-col gap-space-lg max-w-screen-2xl mx-auto">{% block contenido %}{% endblock %}</div><footer class="mt-10 text-body-sm text-on-surface-variant">Prototipo académico · ventas estimadas por día y familia. No es inventario ni recomendación de compra.</footer></main></div></body></html>'
(DESTINO / "base.html").write_text(base, encoding="utf-8")

aviso = '<div class="notice"><strong>Alcance del piloto local</strong><p>Tiendas 1 y 2 · AUTOMOTIVE, BABY CARE y BEAUTY. Consultas retrospectivas con siete días anteriores observados. Las promociones del día se consideran conocidas anticipadamente.</p></div>'
alertas = '{% for aviso in resumen.advertencias %}<p class="notice error" role="alert">{{ aviso }}</p>{% endfor %}'

# Formulario original conservado, suprimiendo opciones/contexto no implementados.
formulario = copy.deepcopy(sopa.find("form", id="prediction-form"))
# Solo conservar los tres campos originales y sus labels, con ids compartidos.
campos = []
for identificador, label in (("store-select", "Tienda"), ("family-select", "Familia"), ("date-input", "Fecha histórica")):
    original = formulario.find(id=identificador)
    original.attrs.pop("value", None)
    original.attrs.pop("min", None)
    original.attrs.pop("max", None)
    original["required"] = ""
    original["disabled"] = ""
    if original.name == "select":
        original.clear()
        opcion = sopa.new_tag("option", value="")
        opcion.string = "Cargando opciones…"
        original.append(opcion)
    else:
        original["type"] = "date"
    campos.append('<div class="flex flex-col gap-2"><label for="' + identificador + '" class="font-label-lg">' + label + '</label>' + str(original) + '</div>')
formulario.clear()
formulario.append(BeautifulSoup(''.join(campos) + '<p id="rango-fechas" class="text-body-sm text-on-surface-variant"></p><div class="flex flex-wrap gap-3"><button id="btn-predict" class="primary-button" type="submit" disabled>Generar predicción</button><button class="secondary-button" id="btn-pilot-case" type="button" disabled>Usar caso del piloto</button></div>', "html.parser"))
formulario['aria-describedby'] = 'consulta-estado'
(DESTINO / "formulario.html").write_text(str(formulario), encoding="utf-8")
(DESTINO / "resultado.html").write_text('<div id="consulta-estado" role="status" aria-live="polite">Preparando opciones del piloto…</div><div id="resultado" hidden><p class="text-label-md text-on-surface-variant">Ventas estimadas por día y familia</p><p class="text-metric-display text-primary my-4" id="ventas-estimadas"></p><p id="resultado-contexto"></p><p class="text-body-sm mt-4" id="resultado-modelo"></p></div><p class="notice mt-6">No representa stock disponible ni una recomendación de compra. Consulta histórica con modelo fijo; sin intervalos de confianza.</p>',encoding="utf-8")

bs = bloques(sopa)
layout = copy.deepcopy(bs[2])
paneles = layout.find_all(recursive=False)
layout.clear()
layout.append(BeautifulSoup(contenido(paneles[0], '<h2 class="text-headline-md mb-4">Parámetros de la consulta</h2>{% include "formulario.html" %}'), "html.parser"))
layout.append(BeautifulSoup(contenido(paneles[1], '<div class="bg-surface-container-lowest rounded-xl p-space-lg shadow-sm"><h2 class="text-headline-md mb-5">Resultado del modelo</h2>{% include "resultado.html" %}</div>'), "html.parser"))
flujo = contenido(bs[3], '<h2 class="text-headline-md">Cómo se genera la predicción</h2><div class="grid md:grid-cols-3 gap-space-md"><div class="card"><h3>1. Historia disponible</h3><p>Ventas de t−1, t−7 y promedio de los siete días anteriores; promoción planificada del día.</p></div><div class="card"><h3>2. Transformaciones guardadas</h3><p>StandardScaler, ciclos seno/coseno y one-hot. Sin reajuste ni información futura.</p></div><div class="card"><h3>3. MLP del piloto</h3><p>13 entradas → 32 neuronas ReLU → 1 salida. Ventas en su escala original, sin recorte.</p></div></div>')
guardar("predicciones", titulo(sopa, "Seleccioná una tienda, familia y fecha compatibles con el modelo entrenado.") + aviso + str(layout) + flujo)

# Inicio: tarjetas y panel de consulta del exportado, con datos de la corrida local.
inicio = leer("inicio"); bi = bloques(inicio)
kpis = contenido(bi[1], """{% set r = resumen.corridas[-1] if resumen.corridas else none %}<article class="card"><p class="eyebrow">Modelo local</p><h2>Red neuronal MLP</h2><p>Una capa oculta de 32 neuronas · ReLU</p><span class="badge">PyTorch · CPU</span></article><article class="card"><p class="eyebrow">Progreso guardado</p><p class="metric">{{ r.epocas if r else '—' }}</p><p>Épocas completadas{% if r %} · mejor época {{ r.mejor_epoca }}{% endif %}</p></article><article class="card"><p class="eyebrow">Alcance del piloto</p><p class="metric">2 × 3</p><p>Tiendas × familias</p><span class="badge">Corporación Favorita</span></article><article class="card"><p class="eyebrow">Objetivo</p><h2>Ventas diarias</h2><p>Por tienda y familia, a un paso.</p><span class="badge">Sin recomendaciones de stock</span></article>""")
consulta = contenido(bi[2], '<div class="flex items-center justify-between mb-6"><h2 class="text-headline-md">Consultar predicción</h2><a href="/predicciones" class="text-primary">Ver detalle →</a></div><div class="grid lg:grid-cols-2 gap-space-xl"><div>{% include "formulario.html" %}</div><div>{% include "resultado.html" %}</div></div>')
guardar("inicio", titulo(inicio, "Consultá el modelo entrenado y revisá sus resultados locales verificables.") + alertas + kpis + aviso + consulta + '<div class="notice"><strong>Entrenamiento completo en Colab / Drive</strong><p>{{ resumen.drive }} Los checkpoints mostrados pertenecen únicamente al piloto local.</p><a href="/historial" class="text-primary">Explorar historial →</a></div>')

# Evaluacion: mismo layout de KPI y tabla, sin pruebas estadisticas inventadas.
eval_s = leer("evaluacion"); be = bloques(eval_s)
metricas = contenido(be[2], """{% for nombre, valor in [('MAE · MLP', e.mlp.mae), ('RMSE · MLP', e.mlp.rmse), ('RMSE · baseline semanal', e.baseline.rmse)] %}<article class="card"><p class="eyebrow">{{ nombre }}</p><p class="metric">{{ '%.6f'|format(valor) }}</p><p>Ventas en unidades originales</p></article>{% endfor %}<article class="card"><p class="eyebrow">Reducción del RMSE</p><p class="metric text-secondary">{{ '%.2f'|format(e.mejora) }} %</p><p>Comparación sobre las mismas filas</p></article>""")
tabla = copy.deepcopy(eval_s.find("table")); tabla.clear()
tabla.append(BeautifulSoup("""<caption>Comparación de rendimiento sobre el test interno</caption><thead><tr><th>Modelo</th><th>MAE</th><th>RMSE</th><th>Ejemplos</th></tr></thead><tbody><tr><th>MLP · época 4</th><td>{{ '%.6f'|format(e.mlp.mae) }}</td><td>{{ '%.6f'|format(e.mlp.rmse) }}</td><td>{{ e.mlp.n }}</td></tr><tr><th>Baseline: ventas de hace 7 días</th><td>{{ '%.6f'|format(e.baseline.mae) }}</td><td>{{ '%.6f'|format(e.baseline.rmse) }}</td><td>{{ e.baseline.n }}</td></tr></tbody>""", "html.parser"))
comparacion = """<section class="card"><h2>Comparación de errores</h2>{% for metrica in ['mae', 'rmse'] %}<div class="mt-6"><h3>{{ metrica|upper }}</h3>{% for etiqueta, valor, color in [('MLP', e.mlp[metrica], 'bg-primary-container'), ('Baseline', e.baseline[metrica], 'bg-tertiary')] %}<div class="my-3"><div class="flex justify-between text-label-md"><span>{{ etiqueta }}</span><span>{{ '%.6f'|format(valor) }}</span></div><div class="bg-surface-container-low rounded-lg h-4 mt-1"><div class="{{ color }} rounded-lg h-4" style="width: {{ 100 * valor / ([e.mlp[metrica], e.baseline[metrica]]|max) }}%"></div></div></div>{% endfor %}</div>{% endfor %}</section>"""
guardar("evaluacion", titulo(eval_s, "Resultados leídos del JSON de evaluación local; modelo elegido previamente por validación.") + alertas + aviso + '{% set e = resumen.evaluacion %}{% if e %}' + metricas + contenido(be[3], '<div class="table-scroll">' + str(tabla) + '</div>') + comparacion + '<div class="notice"><strong>Test interno: {{ e.inicio }} a {{ e.fin }}</strong><p>{{ e.mlp.n }} observaciones. No es el test.csv sin objetivos de Kaggle. Baseline = ventas(t−7), no media móvil. No se realizaron pruebas de significancia estadística.</p></div>{% else %}<div class="card">No hay una evaluación verificable disponible para este checkpoint.</div>{% endif %}')

# Historial: tabla y panel del exportado; dos corridas locales y curva de 4 epocas.
hist = leer("historial"); bh = bloques(hist)
layout_h = copy.deepcopy(bh[2]); hijos = layout_h.find_all(recursive=False); layout_h.clear()
tabla_h = copy.deepcopy(hist.find("table")); tabla_h.clear()
tabla_h.append(BeautifulSoup("""<caption>Corridas encontradas en el almacenamiento local</caption><thead><tr><th>Corrida</th><th>Épocas</th><th>Mejor época</th><th>MAE validación</th><th>RMSE validación</th></tr></thead><tbody>{% for r in resumen.corridas %}<tr><th>{{ r.nombre }}</th><td>{{ r.epocas }}</td><td>{{ r.mejor_epoca }}</td><td>{{ '%.6f'|format(r.mae) }}</td><td>{{ '%.6f'|format(r.rmse) }}</td></tr>{% else %}<tr><td colspan="5">No hay resultados locales disponibles.</td></tr>{% endfor %}</tbody>""", "html.parser"))
curva = """{% if resumen.corridas %}{% set r = resumen.corridas[-1] %}<section class="card mt-6"><h2>Historia de la corrida local más reciente</h2><p class="mb-4">MSE por época: entrenamiento y validación. Valores leídos de metricas.csv.</p><div id="curva-entrenamiento" aria-label="Curva de MSE de entrenamiento y validación"></div><div class="table-scroll"><table id="tabla-epocas"><thead><tr><th>época</th><th>MSE entrenamiento</th><th>MSE validación</th><th>MAE validación</th><th>RMSE validación</th></tr></thead><tbody>{% for f in r.historial %}<tr data-epoca="{{ f.epoca }}" data-train="{{ f.train_mse }}" data-val="{{ f.val_mse }}"><td>{{ f.epoca|int }}</td><td>{{ '%.6f'|format(f.train_mse) }}</td><td>{{ '%.6f'|format(f.val_mse) }}</td><td>{{ '%.6f'|format(f.val_mae) }}</td><td>{{ '%.6f'|format(f.val_rmse) }}</td></tr>{% endfor %}</tbody></table></div></section>{% endif %}"""
layout_h.append(BeautifulSoup(contenido(hijos[0], '<section class="card"><div class="table-scroll">' + str(tabla_h) + '</div></section>' + curva), "html.parser"))
layout_h.append(BeautifulSoup(contenido(hijos[1], '<section class="card"><h2>Checkpoint de consultas</h2><p class="badge">Piloto reanudado · best.pt</p><dl class="datos"><dt>Arquitectura</dt><dd>13 → 32 (ReLU) → 1</dd><dt>Entrenamiento</dt><dd>Adam · MSE · lr 0.001</dd><dt>Semilla</dt><dd>42</dd><dt>Mejor época local</dt><dd>4, seleccionada por validación</dd><dt>Alcance</dt><dd>2 tiendas · 3 familias</dd></dl><p>La API no permite descargar ni reemplazar checkpoints.</p></section><section class="notice mt-6"><h2>Modelo completo / Drive</h2><p>{{ resumen.drive }}</p><p>El repositorio incluye un notebook de Colab. No se muestran métricas ni estado de entrenamiento remoto sin evidencia local.</p></section>'), "html.parser"))
guardar("historial", titulo(hist, "Dos registros del piloto: corrida inicial y continuación hasta la época 4; no son experimentos independientes.") + alertas + str(layout_h))

# Acerca: conservar las rejillas y tarjetas de objetivos, proceso y stack.
acerca = leer("acerca"); ba = bloques(acerca)
objetivo = contenido(ba[1], '<article class="card lg:col-span-8"><p class="eyebrow">Objetivo del proyecto</p><h2>Estimar ventas diarias con redes neuronales</h2><p>Usar ventas históricas y contexto temporal/comercial para estimar ventas del día siguiente por tienda y familia. Las recomendaciones de reposición pertenecen a una etapa futura.</p><div class="grid sm:grid-cols-3 gap-4 mt-6"><div class="notice">Paso diario</div><div class="notice">Tienda · familia</div><div class="notice">MLP / PyTorch</div></div></article><aside class="card lg:col-span-4"><h2>Alcance implementado</h2><p>Lectura y validación, historia causal, one-hot, escalado, entrenamiento, reanudación, evaluación y consultas locales.</p><p class="notice mt-4">Sin stock actual, pedidos sugeridos ni intervalos de confianza.</p></aside>')
proceso = contenido(ba[2], '<h2 class="text-headline-md">Proceso de extremo a extremo</h2><div class="grid sm:grid-cols-2 xl:grid-cols-4 gap-4"><article class="card"><p class="eyebrow">01 / Datos</p><h3>Historia causal</h3><p>train.csv, siete días previos observados. Los huecos no son ventas cero.</p></article><article class="card"><p class="eyebrow">02 / Entrenamiento</p><h3>MLP baseline</h3><p>Adam y MSE. Escalador y categorías ajustados solo con entrenamiento.</p></article><article class="card"><p class="eyebrow">03 / Evaluación</p><h3>MAE y RMSE</h3><p>Test cronológico reservado y comparación con ventas de hace siete días.</p></article><article class="card"><p class="eyebrow">04 / Consulta</p><h3>Predicción real</h3><p>Flask reutiliza Python y el checkpoint existente. Sin entrenamiento web.</p></article></div>')
stack = contenido(ba[3], '<h2 class="text-headline-md">Tecnologías y límites</h2><div class="grid md:grid-cols-3 gap-4"><article class="card"><h3>Python y PyTorch</h3><p>Modelo MLP, pandas para datos y scikit-learn para transformaciones.</p></article><article class="card"><h3>HTML Stitch + Flask</h3><p>Interfaz adaptada del diseño exportado. API exclusivamente local con validación de entradas.</p></article><article class="card"><h3>Colab y Drive</h3><p>Notebook preparado para entrenamiento y recuperación. Estado remoto no verificado por esta aplicación.</p></article></div>')
guardar("acerca", titulo(acerca, "Proyecto académico de predicción de ventas con datos reales de Corporación Favorita.") + objetivo + proceso + stack + '<div class="notice">El piloto usa tiendas 1 y 2 y familias AUTOMOTIVE, BABY CARE y BEAUTY. No integra todavía petróleo, feriados ni transacciones externas. No se atribuyen investigadores, licencias o publicaciones no documentados.</div>')
guardar("error", '<div class="card"><h1>{{ titulo }}</h1><p>Usá la navegación para volver a una pantalla disponible.</p><a class="primary-button" href="/inicio">Volver al inicio</a></div>')
print("Plantillas generadas desde los cinco HTML de Stitch.")
