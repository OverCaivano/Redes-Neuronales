"use strict";
const menu = document.getElementById("menu-toggle");
const nav = document.getElementById("navegacion");
if (menu && nav) {
  menu.addEventListener("click", () => {
    const abierto = nav.classList.toggle("abierto");
    menu.setAttribute("aria-expanded", String(abierto));
  });
  document.addEventListener("keydown", event => {
    if (event.key === "Escape") { nav.classList.remove("abierto"); menu.setAttribute("aria-expanded", "false"); }
  });
}

const form = document.getElementById("prediction-form");
if (form) {
  const tienda = document.getElementById("store-select");
  const familia = document.getElementById("family-select");
  const fecha = document.getElementById("date-input");
  const boton = document.getElementById("btn-predict");
  const piloto = document.getElementById("btn-pilot-case");
  const estado = document.getElementById("consulta-estado");
  const resultado = document.getElementById("resultado");
  const rango = document.getElementById("rango-fechas");
  let series = [], fechas = [], ocupado = false;
  const controles = [tienda, familia, fecha, boton, piloto];
  function mostrarEstado(texto, error = false) {
    estado.textContent = texto;
    estado.className = error ? "notice error" : "notice";
    estado.setAttribute("role", error ? "alert" : "status");
  }
  function opciones(select, valores) {
    select.replaceChildren(...valores.map(valor => new Option(String(valor), String(valor))));
  }
  function comprobarFecha() {
    fecha.setCustomValidity(fechas.includes(fecha.value) ? "" : "Esta fecha no tiene los siete días previos observados. Elegí otra dentro del intervalo.");
  }
  function actualizarFecha() {
    const serie = series.find(s => s.tienda === Number(tienda.value) && s.familia === familia.value);
    fechas = serie ? serie.fechas : [];
    fecha.min = fechas[0] || "";
    fecha.max = fechas[fechas.length - 1] || "";
    if (!fechas.includes(fecha.value)) fecha.value = fechas.includes("2017-07-16") ? "2017-07-16" : (fechas[fechas.length - 1] || "");
    rango.textContent = fechas.length ? `${fechas.length} fechas compatibles entre ${fecha.min} y ${fecha.max}. Los huecos de calendario no son válidos.` : "Sin fechas disponibles.";
    comprobarFecha();
  }
  function actualizarFamilias() {
    opciones(familia, series.filter(s => s.tienda === Number(tienda.value)).map(s => s.familia));
    actualizarFecha();
  }
  function limpiarResultado() {
    resultado.hidden = true;
    mostrarEstado("Seleccioná los parámetros y generá una predicción real.");
  }
  tienda.addEventListener("change", () => { actualizarFamilias(); limpiarResultado(); });
  familia.addEventListener("change", () => { actualizarFecha(); limpiarResultado(); });
  fecha.addEventListener("input", () => { comprobarFecha(); limpiarResultado(); });
  piloto.addEventListener("click", () => {
    const ejemplo = series.find(s => s.tienda === 1 && s.familia === "BEAUTY" && s.fechas.includes("2017-07-16"));
    if (!ejemplo) { mostrarEstado("El caso de ejemplo no está disponible en este checkpoint.", true); return; }
    tienda.value = "1"; actualizarFamilias(); familia.value = "BEAUTY";
    fecha.value = "2017-07-16"; actualizarFecha(); limpiarResultado();
  });
  fetch("/api/opciones").then(async respuesta => {
    const datos = await respuesta.json();
    if (!respuesta.ok) throw new Error(datos.error || "No se pudieron cargar las opciones.");
    series = datos.series;
    opciones(tienda, [...new Set(series.map(s => s.tienda))]);
    actualizarFamilias();
    controles.forEach(c => { c.disabled = false; });
    limpiarResultado();
  }).catch(error => mostrarEstado(error.message + " Recargá la página después de revisar el servicio.", true));
  form.addEventListener("submit", async event => {
    event.preventDefault();
    comprobarFecha();
    if (ocupado || !form.reportValidity()) return;
    ocupado = true; form.setAttribute("aria-busy", "true");
    resultado.hidden = true; controles.forEach(c => { c.disabled = true; });
    boton.textContent = "Calculando…";
    mostrarEstado("Verificando historia y ejecutando el MLP local. Esperá el resultado.");
    try {
      const respuesta = await fetch("/api/predicciones", {method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify({tienda: Number(tienda.value), familia: familia.value, fecha: fecha.value})});
      const datos = await respuesta.json();
      if (!respuesta.ok) throw new Error(datos.error || "No se pudo generar la predicción.");
      document.getElementById("ventas-estimadas").textContent = Number(datos.ventas_estimadas).toLocaleString("es-AR", {minimumFractionDigits: 6, maximumFractionDigits: 6});
      document.getElementById("resultado-contexto").textContent = `Tienda ${datos.tienda} · ${datos.familia} · ${datos.fecha}`;
      document.getElementById("resultado-modelo").textContent = `${datos.modelo} · época ${datos.epoca}. Promoción planificada conocida anticipadamente.`;
      resultado.hidden = false;
      mostrarEstado(datos.ventas_estimadas < 0 ? "El modelo devolvió una estimación negativa; se muestra sin recorte como limitación del MLP." : "Predicción real completada. No se ejecutó entrenamiento.", datos.ventas_estimadas < 0);
    } catch (error) { mostrarEstado(error.message, true); }
    finally {
      ocupado = false; form.removeAttribute("aria-busy");
      controles.forEach(c => { c.disabled = false; }); boton.textContent = "Generar predicción";
    }
  });
}

// Curva trazada exclusivamente con las filas de metricas.csv renderizadas por el servidor.
const grafico = document.getElementById("curva-entrenamiento");
if (grafico) {
  const filas = [...document.querySelectorAll("#tabla-epocas tbody tr")].map(f => ({epoca: Number(f.dataset.epoca), train: Number(f.dataset.train), val: Number(f.dataset.val)}));
  if (filas.length && filas.every(f => Object.values(f).every(Number.isFinite))) {
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg"); svg.setAttribute("viewBox", "0 0 680 270"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", "MSE por época. Los valores exactos están en la tabla siguiente.");
    const max = Math.max(...filas.flatMap(f => [f.train, f.val]), 1);
    function nodo(tipo, attrs, texto) { const n = document.createElementNS(ns, tipo); for (const [k,v] of Object.entries(attrs)) n.setAttribute(k,String(v)); if (texto !== undefined) n.textContent=texto; svg.appendChild(n); return n; }
    const x = i => 60 + i * 570 / Math.max(filas.length - 1, 1);
    const y = v => 215 - v * 170 / max;
    for (let i=0;i<=4;i++) { const v=max*i/4; nodo("line",{x1:60,x2:630,y1:y(v),y2:y(v),stroke:"var(--chart-grid)"}); nodo("text",{x:8,y:y(v)+4,fill:"var(--chart-label)","font-size":12},v.toFixed(1)); }
    for (const [clave, color, etiqueta, posicion] of [["train","var(--chart-train)","Entrenamiento",60],["val","var(--chart-validation)","Validación",240]]) {
      nodo("polyline",{points:filas.map((f,i)=>`${x(i)},${y(f[clave])}`).join(" "),fill:"none",stroke:color,"stroke-width":3});
      filas.forEach((f,i)=>nodo("circle",{cx:x(i),cy:y(f[clave]),r:4,fill:color}));
      nodo("text",{x:posicion,y:22,fill:color,"font-size":13},etiqueta);
    }
    filas.forEach((f,i)=>nodo("text",{x:x(i),y:243,"text-anchor":"middle",fill:"var(--chart-label)","font-size":12},`Época ${f.epoca}`));
    grafico.appendChild(svg);
  }
}
