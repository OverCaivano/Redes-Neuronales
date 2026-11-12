"use strict";
const presentacion = window.StockPredictPresentacion;
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
  const modelo = document.getElementById("model-select");
  const tienda = document.getElementById("store-select");
  const familia = document.getElementById("family-select");
  const fecha = document.getElementById("date-input");
  const boton = document.getElementById("btn-predict");
  const piloto = document.getElementById("btn-pilot-case");
  const estado = document.getElementById("consulta-estado");
  const resultado = document.getElementById("resultado");
  const rango = document.getElementById("rango-fechas");
  let series = [], fechas = [], ocupado = false;
  const controles = [modelo, tienda, familia, fecha, boton, piloto];
  let serieCompleta = false;
  function mostrarEstado(texto, error = false) {
    estado.textContent = texto;
    estado.className = error ? "notice error" : "notice";
    estado.setAttribute("role", error ? "alert" : "status");
  }
  function opciones(select, valores, etiqueta = String) {
    select.replaceChildren(...valores.map(valor => new Option(etiqueta(valor), String(valor))));
  }
  function comprobarFecha() {
    const rangoCompleto = serieCompleta && fechas.length === 0
      && fecha.value >= fecha.min && fecha.value <= fecha.max;
    fecha.setCustomValidity(fechas.includes(fecha.value) || rangoCompleto
      ? ""
      : "Esta fecha no tiene los siete días previos observados. Elegí otra dentro del intervalo.");
  }
  function actualizarFecha() {
    const serie = series.find(s => s.tienda === Number(tienda.value) && s.familia === familia.value);
    fechas = serie && serie.fechas ? serie.fechas : [];
    fecha.min = serie ? (serie.fecha_min || fechas[0] || "") : "";
    fecha.max = serie ? (serie.fecha_max || fechas[fechas.length - 1] || "") : "";
    if (!fechas.includes(fecha.value) && !(serieCompleta && fecha.value >= fecha.min && fecha.value <= fecha.max)) {
      fecha.value = fechas.includes("2017-07-16") ? "2017-07-16" : (fechas[fechas.length - 1] || "");
    }
    rango.textContent = !serie
      ? "Sin fechas disponibles."
      : serieCompleta
        ? `Intervalo histórico ${fecha.min} a ${fecha.max}. El servidor confirma que la fecha y los siete días previos tengan observaciones.`
        : `${fechas.length} fechas compatibles entre ${fecha.min} y ${fecha.max}. Los huecos de calendario no son válidos.`;
    comprobarFecha();
  }
  function actualizarFamilias() {
    opciones(familia, series.filter(s => s.tienda === Number(tienda.value))
      .map(s => s.familia), presentacion.nombreFamilia);
    actualizarFecha();
  }
  function limpiarResultado() {
    resultado.hidden = true;
    mostrarEstado("Seleccioná los parámetros y generá una predicción real.");
  }
  async function cargarOpciones() {
    controles.forEach(c => { c.disabled = true; });
    mostrarEstado("Cargando opciones del modelo seleccionado.");
    try {
      const respuesta = await fetch(`/api/opciones?modelo=${encodeURIComponent(modelo.value)}`);
      const datos = await respuesta.json();
      if (!respuesta.ok) throw new Error(datos.error || "No se pudieron cargar las opciones.");
      series = datos.series;
      serieCompleta = modelo.value === "completo";
      opciones(tienda, [...new Set(series.map(s => s.tienda))]);
      actualizarFamilias();
      controles.forEach(c => { c.disabled = false; });
      limpiarResultado();
    } catch (error) {
      series = [];
      fechas = [];
      opciones(tienda, []);
      opciones(familia, []);
      fecha.value = "";
      rango.textContent = "";
      controles.forEach(c => { c.disabled = c !== modelo; });
      mostrarEstado(error.message + " Recargá la página después de revisar el servicio.", true);
    }
  }
  modelo.addEventListener("change", () => {
    fechas = [];
    fecha.value = "";
    limpiarResultado();
    cargarOpciones();
  });
  tienda.addEventListener("change", () => { actualizarFamilias(); limpiarResultado(); });
  familia.addEventListener("change", () => { actualizarFecha(); limpiarResultado(); });
  fecha.addEventListener("input", () => { comprobarFecha(); limpiarResultado(); });
  piloto.addEventListener("click", () => {
    const ejemplo = series.find(s => s.tienda === 1 && s.familia === "BEAUTY"
      && (s.fechas ? s.fechas.includes("2017-07-16")
        : s.fecha_min <= "2017-07-16" && s.fecha_max >= "2017-07-16"));
    if (!ejemplo) { mostrarEstado("El caso de ejemplo no está disponible en este checkpoint.", true); return; }
    tienda.value = "1"; actualizarFamilias(); familia.value = "BEAUTY";
    fecha.value = "2017-07-16"; actualizarFecha(); limpiarResultado();
  });
  cargarOpciones();
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
        body: JSON.stringify({
          tienda: Number(tienda.value), familia: familia.value, fecha: fecha.value, modelo: modelo.value,
        })});
      const datos = await respuesta.json();
      if (!respuesta.ok) throw new Error(datos.error || "No se pudo generar la predicción.");
      document.getElementById("ventas-estimadas").textContent =
        presentacion.formatearUnidadesEstimadas(datos.ventas_estimadas);
      document.getElementById("resultado-contexto").textContent =
        `Tienda ${datos.tienda} · ${presentacion.nombreFamilia(datos.familia)} · ${datos.fecha}`;
      document.getElementById("resultado-modelo").textContent = `${datos.modelo} · época ${datos.epoca}. Promoción del día conocida anticipadamente.`;
      resultado.hidden = false;
      mostrarEstado(datos.ventas_estimadas < 0
        ? "Predicción real completada. La estimación negativa se presenta como 0 unidades."
        : "Predicción real completada. No se ejecutó entrenamiento.");
    } catch (error) { mostrarEstado(error.message, true); }
    finally {
      ocupado = false; form.removeAttribute("aria-busy");
      controles.forEach(c => { c.disabled = false; }); boton.textContent = "Generar predicción";
    }
  });
}

const forecastForm = document.getElementById("forecast-form");
if (forecastForm) {
  const store = document.getElementById("forecast-store");
  const family = document.getElementById("forecast-family");
  const origin = document.getElementById("forecast-origin");
  const horizon = document.getElementById("forecast-horizon");
  const submit = document.getElementById("forecast-submit");
  const status = document.getElementById("pronostico-estado");
  const results = document.getElementById("forecast-results");
  const rows = document.getElementById("forecast-rows");
  const context = document.getElementById("forecast-context");
  let series = [], busy = false;
  const controls = [store, family, horizon, submit];
  function setStatus(message, error = false) {
    status.textContent = message;
    status.className = error ? "notice error mt-4" : "notice mt-4";
    status.setAttribute("role", error ? "alert" : "status");
  }
  function setOptions(select, values, label = String) {
    select.replaceChildren(...values.map(value => new Option(label(value), String(value))));
  }
  function updateFamilies() {
    const families = [...new Set(series.filter(item => item.tienda === Number(store.value))
      .map(item => item.familia))];
    setOptions(family, families, presentacion.nombreFamilia);
    if (families.includes("BEAUTY")) family.value = "BEAUTY";
  }
  function clearForecast() {
    results.hidden = true;
    rows.replaceChildren();
    setStatus("Elegí tienda, familia y horizonte para generar el pronóstico.");
  }
  async function loadOptions() {
    controls.forEach(control => { control.disabled = true; });
    try {
      const response = await fetch("/api/opciones?modelo=completo");
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "No se pudieron cargar las opciones.");
      series = data.series;
      origin.value = data.fecha_max_historica;
      setOptions(store, [...new Set(series.map(item => item.tienda))]);
      if (series.some(item => item.tienda === 1)) store.value = "1";
      updateFamilies();
      controls.forEach(control => { control.disabled = false; });
      clearForecast();
    } catch (error) {
      series = [];
      setOptions(store, []);
      setOptions(family, []);
      controls.forEach(control => { control.disabled = true; });
      setStatus(error.message + " Revisá los artefactos locales y recargá la página.", true);
    }
  }
  store.addEventListener("change", () => { updateFamilies(); clearForecast(); });
  family.addEventListener("change", clearForecast);
  horizon.addEventListener("change", clearForecast);
  loadOptions();
  forecastForm.addEventListener("submit", async event => {
    event.preventDefault();
    if (busy || !forecastForm.reportValidity()) return;
    busy = true;
    forecastForm.setAttribute("aria-busy", "true");
    controls.forEach(control => { control.disabled = true; });
    results.hidden = true;
    setStatus("Verificando historia y promociones; calculando el horizonte recursivo.");
    try {
      const response = await fetch("/api/pronosticos", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          tienda: Number(store.value),
          familia: family.value,
          fecha_origen: origin.value,
          horizonte: Number(horizon.value),
        }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "No se pudo generar el pronóstico.");
      context.textContent = `Tienda ${data.tienda} · ${presentacion.nombreFamilia(data.familia)} · origen ${data.fecha_origen} · horizonte ${data.horizonte} días · ${data.modelo}, época ${data.epoca}.`;
      rows.replaceChildren(...data.pronosticos.map(item => {
        const row = document.createElement("tr");
        for (const value of [
          item.fecha,
          presentacion.formatearUnidadesEstimadas(item.ventas_estimadas),
          Number(item.cantidad_en_promocion).toLocaleString("es-AR"),
        ]) {
          const cell = document.createElement("td");
          cell.textContent = value;
          row.appendChild(cell);
        }
        return row;
      }));
      results.hidden = false;
      setStatus(data.aviso);
    } catch (error) {
      setStatus(error.message, true);
    } finally {
      busy = false;
      forecastForm.removeAttribute("aria-busy");
      controls.forEach(control => { control.disabled = false; });
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
