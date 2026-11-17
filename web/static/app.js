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
    if (event.key === "Escape") {
      nav.classList.remove("abierto");
      menu.setAttribute("aria-expanded", "false");
    }
  });
}

const form = document.getElementById("prediction-form");
if (form) {
  const tienda = document.getElementById("store-select");
  const producto = document.getElementById("product-select");
  const fecha = document.getElementById("date-input");
  const boton = document.getElementById("btn-predict");
  const estado = document.getElementById("consulta-estado");
  const resultado = document.getElementById("resultado");
  const rango = document.getElementById("rango-fechas");
  const controles = [tienda, producto, fecha, boton];
  let ocupado = false;

  function mostrarEstado(texto, error = false) {
    estado.textContent = texto;
    estado.className = error ? "notice error" : "notice";
    estado.setAttribute("role", error ? "alert" : "status");
  }

  function asignarOpciones(select, valores, etiqueta, identificador = valor => valor) {
    select.replaceChildren(...valores.map(valor =>
      new Option(etiqueta(valor), String(identificador(valor)))));
  }

  function limpiarResultado() {
    resultado.hidden = true;
    mostrarEstado("Seleccioná tienda, producto y fecha para generar una predicción.");
  }

  async function cargarOpciones() {
    controles.forEach(control => { control.disabled = true; });
    mostrarEstado("Cargando tiendas, productos y fechas disponibles.");
    try {
      const [respuestaOpciones, respuestaProductos] = await Promise.all([
        fetch("/api/synthetic-v2/opciones"),
        fetch("/api/synthetic-v2/productos"),
      ]);
      const [opciones, catalogo] = await Promise.all([
        respuestaOpciones.json(),
        respuestaProductos.json(),
      ]);
      if (!respuestaOpciones.ok) {
        throw new Error(opciones.error || "No se pudieron cargar las opciones del modelo.");
      }
      if (!respuestaProductos.ok) {
        throw new Error(catalogo.error || "No se pudo cargar el catálogo de productos.");
      }

      asignarOpciones(tienda, opciones.tiendas, id => `Tienda ${id}`);
      asignarOpciones(producto, catalogo.productos, item =>
        `${item.nombre} (${item.producto_id}) · ${item.categoria}`,
      item => item.producto_id);
      fecha.min = opciones.fecha_minima;
      fecha.value = opciones.fecha_minima;
      rango.textContent = `Fecha mínima: ${opciones.fecha_minima}. Último registro sintético: ${opciones.fecha_corte}.`;
      controles.forEach(control => { control.disabled = false; });
      limpiarResultado();
    } catch (error) {
      asignarOpciones(tienda, [], String);
      asignarOpciones(producto, [], item => String(item));
      fecha.value = "";
      rango.textContent = "";
      mostrarEstado(error.message + " Revisá el modelo y el catálogo locales.", true);
    }
  }

  for (const control of [tienda, producto, fecha]) {
    control.addEventListener("change", limpiarResultado);
  }
  cargarOpciones();

  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (ocupado || !form.reportValidity()) return;
    ocupado = true;
    form.setAttribute("aria-busy", "true");
    controles.forEach(control => { control.disabled = true; });
    resultado.hidden = true;
    boton.textContent = "Calculando…";
    mostrarEstado("Ejecutando el modelo sintético local.");
    try {
      const respuesta = await fetch("/api/synthetic-v2/predicciones", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({
          tienda_id: Number(tienda.value),
          producto_id: producto.value,
          fecha: fecha.value,
        }),
      });
      const datos = await respuesta.json();
      if (!respuesta.ok) throw new Error(datos.error || "No se pudo generar la predicción.");
      document.getElementById("ventas-estimadas").textContent =
        presentacion.formatearUnidadesEstimadas(datos.unidades_estimadas);
      document.getElementById("resultado-contexto").textContent =
        `Tienda ${datos.tienda_id} · ${datos.producto.nombre} (${datos.producto.producto_id}) · ${datos.fecha}`;
      document.getElementById("resultado-modelo").textContent =
        "MLP PyTorch · datos sintéticos · estimación no condicionada por precio ni promoción.";
      resultado.hidden = false;
      mostrarEstado("Predicción completada con el modelo local; no se ejecutó entrenamiento.");
    } catch (error) {
      mostrarEstado(error.message, true);
    } finally {
      ocupado = false;
      form.removeAttribute("aria-busy");
      controles.forEach(control => { control.disabled = false; });
      boton.textContent = "Generar predicción";
    }
  });
}
