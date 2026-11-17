"use strict";
const presentacion = window.StockPredictPresentacion;

const form = document.getElementById("prediction-form");
if (form) {
  const tienda = document.getElementById("store-select");
  const producto = document.getElementById("product-select");
  const fecha = document.getElementById("date-input");
  const estrategia = document.getElementById("strategy-select");
  const boton = document.getElementById("btn-predict");
  const estado = document.getElementById("consulta-estado");
  const resultado = document.getElementById("resultado");
  const resultadoReposicion = document.getElementById("resultado-reposicion");
  const rango = document.getElementById("rango-fechas");
  const controles = [tienda, producto, fecha, boton];
  let estrategiaDisponible = false;
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
    if (resultado) resultado.hidden = true;
    if (resultadoReposicion) resultadoReposicion.hidden = true;
    mostrarEstado("Seleccioná tienda, producto y fecha para generar una predicción.");
  }

  async function cargarOpciones() {
    controles.forEach(control => { control.disabled = true; });
    if (estrategia) estrategia.disabled = true;
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
      fecha.max = opciones.fecha_maxima;
      fecha.value = opciones.fecha_minima;
      rango.textContent = `Fechas disponibles: ${opciones.fecha_minima} a ${opciones.fecha_maxima}. Último registro sintético: ${opciones.fecha_corte}.`;
      controles.forEach(control => { control.disabled = false; });

      if (estrategia) {
        try {
          const respuestaEstrategias = await fetch(
            "/api/synthetic-v2/reposicion/estrategias",
          );
          const datosEstrategias = await respuestaEstrategias.json();
          if (!respuestaEstrategias.ok) {
            throw new Error(
              datosEstrategias.error || "No se pudieron cargar las estrategias.",
            );
          }
          if (!Array.isArray(datosEstrategias.estrategias) || !datosEstrategias.estrategias.length) {
            throw new Error("La respuesta no contiene estrategias de reposición.");
          }
          asignarOpciones(
            estrategia,
            datosEstrategias.estrategias,
            item => item.nombre,
            item => item.codigo,
          );
          estrategiaDisponible = true;
          estrategia.disabled = false;
        } catch (_error) {
          asignarOpciones(estrategia, [], item => String(item));
          estrategiaDisponible = false;
        }
      }

      limpiarResultado();
      boton.textContent = estrategiaDisponible
        ? "Predecir y calcular reposición"
        : "Generar predicción";
      if (!estrategiaDisponible && estrategia) {
        mostrarEstado(
          "La predicción individual está disponible; las estrategias de reposición no se pudieron cargar.",
          true,
        );
      }
    } catch (error) {
      asignarOpciones(tienda, [], String);
      asignarOpciones(producto, [], item => String(item));
      if (estrategia) {
        asignarOpciones(estrategia, [], item => String(item));
        estrategiaDisponible = false;
      }
      fecha.value = "";
      rango.textContent = "";
      boton.textContent = "Generar predicción";
      mostrarEstado(error.message + " Revisá el modelo y el catálogo locales.", true);
    }
  }

  for (const control of [tienda, producto, fecha, estrategia]) {
    if (control) control.addEventListener("change", limpiarResultado);
  }
  cargarOpciones();

  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (ocupado || !form.reportValidity()) return;
    ocupado = true;
    form.setAttribute("aria-busy", "true");
    controles.forEach(control => { control.disabled = true; });
    if (estrategia) estrategia.disabled = true;
    if (resultado) resultado.hidden = true;
    if (resultadoReposicion) resultadoReposicion.hidden = true;
    boton.textContent = "Calculando…";
    mostrarEstado("Ejecutando el modelo sintético local.");
    try {
      const usarReposicion = Boolean(
        estrategia && estrategiaDisponible && estrategia.value,
      );
      const solicitud = {
        tienda_id: Number(tienda.value),
        producto_id: producto.value,
        fecha: fecha.value,
      };
      if (usarReposicion) solicitud.estrategia = estrategia.value;
      const ruta = usarReposicion
        ? "/api/synthetic-v2/reposiciones"
        : "/api/synthetic-v2/predicciones";
      const respuesta = await fetch(ruta, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(solicitud),
      });
      const datos = await respuesta.json();
      if (!respuesta.ok) throw new Error(datos.error || "No se pudo generar la predicción.");
      document.getElementById("ventas-estimadas").textContent =
        presentacion.formatearUnidadesEstimadas(
          usarReposicion ? datos.demanda_fecha_seleccionada : datos.unidades_estimadas,
        );
      document.getElementById("resultado-contexto").textContent =
        `Tienda ${datos.tienda_id} · ${datos.producto.nombre} (${datos.producto.producto_id}) · ${datos.fecha}`;
      document.getElementById("resultado-modelo").textContent =
        "MLP PyTorch · datos sintéticos · estimación no condicionada por precio ni promoción.";
      if (usarReposicion) {
        const inventario = datos.inventario;
        document.getElementById("stock-actual").textContent =
          Number(inventario.stock_actual).toLocaleString("es-AR");
        document.getElementById("stock-minimo").textContent =
          Number(inventario.stock_minimo).toLocaleString("es-AR");
        document.getElementById("plazo-reposicion").textContent =
          `${inventario.tiempo_reposicion} ${
            inventario.tiempo_reposicion === 1 ? "día" : "días"
          }`;
        document.getElementById("demanda-acumulada").textContent =
          `${Number(datos.demanda_acumulada).toLocaleString("es-AR", {
            maximumFractionDigits: 2,
          })} unidades`;
        document.getElementById("periodo-cobertura").textContent =
          `Cobertura: ${datos.periodo_cobertura.desde} a ${datos.periodo_cobertura.hasta} (incluye la fecha inicial).`;
        document.getElementById("estrategia-reposicion").textContent =
          `Estrategia aplicada: ${datos.estrategia.nombre}.`;
        document.getElementById("cantidad-recomendada").textContent =
          presentacion.formatearUnidadesEstimadas(datos.cantidad_recomendada);
        document.getElementById("mensaje-reposicion").textContent = datos.mensaje;
        if (resultadoReposicion) resultadoReposicion.hidden = false;
      }
      resultado.hidden = false;
      mostrarEstado(
        usarReposicion
          ? "Predicción y recomendación calculadas; el inventario no fue modificado."
          : "Predicción individual completada con el modelo sintético.",
      );
    } catch (error) {
      mostrarEstado(error.message, true);
    } finally {
      ocupado = false;
      form.removeAttribute("aria-busy");
      controles.forEach(control => { control.disabled = false; });
      if (estrategia) estrategia.disabled = !estrategiaDisponible;
      boton.textContent = estrategiaDisponible
        ? "Predecir y calcular reposición"
        : "Generar predicción";
    }
  });
}
