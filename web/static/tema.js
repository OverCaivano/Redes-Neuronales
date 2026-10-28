"use strict";

const claveTema = "stockpredict-theme";
const temaGuardado = localStorage.getItem(claveTema);
const temaInicial = temaGuardado === "light" || temaGuardado === "dark"
  ? temaGuardado
  : (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");

document.documentElement.dataset.theme = temaInicial;

function actualizarControlTema(tema) {
  const boton = document.getElementById("theme-toggle");
  if (!boton) return;
  const oscuro = tema === "dark";
  const etiqueta = oscuro ? "Activar tema claro" : "Activar tema oscuro";
  boton.setAttribute("aria-pressed", String(oscuro));
  boton.setAttribute("aria-label", etiqueta);
  boton.title = etiqueta;
}

document.addEventListener("DOMContentLoaded", () => actualizarControlTema(temaInicial), {once: true});
document.addEventListener("click", event => {
  if (!(event.target instanceof Element)) return;
  const boton = event.target.closest("#theme-toggle");
  if (!boton) return;

  const tema = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = tema;
  localStorage.setItem(claveTema, tema);
  actualizarControlTema(tema);
});
