"use strict";

(function (global) {
  function formatearUnidadesEstimadas(valor) {
    const numero = Number(valor);
    if (!Number.isFinite(numero)) {
      throw new TypeError("La estimación debe ser un número finito.");
    }
    const unidades = Math.max(0, Math.floor(numero + 0.5));
    return `${unidades} ${unidades === 1 ? "unidad" : "unidades"}`;
  }

  const presentacion = Object.freeze({formatearUnidadesEstimadas});
  global.StockPredictPresentacion = presentacion;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = presentacion;
  }
})(globalThis);
