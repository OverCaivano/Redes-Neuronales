"use strict";

(function (global) {
const nombresFamilias = Object.freeze({
  "AUTOMOTIVE": "Automotriz",
  "BABY CARE": "Bebés y cuidado infantil",
  "BEAUTY": "Belleza",
  "BEVERAGES": "Bebidas",
  "BOOKS": "Libros",
  "BREAD/BAKERY": "Panadería",
  "CELEBRATION": "Celebraciones",
  "CLEANING": "Limpieza",
  "DAIRY": "Lácteos",
  "DELI": "Delicatessen",
  "EGGS": "Huevos",
  "FROZEN FOODS": "Alimentos congelados",
  "GROCERY I": "Abarrotes I",
  "GROCERY II": "Abarrotes II",
  "HARDWARE": "Ferretería",
  "HOME AND KITCHEN I": "Hogar y cocina I",
  "HOME AND KITCHEN II": "Hogar y cocina II",
  "HOME APPLIANCES": "Electrodomésticos",
  "HOME CARE": "Cuidado del hogar",
  "LADIESWEAR": "Ropa de mujer",
  "LAWN AND GARDEN": "Jardinería",
  "LINGERIE": "Lencería",
  "LIQUOR,WINE,BEER": "Licores, vinos y cervezas",
  "MAGAZINES": "Revistas",
  "MEATS": "Carnes",
  "PERSONAL CARE": "Cuidado personal",
  "PET SUPPLIES": "Artículos para mascotas",
  "PLAYERS AND ELECTRONICS": "Reproductores y electrónica",
  "POULTRY": "Aves de corral",
  "PREPARED FOODS": "Comidas preparadas",
  "PRODUCE": "Frutas y verduras",
  "SCHOOL AND OFFICE SUPPLIES": "Útiles escolares y de oficina",
  "SEAFOOD": "Pescados y mariscos",
});

function nombreFamilia(id) {
  return Object.prototype.hasOwnProperty.call(nombresFamilias, id)
    ? nombresFamilias[id]
    : String(id);
}

function formatearUnidadesEstimadas(valor) {
  const numero = Number(valor);
  if (!Number.isFinite(numero)) {
    throw new TypeError("La estimación debe ser un número finito.");
  }
  const unidades = Math.max(0, Math.floor(numero + 0.5));
  return `${unidades} ${unidades === 1 ? "unidad" : "unidades"}`;
}

const presentacion = Object.freeze({
  familias: nombresFamilias,
  nombreFamilia,
  formatearUnidadesEstimadas,
});

global.StockPredictPresentacion = presentacion;
if (typeof module !== "undefined" && module.exports) {
  module.exports = presentacion;
}
})(globalThis);
