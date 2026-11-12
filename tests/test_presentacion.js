"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const { familias, nombreFamilia, formatearUnidadesEstimadas } =
  require("../web/static/presentacion.js");

const idsFavorita = [
  "AUTOMOTIVE", "BABY CARE", "BEAUTY", "BEVERAGES", "BOOKS", "BREAD/BAKERY",
  "CELEBRATION", "CLEANING", "DAIRY", "DELI", "EGGS", "FROZEN FOODS",
  "GROCERY I", "GROCERY II", "HARDWARE", "HOME AND KITCHEN I",
  "HOME AND KITCHEN II", "HOME APPLIANCES", "HOME CARE", "LADIESWEAR",
  "LAWN AND GARDEN", "LINGERIE", "LIQUOR,WINE,BEER", "MAGAZINES", "MEATS",
  "PERSONAL CARE", "PET SUPPLIES", "PLAYERS AND ELECTRONICS", "POULTRY",
  "PREPARED FOODS", "PRODUCE", "SCHOOL AND OFFICE SUPPLIES", "SEAFOOD",
];

test("traduce las 33 categorías sin reemplazar sus identificadores", () => {
  assert.deepEqual(Object.keys(familias).sort(), [...idsFavorita].sort());
  assert.equal(nombreFamilia("BEAUTY"), "Belleza");
  assert.equal(nombreFamilia("PLAYERS AND ELECTRONICS"), "Reproductores y electrónica");
  assert.equal(nombreFamilia("UNKNOWN"), "UNKNOWN");
});

test("redondea a unidades enteras sin redondeo bancario y limita la vista a cero", () => {
  for (const [valor, esperado] of [
    [30.49, "30 unidades"],
    [30.5, "31 unidades"],
    [30.96, "31 unidades"],
    [0, "0 unidades"],
    [-0.5, "0 unidades"],
    [-12.8, "0 unidades"],
    [1, "1 unidad"],
  ]) {
    assert.equal(formatearUnidadesEstimadas(valor), esperado);
  }
});
