"use strict";

const assert = require("node:assert/strict");
const test = require("node:test");
const presentacion = require("../web/static/presentacion.js");

test("expone solo formato de unidades, sin catalogos visuales Favorita", () => {
  assert.deepEqual(Object.keys(presentacion), ["formatearUnidadesEstimadas"]);
});

test("redondea a unidades enteras y limita la vista a cero", () => {
  for (const [valor, esperado] of [
    [30.49, "30 unidades"],
    [30.5, "31 unidades"],
    [30.96, "31 unidades"],
    [0, "0 unidades"],
    [-0.5, "0 unidades"],
    [-12.8, "0 unidades"],
    [1, "1 unidad"],
  ]) {
    assert.equal(presentacion.formatearUnidadesEstimadas(valor), esperado);
  }
});
