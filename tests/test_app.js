"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const presentacion = require("../web/static/presentacion.js");

class FakeSelect {
  constructor() {
    this.options = [];
    this._selectedIndex = -1;
    this.listeners = {};
    this.disabled = true;
  }

  replaceChildren(...options) {
    this.options = options;
    this._selectedIndex = options.length ? 0 : -1;
  }

  get selectedIndex() {
    return this._selectedIndex;
  }

  set selectedIndex(index) {
    this._selectedIndex = index;
  }

  get value() {
    return this.options[this._selectedIndex]?.value || "";
  }

  set value(value) {
    this._selectedIndex = this.options.findIndex(
      option => option.value === String(value),
    );
  }

  addEventListener(name, callback) {
    this.listeners[name] = callback;
  }
}

class FakeElement {
  constructor() {
    this.listeners = {};
    this.attributes = {};
    this.disabled = true;
    this.hidden = true;
    this.textContent = "";
    this.className = "";
  }

  addEventListener(name, callback) {
    this.listeners[name] = callback;
  }

  setAttribute(name, value) {
    this.attributes[name] = value;
  }

  removeAttribute(name) {
    delete this.attributes[name];
  }

  reportValidity() {
    return true;
  }
}

test("BEB-002 selection posts its string ID and renders the model result", async () => {
  const ids = [
    "store-select",
    "product-select",
    "date-input",
    "btn-predict",
    "consulta-estado",
    "resultado",
    "rango-fechas",
    "ventas-estimadas",
    "resultado-contexto",
    "resultado-modelo",
  ];
  const elements = new Map(ids.map(id => [
    id,
    id === "store-select" || id === "product-select"
      ? new FakeSelect()
      : new FakeElement(),
  ]));
  const form = new FakeElement();
  form.reportValidity = () => true;
  elements.set("prediction-form", form);
  const requests = [];
  const document = {
    getElementById: id => elements.get(id) || null,
    addEventListener() {},
  };
  const payloads = [];
  const fetch = async (url, init = {}) => {
    requests.push(url);
    if (url.endsWith("/opciones")) {
      return {
        ok: true,
        json: async () => ({
          tiendas: [1, 2, 3],
          fecha_corte: "2026-11-16",
          fecha_minima: "2026-11-17",
        }),
      };
    }
    if (url.endsWith("/productos")) {
      return {
        ok: true,
        json: async () => ({
          productos: [
            {producto_id: "BEB-001", nombre: "Agua mineral 500 ml", categoria: "bebidas"},
            {producto_id: "BEB-002", nombre: "Agua mineral 1.5 l", categoria: "bebidas"},
          ],
        }),
      };
    }
    if (url.endsWith("/predicciones")) {
      payloads.push(JSON.parse(init.body));
      return {
        ok: true,
        json: async () => ({
          tienda_id: 2,
          producto: {
            producto_id: "BEB-002",
            nombre: "Agua mineral 1.5 l",
            categoria: "bebidas",
          },
          fecha: "2026-11-25",
          unidades_estimadas: 18.2,
          datos_sinteticos: true,
        }),
      };
    }
    throw new Error(`URL inesperada: ${url}`);
  };
  class FakeOption {
    constructor(label, value) {
      this.label = String(label);
      this.value = String(value);
    }
  }

  vm.runInNewContext(
    fs.readFileSync(path.join(__dirname, "..", "web", "static", "app.js"), "utf8"),
    {
      window: {StockPredictPresentacion: presentacion},
      document,
      fetch,
      Option: FakeOption,
      Number,
      String,
      JSON,
      Error,
    },
  );
  await new Promise(resolve => setImmediate(resolve));

  const store = elements.get("store-select");
  const product = elements.get("product-select");
  const date = elements.get("date-input");
  assert.equal(store.options[1].value, "2");
  assert.equal(product.options[1].value, "BEB-002");
  store.selectedIndex = 1;
  product.selectedIndex = 1;
  date.value = "2026-11-25";

  await form.listeners.submit({preventDefault() {}});

  assert.deepEqual(payloads, [{
    tienda_id: 2,
    producto_id: "BEB-002",
    fecha: "2026-11-25",
  }]);
  assert.equal(typeof payloads[0].producto_id, "string");
  assert.equal(elements.get("ventas-estimadas").textContent, "18 unidades");
  assert.doesNotMatch(elements.get("resultado-contexto").textContent, /\[object Object\]/);
  assert.equal(elements.get("resultado").hidden, false);
  assert.deepEqual(requests, [
    "/api/synthetic-v2/opciones",
    "/api/synthetic-v2/productos",
    "/api/synthetic-v2/predicciones",
  ]);
});
