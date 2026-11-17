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

async function reproducirInicializacionSinEstrategias({incluirSelector = true} = {}) {
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
  if (incluirSelector) ids.push("strategy-select", "resultado-reposicion");
  const elements = new Map(ids.map(id => [
    id,
    ["store-select", "product-select", "strategy-select"].includes(id)
      ? new FakeSelect()
      : new FakeElement(),
  ]));
  const form = new FakeElement();
  form.reportValidity = () => true;
  elements.set("prediction-form", form);
  const requests = [];
  const payloads = [];
  const document = {
    getElementById: id => elements.get(id) || null,
    addEventListener() {},
  };
  const fetch = async (url, init = {}) => {
    requests.push(url);
    if (url.endsWith("/opciones")) {
      return {
        ok: true,
        json: async () => ({
          tiendas: [1, 2, 3],
          fecha_corte: "2026-11-16",
          fecha_minima: "2026-11-17",
          fecha_maxima: "2027-02-14",
        }),
      };
    }
    if (url.endsWith("/productos")) {
      return {
        ok: true,
        json: async () => ({
          productos: Array.from({length: 30}, (_, index) => ({
            producto_id: index === 0 ? "BEB-001" : `PRO-${index}`,
            nombre: index === 0 ? "Agua mineral 500 ml" : `Producto ${index}`,
            categoria: "bebidas",
          })),
        }),
      };
    }
    if (url.endsWith("/estrategias")) {
      return {
        ok: false,
        json: async () => ({error: "No se encontró la ruta."}),
      };
    }
    if (url.endsWith("/predicciones")) {
      payloads.push(JSON.parse(init.body));
      return {
        ok: true,
        json: async () => ({
          tienda_id: 1,
          producto: {
            producto_id: "BEB-001",
            nombre: "Agua mineral 500 ml",
            categoria: "bebidas",
          },
          fecha: "2026-11-25",
          unidades_estimadas: 16.586666107177734,
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
  const button = elements.get("btn-predict");
  assert.equal(store.options.length, 3);
  assert.equal(product.options.length, 30);
  assert.equal(date.value, "2026-11-17");
  assert.equal(date.min, "2026-11-17");
  assert.equal(date.max, "2027-02-14");
  assert.equal(button.disabled, false);
  if (incluirSelector) {
    assert.equal(elements.get("strategy-select").disabled, true);
    assert.match(elements.get("consulta-estado").textContent, /predicción individual está disponible/);
  }

  store.selectedIndex = 0;
  product.selectedIndex = 0;
  date.value = "2026-11-25";
  await form.listeners.submit({preventDefault() {}});

  assert.deepEqual(payloads, [{
    tienda_id: 1,
    producto_id: "BEB-001",
    fecha: "2026-11-25",
  }]);
  assert.equal(elements.get("ventas-estimadas").textContent, "17 unidades");
  assert.equal(elements.get("resultado").hidden, false);
  if (elements.has("resultado-reposicion")) {
    assert.equal(elements.get("resultado-reposicion").hidden, true);
  }
  assert.equal(requests.includes("/api/synthetic-v2/predicciones"), true);
  assert.equal(requests.includes("/api/synthetic-v2/reposiciones"), false);
  return {requests, elements};
}

test("BEB-002 selection renders predictions and both replenishment outcomes", async () => {
  const ids = [
    "store-select",
    "product-select",
    "date-input",
    "strategy-select",
    "btn-predict",
    "consulta-estado",
    "resultado",
    "rango-fechas",
    "ventas-estimadas",
    "resultado-contexto",
    "resultado-modelo",
    "stock-actual",
    "stock-minimo",
    "plazo-reposicion",
    "demanda-acumulada",
    "periodo-cobertura",
    "estrategia-reposicion",
    "cantidad-recomendada",
    "mensaje-reposicion",
  ];
  const elements = new Map(ids.map(id => [
    id,
    id === "store-select" || id === "product-select" || id === "strategy-select"
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
  let respuestaReposicion = {
    tienda_id: 2,
    producto: {
      producto_id: "BEB-002",
      nombre: "Agua mineral 1.5 l",
      categoria: "bebidas",
    },
    fecha: "2026-11-25",
    demanda_fecha_seleccionada: 18.2,
    periodo_cobertura: {
      desde: "2026-11-25",
      hasta: "2026-11-27",
      dias: 3,
    },
    inventario: {
      stock_actual: 4,
      stock_minimo: 2,
      tiempo_reposicion: 3,
    },
    demanda_acumulada: 6.31,
    estrategia: {codigo: "basica", nombre: "Reposición básica"},
    cantidad_recomendada: 3,
    requiere_reposicion: true,
    mensaje: "Se recomienda reponer 3 unidades.",
    datos_sinteticos: true,
  };
  const fetch = async (url, init = {}) => {
    requests.push(url);
    if (url.endsWith("/opciones")) {
      return {
        ok: true,
        json: async () => ({
          tiendas: [1, 2, 3],
          fecha_corte: "2026-11-16",
          fecha_minima: "2026-11-17",
          fecha_maxima: "2027-02-14",
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
    if (url.endsWith("/estrategias")) {
      return {
        ok: true,
        json: async () => ({
          estrategias: [
            {codigo: "basica", nombre: "Reposición básica"},
            {codigo: "seguridad", nombre: "Reposición con stock de seguridad"},
          ],
        }),
      };
    }
    if (url.endsWith("/reposiciones")) {
      payloads.push(JSON.parse(init.body));
      return {
        ok: true,
        json: async () => respuestaReposicion,
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
  const strategy = elements.get("strategy-select");
  assert.equal(store.options[1].value, "2");
  assert.equal(product.options[1].value, "BEB-002");
  assert.equal(strategy.options[0].value, "basica");
  store.selectedIndex = 1;
  product.selectedIndex = 1;
  strategy.selectedIndex = 0;
  date.value = "2026-11-25";

  await form.listeners.submit({preventDefault() {}});

  assert.deepEqual(payloads, [{
    tienda_id: 2,
    producto_id: "BEB-002",
    fecha: "2026-11-25",
    estrategia: "basica",
  }]);
  assert.equal(typeof payloads[0].producto_id, "string");
  assert.equal(elements.get("ventas-estimadas").textContent, "18 unidades");
  assert.equal(elements.get("stock-actual").textContent, "4");
  assert.equal(elements.get("stock-minimo").textContent, "2");
  assert.equal(elements.get("plazo-reposicion").textContent, "3 días");
  assert.equal(elements.get("demanda-acumulada").textContent, "6,31 unidades");
  assert.equal(elements.get("cantidad-recomendada").textContent, "3 unidades");
  assert.equal(
    elements.get("mensaje-reposicion").textContent,
    "Se recomienda reponer 3 unidades.",
  );
  assert.doesNotMatch(elements.get("resultado-contexto").textContent, /\[object Object\]/);
  assert.equal(elements.get("resultado").hidden, false);

  respuestaReposicion = {
    ...respuestaReposicion,
    cantidad_recomendada: 0,
    requiere_reposicion: false,
    mensaje: "No es necesario reponer.",
  };
  await form.listeners.submit({preventDefault() {}});
  assert.equal(elements.get("cantidad-recomendada").textContent, "0 unidades");
  assert.equal(
    elements.get("mensaje-reposicion").textContent,
    "No es necesario reponer.",
  );
  respuestaReposicion = {
    ...respuestaReposicion,
    inventario: {...respuestaReposicion.inventario, tiempo_reposicion: 1},
    periodo_cobertura: {
      desde: "2026-11-25",
      hasta: "2026-11-25",
      dias: 1,
    },
  };
  await form.listeners.submit({preventDefault() {}});
  assert.equal(elements.get("plazo-reposicion").textContent, "1 día");
  assert.equal(payloads.length, 3);
  assert.deepEqual(requests, [
    "/api/synthetic-v2/opciones",
    "/api/synthetic-v2/productos",
    "/api/synthetic-v2/reposicion/estrategias",
    "/api/synthetic-v2/reposiciones",
    "/api/synthetic-v2/reposiciones",
    "/api/synthetic-v2/reposiciones",
  ]);
});

test("una falla en estrategias no bloquea la predicción ni la fecha", async () => {
  const {requests} = await reproducirInicializacionSinEstrategias();

  assert.deepEqual(requests, [
    "/api/synthetic-v2/opciones",
    "/api/synthetic-v2/productos",
    "/api/synthetic-v2/reposicion/estrategias",
    "/api/synthetic-v2/predicciones",
  ]);
});

test("tolera un formulario sin selector de estrategia", async () => {
  const {requests} = await reproducirInicializacionSinEstrategias({
    incluirSelector: false,
  });

  assert.deepEqual(requests, [
    "/api/synthetic-v2/opciones",
    "/api/synthetic-v2/productos",
    "/api/synthetic-v2/predicciones",
  ]);
});
