let catalogo = [];
let seleccion = new Set();
let archivoActual = null;

const estadoEl = document.getElementById("estado-modelos");
const lightbox = document.getElementById("lightbox");

function entero(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return String(n);
}

function pct(n) {
  if (n == null || Number.isNaN(Number(n))) return "—";
  return `${(Number(n) * 100).toFixed(1)}%`;
}

function params(n) {
  if (n == null) return "—";
  if (n >= 1e5) return `${(n / 1e6).toFixed(2)} M`;
  return entero(n);
}

function mb(n) {
  if (n == null) return "—";
  return `${Number(n).toFixed(2)} MB`;
}

function minutos(n) {
  if (n == null) return "—";
  return `${n} min`;
}

function f1(n) {
  if (n == null) return "—";
  return Number(n).toFixed(3);
}

function escapar(texto) {
  return String(texto)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function pintarCifras() {
  const caja = document.getElementById("cifras");
  caja.innerHTML = catalogo
    .map(
      (m) => `<article class="cifra tono-${m.tono}">
        <span class="tag">${escapar(m.nombre)}</span>
        <strong>${pct(m.accuracy)}</strong>
        <p>${escapar(m.dataset)} · ${entero(m.clases)} especies</p>
      </article>`
    )
    .join("");
}

function pintarTabla() {
  const head = document.getElementById("tabla-head");
  head.innerHTML = `<th></th>${catalogo.map((m) => `<th>${escapar(m.nombre)}</th>`).join("")}`;
  const filas = [
    ["Arquitectura", (m) => escapar(m.arquitectura)],
    ["Especies", (m) => entero(m.clases)],
    ["Train / val / test", (m) =>
      m.train != null && m.val != null && m.test != null
        ? `${entero(m.train)} / ${entero(m.val)} / ${entero(m.test)}`
        : "—"],
    ["Accuracy test", (m) => {
      if (m.accuracy == null) return "—";
      const errores = m.errores != null ? ` (${entero(m.errores)} errores)` : "";
      return `${pct(m.accuracy)}${errores}`;
    }],
    ["Top-3", (m) => pct(m.top3)],
    ["F1 ponderado", (m) => f1(m.f1)],
    ["Parámetros", (m) => params(m.parametros)],
    ["Tamaño", (m) => mb(m.tamano_mb)],
    ["Entrenamiento", (m) => minutos(m.minutos)],
    ["Fotos", (m) => escapar(m.fotos)],
    ["Límite", (m) => escapar(m.limite)],
  ];
  document.getElementById("tabla-body").innerHTML = filas
    .map(
      ([titulo, celda]) =>
        `<tr><th>${titulo}</th>${catalogo.map((m) => `<td>${celda(m)}</td>`).join("")}</tr>`
    )
    .join("");
}

function pintarPaneles() {
  const pills = document.getElementById("pills-graficas");
  const paneles = document.getElementById("paneles");
  pills.innerHTML = "";
  paneles.innerHTML = "";
  catalogo.forEach((m, i) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `pill${i === 0 ? " activo" : ""}`;
    btn.dataset.tab = m.id;
    btn.textContent = m.nombre;
    pills.appendChild(btn);

    const f1html = (m.f1_por_clase || [])
      .map((item) => {
        const v = Math.round(item.f1 * 100);
        return `<li><span>${escapar(item.comun)}</span><b>${f1(item.f1)}</b><i style="--v:${v}%"></i></li>`;
      })
      .join("");
    const panel = document.createElement("div");
    panel.className = `panel${i === 0 ? " activo" : ""}`;
    panel.id = `panel-${m.id}`;
    panel.innerHTML = `<div class="graficas">
        <figure>
          <img src="/static/${m.curvas}" alt="Curvas de ${escapar(m.nombre)}">
          <figcaption>${escapar(m.curvas_nota)}</figcaption>
        </figure>
        <figure>
          <img src="/static/${m.matriz}" alt="Matriz de confusión de ${escapar(m.nombre)}">
          <figcaption>${escapar(m.matriz_nota)}</figcaption>
        </figure>
      </div>
      ${f1html ? `<div class="f1s"><h3>F1 por especie (test)</h3><ul>${f1html}</ul></div>` : ""}`;
    paneles.appendChild(panel);
  });
}

function pintarChecks() {
  const cont = document.getElementById("pills-modelo");
  cont.innerHTML = "";
  const todos = document.createElement("button");
  todos.type = "button";
  todos.className = "pill";
  todos.dataset.todos = "1";
  todos.textContent = "Todos";
  todos.addEventListener("click", () => {
    catalogo.filter((m) => m.archivo).forEach((m) => seleccion.add(m.id));
    syncChecks();
    if (archivoActual) identificar(archivoActual);
  });
  cont.appendChild(todos);

  catalogo.forEach((m) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "pill";
    btn.dataset.id = m.id;
    btn.textContent = m.nombre;
    btn.disabled = !m.archivo;
    btn.addEventListener("click", () => {
      if (seleccion.has(m.id)) {
        if (seleccion.size === 1) return;
        seleccion.delete(m.id);
      } else {
        seleccion.add(m.id);
      }
      syncChecks();
      if (archivoActual) identificar(archivoActual);
    });
    cont.appendChild(btn);
  });
  syncChecks();
}

function syncChecks() {
  const disponibles = catalogo.filter((m) => m.archivo).map((m) => m.id);
  document.querySelectorAll("#pills-modelo .pill").forEach((btn) => {
    if (btn.dataset.todos) {
      btn.classList.toggle("activo", disponibles.every((id) => seleccion.has(id)));
      return;
    }
    btn.classList.toggle("activo", seleccion.has(btn.dataset.id));
  });
}

function pintarGaleria(grupos) {
  const caja = document.getElementById("galerias");
  caja.innerHTML = "";
  (grupos || []).forEach((grupo) => {
    const bloque = document.createElement("div");
    bloque.className = "grupo-esp";
    const titulo = document.createElement("h3");
    titulo.textContent = grupo.titulo;
    const grid = document.createElement("div");
    grid.className = "galeria";
    grupo.especies.forEach((esp) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "tarjeta";
      btn.innerHTML = `<img src="${escapar(esp.img)}" alt="${escapar(esp.comun)}"><div><strong>${escapar(esp.comun)}</strong><small>${escapar(esp.cientifico)}</small></div>`;
      btn.addEventListener("click", async () => {
        const blob = await fetch(esp.img).then((r) => r.blob());
        const file = new File([blob], `${esp.id}.jpg`, { type: blob.type || "image/jpeg" });
        document.getElementById("identificar").scrollIntoView({ behavior: "smooth" });
        identificar(file);
      });
      grid.appendChild(btn);
    });
    bloque.appendChild(titulo);
    bloque.appendChild(grid);
    caja.appendChild(bloque);
  });
}

document.getElementById("pills-graficas").addEventListener("click", (ev) => {
  const btn = ev.target.closest(".pill");
  if (!btn || !btn.dataset.tab) return;
  document.querySelectorAll("#pills-graficas .pill").forEach((p) => p.classList.remove("activo"));
  btn.classList.add("activo");
  document.querySelectorAll("#paneles .panel").forEach((p) => p.classList.remove("activo"));
  document.getElementById(`panel-${btn.dataset.tab}`).classList.add("activo");
});

document.getElementById("paneles").addEventListener("click", (ev) => {
  const img = ev.target.closest("img");
  if (!img) return;
  lightbox.querySelector("img").src = img.src;
  lightbox.querySelector("img").alt = img.alt;
  lightbox.classList.remove("oculto");
});
lightbox.addEventListener("click", () => lightbox.classList.add("oculto"));

function carta(bloque) {
  const tono = bloque.tono || "leaf";
  if (!bloque || !bloque.disponible) {
    return `<article class="veredicto tono-${tono}"><p class="etiqueta">${escapar(bloque.nombre || "Modelo")}</p><p>${escapar(bloque.error || "Este modelo no está disponible.")}</p></article>`;
  }
  const top = bloque.top || [];
  const g = top[0];
  if (!g) {
    return `<article class="veredicto tono-${tono}"><p class="etiqueta">${escapar(bloque.nombre || "Modelo")}</p><p>El modelo no devolvió una especie.</p></article>`;
  }
  const filas = top
    .map(
      (t) => `<div class="top-item"><span>${escapar(t.comun)} <em>(${escapar(t.cientifico)})</em></span><b>${pct(t.prob)}</b></div>
        <div class="barra-p"><span style="width:${(t.prob * 100).toFixed(1)}%"></span></div>`
    )
    .join("");
  return `<article class="veredicto tono-${tono}">
    <p class="etiqueta">${escapar(bloque.nombre)}</p>
    <h3 class="especie">${escapar(g.comun)}</h3>
    <p class="cientifico">${escapar(g.cientifico)}</p>
    <p class="confianza">Confianza ${pct(g.prob)}</p>
    ${filas}
  </article>`;
}

function acuerdoHtml(acuerdo) {
  if (!acuerdo) return "";
  const partes = [];
  if (acuerdo.herbify === true) {
    partes.push('<p class="badge">Los modelos de Herbify coinciden en la especie</p>');
  }
  if (acuerdo.nombre_cientifico === true) {
    partes.push('<p class="badge">El regional y un modelo de Herbify señalan el mismo nombre científico</p>');
  }
  return partes.join("");
}

const dropUna = document.getElementById("drop-una");
const fileUna = document.getElementById("file-una");
const previewUna = document.getElementById("preview-una");
const caja = document.getElementById("caja-identifica");
const avisoUna = document.getElementById("aviso-una");
const carga = document.getElementById("carga");
const veredicto = document.getElementById("veredicto");

function mostrarFoto(file) {
  archivoActual = file;
  previewUna.src = URL.createObjectURL(file);
  caja.classList.remove("oculto");
  dropUna.classList.add("lleno");
  dropUna.querySelector("span").textContent = file.name;
}

["dragenter", "dragover"].forEach((ev) => {
  dropUna.addEventListener(ev, (e) => {
    e.preventDefault();
    dropUna.classList.add("lleno");
  });
});
dropUna.addEventListener("dragleave", () => {
  if (!archivoActual) dropUna.classList.remove("lleno");
});
dropUna.addEventListener("drop", (e) => {
  e.preventDefault();
  const f = e.dataTransfer.files[0];
  if (f) identificar(f);
});
fileUna.addEventListener("change", () => {
  if (fileUna.files[0]) identificar(fileUna.files[0]);
});

async function identificar(file) {
  mostrarFoto(file);
  avisoUna.classList.add("oculto");
  veredicto.innerHTML = "";
  carga.classList.remove("oculto");
  const fd = new FormData();
  fd.append("imagen", file);
  fd.append("modelos", [...seleccion].join(","));
  try {
    const r = await fetch("/api/predecir", { method: "POST", body: fd });
    const data = await r.json();
    if (!data.ok) throw new Error(data.error || "No se pudo identificar");
    const orden = catalogo.map((m) => m.id);
    const bloques = orden
      .filter((id) => data.modelos[id])
      .map((id) => carta(data.modelos[id]));
    veredicto.innerHTML = bloques.join("") + acuerdoHtml(data.acuerdo);
  } catch (err) {
    avisoUna.textContent = err.message;
    avisoUna.classList.remove("oculto");
  } finally {
    carga.classList.add("oculto");
  }
}

async function arrancar() {
  try {
    const [mods, esps, est] = await Promise.all([
      fetch("/api/modelos").then((r) => r.json()),
      fetch("/api/especies").then((r) => r.json()),
      fetch("/api/estado").then((r) => r.json()),
    ]);
    catalogo = mods.modelos || [];
    seleccion = new Set(catalogo.filter((m) => m.archivo).map((m) => m.id));
    pintarCifras();
    pintarTabla();
    pintarPaneles();
    pintarChecks();
    pintarGaleria(esps.grupos);
    if (est.listo) {
      estadoEl.textContent = "Catálogo listo. Sube una foto y los modelos elegidos dicen a qué especie pertenece.";
    } else {
      estadoEl.textContent = est.error || "No hay modelos en disco.";
    }
  } catch {
    estadoEl.textContent = "El servidor no responde.";
  }
}

arrancar();
