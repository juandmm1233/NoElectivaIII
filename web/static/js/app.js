const ESPECIES = [
  { id: "Baccharis_latifolia__chilca", comun: "chilca", cientifico: "Baccharis latifolia", img: "/static/muestras/reg_Baccharis_latifolia__chilca.jpg" },
  { id: "Cecropia_peltata__yarumo", comun: "yarumo", cientifico: "Cecropia peltata", img: "/static/muestras/reg_Cecropia_peltata__yarumo.jpg" },
  { id: "Equisetum_bogotense__cola_de_caballo", comun: "cola de caballo", cientifico: "Equisetum bogotense", img: "/static/muestras/reg_Equisetum_bogotense__cola_de_caballo.jpg" },
  { id: "Eryngium_foetidum__cilantro_cimarron", comun: "cilantro cimarrón", cientifico: "Eryngium foetidum", img: "/static/muestras/reg_Eryngium_foetidum__cilantro_cimarron.jpg" },
  { id: "Gliricidia_sepium__matarraton", comun: "matarratón", cientifico: "Gliricidia sepium", img: "/static/muestras/reg_Gliricidia_sepium__matarraton.jpg" },
  { id: "Petiveria_alliacea__anamu", comun: "anamú", cientifico: "Petiveria alliacea", img: "/static/muestras/reg_Petiveria_alliacea__anamu.jpg" },
  { id: "Piper_aduncum__cordoncillo", comun: "cordoncillo", cientifico: "Piper aduncum", img: "/static/muestras/reg_Piper_aduncum__cordoncillo.jpg" },
  { id: "Plantago_major__llanten", comun: "llantén", cientifico: "Plantago major", img: "/static/muestras/reg_Plantago_major__llanten.jpg" },
  { id: "Psidium_guajava__guayaba", comun: "guayaba", cientifico: "Psidium guajava", img: "/static/muestras/reg_Psidium_guajava__guayaba.jpg" },
  { id: "Ruta_graveolens__ruda", comun: "ruda", cientifico: "Ruta graveolens", img: "/static/muestras/reg_Ruta_graveolens__ruda.jpg" },
];

let redActiva = "regional";
let archivoActual = null;

const galeria = document.getElementById("galeria");
ESPECIES.forEach((esp) => {
  const btn = document.createElement("button");
  btn.className = "tarjeta";
  btn.type = "button";
  btn.innerHTML = `<img src="${esp.img}" alt="${esp.comun}"><div><strong>${esp.comun}</strong><small>${esp.cientifico}</small></div>`;
  btn.addEventListener("click", async () => {
    const blob = await fetch(esp.img).then((r) => r.blob());
    const file = new File([blob], `${esp.id}.jpg`, { type: blob.type || "image/jpeg" });
    document.getElementById("identificar").scrollIntoView({ behavior: "smooth" });
    identificar(file);
  });
  galeria.appendChild(btn);
});

document.querySelectorAll(".pills").forEach((grupo) => {
  grupo.addEventListener("click", (ev) => {
    const btn = ev.target.closest(".pill");
    if (!btn) return;
    grupo.querySelectorAll(".pill").forEach((p) => p.classList.remove("activo"));
    btn.classList.add("activo");
    if (btn.dataset.tab) {
      document.querySelectorAll(".panel").forEach((p) => p.classList.remove("activo"));
      document.getElementById(`panel-${btn.dataset.tab}`).classList.add("activo");
    }
    if (btn.dataset.red) {
      redActiva = btn.dataset.red;
      if (archivoActual) identificar(archivoActual);
    }
  });
});

const lightbox = document.getElementById("lightbox");
document.querySelectorAll(".graficas img").forEach((img) => {
  img.addEventListener("click", () => {
    lightbox.querySelector("img").src = img.src;
    lightbox.classList.remove("oculto");
  });
});
lightbox.addEventListener("click", () => lightbox.classList.add("oculto"));

const estadoEl = document.getElementById("estado-modelos");
async function checarEstado() {
  try {
    const r = await fetch("/api/estado").then((x) => x.json());
    if (r.listo) {
      estadoEl.textContent = "Modelo conectado. Sube una foto y te dice a qué especie pertenece.";
      return true;
    }
    if (r.error) {
      estadoEl.textContent = `No se pudo conectar el modelo: ${r.error}`;
      return false;
    }
    estadoEl.textContent = "Conectando el modelo entrenado… la primera vez tarda un poco.";
    return false;
  } catch {
    estadoEl.textContent = "El servidor no responde.";
    return false;
  }
}
checarEstado();
const poll = setInterval(async () => {
  if (await checarEstado()) clearInterval(poll);
}, 2000);

function pct(n) {
  return `${Math.round(n * 1000) / 10}%`;
}

function carta(titulo, bloque, alt) {
  if (!bloque || !bloque.disponible) {
    return `<article class="veredicto${alt ? " alt" : ""}"><p class="etiqueta">${titulo}</p><p>Este modelo no está disponible.</p></article>`;
  }
  const top = bloque.top || [];
  const g = top[0];
  const resto = top
    .slice(1)
    .map(
      (t) =>
        `<div class="top-item"><span>${t.comun} <em>(${t.cientifico})</em></span><b>${pct(t.prob)}</b></div>
         <div class="barra-p"><span style="width:${(t.prob * 100).toFixed(1)}%"></span></div>`
    )
    .join("");
  return `<article class="veredicto${alt ? " alt" : ""}">
    <p class="etiqueta">${titulo}</p>
    <h3 class="especie">${g.comun}</h3>
    <p class="cientifico">${g.cientifico}</p>
    <p class="confianza">Confianza ${pct(g.prob)}</p>
    <div class="top-item"><span>${g.comun}</span><b>${pct(g.prob)}</b></div>
    <div class="barra-p"><span style="width:${(g.prob * 100).toFixed(1)}%"></span></div>
    ${resto}
  </article>`;
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
  try {
    const r = await fetch("/api/predecir", { method: "POST", body: fd });
    const data = await r.json();
    if (!data.ok) throw new Error(data.error || "No se pudo identificar");
    const partes = [];
    if (redActiva !== "herbify") {
      partes.push(carta("El modelo regional dice que es", data.modelos.regional, true));
    }
    if (redActiva !== "regional") {
      partes.push(carta("Herbify dice que es", data.modelos.herbify, false));
    }
    if (redActiva === "ambos" && data.coinciden) {
      partes.push('<p class="badge">Los dos modelos coinciden en la especie</p>');
    }
    veredicto.innerHTML = partes.join("");
  } catch (err) {
    avisoUna.textContent = err.message;
    avisoUna.classList.remove("oculto");
  } finally {
    carga.classList.add("oculto");
  }
}
