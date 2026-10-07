"""Sitio local de resultados y comparador de imagenes.

Desde la raiz del repo:

    .\\.venv\\Scripts\\python.exe web\\app.py

Luego abre http://127.0.0.1:5050
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

WEB = Path(__file__).resolve().parent
RAIZ = WEB.parent
UPLOADS = WEB / "uploads"
UPLOADS.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    static_folder=str(WEB / "static"),
    template_folder=str(WEB / "templates"),
)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

IMG = (224, 224)
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

MODELOS = {
    "herbify": {
        "ruta": RAIZ / "modelo_plantas.h5",
        "clases": RAIZ / "clases.json",
        "objeto": None,
        "nombres": [],
    },
    "regional": {
        "ruta": RAIZ / "regional_colombia" / "resultados" / "modelo_regional.h5",
        "clases": RAIZ / "regional_colombia" / "resultados" / "clases.json",
        "objeto": None,
        "nombres": [],
    },
}

_lock = threading.Lock()
_estado = {"listo": False, "error": None, "cargando": False}


def _etiqueta(clase: str) -> dict:
    if "__" in clase:
        cientifico, comun = clase.split("__", 1)
        return {
            "id": clase,
            "cientifico": cientifico.replace("_", " "),
            "comun": comun.replace("_", " "),
        }
    if "(" in clase and clase.endswith(")"):
        cientifico, comun = clase.rsplit("(", 1)
        return {
            "id": clase,
            "cientifico": cientifico.strip(),
            "comun": comun[:-1].replace("_", " ").strip(),
        }
    return {"id": clase, "cientifico": clase, "comun": clase}


def _clave(nombre: str) -> str:
    base = nombre.split("(")[0].split("__")[0]
    return " ".join(base.replace("_", " ").lower().split())


def cargar_modelos() -> None:
    global _estado
    with _lock:
        if _estado["listo"] or _estado["cargando"]:
            return
        _estado["cargando"] = True
    try:
        import tensorflow as tf

        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices(gpus[0], "GPU")
            tf.config.experimental.set_memory_growth(gpus[0], True)

        for clave, info in MODELOS.items():
            if not info["ruta"].is_file():
                continue
            info["nombres"] = json.loads(info["clases"].read_text(encoding="utf-8"))
            info["objeto"] = tf.keras.models.load_model(str(info["ruta"]))

        with _lock:
            _estado = {
                "listo": any(m["objeto"] is not None for m in MODELOS.values()),
                "error": None if any(m["objeto"] is not None for m in MODELOS.values()) else "No hay modelos .h5",
                "cargando": False,
                "herbify": MODELOS["herbify"]["objeto"] is not None,
                "regional": MODELOS["regional"]["objeto"] is not None,
            }
    except Exception as exc:
        with _lock:
            _estado = {"listo": False, "error": str(exc), "cargando": False}


def _predecir_uno(modelo, nombres, imagen: Image.Image) -> list[dict]:
    import numpy as np

    arr = np.expand_dims(np.array(imagen.convert("RGB").resize(IMG)), 0)
    probs = modelo.predict(arr, verbose=0)[0]
    orden = probs.argsort()[::-1][:3]
    salida = []
    for i in orden:
        etq = _etiqueta(nombres[int(i)])
        etq["prob"] = float(probs[int(i)])
        salida.append(etq)
    return salida


def _abrir_imagen(archivo) -> Image.Image:
    nombre = secure_filename(archivo.filename or "foto.jpg")
    if Path(nombre).suffix.lower() not in EXTS:
        raise ValueError("Usa JPG, PNG o WEBP.")
    img = Image.open(archivo.stream)
    img.load()
    return img.convert("RGB")


@app.get("/")
def inicio():
    return render_template("index.html")


@app.get("/api/estado")
def estado():
    if not _estado.get("listo") and not _estado.get("cargando") and _estado.get("error") is None:
        threading.Thread(target=cargar_modelos, daemon=True).start()
    return jsonify(_estado)


@app.post("/api/predecir")
def predecir():
    if not _estado.get("listo"):
        cargar_modelos()
    if not _estado.get("listo"):
        return jsonify({"ok": False, "error": _estado.get("error") or "Modelos no disponibles"}), 503

    archivo = request.files.get("imagen")
    if archivo is None or not archivo.filename:
        return jsonify({"ok": False, "error": "Falta la imagen"}), 400
    try:
        img = _abrir_imagen(archivo)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    resultado = {"ok": True, "modelos": {}}
    claves = []
    for clave, info in MODELOS.items():
        if info["objeto"] is None:
            resultado["modelos"][clave] = {"disponible": False}
            continue
        top = _predecir_uno(info["objeto"], info["nombres"], img)
        resultado["modelos"][clave] = {"disponible": True, "top": top}
        if top:
            claves.append(_clave(top[0]["cientifico"]))

    resultado["coinciden"] = len(claves) == 2 and claves[0] == claves[1]
    return jsonify(resultado)


@app.post("/api/comparar")
def comparar():
    if not _estado.get("listo"):
        cargar_modelos()
    if not _estado.get("listo"):
        return jsonify({"ok": False, "error": _estado.get("error") or "Modelos no disponibles"}), 503

    a = request.files.get("imagen_a")
    b = request.files.get("imagen_b")
    if a is None or b is None or not a.filename or not b.filename:
        return jsonify({"ok": False, "error": "Sube las dos fotos"}), 400
    try:
        img_a = _abrir_imagen(a)
        img_b = _abrir_imagen(b)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    def bloque(img):
        out = {}
        for clave, info in MODELOS.items():
            if info["objeto"] is None:
                out[clave] = {"disponible": False}
            else:
                out[clave] = {
                    "disponible": True,
                    "top": _predecir_uno(info["objeto"], info["nombres"], img),
                }
        return out

    return jsonify({"ok": True, "foto_a": bloque(img_a), "foto_b": bloque(img_b)})


@app.get("/muestras/<path:nombre>")
def muestras(nombre):
    return send_from_directory(WEB / "static" / "muestras", nombre)


if __name__ == "__main__":
    threading.Thread(target=cargar_modelos, daemon=True).start()
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
