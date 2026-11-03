"""Comparador local de los modelos de plantas.

Desde la raiz del repo, con el entorno de la web (Python 3.12, TensorFlow 2.21, Keras 3.15):

    .\\.venv-web\\Scripts\\python.exe web\\app.py

Luego abre http://127.0.0.1:5050

El .venv de entrenamiento sigue en TensorFlow 2.10 y no abre los .keras.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image
from werkzeug.utils import secure_filename

WEB = Path(__file__).resolve().parent
RAIZ = WEB.parent
LIVIANOS = RAIZ / "modelos_livianos"
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

# Cifras publicadas que no estan en un json de metricas.
_EQUIPO = {
    "accuracy_test": 0.912,
    "f1_ponderado": 0.914,
    "imagenes_test": 761,
    "errores_test": 67,
    "train": 3408,
    "val": 734,
    "clases": 57,
}
_SPLIT_HERBIFY = {"train": 3408, "val": 734}
_SPLIT_REGIONAL = {"train": 10201, "val": 2178}


def _ficha(
    modelo_id,
    nombre,
    arquitectura,
    vocab,
    dataset,
    fotos,
    limite,
    ruta,
    clases_ruta,
    curvas,
    matriz,
    curvas_nota,
    matriz_nota,
    tono,
    metricas_ruta=None,
    metricas_fijas=None,
    split=None,
):
    return {
        "id": modelo_id,
        "nombre": nombre,
        "arquitectura": arquitectura,
        "vocab": vocab,
        "dataset": dataset,
        "fotos": fotos,
        "limite": limite,
        "ruta": ruta,
        "clases_ruta": clases_ruta,
        "curvas": curvas,
        "matriz": matriz,
        "curvas_nota": curvas_nota,
        "matriz_nota": matriz_nota,
        "tono": tono,
        "metricas_ruta": metricas_ruta,
        "metricas_fijas": metricas_fijas or {},
        "split": split or {},
        "objeto": None,
        "nombres": [],
        "error_carga": None,
    }


CATALOGO = [
    _ficha(
        "v2_equipo",
        "MobileNetV2 · intento 1",
        "MobileNetV2",
        "herbify",
        "Herbify · India",
        "Kaggle Herbify, estudio, fondo uniforme",
        "Especies de la India; falla fuera de dominio",
        RAIZ / "modelo_plantas.h5",
        RAIZ / "clases.json",
        "img/herbify_curvas.png",
        "img/herbify_matriz.png",
        "Curvas · fase 1 congelada y fine-tuning",
        "Matriz de confusión en test (57 especies)",
        "deep",
        metricas_fijas=_EQUIPO,
    ),
    _ficha(
        "v2_companero",
        "MobileNetV2 · intento 2",
        "MobileNetV2",
        "herbify",
        "Herbify · India",
        "El mismo split Herbify (57 especies, semilla 42)",
        "Misma flora que el intento 1; otra corrida, en CPU",
        LIVIANOS / "modelo_mobilenetv2.keras",
        LIVIANOS / "clases.json",
        "img/v2_companero_curvas.png",
        "img/v2_companero_matriz.png",
        "Curvas · MobileNetV2, intento 2",
        "Matriz de confusión en test (57 especies)",
        "leaf",
        metricas_ruta=LIVIANOS / "metricas_mobilenetv2.json",
        split=_SPLIT_HERBIFY,
    ),
    _ficha(
        "v3",
        "MobileNetV3Small",
        "MobileNetV3Small",
        "herbify",
        "Herbify · India",
        "El mismo split Herbify (57 especies, semilla 42)",
        "Más liviano: menos parámetros y menos accuracy",
        LIVIANOS / "modelo_mobilenetv3.keras",
        LIVIANOS / "clases.json",
        "img/v3_curvas.png",
        "img/v3_matriz.png",
        "Curvas · MobileNetV3Small",
        "Matriz de confusión en test (57 especies)",
        "gold",
        metricas_ruta=LIVIANOS / "metricas_mobilenetv3.json",
        split=_SPLIT_HERBIFY,
    ),
    _ficha(
        "regional",
        "MobileNetV2 · regional",
        "MobileNetV2",
        "regional",
        "Colombia y huerta",
        "iNaturalist, Herbify en guayaba y ruda, y fotos de huerta del equipo",
        "Fotos de campo; hojas parecidas (matarratón, roble, moringa)",
        RAIZ / "regional_colombia" / "resultados" / "modelo_regional_29clases.h5",
        RAIZ / "regional_colombia" / "resultados" / "clases_29.json",
        "img/regional_curvas.png",
        "img/regional_matriz.png",
        "Curvas · 17 épocas + 20 de fine-tuning",
        "Matriz de confusión en test (29 especies)",
        "clay",
        metricas_ruta=RAIZ / "regional_colombia" / "resultados" / "metricas_regional_29.json",
        split=_SPLIT_REGIONAL,
    ),
]

_lock = threading.Lock()
_por_id = {m["id"]: m for m in CATALOGO}


def _etiqueta(clase: str) -> dict:
    if "__" in clase:
        cientifico, comun = clase.split("__", 1)
        return {
            "id": clase,
            "cientifico": cientifico.replace("_", " "),
            "comun": comun.replace("_", " "),
        }
    if "(" in clase:
        cientifico, comun = clase.split("(", 1)
        comun = comun.replace("(", " ").replace(")", " ").replace("_", " ")
        return {
            "id": clase,
            "cientifico": cientifico.strip(),
            "comun": " ".join(comun.split()),
        }
    return {"id": clase, "cientifico": clase, "comun": clase}


def _clave(nombre: str) -> str:
    base = nombre.split("(")[0].split("__")[0]
    return " ".join(base.replace("_", " ").lower().split())


def _leer_json(ruta: Path | None) -> dict:
    if ruta is None or not ruta.is_file():
        return {}
    return json.loads(ruta.read_text(encoding="utf-8"))


def _metricas(info: dict) -> dict:
    datos = dict(info["metricas_fijas"])
    datos.update(_leer_json(info["metricas_ruta"]))
    datos.update(info["split"])
    if info["ruta"].is_file():
        datos["tamano_mb"] = round(info["ruta"].stat().st_size / (1024 * 1024), 2)
    if "clases" not in datos and info["clases_ruta"].is_file():
        datos["clases"] = len(json.loads(info["clases_ruta"].read_text(encoding="utf-8")))
    return datos


def _f1_publico(info: dict, metricas: dict) -> list[dict]:
    crudo = metricas.get("f1_por_clase") or {}
    if not isinstance(crudo, dict):
        return []
    salida = []
    for clase, valor in crudo.items():
        etq = _etiqueta(str(clase))
        etq["f1"] = float(valor)
        salida.append(etq)
    salida.sort(key=lambda item: item["f1"], reverse=True)
    return salida


def _publico(info: dict) -> dict:
    metricas = _metricas(info)
    return {
        "id": info["id"],
        "nombre": info["nombre"],
        "arquitectura": info["arquitectura"],
        "vocab": info["vocab"],
        "dataset": info["dataset"],
        "fotos": info["fotos"],
        "limite": info["limite"],
        "tono": info["tono"],
        "curvas": info["curvas"],
        "matriz": info["matriz"],
        "curvas_nota": info["curvas_nota"],
        "matriz_nota": info["matriz_nota"],
        "archivo": info["ruta"].is_file(),
        "accuracy": metricas.get("accuracy_test"),
        "top3": metricas.get("top3_accuracy_test"),
        "f1": metricas.get("f1_ponderado"),
        "parametros": metricas.get("parametros"),
        "tamano_mb": metricas.get("tamano_mb"),
        "minutos": metricas.get("minutos_entrenamiento"),
        "clases": metricas.get("clases"),
        "train": metricas.get("train"),
        "val": metricas.get("val"),
        "test": metricas.get("imagenes_test"),
        "errores": metricas.get("errores_test"),
        "f1_por_clase": _f1_publico(info, metricas),
    }


def _especies() -> dict:
    grupos = []
    manifest_ruta = LIVIANOS / "muestras_herbify.json"
    if manifest_ruta.is_file():
        herb = []
        for item in json.loads(manifest_ruta.read_text(encoding="utf-8")):
            etq = _etiqueta(item["clase"])
            etq["img"] = f"/static/muestras/{item['img']}"
            herb.append(etq)
        herb.sort(key=lambda item: item["comun"].lower())
        grupos.append(
            {
                "id": "herbify",
                "titulo": "Herbify · 57 especies de la India",
                "especies": herb,
            }
        )

    regional_clases = RAIZ / "regional_colombia" / "resultados" / "clases_29.json"
    if regional_clases.is_file():
        regional = []
        for clase in json.loads(regional_clases.read_text(encoding="utf-8")):
            archivo = f"reg_{clase}.jpg"
            if not (WEB / "static" / "muestras" / archivo).is_file():
                continue
            etq = _etiqueta(clase)
            etq["img"] = f"/static/muestras/{archivo}"
            regional.append(etq)
        regional.sort(key=lambda item: item["comun"].lower())
        grupos.append(
            {
                "id": "regional",
                "titulo": "Colombia y huerta · 29 especies",
                "especies": regional,
            }
        )
    return {"grupos": grupos}


def _normalizar_config(obj):
    """Adapta un .h5 de Keras 2 para que Keras 3 pueda reconstruir el grafo."""
    if isinstance(obj, dict):
        if obj.get("class_name") == "DepthwiseConv2D":
            config = obj.get("config")
            if isinstance(config, dict):
                config.pop("groups", None)
        nodos = obj.get("inbound_nodes")
        if isinstance(nodos, list):
            obj["inbound_nodes"] = [
                [nodo]
                if isinstance(nodo, list) and nodo and isinstance(nodo[0], str)
                else nodo
                for nodo in nodos
            ]
        for valor in obj.values():
            _normalizar_config(valor)
    elif isinstance(obj, list):
        for valor in obj:
            _normalizar_config(valor)


def _bajar_indices(model_cfg):
    if not isinstance(model_cfg, dict):
        return
    capas = (model_cfg.get("config") or {}).get("layers") or []
    un_nodo = {
        capa.get("name")
        for capa in capas
        if len(capa.get("inbound_nodes") or []) <= 1
    }

    def recorrer(obj):
        if isinstance(obj, list):
            if (
                len(obj) >= 4
                and isinstance(obj[0], str)
                and obj[1] == 1
                and obj[0] in un_nodo
            ):
                obj[1] = 0
            for valor in obj:
                recorrer(valor)
        elif isinstance(obj, dict):
            for valor in obj.values():
                recorrer(valor)

    for capa in capas:
        recorrer(capa.get("inbound_nodes"))
        if capa.get("class_name") in {"Functional", "Sequential", "Model"}:
            _bajar_indices(capa)


def _ruta_h5(ruta: Path) -> str:
    if ruta.suffix.lower() != ".h5":
        return str(ruta)
    destino = WEB / ".cache_h5" / f"{ruta.stem}.h5"
    if destino.is_file() and destino.stat().st_mtime >= ruta.stat().st_mtime:
        return str(destino)
    import h5py

    destino.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ruta, destino)
    with h5py.File(destino, "a") as archivo:
        crudo = archivo.attrs["model_config"]
        if isinstance(crudo, bytes):
            crudo = crudo.decode("utf-8")
        cfg = json.loads(crudo)
        _normalizar_config(cfg)
        _bajar_indices(cfg)
        archivo.attrs["model_config"] = json.dumps(cfg)
    return str(destino)


def _capa_lambda():
    import keras
    import tensorflow as tf

    class TFOpLambda(keras.layers.Layer):
        def __init__(self, function, **kwargs):
            kwargs.pop("trainable", None)
            super().__init__(**kwargs)
            self._fn = function

        def call(self, inputs, **kwargs):
            fn = self._fn
            if isinstance(fn, str):
                obj = tf
                for parte in fn.split("."):
                    obj = getattr(obj, parte)
                fn = obj
            for clave in ("name", "mask", "training"):
                kwargs.pop(clave, None)
            return fn(inputs, **kwargs)

    return TFOpLambda


def cargar_uno(info: dict):
    with _lock:
        if info["objeto"] is not None:
            return
        if not info["ruta"].is_file():
            raise FileNotFoundError(f"No está el archivo de {info['nombre']}")
        if not info["clases_ruta"].is_file():
            raise FileNotFoundError(f"No están las clases de {info['nombre']}")
        import keras

        info["nombres"] = json.loads(info["clases_ruta"].read_text(encoding="utf-8"))
        with keras.utils.custom_object_scope({"TFOpLambda": _capa_lambda()}):
            info["objeto"] = keras.models.load_model(_ruta_h5(info["ruta"]), compile=False)
        info["error_carga"] = None


def _predecir_uno(modelo, nombres, imagen: Image.Image) -> list[dict]:
    import numpy as np

    arr = np.asarray(imagen.convert("RGB").resize(IMG), dtype="float32")
    probs = modelo.predict(np.expand_dims(arr, 0), verbose=0)[0]
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


def _ids_pedidos() -> list[str]:
    crudo = (request.form.get("modelos") or "").strip()
    if not crudo:
        return [m["id"] for m in CATALOGO]
    ids = [parte.strip() for parte in crudo.split(",") if parte.strip()]
    desconocidos = [modelo_id for modelo_id in ids if modelo_id not in _por_id]
    if desconocidos:
        raise ValueError("Modelo desconocido: " + ", ".join(desconocidos))
    if not ids:
        raise ValueError("Elige al menos un modelo")
    return ids


def _acuerdo(bloques: dict) -> dict:
    herbify = []
    regional = []
    for info in CATALOGO:
        bloque = bloques.get(info["id"]) or {}
        top = bloque.get("top") or []
        if not bloque.get("disponible") or not top:
            continue
        if info["vocab"] == "herbify":
            herbify.append(top[0])
        elif info["vocab"] == "regional":
            regional.append(top[0])

    acuerdo_herbify = None
    if len(herbify) >= 2:
        acuerdo_herbify = len({item["id"] for item in herbify}) == 1

    acuerdo_nombre = None
    if herbify and regional:
        claves_h = {_clave(item["cientifico"]) for item in herbify}
        acuerdo_nombre = _clave(regional[0]["cientifico"]) in claves_h

    return {"herbify": acuerdo_herbify, "nombre_cientifico": acuerdo_nombre}


@app.get("/")
def inicio():
    return render_template("index.html")


@app.get("/api/modelos")
def modelos():
    return jsonify({"modelos": [_publico(info) for info in CATALOGO]})


@app.get("/api/especies")
def especies():
    return jsonify(_especies())


@app.get("/api/estado")
def estado():
    presentes = {info["id"]: info["ruta"].is_file() for info in CATALOGO}
    return jsonify(
        {
            "listo": any(presentes.values()),
            "error": None if any(presentes.values()) else "No hay modelos en disco",
            "modelos": presentes,
        }
    )


@app.post("/api/predecir")
def predecir():
    archivo = request.files.get("imagen")
    if archivo is None or not archivo.filename:
        return jsonify({"ok": False, "error": "Falta la imagen"}), 400
    try:
        ids = _ids_pedidos()
        img = _abrir_imagen(archivo)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400

    bloques = {}
    for modelo_id in ids:
        info = _por_id[modelo_id]
        if not info["ruta"].is_file():
            bloques[modelo_id] = {
                "disponible": False,
                "nombre": info["nombre"],
                "vocab": info["vocab"],
                "error": "No está el archivo del modelo",
            }
            continue
        try:
            cargar_uno(info)
            bloques[modelo_id] = {
                "disponible": True,
                "nombre": info["nombre"],
                "vocab": info["vocab"],
                "tono": info["tono"],
                "top": _predecir_uno(info["objeto"], info["nombres"], img),
            }
        except Exception as exc:
            info["error_carga"] = str(exc)
            bloques[modelo_id] = {
                "disponible": False,
                "nombre": info["nombre"],
                "vocab": info["vocab"],
                "error": str(exc),
            }

    return jsonify(
        {
            "ok": any(b.get("disponible") for b in bloques.values()),
            "modelos": bloques,
            "acuerdo": _acuerdo(bloques),
            "error": None
            if any(b.get("disponible") for b in bloques.values())
            else "No se pudo usar ningún modelo",
        }
    )


@app.get("/muestras/<path:nombre>")
def muestras(nombre):
    return send_from_directory(WEB / "static" / "muestras", nombre)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False, threaded=True)
