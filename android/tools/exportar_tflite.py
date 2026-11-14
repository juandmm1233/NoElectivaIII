"""Exporta a TFLite float16 los cinco modelos que corren en el teléfono.

Usa el Python de .venv-web (TensorFlow 2.21 / Keras 3.15):

    .\\.venv-web\\Scripts\\python.exe android\\tools\\exportar_tflite.py

La entrada queda en float32, 224x224, RGB, rango 0-255. El preprocesado
sigue dentro del grafo. La Sequential de augmentación se quita: en
inferencia no debe tocar la foto y suele romper el conversor.
"""
from __future__ import annotations

import gc
import json
import shutil
import traceback
from pathlib import Path

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

RAIZ = Path(__file__).resolve().parents[2]
WEB = RAIZ / "web"
ASSETS = RAIZ / "android" / "app" / "src" / "main" / "assets"
SALIDA = ASSETS / "modelos"
CACHE = WEB / ".cache_h5"

CAPAS_AUG = {
    "RandomFlip",
    "RandomRotation",
    "RandomZoom",
    "RandomTranslation",
    "RandomContrast",
    "RandomBrightness",
    "RandomCrop",
    "RandomHeight",
    "RandomWidth",
    "RandomShear",
    "RandomSaturation",
    "RandomHue",
}

MODELOS = [
    {
        "id": "herbify_v2",
        "nombre": "MobileNetV2",
        "arquitectura": "MobileNetV2",
        "vocab": "herbify",
        "dataset": "Herbify · India",
        "accuracy": 0.912,
        "ruta": RAIZ / "modelo_plantas.h5",
        "clases": RAIZ / "clases.json",
        "archivo_clases": "clases.json",
        "tflite": "herbify_v2.tflite",
        "salida": 57,
    },
    {
        "id": "herbify_v3",
        "nombre": "MobileNetV3Small",
        "arquitectura": "MobileNetV3Small",
        "vocab": "herbify",
        "dataset": "Herbify · India",
        "accuracy": 0.816,
        "ruta": RAIZ / "modelos_livianos" / "modelo_mobilenetv3.keras",
        "clases": RAIZ / "clases.json",
        "archivo_clases": "clases.json",
        "tflite": "herbify_v3.tflite",
        "salida": 57,
    },
    {
        "id": "herbify_efficientnet",
        "nombre": "EfficientNetB0",
        "arquitectura": "EfficientNetB0",
        "vocab": "herbify",
        "dataset": "Herbify · India",
        "accuracy": 0.894,
        "ruta": RAIZ / "resultados_comparacion" / "modelo_efficientnetb0.h5",
        "clases": RAIZ / "clases.json",
        "archivo_clases": "clases.json",
        "tflite": "herbify_efficientnet.tflite",
        "salida": 57,
    },
    {
        "id": "regional_v2",
        "nombre": "MobileNetV2 regional",
        "arquitectura": "MobileNetV2",
        "vocab": "regional",
        "dataset": "Colombia y huerta",
        "accuracy": 0.680,
        "ruta": RAIZ / "regional_colombia" / "resultados" / "modelo_regional_29clases.h5",
        "clases": RAIZ / "regional_colombia" / "resultados" / "clases_29.json",
        "archivo_clases": "clases_29.json",
        "tflite": "regional_v2.tflite",
        "salida": 29,
    },
    {
        "id": "regional_estudiante",
        "nombre": "Estudiante regional",
        "arquitectura": "MobileNetV2",
        "vocab": "regional",
        "dataset": "Colombia y huerta",
        "accuracy": 0.695,
        "ruta": RAIZ / "regional_colombia" / "resultados" / "modelo_mobilenetv2_estudiante_regional.h5",
        "clases": RAIZ / "regional_colombia" / "resultados" / "clases_29.json",
        "archivo_clases": "clases_29.json",
        "tflite": "regional_estudiante.tflite",
        "salida": 29,
    },
]

REMOTOS = [
    {
        "id": "resnet50",
        "nombre": "ResNet50",
        "arquitectura": "ResNet50",
        "vocab": "herbify",
        "dataset": "Herbify · India",
        "accuracy": 0.930,
        "archivo_tflite": None,
        "archivo_clases": "clases.json",
        "remoto": True,
    },
    {
        "id": "resnet50_regional",
        "nombre": "ResNet50 regional",
        "arquitectura": "ResNet50",
        "vocab": "regional",
        "dataset": "Colombia y huerta",
        "accuracy": 0.765,
        "archivo_tflite": None,
        "archivo_clases": "clases_29.json",
        "remoto": True,
    },
]


def _normalizar_config(obj):
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
    destino = CACHE / f"{ruta.stem}.h5"
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

        def get_config(self):
            config = super().get_config()
            config["function"] = self._fn if isinstance(self._fn, str) else getattr(self._fn, "__name__", "fn")
            return config

    return TFOpLambda


def _cargar(ruta: Path):
    import keras

    with keras.utils.custom_object_scope({"TFOpLambda": _capa_lambda()}):
        return keras.models.load_model(_ruta_h5(ruta), compile=False)


def _es_random(capa) -> bool:
    tipo = capa.__class__.__name__
    return tipo in CAPAS_AUG or tipo.startswith("Random")


def _contar_random(capa, vistos=None) -> int:
    vistos = vistos or set()
    marca = id(capa)
    if marca in vistos:
        return 0
    vistos.add(marca)
    total = 1 if _es_random(capa) else 0
    for hijo in getattr(capa, "layers", []) or []:
        total += _contar_random(hijo, vistos)
    return total


def _es_augmentacion(capa) -> bool:
    if _es_random(capa):
        return True
    hijos = [hijo for hijo in (getattr(capa, "layers", []) or []) if hijo.__class__.__name__ != "InputLayer"]
    if not hijos:
        return False
    tipo = capa.__class__.__name__
    if tipo in {"Functional", "Model"} and len(hijos) > 40:
        return False
    if len(hijos) > 30:
        return False
    return _contar_random(capa) > 0


def _constantes(capa) -> dict:
    nodos = list(getattr(capa, "_inbound_nodes", []) or [])
    if not nodos:
        return {}
    kwargs = dict(getattr(nodos[0].arguments, "kwargs", {}) or {})
    return {
        clave: valor
        for clave, valor in kwargs.items()
        if clave not in {"name", "mask", "training"} and valor is not None
    }


def _operacion_fija(funcion, y, nombre):
    import keras
    import tensorflow as tf

    class OperacionFija(keras.layers.Layer):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)

        def call(self, inputs, **kwargs):
            fn = funcion
            if isinstance(fn, str):
                obj = tf
                for parte in fn.split("."):
                    obj = getattr(obj, parte)
                fn = obj
            return fn(inputs, y=kwargs.get("y", y))

    return OperacionFija(name=nombre)


def _clonar_capa(capa):
    import keras

    if _es_augmentacion(capa):
        return keras.layers.Identity(name=f"sin_{capa.name}")
    if capa.__class__.__name__ == "TFOpLambda":
        constantes = _constantes(capa)
        if "y" not in constantes:
            raise RuntimeError(f"{capa.name} no tiene la constante del preprocesado")
        return _operacion_fija(capa._fn, constantes["y"], f"{capa.name}_fijo")
    return capa.__class__.from_config(capa.get_config())


def _llamar(capa, *args, **kwargs):
    kwargs.pop("name", None)
    kwargs.pop("mask", None)
    try:
        return capa(*args, training=False, **kwargs)
    except TypeError:
        try:
            return capa(*args, **kwargs)
        except TypeError as exc:
            raise TypeError(f"{capa.name} ({capa.__class__.__name__}): {exc}") from exc


def _modelo_inferencia(modelo):
    import keras

    saltadas = [
        f"{capa.name}:{capa.__class__.__name__}"
        for capa in modelo.layers
        if _es_augmentacion(capa)
    ]
    if any(getattr(capa, "weights", None) for capa in modelo.layers if _es_augmentacion(capa)):
        raise RuntimeError("La augmentación tiene pesos; no se puede quitar sin perder el modelo")
    with keras.utils.custom_object_scope({"TFOpLambda": _capa_lambda()}):
        nuevo = keras.models.clone_model(
            modelo,
            clone_function=_clonar_capa,
            call_function=_llamar,
        )
    if nuevo.count_params() != modelo.count_params():
        raise RuntimeError(
            f"El clon tiene {nuevo.count_params()} parámetros y el original {modelo.count_params()}"
        )
    nuevo.set_weights(modelo.get_weights())
    return nuevo, saltadas


def _predecir(modelo, lote):
    import tensorflow as tf

    salida = modelo(tf.constant(lote), training=False)
    vector = np.asarray(salida)[0].astype("float64")
    return int(vector.argmax()), vector


def _opciones(converter):
    import tensorflow as tf

    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    try:
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    except Exception:
        pass


def _convertir(modelo, destino: Path, imagenes: list[tuple[str, np.ndarray]]) -> tuple[str, list[dict]]:
    import tensorflow as tf

    errores = []
    modelo(tf.zeros((1, 224, 224, 3)), training=False)
    carpeta = destino.with_name(destino.stem + "_saved")

    def desde_keras():
        return tf.lite.TFLiteConverter.from_keras_model(modelo)

    def desde_saved():
        if carpeta.exists():
            shutil.rmtree(carpeta)
        modelo.export(str(carpeta))
        return tf.lite.TFLiteConverter.from_saved_model(str(carpeta))

    try:
        for nombre, fabrica in (("keras", desde_keras), ("saved_model", desde_saved)):
            try:
                converter = fabrica()
                _opciones(converter)
                destino.write_bytes(converter.convert())
            except Exception as exc:
                errores.append(f"{nombre}: {type(exc).__name__}: {exc}")
                if destino.exists():
                    destino.unlink()
                continue
            tamano = destino.stat().st_size
            if tamano < 400_000:
                errores.append(f"{nombre}: archivo de {tamano} bytes, demasiado pequeño")
                destino.unlink()
                continue
            chequeos = []
            malo = None
            for imagen, lote in imagenes:
                indice_k, probs_k = _predecir(modelo, lote)
                indice_t, probs_t = _argmax_tflite(destino, lote)
                coincide = indice_k == indice_t and bool(np.isfinite(probs_t).all())
                diff = float(np.max(np.abs(probs_k - probs_t))) if np.isfinite(probs_t).all() else float("nan")
                chequeos.append(
                    {
                        "imagen": imagen,
                        "argmax_keras": indice_k,
                        "argmax_tflite": indice_t,
                        "coincide": coincide,
                        "max_abs": None if diff != diff else round(diff, 6),
                        "prob_keras": round(float(probs_k[indice_k]), 6),
                    }
                )
                estado = "OK" if coincide else "DISTINTO"
                print(
                    f"  {estado} {imagen}: keras={indice_k} tflite={indice_t} max_abs={diff:.6f}",
                    flush=True,
                )
                if not coincide and malo is None:
                    malo = imagen
            if malo is None:
                return nombre, chequeos
            errores.append(f"{nombre}: argmax distinto en {malo}")
            destino.unlink()
        raise RuntimeError(" | ".join(errores))
    finally:
        if carpeta.exists():
            shutil.rmtree(carpeta, ignore_errors=True)


def _argmax_tflite(ruta: Path, lote: np.ndarray):
    import tensorflow as tf

    interprete = tf.lite.Interpreter(model_path=str(ruta), num_threads=4)
    interprete.allocate_tensors()
    entrada = interprete.get_input_details()[0]
    salida = interprete.get_output_details()[0]
    datos = np.asarray(lote, dtype=np.dtype(entrada["dtype"].name) if hasattr(entrada["dtype"], "name") else np.float32)
    if tuple(entrada["shape"]) != tuple(datos.shape):
        raise RuntimeError(f"Entrada TFLite {entrada['shape']} distinta de {datos.shape}")
    interprete.set_tensor(entrada["index"], datos)
    interprete.invoke()
    vector = np.asarray(interprete.get_tensor(salida["index"]))[0].astype("float64")
    return int(vector.argmax()), vector


def _imagenes() -> list[tuple[str, np.ndarray]]:
    from PIL import Image

    yy, xx = np.mgrid[0:224, 0:224]
    sintetica = np.stack(
        [
            xx * (255.0 / 223.0),
            yy * (180.0 / 223.0),
            ((xx + yy) * 127.0 / 446.0),
        ],
        axis=-1,
    ).astype("float32")
    lote = [("sintetica", sintetica[None, ...])]
    muestras = RAIZ / "web" / "static" / "muestras"
    reales = []
    if muestras.is_dir():
        for ruta in sorted(muestras.glob("*.jpg")):
            reales.append(ruta)
            if len(reales) >= 2:
                break
    for ruta in reales:
        imagen = Image.open(ruta).convert("RGB").resize((224, 224))
        arr = np.asarray(imagen, dtype="float32")
        lote.append((ruta.name, arr[None, ...]))
    return lote


def _ficha_publica(info: dict, archivo: str | None) -> dict:
    return {
        "id": info["id"],
        "nombre": info["nombre"],
        "arquitectura": info["arquitectura"],
        "vocab": info["vocab"],
        "dataset": info["dataset"],
        "accuracy": info["accuracy"],
        "archivo_tflite": archivo,
        "archivo_clases": info["archivo_clases"],
        "remoto": archivo is None,
    }


def _limpiar():
    import keras

    keras.backend.clear_session()
    gc.collect()


def main() -> int:
    import sys

    import tensorflow as tf

    solo = sys.argv[1] if len(sys.argv) > 1 else None
    print("TensorFlow", tf.__version__, flush=True)
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
    SALIDA.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(RAIZ / "clases.json", ASSETS / "clases.json")
    shutil.copyfile(
        RAIZ / "regional_colombia" / "resultados" / "clases_29.json",
        ASSETS / "clases_29.json",
    )
    imagenes = _imagenes()
    print("Imágenes de chequeo:", ", ".join(nombre for nombre, _ in imagenes), flush=True)
    reporte = []
    fichas = []
    fallos = 0
    for info in MODELOS:
        if solo and info["id"] != solo:
            continue
        print(f"\n== {info['id']} ==", flush=True)
        destino = SALIDA / info["tflite"]
        entrada = {
            "id": info["id"],
            "ok": False,
            "error": None,
            "via": None,
            "saltadas": [],
            "mb": None,
            "chequeos": [],
        }
        if not info["ruta"].is_file():
            entrada["error"] = f"No está el archivo {info['ruta']}"
            print(entrada["error"], flush=True)
            reporte.append(entrada)
            fallos += 1
            continue
        modelo = None
        inferencia = None
        try:
            modelo = _cargar(info["ruta"])
            print("capas:", flush=True)
            for capa in modelo.layers:
                extra = ""
                if capa.__class__.__name__ == "TFOpLambda":
                    extra = f" fn={getattr(capa, '_fn', None)!r}"
                print(f"  - {capa.name} {capa.__class__.__name__}{extra}", flush=True)
            sonda = imagenes[0][1]
            a1, _ = _predecir(modelo, sonda)
            a2, _ = _predecir(modelo, sonda)
            inferencia, saltadas = _modelo_inferencia(modelo)
            entrada["saltadas"] = saltadas
            print("sin augmentación:", ", ".join(saltadas) or "(no había)", flush=True)
            forma = tuple(inferencia.output_shape)
            print("salida", forma, flush=True)
            if forma[-1] != info["salida"]:
                raise RuntimeError(f"Salida {forma[-1]} y se esperaban {info['salida']} clases")
            b, _ = _predecir(inferencia, sonda)
            if a1 == a2 and a1 != b:
                raise RuntimeError(
                    f"Quitar la augmentación cambió el argmax de Keras ({a1} -> {b}) en {imagenes[0][0]}"
                )
            if a1 != a2:
                print("El modelo original no es determinista en inferencia; se compara contra el grafo sin augmentación.", flush=True)
            via, chequeos = _convertir(inferencia, destino, imagenes)
            entrada["via"] = via
            entrada["chequeos"] = chequeos
            entrada["mb"] = round(destino.stat().st_size / (1024 * 1024), 2)
            print(f"TFLite {via} {entrada['mb']} MB", flush=True)
            entrada["ok"] = True
            fichas.append(_ficha_publica(info, f"modelos/{info['tflite']}"))
            print("argmax coincide", flush=True)
        except Exception as exc:
            fallos += 1
            entrada["error"] = str(exc)
            print("FALLO", exc, flush=True)
            traceback.print_exc()
            if destino.exists():
                destino.unlink()
        finally:
            reporte.append(entrada)
            del modelo
            del inferencia
            _limpiar()

    if solo is None:
        for remoto in REMOTOS:
            fichas.append(remoto)
        catalogo = {"modelos": fichas}
        (ASSETS / "catalogo.json").write_text(
            json.dumps(catalogo, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (RAIZ / "android" / "tools" / "reporte_exportacion.json").write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    total = sum(item["mb"] or 0 for item in reporte)
    print(f"\nListos {sum(1 for item in reporte if item['ok'])} de {len(MODELOS)}. {total:.1f} MB.", flush=True)
    return 1 if fallos else 0


if __name__ == "__main__":
    raise SystemExit(main())
