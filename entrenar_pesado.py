"""
Entrenamiento pesado para clasificar plantas medicinales (Herbify).

Arquitectura por defecto: ResNet50 (ImageNet).
Tambien puedes usar DenseNet121 o EfficientNetB0.
ConvNeXt no esta en TensorFlow 2.10, que es el de este equipo.

Usa la carpeta dataset/ de la raiz, el mismo split que para_companero
(57 clases, 70/15/15, semilla 42).

Desde la raiz del proyecto, con el entorno de esta maquina:

    .\.venv\Scripts\activate
    python entrenar_pesado.py

Otra arquitectura de la lista:

    python entrenar_pesado.py --arquitectura densenet121
    python entrenar_pesado.py --arquitectura efficientnetb0

Prueba de 1 epoca (no vale para el informe):

    python entrenar_pesado.py --prueba
"""

import argparse
import json
import os
import time
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

PROYECTO = Path(__file__).resolve().parent
DATASET = PROYECTO / "dataset"
RESULTADOS = PROYECTO / "resultados_pesado"
IMG = (224, 224)
SEMILLA = 42


def crear_resnet50():
    return tf.keras.applications.ResNet50(
        input_shape=IMG + (3,), include_top=False, weights="imagenet"
    )


def crear_densenet121():
    return tf.keras.applications.DenseNet121(
        input_shape=IMG + (3,), include_top=False, weights="imagenet"
    )


def crear_efficientnetb0():
    return tf.keras.applications.EfficientNetB0(
        input_shape=IMG + (3,), include_top=False, weights="imagenet"
    )


ARQUITECTURAS = {
    "resnet50": (crear_resnet50, tf.keras.applications.resnet50.preprocess_input, "ResNet50"),
    "densenet121": (crear_densenet121, tf.keras.applications.densenet.preprocess_input, "DenseNet121"),
    "efficientnetb0": (crear_efficientnetb0, tf.keras.applications.efficientnet.preprocess_input, "EfficientNetB0"),
}


def configurar_gpu():
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError:
            pass
    print("TensorFlow", tf.__version__)
    if gpus:
        print("GPU detectada:", gpus)
    else:
        print("Sin GPU: el entrenamiento corre en CPU y va a tardar mucho mas.")


def cargar_splits(batch):
    for split in ("train", "val", "test"):
        carpeta = DATASET / split
        if not carpeta.is_dir():
            raise SystemExit(
                f"No esta la carpeta {carpeta}.\n"
                "El dataset partido debe estar en dataset/train, dataset/val y dataset/test."
            )

    def cargar(split, shuffle):
        return tf.keras.utils.image_dataset_from_directory(
            DATASET / split,
            image_size=IMG,
            batch_size=batch,
            shuffle=shuffle,
            seed=SEMILLA,
            label_mode="int",
        )

    train_ds = cargar("train", True)
    val_ds = cargar("val", False)
    test_ds = cargar("test", False)
    clases = train_ds.class_names
    if clases != val_ds.class_names or clases != test_ds.class_names:
        raise SystemExit("Las clases de train, val y test no coinciden.")

    print(f"Clases: {len(clases)}")
    return train_ds, val_ds, test_ds, clases


def preparar_pipeline(train_ds, val_ds, test_ds, batch, clave):
    autotune = tf.data.AUTOTUNE
    cache = RESULTADOS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    train_ds = train_ds.cache(str(cache / f"{clave}_train_b{batch}")).shuffle(500, seed=SEMILLA).prefetch(autotune)
    val_ds = val_ds.cache(str(cache / f"{clave}_val_b{batch}")).prefetch(autotune)
    test_ds = test_ds.cache(str(cache / f"{clave}_test_b{batch}")).prefetch(autotune)
    return train_ds, val_ds, test_ds


def construir_modelo(num_clases, crear_base, preproceso, nombre):
    augmentacion = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.25),
            tf.keras.layers.RandomZoom(0.25),
            tf.keras.layers.RandomTranslation(0.15, 0.15),
            tf.keras.layers.RandomContrast(0.3),
            tf.keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
        ],
        name="augmentacion",
    )

    base = crear_base()
    base.trainable = False

    entradas = tf.keras.Input(shape=IMG + (3,))
    x = augmentacion(entradas)
    x = preproceso(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    salidas = tf.keras.layers.Dense(num_clases, activation="softmax")(x)

    modelo = tf.keras.Model(entradas, salidas, name=f"plantas_{nombre.lower()}")
    return modelo, base


def callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
        ),
    ]


def unir_historial(hist1, hist2):
    unidos = {}
    for clave in ("accuracy", "val_accuracy", "loss", "val_loss"):
        unidos[clave] = hist1.history.get(clave, []) + hist2.history.get(clave, [])
    return unidos, len(hist1.history.get("accuracy", []))


def guardar_curvas(historial, corte, nombre, clave):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(historial["accuracy"], label="train")
    ax[0].plot(historial["val_accuracy"], label="val")
    ax[0].axvline(corte - 0.5, color="gray", linestyle="--", label="fine-tuning")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoca")
    ax[0].legend()
    ax[1].plot(historial["loss"], label="train")
    ax[1].plot(historial["val_loss"], label="val")
    ax[1].axvline(corte - 0.5, color="gray", linestyle="--", label="fine-tuning")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoca")
    ax[1].legend()
    fig.suptitle(nombre)
    fig.tight_layout()
    ruta = RESULTADOS / f"curvas_{clave}.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    print("Curvas:", ruta)


def evaluar(modelo, test_ds, clases, nombre, clave):
    inicio = time.perf_counter()
    perdida, accuracy = modelo.evaluate(test_ds, verbose=1)
    segundos_eval = time.perf_counter() - inicio

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_prob = modelo.predict(test_ds, verbose=1)
    y_pred = y_prob.argmax(axis=1)
    top3 = float(np.mean(tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_prob, k=3).numpy()))

    reporte = classification_report(y_true, y_pred, target_names=clases, digits=3)
    print(reporte)
    (RESULTADOS / f"reporte_{clave}.txt").write_text(reporte, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    lado = max(12, min(28, 0.28 * len(clases)))
    plt.figure(figsize=(lado, lado))
    sns.heatmap(cm, cmap="Blues", xticklabels=clases, yticklabels=clases, cbar=False)
    plt.xticks(fontsize=6, rotation=90)
    plt.yticks(fontsize=6)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title(f"Matriz de confusion - {nombre}")
    plt.tight_layout()
    plt.savefig(RESULTADOS / f"matriz_{clave}.png", dpi=120)
    plt.close()

    return {
        "loss_test": float(perdida),
        "accuracy_test": float(accuracy),
        "top3_accuracy_test": top3,
        "imagenes_test": int(len(y_true)),
        "errores_test": int(np.sum(y_true != y_pred)),
        "segundos_evaluacion": round(segundos_eval, 1),
    }


def _a_lista(valor):
    if isinstance(valor, tf.Tensor):
        return valor.numpy().tolist()
    if isinstance(valor, dict):
        return {clave: _a_lista(item) for clave, item in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [_a_lista(item) for item in valor]
    return valor


def guardar_modelo(modelo, ruta):
    vistos = set()

    def recorrer(capa):
        clase = capa.__class__
        if clase not in vistos:
            vistos.add(clase)
            original = clase.get_config

            def get_config(self, _original=original):
                return _a_lista(_original(self))

            clase.get_config = get_config
        if hasattr(capa, "layers"):
            for sub in capa.layers:
                recorrer(sub)

    recorrer(modelo)
    modelo.save(ruta, include_optimizer=False)


def main():
    parser = argparse.ArgumentParser(description="Entrena el modelo pesado sobre plantas medicinales.")
    parser.add_argument(
        "--arquitectura",
        choices=sorted(ARQUITECTURAS),
        default="resnet50",
        help="ResNet50 es la arquitectura pesada por defecto.",
    )
    parser.add_argument("--epochs-fase1", type=int, default=20)
    parser.add_argument("--epochs-fase2", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument(
        "--prueba",
        action="store_true",
        help="1 epoca por fase, solo para comprobar que la GPU puede entrenar.",
    )
    args = parser.parse_args()

    if args.prueba:
        args.epochs_fase1 = 1
        args.epochs_fase2 = 1
        print("Modo prueba: 1 epoca por fase. Estas metricas no van al informe.")

    crear_base, preproceso, nombre = ARQUITECTURAS[args.arquitectura]
    clave = args.arquitectura

    tf.keras.utils.set_random_seed(SEMILLA)
    configurar_gpu()
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    print("Arquitectura:", nombre)

    train_ds, val_ds, test_ds, clases = cargar_splits(args.batch)
    train_ds, val_ds, test_ds = preparar_pipeline(train_ds, val_ds, test_ds, args.batch, clave)

    modelo, base = construir_modelo(len(clases), crear_base, preproceso, nombre)
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("Fase 1: base ImageNet congelada")
    modelo.summary()

    t0 = time.perf_counter()
    hist1 = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_fase1,
        callbacks=callbacks(),
    )

    base.trainable = True
    for capa in base.layers[:-40]:
        capa.trainable = False
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("Fase 2: fine-tuning de las ultimas 40 capas")
    hist2 = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_fase2,
        callbacks=callbacks(),
    )
    minutos = (time.perf_counter() - t0) / 60

    historial, corte = unir_historial(hist1, hist2)
    guardar_curvas(historial, corte, nombre, clave)

    ruta_modelo = RESULTADOS / f"modelo_{clave}.h5"
    guardar_modelo(modelo, ruta_modelo)
    (RESULTADOS / "clases.json").write_text(
        json.dumps(clases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    metricas = evaluar(modelo, test_ds, clases, nombre, clave)
    metricas.update(
        {
            "modelo": nombre,
            "parametros": int(modelo.count_params()),
            "minutos_entrenamiento": round(minutos, 1),
            "tamano_mb": round(ruta_modelo.stat().st_size / (1024 * 1024), 2),
            "batch": args.batch,
            "epochs_fase1_pedidas": args.epochs_fase1,
            "epochs_fase2_pedidas": args.epochs_fase2,
            "epocas_fase1_corridas": len(hist1.history.get("accuracy", [])),
            "epocas_fase2_corridas": len(hist2.history.get("accuracy", [])),
            "mejor_val_accuracy": float(max(historial["val_accuracy"])) if historial["val_accuracy"] else None,
            "imagen": 224,
            "split": "70/15/15",
            "semilla": SEMILLA,
            "modo_prueba": bool(args.prueba),
        }
    )
    ruta_metricas = RESULTADOS / f"metricas_{clave}.json"
    ruta_metricas.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nListo.")
    print(f"  Accuracy test: {metricas['accuracy_test']:.3f}")
    print(f"  Top-3 test:    {metricas['top3_accuracy_test']:.3f}")
    print(f"  Tiempo:        {metricas['minutos_entrenamiento']} min")
    print(f"  Modelo:        {ruta_modelo}")
    print(f"  Metricas:      {ruta_metricas}")
    if args.prueba:
        print("Esto fue solo una prueba. Vuelve a correr sin --prueba para el resultado del trabajo.")


if __name__ == "__main__":
    main()
