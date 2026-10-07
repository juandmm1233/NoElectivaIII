"""
Entrenamiento regional: plantas medicinales de Colombia / Tolima.

Las fotos NO son Herbify. Primero corre:

    python descargar_inaturalist.py

Modelo: MobileNetV2 (mismo esquema de dos fases que el entrenamiento ligero).

    python entrenar_regional.py --prueba
    python entrenar_regional.py
    python entrenar_regional.py --batch 8
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
RESULTADOS = PROYECTO / "resultados"
IMG = (224, 224)
SEMILLA = 42
MODELO_NOMBRE = "MobileNetV2_regional"


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
        print("Sin GPU: el entrenamiento corre en CPU.")


def cargar_splits(batch):
    for split in ("train", "val", "test"):
        carpeta = DATASET / split
        if not carpeta.is_dir():
            raise SystemExit(
                f"No esta {carpeta}.\n"
                "Primero: python descargar_inaturalist.py"
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
    for c in clases:
        print(" -", c)
    return train_ds, val_ds, test_ds, clases


def preparar_pipeline(train_ds, val_ds, test_ds, batch):
    autotune = tf.data.AUTOTUNE
    cache = RESULTADOS / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    train_ds = train_ds.cache(str(cache / f"train_b{batch}")).shuffle(500, seed=SEMILLA).prefetch(autotune)
    val_ds = val_ds.cache(str(cache / f"val_b{batch}")).prefetch(autotune)
    test_ds = test_ds.cache(str(cache / f"test_b{batch}")).prefetch(autotune)
    return train_ds, val_ds, test_ds


def construir_modelo(num_clases):
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

    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    entradas = tf.keras.Input(shape=IMG + (3,))
    x = augmentacion(entradas)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    salidas = tf.keras.layers.Dense(num_clases, activation="softmax")(x)
    modelo = tf.keras.Model(entradas, salidas, name="plantas_regional")
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


def guardar_curvas(historial, corte):
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
    fig.suptitle(MODELO_NOMBRE)
    fig.tight_layout()
    ruta = RESULTADOS / "curvas_regional.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    print("Curvas:", ruta)


def evaluar(modelo, test_ds, clases):
    inicio = time.perf_counter()
    perdida, accuracy = modelo.evaluate(test_ds, verbose=1)
    segundos_eval = time.perf_counter() - inicio

    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_prob = modelo.predict(test_ds, verbose=1)
    y_pred = y_prob.argmax(axis=1)
    top3 = float(np.mean(tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_prob, k=3).numpy()))

    reporte = classification_report(y_true, y_pred, target_names=clases, digits=3)
    print(reporte)
    (RESULTADOS / "reporte_regional.txt").write_text(reporte, encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    lado = max(10, min(24, 0.35 * len(clases)))
    plt.figure(figsize=(lado, lado))
    sns.heatmap(cm, cmap="Greens", xticklabels=clases, yticklabels=clases, cbar=True)
    plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=7)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusion - subset Colombia/Tolima")
    plt.tight_layout()
    plt.savefig(RESULTADOS / "matriz_regional.png", dpi=120)
    plt.close()

    return {
        "loss_test": float(perdida),
        "accuracy_test": float(accuracy),
        "top3_accuracy_test": top3,
        "imagenes_test": int(len(y_true)),
        "errores_test": int(np.sum(y_true != y_pred)),
        "segundos_evaluacion": round(segundos_eval, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Entrena MobileNetV2 sobre plantas de Colombia/Tolima.")
    parser.add_argument("--epochs-fase1", type=int, default=20)
    parser.add_argument("--epochs-fase2", type=int, default=15)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--prueba", action="store_true")
    args = parser.parse_args()

    if args.prueba:
        args.epochs_fase1 = 1
        args.epochs_fase2 = 1
        print("Modo prueba: 1 epoca por fase.")

    tf.keras.utils.set_random_seed(SEMILLA)
    configurar_gpu()
    RESULTADOS.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds, clases = cargar_splits(args.batch)
    train_ds, val_ds, test_ds = preparar_pipeline(train_ds, val_ds, test_ds, args.batch)

    modelo, base = construir_modelo(len(clases))
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("Fase 1: base ImageNet congelada")
    modelo.summary()

    t0 = time.perf_counter()
    hist1 = modelo.fit(train_ds, validation_data=val_ds, epochs=args.epochs_fase1, callbacks=callbacks())

    base.trainable = True
    for capa in base.layers[:-40]:
        capa.trainable = False
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("Fase 2: fine-tuning de las ultimas 40 capas")
    hist2 = modelo.fit(train_ds, validation_data=val_ds, epochs=args.epochs_fase2, callbacks=callbacks())
    minutos = (time.perf_counter() - t0) / 60

    historial, corte = unir_historial(hist1, hist2)
    guardar_curvas(historial, corte)

    ruta_modelo = RESULTADOS / "modelo_regional.h5"
    modelo.save(ruta_modelo)
    (RESULTADOS / "clases.json").write_text(json.dumps(clases, ensure_ascii=False, indent=2), encoding="utf-8")

    metricas = evaluar(modelo, test_ds, clases)
    metricas.update(
        {
            "modelo": MODELO_NOMBRE,
            "dataset": "iNaturalist Colombia/Tolima (licencias abiertas, Research Grade)",
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
    ruta_metricas = RESULTADOS / "metricas_regional.json"
    ruta_metricas.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nListo subset regional.")
    print(f"  Accuracy test: {metricas['accuracy_test']:.3f}")
    print(f"  Top-3 test:    {metricas['top3_accuracy_test']:.3f}")
    print(f"  Tiempo:        {metricas['minutos_entrenamiento']} min")
    print(f"  Modelo:        {ruta_modelo}")
    if args.prueba:
        print("Esto fue solo una prueba. Corre sin --prueba para el informe.")


if __name__ == "__main__":
    main()
