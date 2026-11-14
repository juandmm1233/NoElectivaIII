"""
MobileNetV2 estudiante del ResNet50 regional ya entrenado.

Misma receta de datos que el MobileNetV2 de 29 clases:
10201 train / 2178 val / 2201 test, semilla 42.
No reemplaza modelo_regional_29clases.h5 ni modelo_resnet50_regional_29.h5.

La primera accion es compilar el ResNet50. Despues, si se pasa --prueba,
corre una sola epoca de la fase 1. Sin esa bandera corre el entrenamiento completo.

    python entrenar_estudiante_v2.py --prueba
    python entrenar_estudiante_v2.py
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["DML_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

PROYECTO = Path(__file__).resolve().parent
RAIZ = PROYECTO.parent
DATASET = PROYECTO / "dataset_29"
RESULTADOS = PROYECTO / "resultados"
RUTA_PROFESOR = RESULTADOS / "modelo_resnet50_regional_29.h5"
IMG = (224, 224)
SEMILLA = 42
TEMPERATURA = 4.0
PESO_ETIQUETA = 0.5
PREFIJO = "mobilenetv2_estudiante_regional"
ESPERADO = {"train": 10201, "val": 2178, "test": 2201}


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


def compilar_resnet():
    if not RUTA_PROFESOR.is_file():
        raise SystemExit(f"No esta el profesor: {RUTA_PROFESOR}")
    print("Cargando ResNet50 regional para compilarlo antes de cualquier otro paso.")
    profesor = tf.keras.models.load_model(RUTA_PROFESOR, compile=False)
    profesor.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("ResNet50 compilado.", profesor.name, "salida", profesor.output_shape)
    return profesor


def cargar_datos(batch):
    for split, cantidad in ESPERADO.items():
        carpeta = DATASET / split
        if not carpeta.is_dir():
            raise SystemExit(f"No esta {carpeta}")
        fotos = sum(
            1
            for ruta in carpeta.rglob("*")
            if ruta.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        if fotos != cantidad:
            raise SystemExit(f"{split} tiene {fotos} fotos; se esperaban {cantidad}.")
        print(f"{split}: {fotos}")

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
    clases = list(train_ds.class_names)
    if clases != list(val_ds.class_names) or clases != list(test_ds.class_names):
        raise SystemExit("Las clases de train, val y test no coinciden.")
    autotune = tf.data.AUTOTUNE
    cache = RESULTADOS / "cache_resnet50_29"
    cache.mkdir(parents=True, exist_ok=True)
    train_ds = train_ds.cache(str(cache / f"train_b{batch}")).shuffle(800, seed=SEMILLA).prefetch(autotune)
    val_ds = val_ds.cache(str(cache / f"val_b{batch}")).prefetch(autotune)
    test_ds = test_ds.cache(str(cache / f"test_b{batch}")).prefetch(autotune)
    return train_ds, val_ds, test_ds, clases


def verificar_profesor(profesor, val_ds, num_clases):
    salida = profesor.output_shape
    if salida[-1] != num_clases:
        raise SystemExit(f"ResNet50 tiene salida {salida} y el dataset tiene {num_clases} clases.")
    for imagenes, _etiquetas in val_ds.take(1):
        pred = profesor(imagenes, training=False)
        print("Lote de comprobacion del ResNet50 compilado:", tuple(pred.shape))
        break


def construir_estudiante(num_clases):
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
    salidas = tf.keras.layers.Dense(num_clases, activation="softmax", name="cabeza")(x)
    estudiante = tf.keras.Model(entradas, salidas, name="mobilenetv2_estudiante")
    return estudiante, base


def perdida_transferencia(etiquetas, pred_estudiante, pred_profesor):
    dura = tf.keras.losses.sparse_categorical_crossentropy(etiquetas, pred_estudiante)
    blanda_profesor = tf.nn.softmax(tf.math.log(pred_profesor + 1e-8) / TEMPERATURA)
    blanda_estudiante = tf.nn.softmax(tf.math.log(pred_estudiante + 1e-8) / TEMPERATURA)
    blanda = tf.keras.losses.kl_divergence(blanda_profesor, blanda_estudiante) * (TEMPERATURA ** 2)
    return PESO_ETIQUETA * tf.reduce_mean(dura) + (1.0 - PESO_ETIQUETA) * tf.reduce_mean(blanda)


class Alumno(tf.keras.Model):
    def __init__(self, estudiante, profesor):
        super().__init__(name="alumno_mobilenetv2")
        self.estudiante = estudiante
        self.profesor = profesor

    def call(self, inputs, training=False):
        return self.estudiante(inputs, training=training)

    def train_step(self, data):
        imagenes, etiquetas = data
        with tf.GradientTape() as cinta:
            pred_profesor = tf.stop_gradient(self.profesor(imagenes, training=False))
            pred_estudiante = self.estudiante(imagenes, training=True)
            perdida = perdida_transferencia(etiquetas, pred_estudiante, pred_profesor)
        variables = self.estudiante.trainable_variables
        gradientes = cinta.gradient(perdida, variables)
        self.optimizer.apply_gradients(zip(gradientes, variables))
        self.compiled_metrics.update_state(etiquetas, pred_estudiante)
        resultados = {metrica.name: metrica.result() for metrica in self.metrics}
        resultados["loss"] = perdida
        return resultados

    def test_step(self, data):
        imagenes, etiquetas = data
        pred = self.estudiante(imagenes, training=False)
        perdida = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(etiquetas, pred))
        self.compiled_metrics.update_state(etiquetas, pred)
        resultados = {metrica.name: metrica.result() for metrica in self.metrics}
        resultados["loss"] = perdida
        return resultados


def callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=6,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
        ),
    ]


def abrir_fase(estudiante, base, alumno, fase):
    if fase == 1:
        base.trainable = False
        optimizador = tf.keras.optimizers.Adam(1e-3)
        print("Fase 1: MobileNetV2 congelado, aprende la cabeza con ResNet50")
    else:
        base.trainable = True
        for capa in base.layers[:-40]:
            capa.trainable = False
        optimizador = tf.keras.optimizers.Adam(1e-5)
        print("Fase 2: ultimas 40 capas de MobileNetV2")
    alumno.compile(optimizer=optimizador, metrics=["accuracy"])
    return alumno


def guardar_curvas(hist1, hist2):
    historial = {}
    for clave in ("accuracy", "val_accuracy", "loss", "val_loss"):
        historial[clave] = hist1.history.get(clave, []) + hist2.history.get(clave, [])
    corte = len(hist1.history.get("accuracy", []))
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(historial["accuracy"], label="train")
    ax[0].plot(historial["val_accuracy"], label="val")
    ax[0].axvline(max(corte - 0.5, 0), color="gray", linestyle="--", label="fine-tuning")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoca")
    ax[0].legend()
    ax[1].plot(historial["loss"], label="train")
    ax[1].plot(historial["val_loss"], label="val")
    ax[1].axvline(max(corte - 0.5, 0), color="gray", linestyle="--", label="fine-tuning")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoca")
    ax[1].legend()
    fig.suptitle("MobileNetV2 estudiante de ResNet50 regional")
    fig.tight_layout()
    ruta = RESULTADOS / f"curvas_{PREFIJO}.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    print("Curvas:", ruta)
    return historial, corte


def evaluar(estudiante, test_ds, clases):
    if not getattr(estudiante, "optimizer", None):
        estudiante.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    perdida, accuracy = estudiante.evaluate(test_ds, verbose=1)
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_prob = estudiante.predict(test_ds, verbose=1)
    y_pred = y_prob.argmax(axis=1)
    top3 = float(np.mean(tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_prob, k=3).numpy()))
    reporte = classification_report(y_true, y_pred, target_names=clases, digits=3, output_dict=True)
    texto = classification_report(y_true, y_pred, target_names=clases, digits=3)
    print(texto)
    (RESULTADOS / f"reporte_{PREFIJO}.txt").write_text(texto, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 16))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Greens",
        xticklabels=clases, yticklabels=clases, cbar=True, annot_kws={"size": 6},
    )
    plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=7)
    plt.xlabel("Predicho")
    plt.ylabel("Real")
    plt.title("Matriz de confusion - MobileNetV2 estudiante")
    plt.tight_layout()
    plt.savefig(RESULTADOS / f"matriz_{PREFIJO}.png", dpi=120)
    plt.close()
    return {
        "loss_test": float(perdida),
        "accuracy_test": float(accuracy),
        "top3_accuracy_test": top3,
        "f1_ponderado": round(float(reporte["weighted avg"]["f1-score"]), 3),
        "f1_por_clase": {clase: round(float(reporte[clase]["f1-score"]), 3) for clase in clases},
        "imagenes_test": int(len(y_true)),
        "errores_test": int(np.sum(y_true != y_pred)),
    }


def entrenar(prueba, batch, epochs_fase1, epochs_fase2):
    tf.keras.utils.set_random_seed(SEMILLA)
    configurar_gpu()
    RESULTADOS.mkdir(parents=True, exist_ok=True)
    profesor = compilar_resnet()
    profesor.trainable = False
    train_ds, val_ds, test_ds, clases = cargar_datos(batch)
    verificar_profesor(profesor, val_ds, len(clases))

    estudiante, base = construir_estudiante(len(clases))
    alumno = Alumno(estudiante, profesor)
    abrir_fase(estudiante, base, alumno, 1)

    if prueba:
        print("Prueba: 1 epoca de la fase 1, sin guardar el modelo final.")
        alumno.fit(train_ds, validation_data=val_ds, epochs=1, verbose=1)
        print("PRUEBA_OK")
        return

    t0 = time.perf_counter()
    hist1 = alumno.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_fase1,
        callbacks=callbacks(),
        verbose=1,
    )
    abrir_fase(estudiante, base, alumno, 2)
    hist2 = alumno.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_fase2,
        callbacks=callbacks(),
        verbose=1,
    )
    minutos = (time.perf_counter() - t0) / 60
    historial, _corte = guardar_curvas(hist1, hist2)

    ruta_modelo = RESULTADOS / f"modelo_{PREFIJO}.h5"
    estudiante.save(ruta_modelo, include_optimizer=False)
    metricas = evaluar(estudiante, test_ds, clases)
    metricas.update(
        {
            "modelo": "MobileNetV2 estudiante de ResNet50 regional",
            "profesor": "ResNet50 regional 29",
            "dataset": "iNaturalist Colombia/Tolima + fotos de huerta del equipo (dataset_29)",
            "parametros": int(estudiante.count_params()),
            "minutos_entrenamiento": round(minutos, 1),
            "tamano_mb": round(ruta_modelo.stat().st_size / (1024 * 1024), 2),
            "batch": batch,
            "epochs_fase1_pedidas": epochs_fase1,
            "epochs_fase2_pedidas": epochs_fase2,
            "epocas_fase1_corridas": len(hist1.history.get("accuracy", [])),
            "epocas_fase2_corridas": len(hist2.history.get("accuracy", [])),
            "mejor_val_accuracy": float(max(historial["val_accuracy"])) if historial["val_accuracy"] else None,
            "temperatura": TEMPERATURA,
            "peso_etiqueta": PESO_ETIQUETA,
            "imagen": 224,
            "split": "70/15/15",
            "train": ESPERADO["train"],
            "val": ESPERADO["val"],
            "test": ESPERADO["test"],
            "semilla": SEMILLA,
            "clases": len(clases),
            "regularizacion": "Dropout 0.4, augmentacion y parada temprana. Sin L1/L2. Destilacion compilada.",
        }
    )
    ruta_metricas = RESULTADOS / f"metricas_{PREFIJO}.json"
    ruta_metricas.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nListo MobileNetV2 estudiante.")
    print(f"  Accuracy test: {metricas['accuracy_test']:.3f}")
    print(f"  Top-3 test:    {metricas['top3_accuracy_test']:.3f}")
    print(f"  Tiempo:        {metricas['minutos_entrenamiento']} min")
    print(f"  Modelo:        {ruta_modelo}")


def solo_evaluar(batch):
    tf.keras.utils.set_random_seed(SEMILLA)
    configurar_gpu()
    ruta_modelo = RESULTADOS / f"modelo_{PREFIJO}.h5"
    if not ruta_modelo.is_file():
        raise SystemExit(f"No esta el estudiante guardado: {ruta_modelo}")
    print("Cargando el estudiante ya entrenado y compilando antes de evaluar.")
    estudiante = tf.keras.models.load_model(ruta_modelo, compile=False)
    _train, _val, test_ds, clases = cargar_datos(batch)
    metricas = evaluar(estudiante, test_ds, clases)
    metricas.update(
        {
            "modelo": "MobileNetV2 estudiante de ResNet50 regional",
            "profesor": "ResNet50 regional 29",
            "dataset": "iNaturalist Colombia/Tolima + fotos de huerta del equipo (dataset_29)",
            "parametros": int(estudiante.count_params()),
            "minutos_entrenamiento": 186.2,
            "tamano_mb": round(ruta_modelo.stat().st_size / (1024 * 1024), 2),
            "batch": batch,
            "epochs_fase1_pedidas": 25,
            "epochs_fase2_pedidas": 20,
            "epocas_fase1_corridas": 22,
            "epocas_fase2_corridas": 20,
            "temperatura": TEMPERATURA,
            "peso_etiqueta": PESO_ETIQUETA,
            "imagen": 224,
            "split": "70/15/15",
            "train": ESPERADO["train"],
            "val": ESPERADO["val"],
            "test": ESPERADO["test"],
            "semilla": SEMILLA,
            "clases": len(clases),
            "regularizacion": "Dropout 0.4, augmentacion y parada temprana. Sin L1/L2. Destilacion compilada.",
        }
    )
    ruta_metricas = RESULTADOS / f"metricas_{PREFIJO}.json"
    ruta_metricas.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nEvaluacion lista.")
    print(f"  Accuracy test: {metricas['accuracy_test']:.3f}")
    print(f"  Top-3 test:    {metricas['top3_accuracy_test']:.3f}")
    print(f"  Errores:       {metricas['errores_test']} / {metricas['imagenes_test']}")
    print(f"  Metricas:      {ruta_metricas}")


def main():
    parser = argparse.ArgumentParser(description="Entrena MobileNetV2 estudiante del ResNet50 regional.")
    parser.add_argument("--prueba", action="store_true", help="Solo 1 epoca de la fase 1, para detectar errores.")
    parser.add_argument("--solo-evaluar", action="store_true", help="Evalua el estudiante ya guardado, sin entrenar.")
    parser.add_argument("--epochs-fase1", type=int, default=25)
    parser.add_argument("--epochs-fase2", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    if args.solo_evaluar:
        solo_evaluar(args.batch)
        return
    entrenar(args.prueba, args.batch, args.epochs_fase1, args.epochs_fase2)


if __name__ == "__main__":
    main()
