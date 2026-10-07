import json
from pathlib import Path

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3.10 GPU (NoElectiva)",
            "language": "python",
            "name": "noelectiva-gpu",
        },
        "language_info": {"name": "python", "version": "3.10.11"},
    },
    "cells": [],
}


def md(text):
    nb["cells"].append({"cell_type": "markdown", "metadata": {}, "source": [line + "\n" for line in text.strip("\n").split("\n")]})


def code(text):
    src = text.strip("\n") + "\n"
    nb["cells"].append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": [line + "\n" for line in src.split("\n")]})


md("""# Entrenamiento regional: 10 plantas medicinales de Colombia / Tolima

Responde al comentario del profesor: Herbify es de la India. Este notebook entrena **MobileNetV2** sobre 10 especies de uso medicinal local.

Fotos:
- iNaturalist (Research Grade, licencias abiertas), Tolima primero y Colombia si faltan
- Herbify se mezcla solo donde hay la misma especie (guayaba y ruda), para tener mas imagenes

No sustituye el experimento de 57 clases. Es el subset de impacto regional.""")

md("## 1. Entorno y rutas")

code(r'''
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from pathlib import Path
import json, random, shutil
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

PROYECTO = Path.cwd()
if not (PROYECTO / "regional_colombia").is_dir() and (PROYECTO / "especies.json").exists():
    REG = PROYECTO
    RAIZ = PROYECTO.parent
else:
    REG = PROYECTO / "regional_colombia"
    RAIZ = PROYECTO

CRUDO = REG / "crudo"
DATASET = REG / "dataset"
SALIDA = REG / "resultados"
SALIDA.mkdir(parents=True, exist_ok=True)

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_visible_devices(gpus[0], "GPU")
    tf.config.experimental.set_memory_growth(gpus[0], True)

print("TensorFlow", tf.__version__)
print("GPU:", tf.config.list_physical_devices("GPU"))
print("Regional:", REG)
''')

md("""## 2. Combinar iNaturalist + Herbify (mismas especies)

De Herbify solo entran clases que coinciden: *Psidium guajava* y *Ruta graveolens*.
Luego se vuelve a hacer el split 70/15/15 con semilla 42, solo de las 10 especies.""")

code(r'''
cfg = json.loads((REG / "especies.json").read_text(encoding="utf-8"))
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

def carpeta_clase(comun, cientifico):
    seguro = "".join(c if c.isalnum() or c in "-_" else "_" for c in comun.lower())
    return f"{cientifico.replace(' ', '_')}__{seguro}"

CLASES_OK = {carpeta_clase(e["comun"], e["cientifico"]) for e in cfg["especies"]}
print("Clases objetivo:", len(CLASES_OK))

HERBIFY = Path.home() / ".cache/kagglehub/datasets/ai4a-lab/herb-plant-classification-dataset/versions/1/Herbify-Dataset"
guayaba_orig = HERBIFY / "Psidium guajava (Guava)"
ruta_orig = HERBIFY / "Ruta graveolens (Common rue_naagdalli)"
EXTRAS = {
    "Psidium_guajava__guayaba": [guayaba_orig] if guayaba_orig.is_dir() else [
        RAIZ / "dataset" / "train" / "Psidium guajava (Guava)",
        RAIZ / "dataset" / "val" / "Psidium guajava (Guava)",
        RAIZ / "dataset" / "test" / "Psidium guajava (Guava)",
    ],
    "Ruta_graveolens__ruda": [ruta_orig] if ruta_orig.is_dir() else [],
}

copiadas = {}
for clase, fuentes in EXTRAS.items():
    dest = CRUDO / clase
    dest.mkdir(parents=True, exist_ok=True)
    n = 0
    vistos = {p.name.lower() for p in dest.glob("*")}
    for src in fuentes:
        if not src.is_dir():
            continue
        for f in src.iterdir():
            if f.suffix.lower() not in EXTS:
                continue
            nombre = f"herbify_{src.parent.name}_{f.name}"
            if len(nombre) > 120:
                nombre = f"herbify_{abs(hash(str(f)))}{f.suffix.lower()}"
            if nombre.lower() in vistos:
                continue
            try:
                img = Image.open(f).convert("RGB")
                img.thumbnail((1024, 1024))
                img.save(dest / Path(nombre).with_suffix(".jpg").name, quality=90)
                vistos.add(nombre.lower())
                n += 1
            except OSError:
                continue
    copiadas[clase] = n
    print(f"Herbify -> {clase}: +{n} fotos")

print(copiadas)
''')

md("## 3. Split train / val / test")

code(r'''
random.seed(42)
if DATASET.exists():
    shutil.rmtree(DATASET)

P_TRAIN, P_VAL = 0.70, 0.15
resumen = []
for clase in sorted(CLASES_OK):
    src = CRUDO / clase
    if not src.is_dir():
        print("Falta crudo:", clase)
        continue
    archivos = [f for f in src.iterdir() if f.suffix.lower() in EXTS]
    random.shuffle(archivos)
    n = len(archivos)
    i1, i2 = int(n * P_TRAIN), int(n * (P_TRAIN + P_VAL))
    partes = {"train": archivos[:i1], "val": archivos[i1:i2], "test": archivos[i2:]}
    for split, lista in partes.items():
        dest = DATASET / split / clase
        dest.mkdir(parents=True, exist_ok=True)
        for f in lista:
            Image.open(f).convert("RGB").resize((224, 224)).save(dest / f.with_suffix(".jpg").name, quality=90)
    resumen.append((clase, n, len(partes["train"]), len(partes["val"]), len(partes["test"])))

print(f"{'clase':<50}{'total':>7}{'train':>7}{'val':>6}{'test':>6}")
for r in resumen:
    print(f"{r[0]:<50}{r[1]:>7}{r[2]:>7}{r[3]:>6}{r[4]:>6}")
print("Total imagenes:", sum(x[1] for x in resumen), "| clases:", len(resumen))
''')

md("## 4. Pipeline y augmentacion")

code(r'''
IMG = (224, 224)
BATCH = 16

def cargar(split, shuffle):
    return tf.keras.utils.image_dataset_from_directory(
        DATASET / split, image_size=IMG, batch_size=BATCH,
        shuffle=shuffle, seed=42, label_mode="int")

train_ds = cargar("train", True)
val_ds = cargar("val", False)
test_ds = cargar("test", False)
CLASES = train_ds.class_names
NUM_CLASES = len(CLASES)
print("Clases:", CLASES)

AUTOTUNE = tf.data.AUTOTUNE
augmentacion = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomZoom(0.25),
    tf.keras.layers.RandomTranslation(0.15, 0.15),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
], name="augmentacion")

train_ds = train_ds.cache().shuffle(500).prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)
test_ds = test_ds.cache().prefetch(AUTOTUNE)
''')

md("## 5. Modelo MobileNetV2 — fase 1 (base congelada)")

code(r'''
base = tf.keras.applications.MobileNetV2(
    input_shape=IMG + (3,), include_top=False, weights="imagenet")
base.trainable = False

entradas = tf.keras.Input(shape=IMG + (3,))
x = augmentacion(entradas)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
salidas = tf.keras.layers.Dense(NUM_CLASES, activation="softmax")(x)
modelo = tf.keras.Model(entradas, salidas, name="regional_mobilenetv2")
modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
modelo.summary()
''')

code(r'''
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]
hist1 = modelo.fit(train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks)
''')

md("## 6. Fase 2 — fine-tuning")

code(r'''
base.trainable = True
for capa in base.layers[:-40]:
    capa.trainable = False
modelo.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
               loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
hist2 = modelo.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
''')

md("## 7. Curvas y evaluacion")

code(r'''
acc = hist1.history["accuracy"] + hist2.history["accuracy"]
vacc = hist1.history["val_accuracy"] + hist2.history["val_accuracy"]
loss = hist1.history["loss"] + hist2.history["loss"]
vloss = hist1.history["val_loss"] + hist2.history["val_loss"]
corte = len(hist1.history["accuracy"])

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(acc, label="train"); ax[0].plot(vacc, label="val")
ax[0].axvline(corte, ls="--", c="gray"); ax[0].set_title("Exactitud"); ax[0].legend()
ax[1].plot(loss, label="train")
ax[1].plot(vloss, label="val")
ax[1].axvline(corte, ls="--", c="gray"); ax[1].set_title("Perdida"); ax[1].legend()
fig.suptitle("MobileNetV2 — 10 especies Colombia/Tolima")
plt.tight_layout()
fig.savefig(SALIDA / "curvas_regional.png", dpi=120)
plt.show()
''')

code(r'''
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_prob = modelo.predict(test_ds)
y_pred = y_prob.argmax(axis=1)
reporte = classification_report(y_true, y_pred, target_names=CLASES, digits=3)
print(reporte)
(SALIDA / "reporte_regional.txt").write_text(reporte, encoding="utf-8")

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASES, yticklabels=CLASES)
plt.xticks(rotation=90, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.xlabel("Prediccion"); plt.ylabel("Real")
plt.title("Matriz de confusion (test) — subset regional")
plt.tight_layout()
plt.savefig(SALIDA / "matriz_regional.png", dpi=120)
plt.show()
print(f"{int(np.sum(y_true != y_pred))} errores de {len(y_true)} imagenes de prueba")
''')

md("## 8. Guardar modelo")

code(r'''
modelo.save(str(SALIDA / "modelo_regional.h5"))
(SALIDA / "clases.json").write_text(json.dumps(CLASES, ensure_ascii=False, indent=2), encoding="utf-8")
metricas = {
    "accuracy_test": float(np.mean(y_true == y_pred)),
    "imagenes_test": int(len(y_true)),
    "errores_test": int(np.sum(y_true != y_pred)),
    "clases": CLASES,
    "dataset": "iNaturalist Colombia/Tolima + Herbify (guayaba, ruda)",
}
(SALIDA / "metricas_regional.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
print("Guardado en", SALIDA)
print("Accuracy test:", round(metricas["accuracy_test"], 3))
''')

# fix last newlines becoming extra empty lines in source arrays
for cell in nb["cells"]:
    src = cell["source"]
    if src and src[-1] == "\n":
        src.pop()
    if src and not src[-1].endswith("\n"):
        src[-1] += "\n"

out = Path(r"D:\juandmm1233\NoElectivaIII\entrenamiento_regional_colombia.ipynb")
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Notebook:", out)
print("Celdas:", len(nb["cells"]))
