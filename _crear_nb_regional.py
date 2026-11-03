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


md("""# Entrenamiento regional: plantas de Colombia / Tolima

Responde al comentario del profesor: Herbify es de la India. Este notebook entrena **MobileNetV2** sobre las especies de `especies.json` (medicinales de uso local, emblemáticas andinas y frutales de huerta).

Fotos:
- iNaturalist (Research Grade, licencias abiertas), Tolima primero y Colombia si faltan
- Herbify se mezcla solo donde hay la misma especie (guayaba y ruda), para tener mas imagenes

No sustituye el experimento de 57 clases. Es el subset de impacto regional. El modelo de 10 clases queda archivado en `resultados/modelo_regional_10clases.h5`.""")

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
Luego se vuelve a hacer el split 70/15/15 con semilla 42, solo de las especies de `especies.json`.""")

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

train_ds = train_ds.cache().shuffle(2000).prefetch(AUTOTUNE)
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
fig.suptitle(f"MobileNetV2 — {NUM_CLASES} especies Colombia/Tolima")
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
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASES, yticklabels=CLASES)
plt.xticks(rotation=90, ha="right", fontsize=7)
plt.yticks(fontsize=7)
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

md(r'''
## 9. Regularización L1 y L2

Las secciones 1 a 8 entrenan la línea base: Dropout 0.4, augmentación y parada temprana, sin multa sobre los pesos. Esa corrida es la que publica la web (77.4% en test) y queda en `modelo_regional.h5`.

Esta sección repite el mismo MobileNetV2 y las mismas dos fases, y suma L1 y L2 solo en el kernel de la capa final:

```
entropía cruzada + 1e-5 * ||W||1 + 1e-4 * ||W||2²
```

L2 = 1e-4 encoge pesos grandes. L1 = 1e-5 es menor, para que la cabeza siga pudiendo separar 20 especies. El sesgo no lleva multa. La base ImageNet tampoco: una L1 sobre esos filtros los empujaría a cero.

El mismo cabezal está en `regional_colombia/entrenar_regional.py`. El resultado se guarda aparte (`modelo_regional_l1l2.h5`, curvas, matriz y métricas con el mismo sufijo) y no reemplaza el archivo que usa la web. Estas celdas quedan sin ejecutar.
''')

code(r'''
L1 = 1e-5
L2 = 1e-4

augmentacion_l1l2 = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.25),
    tf.keras.layers.RandomZoom(0.25),
    tf.keras.layers.RandomTranslation(0.15, 0.15),
    tf.keras.layers.RandomContrast(0.3),
    tf.keras.layers.RandomBrightness(0.25, value_range=(0, 255)),
], name="augmentacion_l1l2")

base_l1l2 = tf.keras.applications.MobileNetV2(
    input_shape=IMG + (3,), include_top=False, weights="imagenet")
base_l1l2.trainable = False

entradas_l1l2 = tf.keras.Input(shape=IMG + (3,))
x = augmentacion_l1l2(entradas_l1l2)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_l1l2(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.4)(x)
salidas_l1l2 = tf.keras.layers.Dense(
    NUM_CLASES,
    activation="softmax",
    kernel_regularizer=tf.keras.regularizers.L1L2(l1=L1, l2=L2),
    name="cabeza",
)(x)
modelo_l1l2 = tf.keras.Model(entradas_l1l2, salidas_l1l2, name="regional_mobilenetv2_l1l2")
modelo_l1l2.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
print("Multas de la cabeza:", len(modelo_l1l2.losses))
modelo_l1l2.summary()
''')

code(r'''
hist1_l1l2 = modelo_l1l2.fit(
    train_ds, validation_data=val_ds, epochs=25, callbacks=callbacks)

base_l1l2.trainable = True
for capa in base_l1l2.layers[:-40]:
    capa.trainable = False
modelo_l1l2.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
print("Fase 2: multa de la cabeza sigue en", len(modelo_l1l2.losses), "terminos")
hist2_l1l2 = modelo_l1l2.fit(
    train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
''')

code(r'''
acc = hist1_l1l2.history["accuracy"] + hist2_l1l2.history["accuracy"]
vacc = hist1_l1l2.history["val_accuracy"] + hist2_l1l2.history["val_accuracy"]
loss = hist1_l1l2.history["loss"] + hist2_l1l2.history["loss"]
vloss = hist1_l1l2.history["val_loss"] + hist2_l1l2.history["val_loss"]
corte = len(hist1_l1l2.history["accuracy"])

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(acc, label="train"); ax[0].plot(vacc, label="val")
ax[0].axvline(corte, ls="--", c="gray"); ax[0].set_title("Exactitud"); ax[0].legend()
ax[1].plot(loss, label="train"); ax[1].plot(vloss, label="val")
ax[1].axvline(corte, ls="--", c="gray"); ax[1].set_title("Perdida"); ax[1].legend()
fig.suptitle(f"MobileNetV2 L1+L2 — {NUM_CLASES} especies")
plt.tight_layout()
fig.savefig(SALIDA / "curvas_regional_l1l2.png", dpi=120)
plt.show()

y_true_l1l2 = np.concatenate([y.numpy() for _, y in test_ds])
y_prob_l1l2 = modelo_l1l2.predict(test_ds)
y_pred_l1l2 = y_prob_l1l2.argmax(axis=1)
reporte_l1l2 = classification_report(y_true_l1l2, y_pred_l1l2, target_names=CLASES, digits=3)
print(reporte_l1l2)
(SALIDA / "reporte_regional_l1l2.txt").write_text(reporte_l1l2, encoding="utf-8")

cm = confusion_matrix(y_true_l1l2, y_pred_l1l2)
plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=CLASES, yticklabels=CLASES)
plt.xticks(rotation=90, ha="right", fontsize=7)
plt.yticks(fontsize=7)
plt.xlabel("Prediccion"); plt.ylabel("Real")
plt.title("Matriz de confusion (test) — L1+L2")
plt.tight_layout()
plt.savefig(SALIDA / "matriz_regional_l1l2.png", dpi=120)
plt.show()

modelo_l1l2.save(str(SALIDA / "modelo_regional_l1l2.h5"))
metricas_l1l2 = {
    "accuracy_test": float(np.mean(y_true_l1l2 == y_pred_l1l2)),
    "imagenes_test": int(len(y_true_l1l2)),
    "errores_test": int(np.sum(y_true_l1l2 != y_pred_l1l2)),
    "clases": CLASES,
    "dataset": "iNaturalist Colombia/Tolima + Herbify (guayaba, ruda)",
    "l1": L1,
    "l2": L2,
    "regularizacion": "L1L2 solo en el kernel de la Dense final",
}
(SALIDA / "metricas_regional_l1l2.json").write_text(
    json.dumps(metricas_l1l2, ensure_ascii=False, indent=2), encoding="utf-8")
print("Guardado", SALIDA / "modelo_regional_l1l2.h5")
print("Accuracy test L1+L2:", round(metricas_l1l2["accuracy_test"], 3))
''')

code(r'''
def kernel_cabeza(red):
    densas = [c for c in red.layers if isinstance(c, tf.keras.layers.Dense)]
    if not densas:
        raise RuntimeError("El modelo no tiene capa Dense")
    return densas[-1].kernel.numpy()

def normas_kernel(w):
    return float(np.sum(np.abs(w))), float(np.sum(np.square(w)))

w_l1l2 = kernel_cabeza(modelo_l1l2)
n1, n2 = normas_kernel(w_l1l2)
print(f"L1+L2   ||W||1={n1:.4f}   ||W||2^2={n2:.4f}")
print(f"Multa sobre esos pesos: {L1 * n1 + L2 * n2:.6f}")

ruta_base = SALIDA / "modelo_regional.h5"
if ruta_base.is_file():
    modelo_base = tf.keras.models.load_model(ruta_base)
    w_base = kernel_cabeza(modelo_base)
    b1, b2 = normas_kernel(w_base)
    _, acc_base = modelo_base.evaluate(test_ds, verbose=0)
    _, acc_reg = modelo_l1l2.evaluate(test_ds, verbose=0)
    print(f"Base    ||W||1={b1:.4f}   ||W||2^2={b2:.4f}")
    print(f"Accuracy test base (sin multa de pesos): {acc_base:.3f}")
    print(f"Accuracy test L1+L2:                     {acc_reg:.3f}")
else:
    print("No esta modelo_regional.h5. La linea base no se puede comparar.")
''')

md(r'''
## 10. Defensa técnica del modelo

### Arquitectura

El clasificador es MobileNetV2 con pesos de ImageNet y una cabeza propia. Es el mismo esqueleto que el experimento Herbify, así la diferencia entre 91.2% y 77.4% se lee con la red fija: cambia el conjunto de fotos y el dominio.

La cabeza es pequeña a propósito. `GlobalAveragePooling` resume cada mapa de 1280 canales a un vector, y la `Dense` final tiene 1280 × 20 pesos. Una capa que aplanara el mapa espacial tendría muchos más parámetros y más sitio para memorizar el fondo de la foto.

El entrenamiento va en dos fases. En la primera la base queda congelada y Adam usa 1e-3: solo aprende la cabeza. En la segunda se abren las últimas 40 capas con Adam a 1e-5, para ajustar los filtros de ImageNet sin reescribirlos de un golpe.

Dropout 0.4 apaga activaciones al azar en cada paso de entrenamiento. L1 y L2 suman otra palanca: una multa según el tamaño de los pesos, en todos los pasos. La pérdida de esta variante es entropía cruzada + 1e-5 · ||W||₁ + 1e-4 · ||W||₂². L2 encoge pesos grandes. L1, más chica, empuja pesos pequeños hacia cero. La multa va solo en el kernel de la `Dense` final. El sesgo queda libre. La base ImageNet queda libre: una L1 ahí empujaría a cero filtros que ya sirven para ver hojas.

Con pesos moderados esa multa es pequeña al lado de la entropía cruzada. Crece si la cabeza se infla. La augmentación y la parada temprana siguen en su sitio: una actúa sobre las fotos y la otra sobre cuántas épocas corre el ajuste.

### Datos

Son 20 especies de `especies.json`: medicinales de uso local, emblemáticas andinas y frutales de huerta. El curso pidió impacto en la región. Herbify está armado con plantas de la India.

Las fotos salen de iNaturalist, grado de investigación y licencias abiertas. La búsqueda empieza en Tolima y, cuando faltan imágenes, sigue en Colombia y Suramérica. Herbify solo se mezcla en guayaba y ruda, que son la misma especie en los dos conjuntos.

El reparto es 70/15/15 con semilla 42, a 224×224. El test no entra al ajuste de pesos ni a la parada temprana. La augmentación (giros, rotación, zoom, desplazamiento, contraste y brillo) corre solo en entrenamiento.

Son fotos de campo, con fondos distintos. En Herbify el fondo es de estudio. Por eso la accuracy regional, 77.4% en 1347 fotos de test, queda por debajo del 91.2% de Herbify.

### Métricas

La accuracy de test es el número que se pone al lado de Herbify: 77.4%, 304 errores de 1347.

El F1 por especie muestra que ese error no está repartido parejo. Frailejón queda en 0.971 y palma de cera en 0.956. Roble andino queda en 0.576, yarumo en 0.619 y matarratón en 0.634. La matriz de confusión enseña esos cruces entre hojas parecidas. Un solo promedio los esconde.

La parada temprana mira `val_accuracy`, con paciencia 6 en este notebook. El criterio es acertar en validación. La pérdida de entrenamiento no decide cuándo parar.

Las cifras de la web y el archivo `modelo_regional.h5` corresponden al modelo sin multa sobre los pesos. `modelo_regional_l1l2.h5` es la variante de la sección 9. Hasta correr esas celdas, la comparación de normas y de accuracy queda pendiente.
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
