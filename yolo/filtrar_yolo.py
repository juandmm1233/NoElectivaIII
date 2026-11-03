"""
Filtra con YOLO las fotos de dataset_final donde aparezcan personas,
frascos, vasos, carros, celulares, etc. NO borra nada: las mueve a
dataset_revisar para que las revises y devuelvas las que sí servían.

Uso (en la terminal, dentro de la carpeta del proyecto):
    pip install ultralytics
    python filtrar_yolo.py
"""
import shutil
from pathlib import Path

from ultralytics import YOLO

FINAL = Path("dataset_final")
REVISAR = Path("dataset_revisar")
CONFIANZA = 0.5  # sube a 0.6 si marca demasiadas fotos buenas

# Objetos (nombres del dataset COCO) que NO deberían salir en una foto de planta
MARCAR = {
    "person", "bottle", "cup", "bowl", "wine glass",
    "car", "truck", "bus", "motorcycle",
    "laptop", "cell phone", "tv", "book",
}

if not FINAL.exists():
    raise SystemExit("No encuentro dataset_final. Corre primero python limpiar.py")

model = YOLO("yolov8n.pt")  # se descarga solo la primera vez

total_movidas = 0
for carpeta in sorted(FINAL.iterdir()):
    if not carpeta.is_dir():
        continue

    fotos = sorted(p for p in carpeta.iterdir() if p.suffix.lower() == ".jpg")
    destino = REVISAR / carpeta.name
    movidas = 0

    for ruta in fotos:
        try:
            r = model.predict(str(ruta), conf=CONFIANZA, verbose=False)[0]
        except Exception:
            continue
        encontrados = {r.names[int(c)] for c in r.boxes.cls}
        marcados = encontrados & MARCAR
        if marcados:
            destino.mkdir(parents=True, exist_ok=True)
            motivo = sorted(marcados)[0].replace(" ", "")
            shutil.move(str(ruta), str(destino / f"{motivo}__{ruta.name}"))
            movidas += 1

    total_movidas += movidas
    print(f"{carpeta.name:12s} -> revisadas: {len(fotos)}, movidas a dataset_revisar: {movidas}, quedan: {len(fotos) - movidas}")

print(f"\nListo. Se movieron {total_movidas} fotos a dataset_revisar/.")
print("Míralas: si alguna sí era buena, devuélvela a su carpeta en dataset_final.")
