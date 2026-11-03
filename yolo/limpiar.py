"""
Limpieza de imágenes: quita corruptas, muy pequeñas y duplicadas,
convierte todo a JPG y renombra.

Uso (en la terminal, dentro de la carpeta del proyecto):
    pip install pillow imagehash
    python limpiar.py

Espera encontrar una carpeta llamada dataset_raw con una subcarpeta por planta.
Crea dataset_final con el resultado. No borra tus fotos originales.
"""
from pathlib import Path

from PIL import Image
import imagehash

RAW = Path("dataset_raw")
FINAL = Path("dataset_final")
MIN_LADO = 150           # descarta imágenes con un lado menor a esto (px)
MAX_LADO = 1024          # reduce imágenes más grandes (px)
OBJETIVO = 800           # máximo de imágenes finales por planta
UMBRAL_DUPLICADO = 4     # distancia de hash: menor = más parecidas

if not RAW.exists():
    raise SystemExit("No encuentro la carpeta dataset_raw. Revisa que estés en la carpeta correcta.")

for carpeta in sorted(RAW.iterdir()):
    if not carpeta.is_dir():
        continue

    clase = carpeta.name.lower().replace(" ", "_")
    destino = FINAL / clase
    destino.mkdir(parents=True, exist_ok=True)

    hashes = []
    stats = {"leidas": 0, "corruptas": 0, "pequenas": 0, "duplicadas": 0, "guardadas": 0}

    for ruta in sorted(carpeta.rglob("*")):
        if not ruta.is_file():
            continue
        if stats["guardadas"] >= OBJETIVO:
            break
        stats["leidas"] += 1

        # 1) Corruptas
        try:
            with Image.open(ruta) as img:
                img.load()
                img = img.convert("RGB")
        except Exception:
            stats["corruptas"] += 1
            continue

        # 2) Muy pequeñas
        if min(img.size) < MIN_LADO:
            stats["pequenas"] += 1
            continue

        # 3) Duplicadas (o casi iguales)
        h = imagehash.phash(img)
        if any(h - otro <= UMBRAL_DUPLICADO for otro in hashes):
            stats["duplicadas"] += 1
            continue
        hashes.append(h)

        # 4) Reducir, guardar como JPG y renombrar
        img.thumbnail((MAX_LADO, MAX_LADO))
        stats["guardadas"] += 1
        img.save(destino / f"{clase}_{stats['guardadas']:04d}.jpg", "JPEG", quality=90)

    print(f"{clase:12s} -> {stats}")

print("\nListo. Revisa dataset_final/ y borra a mano lo que no sea la planta.")
