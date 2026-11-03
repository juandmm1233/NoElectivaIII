"""Junta el dataset regional (20 clases) con yolo/dataset_final.

No pone tope de fotos. La ruda de las dos fuentes cae en la misma clase.
Solo descarta una foto de ruda del equipo si es casi la misma que una
regional, para que no quede en train y en test a la vez.

    python preparar_29.py
"""
import json
import random
import shutil
from pathlib import Path

from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
REG = RAIZ / "regional_colombia"
ORIGEN_REG = REG / "dataset"
ORIGEN_EQUIPO = RAIZ / "yolo" / "dataset_final"
DESTINO = REG / "dataset_29"
MUESTRAS = RAIZ / "web" / "static" / "muestras"
SEMILLA = 42
P_TRAIN, P_VAL = 0.70, 0.15
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

NUEVAS = {
    "achiote": "Bixa_orellana__achiote",
    "aloe_vera": "Aloe_vera__aloe_vera",
    "calendula": "Calendula_officinalis__calendula",
    "moringa": "Moringa_oleifera__moringa",
    "paico": "Dysphania_ambrosioides__paico",
    "pronto_alivio": "Lippia_alba__pronto_alivio",
    "romero": "Salvia_rosmarinus__romero",
    "ruda": "Ruta_graveolens__ruda",
    "sauco": "Sambucus_nigra__sauco",
    "toronjil": "Melissa_officinalis__toronjil",
}


def dhash(path, hash_size=8):
    with Image.open(path) as img:
        img = img.convert("L").resize((hash_size + 1, hash_size))
        px = list(img.getdata())
    bits = 0
    bit = 0
    for row in range(hash_size):
        for col in range(hash_size):
            left = px[row * (hash_size + 1) + col]
            right = px[row * (hash_size + 1) + col + 1]
            if left > right:
                bits |= 1 << bit
            bit += 1
    return bits


def guardar_jpg(src, dest):
    with Image.open(src) as img:
        img = img.convert("RGB")
        img = img.resize((224, 224))
        img.save(dest, "JPEG", quality=90)


def fotos_de(carpeta):
    return sorted(p for p in carpeta.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def main():
    if not ORIGEN_REG.is_dir():
        raise SystemExit(f"No esta {ORIGEN_REG}")
    if not ORIGEN_EQUIPO.is_dir():
        raise SystemExit(f"No esta {ORIGEN_EQUIPO}")
    if DESTINO.exists():
        shutil.rmtree(DESTINO)

    por_clase = {}
    for split in ("train", "val", "test"):
        base = ORIGEN_REG / split
        for carpeta in sorted(p for p in base.iterdir() if p.is_dir()):
            por_clase.setdefault(carpeta.name, []).extend(fotos_de(carpeta))

    if "Ruta_graveolens__ruda" not in por_clase:
        raise SystemExit("No esta la clase regional de ruda.")

    hashes_ruda = []
    for ruta in por_clase["Ruta_graveolens__ruda"]:
        try:
            hashes_ruda.append(dhash(ruta))
        except OSError:
            continue

    duplicadas_ruda = 0
    for comun, clase in NUEVAS.items():
        carpeta = ORIGEN_EQUIPO / comun
        if not carpeta.is_dir():
            raise SystemExit(f"Falta {carpeta}")
        nuevas = []
        for ruta in fotos_de(carpeta):
            if clase == "Ruta_graveolens__ruda":
                try:
                    h = dhash(ruta)
                except OSError:
                    continue
                if any((h ^ otro).bit_count() <= 4 for otro in hashes_ruda):
                    duplicadas_ruda += 1
                    continue
                hashes_ruda.append(h)
            nuevas.append(ruta)
        por_clase.setdefault(clase, []).extend(nuevas)

    random.seed(SEMILLA)
    resumen = []
    for clase in sorted(por_clase):
        archivos = list(por_clase[clase])
        random.shuffle(archivos)
        n = len(archivos)
        i1, i2 = int(n * P_TRAIN), int(n * (P_TRAIN + P_VAL))
        partes = {"train": archivos[:i1], "val": archivos[i1:i2], "test": archivos[i2:]}
        for split, lista in partes.items():
            dest = DESTINO / split / clase
            dest.mkdir(parents=True, exist_ok=True)
            for i, src in enumerate(lista):
                guardar_jpg(src, dest / f"{i:04d}.jpg")
        resumen.append({
            "clase": clase,
            "total": n,
            "train": len(partes["train"]),
            "val": len(partes["val"]),
            "test": len(partes["test"]),
        })

    MUESTRAS.mkdir(parents=True, exist_ok=True)
    for comun, clase in NUEVAS.items():
        if clase == "Ruta_graveolens__ruda":
            continue
        muestra = MUESTRAS / f"reg_{clase}.jpg"
        origen = next(iter(fotos_de(ORIGEN_EQUIPO / comun)))
        with Image.open(origen) as img:
            img = img.convert("RGB")
            img.thumbnail((800, 800))
            img.save(muestra, "JPEG", quality=90)

    totales = {
        "clases": len(resumen),
        "imagenes": sum(r["total"] for r in resumen),
        "train": sum(r["train"] for r in resumen),
        "val": sum(r["val"] for r in resumen),
        "test": sum(r["test"] for r in resumen),
        "ruda_duplicadas_omitidas": duplicadas_ruda,
        "por_clase": resumen,
    }
    (REG / "resumen_29.json").write_text(json.dumps(totales, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Clases: {totales['clases']}")
    print(f"Imagenes: {totales['imagenes']}  train {totales['train']}  val {totales['val']}  test {totales['test']}")
    print(f"Ruda casi duplicada, omitida: {duplicadas_ruda}")
    for r in resumen:
        print(f"{r['clase']:<42} {r['total']:5} {r['train']:5} {r['val']:5} {r['test']:5}")


if __name__ == "__main__":
    main()
