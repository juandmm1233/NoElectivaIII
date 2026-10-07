"""
Baja fotos de plantas medicinales de Colombia / Tolima desde iNaturalist
y arma el split train/val/test.

No usa Herbify. Es el subset regional que pidio el profesor
(iNaturalist + especies de uso medicinal local).

  python descargar_inaturalist.py
  python descargar_inaturalist.py --solo-tolima
  python descargar_inaturalist.py --min 25 --max-por-clase 80
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

PROYECTO = Path(__file__).resolve().parent
ESPECIES_JSON = PROYECTO / "especies.json"
CRUDO = PROYECTO / "crudo"
DATASET = PROYECTO / "dataset"
API = "https://api.inaturalist.org/v1"
UA = "NoElectivaIII/1.0 (trabajo academico UCC; plantas medicinales Tolima)"
LICENCIAS_OK = {
    "cc0",
    "cc-by",
    "cc-by-sa",
    "cc-by-nc",
    "cc-by-nc-sa",
    "cc-by-nd",
    "cc-by-nc-nd",
}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def get_json(url: str, intentos: int = 4):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": UA, "Accept": "application/json"},
    )
    ultimo = None
    for i in range(intentos):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            ultimo = exc
            time.sleep(2 * (i + 1))
    raise RuntimeError(f"Fallo GET {url}: {ultimo}")


def bajar_archivo(url: str, destino: Path) -> bool:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            destino.write_bytes(resp.read())
        return True
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def resolver_taxon(nombre: str) -> dict | None:
    q = urllib.parse.urlencode({"q": nombre, "rank": "species"})
    data = get_json(f"{API}/taxa?{q}")
    for taxon in data.get("results", []):
        if taxon.get("name", "").lower() == nombre.lower() and taxon.get("is_active"):
            return taxon
    return data["results"][0] if data.get("results") else None


def url_foto(photo: dict) -> str | None:
    url = photo.get("medium_url") or photo.get("url")
    if not url:
        return None
    return url.replace("/square.", "/large.").replace("/square/", "/large/")


def licencia_ok(photo: dict) -> bool:
    codigo = (photo.get("license_code") or "").lower().strip()
    return codigo in LICENCIAS_OK


def observaciones(taxon_id: int, place_id: int):
    id_below = None
    paginas = 0
    while paginas < 25:
        params = {
            "taxon_id": taxon_id,
            "place_id": place_id,
            "quality_grade": "research",
            "photos": "true",
            "per_page": 200,
            "order": "desc",
            "order_by": "id",
        }
        if id_below is not None:
            params["id_below"] = id_below
        data = get_json(f"{API}/observations?{urllib.parse.urlencode(params)}")
        resultados = data.get("results") or []
        if not resultados:
            break
        for obs in resultados:
            yield obs
        id_below = min(obs["id"] for obs in resultados)
        paginas += 1
        if len(resultados) < 200:
            break
        time.sleep(1.0)


def carpeta_clase(comun: str, cientifico: str) -> str:
    seguro = "".join(c if c.isalnum() or c in "-_" else "_" for c in comun.lower())
    return f"{cientifico.replace(' ', '_')}__{seguro}"


def descargar_especie(esp: dict, place_id: int, lugar: str, max_por_clase: int, filas_meta: list) -> int:
    taxon = resolver_taxon(esp["cientifico"])
    time.sleep(0.5)
    if not taxon:
        print(f"  no se encontro taxon: {esp['cientifico']}", flush=True)
        return 0

    taxon_id = taxon["id"]
    clase = carpeta_clase(esp["comun"], esp["cientifico"])
    destino = CRUDO / clase
    destino.mkdir(parents=True, exist_ok=True)
    existentes = {p.name for p in destino.glob("*.jpg")}
    guardadas = len(existentes)

    for obs in observaciones(taxon_id, place_id):
        if guardadas >= max_por_clase:
            break
        for photo in obs.get("photos") or []:
            if guardadas >= max_por_clase:
                break
            if not licencia_ok(photo):
                continue
            url = url_foto(photo)
            if not url:
                continue
            nombre = f"{lugar}_{obs['id']}_{photo['id']}.jpg"
            if nombre in existentes:
                continue
            archivo = destino / nombre
            tmp = archivo.with_suffix(".part")
            if not bajar_archivo(url, tmp):
                continue
            try:
                with Image.open(tmp) as raw:
                    img = raw.convert("RGB")
                img.thumbnail((1024, 1024))
                img.save(archivo, format="JPEG", quality=90)
            except OSError:
                pass
            else:
                guardadas += 1
                existentes.add(nombre)
                filas_meta.append(
                    {
                        "clase": clase,
                        "cientifico": esp["cientifico"],
                        "comun": esp["comun"],
                        "lugar": lugar,
                        "observacion_id": obs["id"],
                        "foto_id": photo["id"],
                        "licencia": photo.get("license_code"),
                        "url_obs": f"https://www.inaturalist.org/observations/{obs['id']}",
                        "archivo": str(archivo.relative_to(PROYECTO)),
                    }
                )
            for _ in range(6):
                try:
                    tmp.unlink(missing_ok=True)
                    break
                except PermissionError:
                    time.sleep(0.25)
        time.sleep(0.15)
    print(f"  {clase}: {guardadas} fotos ({lugar})", flush=True)
    return guardadas


def partir(min_imgs: int, seed: int = 42, clases_ok: set[str] | None = None):
    random.seed(seed)
    if DATASET.exists():
        import shutil

        shutil.rmtree(DATASET)

    resumen = []
    for carpeta in sorted(p for p in CRUDO.iterdir() if p.is_dir()):
        if clases_ok is not None and carpeta.name not in clases_ok:
            continue
        archivos = [f for f in carpeta.iterdir() if f.suffix.lower() in EXTS]
        if len(archivos) < min_imgs:
            print(f"Descartada {carpeta.name}: {len(archivos)} < {min_imgs}", flush=True)
            continue
        random.shuffle(archivos)
        n = len(archivos)
        i1, i2 = int(n * 0.70), int(n * 0.85)
        partes = {
            "train": archivos[:i1],
            "val": archivos[i1:i2],
            "test": archivos[i2:],
        }
        for split, lista in partes.items():
            dest = DATASET / split / carpeta.name
            dest.mkdir(parents=True, exist_ok=True)
            for f in lista:
                img = Image.open(f).convert("RGB").resize((224, 224))
                img.save(dest / f.with_suffix(".jpg").name, quality=90)
        resumen.append((carpeta.name, n, len(partes["train"]), len(partes["val"]), len(partes["test"])))

    print(f"\n{'clase':<55}{'total':>7}{'train':>7}{'val':>6}{'test':>6}")
    for r in resumen:
        print(f"{r[0]:<55}{r[1]:>7}{r[2]:>7}{r[3]:>6}{r[4]:>6}")
    print("Total clases:", len(resumen), "| imagenes:", sum(x[1] for x in resumen))
    (PROYECTO / "resumen_split.json").write_text(
        json.dumps(
            [{"clase": a, "total": b, "train": c, "val": d, "test": e} for a, b, c, d, e in resumen],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description="Descarga fotos regionales desde iNaturalist.")
    parser.add_argument("--solo-tolima", action="store_true", help="No completar con fotos del resto de Colombia.")
    parser.add_argument("--min", type=int, default=30, help="Minimo de fotos para dejar la clase.")
    parser.add_argument("--max-por-clase", type=int, default=200)
    parser.add_argument("--solo-partir", action="store_true", help="No descarga: solo arma train/val/test desde crudo/.")
    args = parser.parse_args()

    cfg = json.loads(ESPECIES_JSON.read_text(encoding="utf-8"))
    CLASES_OK = {carpeta_clase(e["comun"], e["cientifico"]) for e in cfg["especies"]}
    CRUDO.mkdir(parents=True, exist_ok=True)

    if args.solo_partir:
        partir(args.min, clases_ok=CLASES_OK)
        print("\nListo. Siguiente paso: python entrenar_regional.py")
        return

    meta = []
    ruta_meta = PROYECTO / "fuentes_inaturalist.csv"
    if ruta_meta.exists():
        with ruta_meta.open("r", encoding="utf-8", newline="") as fh:
            meta = list(csv.DictReader(fh))

    print("Descargando iNaturalist (Research Grade, licencias abiertas).", flush=True)
    print("Tolima", cfg["place_tolima_id"], "Colombia", cfg["place_colombia_id"],
          "Suramerica", cfg.get("place_suramerica_id"), flush=True)

    for esp in cfg["especies"]:
        print(f"\n{esp['cientifico']} ({esp['comun']})", flush=True)
        n = descargar_especie(esp, cfg["place_tolima_id"], "tolima", args.max_por_clase, meta)
        if n < args.max_por_clase and not args.solo_tolima:
            print(f"  Tolima {n}; completo con Colombia.", flush=True)
            n = descargar_especie(esp, cfg["place_colombia_id"], "colombia", args.max_por_clase, meta)
        if n < args.max_por_clase and not args.solo_tolima and cfg.get("place_suramerica_id"):
            print(f"  Colombia {n}; completo con Suramerica.", flush=True)
            n = descargar_especie(esp, cfg["place_suramerica_id"], "suramerica", args.max_por_clase, meta)
        time.sleep(0.8)

    if meta:
        vistos = set()
        unicas = []
        for fila in meta:
            clave = (fila.get("observacion_id"), fila.get("foto_id"))
            if clave in vistos:
                continue
            vistos.add(clave)
            unicas.append(fila)
        with ruta_meta.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(unicas[0].keys()))
            w.writeheader()
            w.writerows(unicas)
        print("Citas iNaturalist:", ruta_meta, "filas:", len(unicas))

    partir(args.min, clases_ok=CLASES_OK)
    print("\nListo. Siguiente paso:")
    print("  python entrenar_regional.py --prueba")
    print("  python entrenar_regional.py")


if __name__ == "__main__":
    main()
