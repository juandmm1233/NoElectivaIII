"""Baja fotos de campo tomadas en Colombia desde fuentes distintas a iNaturalist:

- GBIF / SiB Colombia (sin herbario ni iNaturalist)
- Wikimedia Commons (GPS o categoria Colombia)
- Openverse / Flickr con licencia CC y etiqueta Colombia

  python descargar_fuentes_colombia.py --max-por-clase 450
"""
from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from PIL import Image

PROYECTO = Path(__file__).resolve().parent
ESPECIES_JSON = PROYECTO / "especies.json"
CRUDO = PROYECTO / "crudo"
UA = "NoElectivaIII/1.0 (trabajo academico UCC; plantas medicinales Colombia)"
INAT_DATASET = "50c9509d-22c7-4a22-a47d-8c48425ef4a7"
EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_CLASE = 450
COLOMBIA_LAT = (-4.3, 13.6)
COLOMBIA_LON = (-81.9, -66.7)
PALABRAS_CO = (
    "colombia", "colombian", "colombiano", "colombiana", "bogota", "bogotá",
    "tolima", "ibague", "ibagué", "medellin", "medellín", "antioquia",
    "cundinamarca", "boyaca", "boyacá", "cali", "valle del cauca", "huila",
    "narino", "nariño", "cauca", "quindio", "quindío", "risaralda", "caldas",
    "santander", "magdalena", "cartagena", "barranquilla", "meta", "llanos",
    "amazonas", "choco", "chocó", "pereira", "manizales", "neiva", "pasto",
)
LICENCIAS = {
    "cc0", "cc-by", "cc-by-sa", "cc-by-nc", "cc-by-nc-sa", "cc-by-nd",
    "cc-by-nc-nd", "cc by", "cc by-sa", "cc by-nc", "pd", "public domain",
}


def get_json(url: str, intentos: int = 6):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    ultimo = None
    for i in range(intentos):
        try:
            with urllib.request.urlopen(req, timeout=45) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            ultimo = exc
            espera = 20 if exc.code == 429 else 2 * (i + 1)
            time.sleep(espera)
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


def carpeta_clase(comun: str, cientifico: str) -> str:
    seguro = "".join(c if c.isalnum() or c in "-_" else "_" for c in comun.lower())
    return f"{cientifico.replace(' ', '_')}__{seguro}"


def texto_co(texto: str) -> bool:
    t = (texto or "").lower()
    return any(p in t for p in PALABRAS_CO)


def gps_colombia(lat, lon) -> bool:
    try:
        lat = float(lat)
        lon = float(lon)
    except (TypeError, ValueError):
        return False
    return COLOMBIA_LAT[0] <= lat <= COLOMBIA_LAT[1] and COLOMBIA_LON[0] <= lon <= COLOMBIA_LON[1]


def licencia_ok(codigo: str) -> bool:
    c = (codigo or "").lower().replace("_", "-").strip()
    if not c:
        return False
    if "http" in c:
        if "publicdomain" in c or "/zero/" in c:
            return True
        if "creativecommons.org" not in c:
            return False
        return True
    return any(k in c for k in LICENCIAS)


def guardar_jpg(url: str, destino: Path) -> bool:
    tmp = destino.with_suffix(".part")
    if not bajar_archivo(url, tmp):
        return False
    try:
        with Image.open(tmp) as raw:
            img = raw.convert("RGB")
        if min(img.size) < 120:
            return False
        img.thumbnail((1024, 1024))
        img.save(destino, format="JPEG", quality=90)
        return True
    except OSError:
        return False
    finally:
        for _ in range(6):
            try:
                tmp.unlink(missing_ok=True)
                break
            except PermissionError:
                time.sleep(0.2)


def guardar_si_cabe(clase: str, prefijo: str, ident: str, url: str, max_n: int, meta: dict, filas: list) -> bool:
    dest_dir = CRUDO / clase
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = len([p for p in dest_dir.iterdir() if p.suffix.lower() in EXTS])
    if n >= max_n:
        return False
    nombre = f"{prefijo}_{ident}.jpg"
    archivo = dest_dir / nombre
    if archivo.exists():
        return False
    if not guardar_jpg(url, archivo):
        return False
    filas.append({**meta, "archivo": str(archivo.relative_to(PROYECTO))})
    return True


def desde_gbif(esp: dict, clase: str, max_n: int, filas: list) -> int:
    n_ok = 0
    offset = 0
    while offset < 1200:
        dest_n = len([p for p in (CRUDO / clase).glob("*") if p.suffix.lower() in EXTS])
        if dest_n >= max_n:
            break
        q = urllib.parse.urlencode(
            {
                "scientificName": esp["cientifico"],
                "country": "CO",
                "mediaType": "StillImage",
                "limit": 300,
                "offset": offset,
            }
        )
        data = get_json(f"https://api.gbif.org/v1/occurrence/search?{q}")
        results = data.get("results") or []
        if not results:
            break
        for rec in results:
            if rec.get("datasetKey") == INAT_DATASET:
                continue
            if rec.get("basisOfRecord") == "PRESERVED_SPECIMEN":
                continue
            for media in rec.get("media") or []:
                ident = str(media.get("identifier") or "")
                if "inaturalist" in ident.lower():
                    continue
                lic = media.get("license") or rec.get("license") or ""
                if not licencia_ok(lic):
                    continue
                tipo = (media.get("type") or media.get("format") or "").lower()
                if tipo and "image" not in tipo and "jpg" not in tipo and "jpeg" not in tipo:
                    continue
                clave = rec.get("key") or ident[-20:]
                if guardar_si_cabe(
                    clase,
                    "gbifco",
                    f"{clave}_{abs(hash(ident)) % 10_000_000}",
                    ident,
                    max_n,
                    {
                        "fuente": "gbif_sib_colombia",
                        "cientifico": esp["cientifico"],
                        "comun": esp["comun"],
                        "clase": clase,
                        "licencia": lic,
                        "url": ident,
                    },
                    filas,
                ):
                    n_ok += 1
                time.sleep(0.12)
        if len(results) < 300:
            break
        offset += 300
        time.sleep(0.5)
    return n_ok


def desde_commons(esp: dict, clase: str, max_n: int, filas: list) -> int:
    n_ok = 0
    consultas = [
        f'{esp["cientifico"]} Colombia',
        f'{esp["comun"]} Colombia',
        f'{esp["cientifico"]} Tolima',
        f'{esp["cientifico"]} Bogotá',
    ]
    vistos = set()
    for consulta in consultas:
        offset = 0
        while offset < 400:
            dest_n = len([p for p in (CRUDO / clase).glob("*") if p.suffix.lower() in EXTS])
            if dest_n >= max_n:
                return n_ok
            q = urllib.parse.urlencode(
                {
                    "action": "query",
                    "format": "json",
                    "generator": "search",
                    "gsrsearch": consulta,
                    "gsrnamespace": 6,
                    "gsrlimit": 50,
                    "gsroffset": offset,
                    "prop": "imageinfo|categories",
                    "iiprop": "url|size|extmetadata|mime",
                    "iiurlwidth": 1024,
                    "cllimit": 40,
                }
            )
            data = get_json(f"https://commons.wikimedia.org/w/api.php?{q}")
            pages = (data.get("query") or {}).get("pages") or {}
            if not pages:
                break
            for page in pages.values():
                infos = page.get("imageinfo") or []
                if not infos:
                    continue
                info = infos[0]
                if "image" not in (info.get("mime") or ""):
                    continue
                meta = info.get("extmetadata") or {}
                lic = (meta.get("LicenseShortName") or {}).get("value") or ""
                if not licencia_ok(lic):
                    continue
                cats = " ".join(c.get("title", "") for c in page.get("categories") or [])
                lat = (meta.get("GPSLatitude") or {}).get("value")
                lon = (meta.get("GPSLongitude") or {}).get("value")
                titulo = page.get("title") or ""
                desc = (meta.get("ImageDescription") or {}).get("value") or ""
                nacional = gps_colombia(lat, lon) or texto_co(cats) or texto_co(titulo) or texto_co(desc)
                if not nacional:
                    continue
                url = info.get("url")
                if not url or url in vistos:
                    continue
                vistos.add(url)
                pid = page.get("pageid") or abs(hash(url)) % 10_000_000
                if guardar_si_cabe(
                    clase,
                    "commonsco",
                    str(pid),
                    url,
                    max_n,
                    {
                        "fuente": "wikimedia_commons_colombia",
                        "cientifico": esp["cientifico"],
                        "comun": esp["comun"],
                        "clase": clase,
                        "licencia": lic,
                        "url": url,
                    },
                    filas,
                ):
                    n_ok += 1
                time.sleep(0.1)
            if "continue" not in data:
                break
            offset = int(data["continue"].get("gsroffset", offset + 50))
            time.sleep(1.2)
    return n_ok


def desde_openverse(esp: dict, clase: str, max_n: int, filas: list) -> int:
    n_ok = 0
    vistos = set()
    for consulta in (esp["cientifico"], f'{esp["cientifico"]} Colombia', f'{esp["comun"]} Colombia'):
        for page in range(1, 12):
            dest_n = len([p for p in (CRUDO / clase).glob("*") if p.suffix.lower() in EXTS])
            if dest_n >= max_n:
                return n_ok
            q = urllib.parse.urlencode({"q": consulta, "page": page, "page_size": 20})
            try:
                data = get_json(f"https://api.openverse.org/v1/images/?{q}")
            except RuntimeError:
                break
            results = data.get("results") or []
            if not results:
                break
            for it in results:
                tags = " ".join(t.get("name", "") for t in it.get("tags") or [])
                blob = " ".join([it.get("title") or "", tags, it.get("creator") or ""])
                if not texto_co(blob):
                    continue
                if not licencia_ok(it.get("license") or ""):
                    continue
                url = it.get("url")
                if not url or url in vistos or "inaturalist" in url.lower():
                    continue
                vistos.add(url)
                if guardar_si_cabe(
                    clase,
                    "openverseco",
                    (it.get("id") or str(abs(hash(url))))[:40],
                    url,
                    max_n,
                    {
                        "fuente": "openverse_flickr_colombia",
                        "cientifico": esp["cientifico"],
                        "comun": esp["comun"],
                        "clase": clase,
                        "licencia": it.get("license"),
                        "url": url,
                    },
                    filas,
                ):
                    n_ok += 1
                time.sleep(0.12)
            time.sleep(0.4)
    return n_ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-por-clase", type=int, default=MAX_CLASE)
    args = parser.parse_args()
    cfg = json.loads(ESPECIES_JSON.read_text(encoding="utf-8"))
    objetivo = {
        "Baccharis latifolia",
        "Psidium guajava",
        "Ruta graveolens",
    }
    filas = []
    print("Fuentes nacionales distintas de iNaturalist: GBIF/SiB, Wikimedia Commons, Openverse.", flush=True)
    for esp in cfg["especies"]:
        if esp["cientifico"] not in objetivo:
            continue
        clase = carpeta_clase(esp["comun"], esp["cientifico"])
        hay = len(list((CRUDO / clase).glob("*"))) if (CRUDO / clase).exists() else 0
        print(f"\n{esp['cientifico']} ({esp['comun']}) ahora {hay}", flush=True)
        a = desde_gbif(esp, clase, args.max_por_clase, filas)
        print(f"  GBIF/SiB Colombia (sin herbario): +{a}", flush=True)
        b = desde_commons(esp, clase, args.max_por_clase, filas)
        print(f"  Wikimedia Commons Colombia: +{b}", flush=True)
        c = desde_openverse(esp, clase, args.max_por_clase, filas)
        print(f"  Openverse/Flickr Colombia: +{c}", flush=True)
        total = len([p for p in (CRUDO / clase).iterdir() if p.suffix.lower() in EXTS]) if (CRUDO / clase).exists() else 0
        print(f"  total crudo: {total}", flush=True)

    if filas:
        ruta = PROYECTO / "fuentes_colombia_otras.csv"
        with ruta.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(filas[0].keys()))
            w.writeheader()
            w.writerows(filas)
        print("Citas:", ruta, "filas:", len(filas), flush=True)
    print("\nListo. No se entreno.")


if __name__ == "__main__":
    main()
