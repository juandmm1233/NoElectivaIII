import json
import time
import urllib.parse
import urllib.request

API = "https://api.inaturalist.org/v1"
UA = "NoElectivaIII/1.0 (trabajo academico UCC)"
CO = 7196
SA = 97389

YA = {
    "piper aduncum",
    "ruta graveolens",
    "plantago major",
    "petiveria alliacea",
    "cecropia peltata",
    "equisetum bogotense",
    "gliricidia sepium",
    "eryngium foetidum",
    "psidium guajava",
    "baccharis latifolia",
}

CANDIDATAS = [
    ("Ceroxylon quindiuense", "palma de cera", "endémica emblemática"),
    ("Cattleya trianae", "cattleya", "flor nacional"),
    ("Quercus humboldtii", "roble andino", "nativa andina"),
    ("Guadua angustifolia", "guadua", "bambú nativo"),
    ("Espeletia grandiflora", "frailejón", "páramo"),
    ("Espeletia hartwegiana", "frailejón", "páramo"),
    ("Weinmannia tomentosa", "encenillo", "Andes Colombia"),
    ("Drimys granadensis", "canelo de páramo", "andina"),
    ("Brugmansia arborea", "borrachero", "uso tradicional"),
    ("Brugmansia candida", "borrachero blanco", "uso tradicional"),
    ("Brugmansia suaveolens", "borrachero", "uso tradicional"),
    ("Cinchona pubescens", "quina", "medicinal andina"),
    ("Cinchona officinalis", "quina", "medicinal"),
    ("Theobroma cacao", "cacao", "cultivo nativo"),
    ("Bixa orellana", "achiote", "colorante medicinal"),
    ("Annona muricata", "guanábana", "frutal medicinal"),
    ("Passiflora ligularis", "granadilla", "andina"),
    ("Passiflora edulis", "maracuyá", "frutal"),
    ("Passiflora quadrangularis", "badea", "frutal"),
    ("Solanum quitoense", "lulo", "andina"),
    ("Physalis peruviana", "uchuva", "andina"),
    ("Solanum betaceum", "tomate de árbol", "andina"),
    ("Vaccinium meridionale", "agraz", "páramo/andina"),
    ("Rubus glaucus", "mora de Castilla", "andina"),
    ("Lippia alba", "prontoalivio", "medicinal popular"),
    ("Minthostachys mollis", "muña", "andina medicinal"),
    ("Dysphania ambrosioides", "paico", "antiparasitario"),
    ("Sambucus peruviana", "saúco", "andina"),
    ("Sambucus nigra", "saúco", "uso huerta"),
    ("Bocconia frutescens", "trompeto", "andina"),
    ("Piper auritum", "hoja santa", "Piper medicinal"),
    ("Piper umbellatum", "cordoncillo", "Piper"),
    ("Equisetum giganteum", "cola de caballo", "andina"),
    ("Aloe vera", "sábila", "huerta"),
    ("Carica papaya", "papaya", "frutal"),
    ("Persea americana", "aguacate", "nativo"),
    ("Coffea arabica", "café", "cultivo"),
    ("Anacardium occidentale", "marañón", "costa"),
    ("Spondias mombin", "hobo", "nativa"),
    ("Inga edulis", "guamo", "nativa"),
    ("Jatropha curcas", "piñón", "medicinal"),
    ("Tithonia diversifolia", "mirasol", "forraje/medicinal"),
    ("Lantana camara", "venturosa", "popular"),
    ("Ocimum campechianum", "albahaca de monte", "nativa"),
    ("Cymbopogon citratus", "limoncillo", "huerta"),
    ("Uncaria tomentosa", "uña de gato", "amazónica"),
    ("Croton lechleri", "sangre de drago", "amazónica"),
    ("Manihot esculenta", "yuca", "nativa"),
    ("Ipomoea batatas", "batata", "nativa"),
    ("Sechium edule", "cidra", "cultivo"),
    ("Capsicum pubescens", "rocoto", "andina"),
    ("Smallanthus sonchifolius", "yacón", "andina"),
    ("Arracacia xanthorrhiza", "arracacha", "andina"),
    ("Canna indica", "achira", "andina"),
    ("Bejaria aestuans", "carbonero", "páramo"),
    ("Clusia multiflora", "copey", "andina"),
    ("Morella pubescens", "laurel de cera", "andina"),
    ("Hedyosmum racemosum", "granizo", "andina"),
    ("Phytolacca bogotensis", "cargamanto", "andina"),
    ("Erythroxylum coca", "coca", "tradicional"),
]


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode())


def taxon_id(nombre):
    q = urllib.parse.urlencode({"q": nombre, "rank": "species"})
    t = get(f"{API}/taxa?{q}")
    for x in t.get("results") or []:
        if x.get("name", "").lower() == nombre.lower() and x.get("is_active"):
            return x.get("id"), x.get("name")
    if t.get("results"):
        x = t["results"][0]
        return x.get("id"), x.get("name")
    return None, None


def n_obs(tid, place):
    q = urllib.parse.urlencode(
        {
            "taxon_id": tid,
            "place_id": place,
            "quality_grade": "research",
            "photos": "true",
            "per_page": 0,
        }
    )
    d = get(f"{API}/observations?{q}")
    return int(d.get("total_results") or 0)


print(f"{'especie':<32}{'comun':<22}{'CO':>6}{'SA':>7}  nota")
filas = []
for cien, comun, nota in CANDIDATAS:
    if cien.lower() in YA:
        continue
    tid, nom = taxon_id(cien)
    time.sleep(0.25)
    if not tid:
        print(f"{cien:<32}{'NO TAXON'}")
        continue
    co = n_obs(tid, CO)
    time.sleep(0.25)
    sa = n_obs(tid, SA)
    time.sleep(0.2)
    filas.append((cien, comun, co, sa, nota))
    print(f"{cien:<32}{comun:<22}{co:>6}{sa:>7}  {nota}", flush=True)

print("\n=== Con chance de 400-500 fotos (Colombia >= 180 u obs SA >= 280) ===")
ok = [f for f in filas if f[2] >= 180 or f[3] >= 280]
ok.sort(key=lambda x: (-x[2], -x[3]))
for cien, comun, co, sa, nota in ok:
    print(f"  {cien} ({comun})  Colombia={co}  Suramerica={sa}  [{nota}]")
print(f"\nCandidatas viables: {len(ok)}")
