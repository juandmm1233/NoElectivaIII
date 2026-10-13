import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.inaturalist.org/v1"
UA = "NoElectivaIII/1.0"
cfg = json.loads(
    Path(r"D:\juandmm1233\NoElectivaIII\regional_colombia\especies.json").read_text(encoding="utf-8")
)


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=45) as r:
        return json.loads(r.read().decode())


places = [
    ("Tolima", 12744),
    ("Colombia", 7196),
    ("SurAmerica", 97389),
]
print(f"{'especie':<28}{'Tolima':>8}{'Colombia':>10}{'SurAm':>8}")
for esp in cfg["especies"]:
    q = urllib.parse.urlencode({"q": esp["cientifico"], "rank": "species"})
    t = get(API + "/taxa?" + q)
    taxon = None
    for x in t.get("results", []):
        if x.get("name", "").lower() == esp["cientifico"].lower() and x.get("is_active"):
            taxon = x
            break
    if not taxon and t.get("results"):
        taxon = t["results"][0]
    if not taxon:
        print(esp["cientifico"], "NO TAXON")
        continue
    tid = taxon["id"]
    nums = []
    for _name, pid in places:
        params = urllib.parse.urlencode(
            {
                "taxon_id": tid,
                "place_id": pid,
                "quality_grade": "research",
                "photos": "true",
                "per_page": 0,
            }
        )
        d = get(f"{API}/observations?{params}")
        nums.append(d.get("total_results", 0))
        time.sleep(0.3)
    print(f"{esp['cientifico']:<28}{nums[0]:>8}{nums[1]:>10}{nums[2]:>8}")
    time.sleep(0.15)
