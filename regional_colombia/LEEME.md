# Subset regional: plantas medicinales de Colombia / Tolima

Responde al comentario del profesor: Herbify es de la India. Este flujo usa
**iNaturalist** (Research Grade, licencias abiertas) y **10 especies** de uso medicinal
en Colombia/Tolima (Tolima primero; si faltan fotos, completa con Colombia).
La lista está en `especies.json`.

No sustituye el experimento Herbify. Es un segundo entrenamiento para impacto regional.

## 1. Descargar fotos

Desde esta carpeta, con el `.venv` de la raíz del repo:

```powershell
cd D:\juandmm1233\NoElectivaIII
.\.venv\Scripts\Activate.ps1
cd regional_colombia
python descargar_inaturalist.py
```

Opciones:

```powershell
python descargar_inaturalist.py --solo-tolima
python descargar_inaturalist.py --min 25 --max-por-clase 80
```

Genera:

- `crudo/` fotos originales (no se suben a GitHub)
- `dataset/train|val|test` split 70/15/15, 224×224
- `fuentes_inaturalist.csv` para citar cada foto
- `resumen_split.json`

Las especies con pocas fotos con licencia abierta se descartan.

## 2. Entrenar

```powershell
python entrenar_regional.py --prueba
python entrenar_regional.py
```

Sale en `resultados/`: modelo, matriz, curvas y métricas.

## Citar

iNaturalist.org. Observaciones Research Grade con licencia CC0 / CC-BY / CC-BY-SA / CC-BY-NC.
Lista de especies: Humboldt (plantas útiles y medicinales de Colombia) y usos en huertas del Tolima.
Fotos propias del Jardín Botánico San Jorge se pueden copiar a `crudo/<especie>/` y volver a partir con el mismo script (si ya hay crudo, vuelve a correr solo la parte de split: borra `dataset/` y deja `crudo/`).
