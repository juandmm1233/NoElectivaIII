# Subset regional: plantas medicinales de Colombia / Tolima

Responde al comentario del profesor: Herbify es de la India. Este flujo usa
**iNaturalist** (Research Grade, licencias abiertas) y las especies de `especies.json`
(medicinales de uso local, emblemáticas andinas y frutales de huerta).
Tolima primero; si faltan fotos, completa con Colombia y Suramérica.
El identificador usa un solo modelo regional. El de 10 clases queda archivado
en `resultados/modelo_regional_10clases.h5`.

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

Sale en `resultados/` con sufijo `_l1l2`: modelo, matriz, curvas y métricas. No reemplaza `modelo_regional.h5`.

## 3. Regularización L1 y L2

`entrenar_regional.py` multa el kernel de la capa final con L1 = 1e-5 y L2 = 1e-4. Dropout, augmentación y parada temprana siguen igual. Se lanza con los comandos de la sección anterior.

Ese entrenamiento escribe `modelo_regional_l1l2.h5`, `metricas_regional_l1l2.json`, `curvas_regional_l1l2.png` y `matriz_regional_l1l2.png`.

Las cifras de la web (77.4% en test) son el modelo sin esa multa sobre los pesos. La defensa de arquitectura, datos y métricas está en la sección 10 de `entrenamiento_regional_colombia.ipynb`.

## Citar

iNaturalist.org. Observaciones Research Grade con licencia CC0 / CC-BY / CC-BY-SA / CC-BY-NC.
Lista de especies: Humboldt (plantas útiles y medicinales de Colombia) y usos en huertas del Tolima.
Fotos propias del Jardín Botánico San Jorge se pueden copiar a `crudo/<especie>/` y volver a partir con el mismo script (si ya hay crudo, vuelve a correr solo la parte de split: borra `dataset/` y deja `crudo/`).
