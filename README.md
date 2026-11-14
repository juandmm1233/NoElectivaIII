# Identificación de plantas medicinales

La app reconoce una foto de una planta y devuelve la especie más probable. El modelo que va en el teléfono es un MobileNetV2 de 29 especies de Colombia y de huerta (Tolima primero, luego Colombia). Se entrenó con transfer learning desde ImageNet, en dos fases, sobre fotos de iNaturalist con licencia abierta y fotos de huerta del equipo. El dataset no se sube: pesa demasiado.

Hay otras corridas (Herbify, de la India, y redes más pesadas) guardadas en `notebooks/` y dentro de la app, que compara varios modelos. El archivo de la entrega es uno solo: `model/modelo.tflite`.

## Integrantes

- juandmm1233

El repositorio menciona un segundo integrante, que entrenó la otra arquitectura sobre el mismo split de Herbify. Su nombre no está escrito en el código ni en los commits.

## Métricas finales

Medidas sobre el conjunto de prueba del modelo regional de 29 clases (2.201 imágenes, split 70/15/15, semilla 42). El archivo es `model/modelo.tflite`, float16.

| Métrica | Valor |
| --- | --- |
| F1 ponderado | 0,681 |
| Accuracy | 68,0 % |
| Top-3 | 85,4 % |
| Tamaño | 4,37 MB |
| Latencia (CPU de desarrollo) | 9,0 ms de mediana |

La latencia del teléfono no está medida. El valor de la tabla es de esta máquina: TensorFlow Lite con XNNPACK, entrada 224×224, 30 corridas después de 5 de calentamiento (media 9,8 ms, mínimo 7,1 ms). No sustituye una medición en el Pixel.

## Cómo instalar la app

Hace falta Android Studio y el Android SDK 35. La app pide Android 8 (minSdk 26) o superior. Está pensada para un Pixel 7 Pro con Android 13.

1. Clona el repositorio.
2. Android Studio → Open → la carpeta `app/`.
3. Espera a que baje el Gradle wrapper (8.13) y sincronice.
4. Ejecuta el módulo `app` en el teléfono.

Los modelos que la app compara en el dispositivo están en `app/app/src/main/assets/modelos/`. Funcionan sin red. `model/modelo.tflite` es la copia de entrega del MobileNetV2 regional; las etiquetas, en el mismo orden, están en `model/labels.txt`.

## Cómo ejecutar el notebook

El cuaderno principal del entrenamiento Herbify es `notebooks/entrenamiento_plantas_medicinales.ipynb`. El regional está en `notebooks/entrenamiento_regional_colombia.ipynb`. Hace falta Python 3.10 o 3.11 de 64 bits. Python 3.12 puede fallar al instalar TensorFlow.

```powershell
git clone https://github.com/juandmm1233/NoElectivaIII.git
cd NoElectivaIII
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Abre Jupyter o VS Code con la raíz del repositorio como directorio de trabajo. Los cuadernos usan `Path.cwd()` y buscan `dataset/` o `regional_colombia/` ahí, no dentro de `notebooks/`.

`notebooks/evidencia_entrenamientos.ipynb` junta las corridas ya hechas, con las figuras en las salidas. Ábrelo para leerlo. No vuelvas a ejecutar sus celdas: al correrlas se borra la figura guardada.

## Datasets

No están en el repositorio.

**Herbify (India).** [Herb (Plant) Classification Dataset](https://www.kaggle.com/datasets/ai4a-lab/herb-plant-classification-dataset), de Kaggle, `ai4a-lab/herb-plant-classification-dataset`. Son fotos de estudio, fondo bastante uniforme, con el nombre científico y el común en la carpeta. El conjunto trae 91 clases y 6.104 imágenes. El entrenamiento usa las 57 clases que tienen al menos 50 fotos (4.903 imágenes): 3.408 train, 734 validación y 761 test. La lista de esas 57 clases está en `app/app/src/main/assets/clases.json`.

**Regional (Colombia / Tolima).** Fotos Research Grade de iNaturalist con licencia CC0, CC-BY, CC-BY-SA o CC-BY-NC, más fotos de huerta del equipo. 10.201 train, 2.178 validación y 2.201 test. Estas son las 29 clases del modelo de `model/`, en el orden de `labels.txt`:

- Aloe vera (aloe vera)
- Baccharis latifolia (chilca)
- Bixa orellana (achiote)
- Bocconia frutescens (trompeto)
- Calendula officinalis (calendula)
- Cattleya trianae (cattleya)
- Cecropia peltata (yarumo)
- Ceroxylon quindiuense (palma de cera)
- Dysphania ambrosioides (paico)
- Equisetum bogotense (cola de caballo)
- Eryngium foetidum (cilantro cimarron)
- Espeletia grandiflora (frailejon)
- Gliricidia sepium (matarraton)
- Lippia alba (pronto alivio)
- Melissa officinalis (toronjil)
- Moringa oleifera (moringa)
- Petiveria alliacea (anamu)
- Physalis peruviana (uchuva)
- Phytolacca bogotensis (cargamanto)
- Piper aduncum (cordoncillo)
- Plantago major (llanten)
- Psidium guajava (guayaba)
- Quercus humboldtii (roble andino)
- Ruta graveolens (ruda)
- Salvia rosmarinus (romero)
- Sambucus nigra (sauco)
- Solanum quitoense (lulo)
- Theobroma cacao (cacao)
- Weinmannia tomentosa (encenillo)

## Video

Enlace al video: pendiente de publicar.

## Dónde está cada cosa

| Ruta | Contenido |
| --- | --- |
| `notebooks/` | Cuadernos del proyecto. Falta el de la auditoría de la Semana 15. |
| `model/` | `modelo.tflite` y `labels.txt` |
| `app/` | Código de la app Android (Kotlin y Jetpack Compose) |
| `docs/` | Informe de auditoría (PDF), deck y capturas. Todavía no están los archivos. |
