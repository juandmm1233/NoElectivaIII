# Clasificación de plantas medicinales

Entrenamiento de una red para identificar especies del dataset Herbify.
La copia para entrenar en otro PC está en `para_companero/` y usa **MobileNetV2** (modelo ligero, con pesos de ImageNet).

El dataset de esa copia ya viene partido y no hay que volver a dividirlo:

- `para_companero/dataset/train`: 3408 imágenes
- `para_companero/dataset/val`: 734 imágenes
- `para_companero/dataset/test`: 761 imágenes
- 57 especies, split 70/15/15, semilla 42

No hace falta GPU ni cuenta de Kaggle. En CPU el entrenamiento completo suele tardar entre 2 y 4 horas.

## Requisitos

- Windows 10 u 11
- Python 3.10 o 3.11 de 64 bits  
  En el instalador hay que marcar **Add python.exe to PATH**. Python 3.12 puede fallar al instalar TensorFlow.

## Cómo ejecutarlo

Abre PowerShell, clona el repositorio y entra en la carpeta del entrenamiento ligero:

```powershell
git clone https://github.com/juandmm1233/NoElectivaIII.git
cd NoElectivaIII\para_companero
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Primero comprueba que el entorno arranca. Esto corre 1 época y **no** sirve para el informe:

```powershell
python entrenar_ligero.py --prueba
```

Cuando esa prueba termine sin errores, lanza el entrenamiento del trabajo:

```powershell
python entrenar_ligero.py
```

Si el PC se queda sin memoria, corta con Ctrl+C y relanza con un lote más chico:

```powershell
python entrenar_ligero.py --batch 8
```

La primera ejecución descarga los pesos de ImageNet (unos 10 MB).

## Qué devuelve

Al terminar, la carpeta `para_companero/resultados/` queda con:

| Archivo | Contenido |
| --- | --- |
| `modelo_mobilenetv2.h5` | Modelo entrenado |
| `metricas_mobilenetv2.json` | Accuracy de test, top-3, tiempo, parámetros y tamaño |
| `reporte_mobilenetv2.txt` | Precisión y recall por especie |
| `curvas_mobilenetv2.png` | Accuracy y loss de las dos fases |
| `matriz_mobilenetv2.png` | Matriz de confusión |
| `clases.json` | Orden de las 57 clases |

El número para comparar con el otro modelo es `accuracy_test` dentro de `metricas_mobilenetv2.json`.

## Cómo entrena

1. Fase 1: MobileNetV2 congelado, Adam `1e-3`, hasta 20 épocas.
2. Fase 2: se descongelan las últimas 40 capas, Adam `1e-5`, hasta 15 épocas.
3. Early stopping sobre `val_accuracy` (paciencia 5). Si `val_loss` se estanca, baja el learning rate.
4. La augmentación (giros, rotación, zoom, desplazamiento, contraste y brillo) solo se aplica en entrenamiento.

`fichas_medicinales.csv` es la tabla de usos. El modelo solo identifica la especie.

## Notebook del otro equipo

`entrenamiento_plantas_medicinales.ipynb` es el flujo con TensorFlow DirectML de la máquina principal. Sus dependencias están en el `requirements.txt` de la raíz. Las imágenes de esa copia local no se suben al repositorio.
