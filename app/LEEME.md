# Plantas

App Android (Kotlin y Jetpack Compose) para comparar una foto con los modelos del repo. Pensada para un Pixel 7 Pro, Android 13 o superior. minSdk 26.

## Abrir

1. Android Studio → Open → esta carpeta `app/`.
2. Deja que baje el Gradle wrapper (8.11.1) y sincronice. Hace falta el Android SDK 35.
3. Ejecuta el módulo `app` en el teléfono.

Los cinco modelos locales van en `app/src/main/assets` y funcionan sin red. El campo del PC es opcional: si marcas un ResNet, la foto se envía a `{url}/api/predecir`. Ejemplo: `http://192.168.1.10:5050` (el Flask de `web/app.py`, en `0.0.0.0:5050`).

## Exportar los TFLite

Con el entorno de la web (TensorFlow 2.21 / Keras 3.15), desde la raíz del repo:

```
.\.venv-web\Scripts\python.exe app\tools\exportar_tflite.py
```

El script quita la augmentación, escribe float16 en `assets/modelos/` y comprueba que el argmax coincida con Keras.
