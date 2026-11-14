package app.electiva.plantas

import android.Manifest
import android.graphics.Bitmap
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.FilterChip
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import java.io.File
import java.util.Locale
import java.util.concurrent.Executor

@Composable
@OptIn(ExperimentalLayoutApi::class)
fun PantallaPlantas(
    modelo: PlantasViewModel,
    estudiante: String,
    avisoCampo: String?,
    guardando: Boolean,
    alHistorial: () -> Unit,
    alSalir: () -> Unit,
    alGuardar: (Bitmap, Etiqueta, String, String) -> Unit,
) {
    val contexto = LocalContext.current
    val desplazamiento = rememberScrollState()
    var camaraLista by remember { mutableStateOf<ImageCapture?>(null) }
    var permisoCamara by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(contexto, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED,
        )
    }
    val pedirCamara = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { concedido ->
        permisoCamara = concedido
    }
    val galeria = rememberLauncherForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if (uri != null) modelo.usarUri(uri)
    }
    val foto = modelo.foto
    val render = remember(foto) { foto?.asImageBitmap() }
    var especieClave by rememberSaveable { mutableStateOf("") }
    var nota by rememberSaveable { mutableStateOf("") }
    val candidatos = candidatosDe(modelo.resultados)
    val claveActual = especieClave.takeIf { clave ->
        candidatos.any { claveCientifica(it.cientifico) == clave }
    } ?: candidatos.firstOrNull()?.let { claveCientifica(it.cientifico) }.orEmpty()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .imePadding()
            .verticalScroll(desplazamiento)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Plantas", style = MaterialTheme.typography.headlineLarge, color = MaterialTheme.colorScheme.primary)
            OutlinedButton(onClick = alSalir, enabled = !modelo.comparando && !guardando) {
                Text("Salir")
            }
        }
        Text(estudiante, style = MaterialTheme.typography.titleMedium)
        Text(
            "Compara una foto con los modelos del teléfono. Los ResNet, si los marcas, se consultan en el PC.",
            style = MaterialTheme.typography.bodyLarge,
        )
        OutlinedButton(onClick = alHistorial, enabled = !modelo.comparando) {
            Text("Historial")
        }
        if (permisoCamara) {
            VistaCamara(
                activa = true,
                alCaptura = { camaraLista = it },
                alError = modelo::mostrarAviso,
            )
        } else {
            Text("La cámara pide permiso. La galería funciona sin él.")
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            Button(
                onClick = {
                    val captura = camaraLista
                    when {
                        !permisoCamara -> pedirCamara.launch(Manifest.permission.CAMERA)
                        captura != null -> tomarFoto(
                            contexto = contexto,
                            captura = captura,
                            ejecutor = ContextCompat.getMainExecutor(contexto),
                            alFoto = modelo::usarArchivo,
                            alError = modelo::mostrarAviso,
                        )
                        else -> modelo.mostrarAviso("La cámara todavía se está abriendo.")
                    }
                },
                enabled = !modelo.comparando,
            ) {
                Text(if (permisoCamara) "Tomar foto" else "Permitir cámara")
            }
            OutlinedButton(
                onClick = {
                    galeria.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                },
                enabled = !modelo.comparando,
            ) {
                Text("Galería")
            }
        }
        if (render != null) {
            Image(
                bitmap = render,
                contentDescription = "Foto recortada al centro",
                modifier = Modifier
                    .fillMaxWidth()
                    .height(280.dp)
                    .clip(RoundedCornerShape(16.dp)),
                contentScale = ContentScale.Crop,
            )
        }
        Text("Herbify", style = MaterialTheme.typography.titleMedium)
        Chips(modelo, "herbify")
        Text("Colombia", style = MaterialTheme.typography.titleMedium)
        Chips(modelo, "regional")
        Text("Servidor del PC", style = MaterialTheme.typography.titleMedium)
        Text(
            "Opcional. Si marcas un ResNet, la foto se envía a esa dirección.",
            style = MaterialTheme.typography.bodyMedium,
        )
        OutlinedTextField(
            value = modelo.url,
            onValueChange = modelo::cambiarUrl,
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            label = { Text("Dirección") },
            placeholder = { Text("http://192.168.1.10:5050") },
        )
        if (modelo.errorCatalogo != null) {
            Text(modelo.errorCatalogo ?: "", color = MaterialTheme.colorScheme.error)
        }
        Button(
            onClick = modelo::comparar,
            enabled = foto != null && modelo.seleccion.isNotEmpty() && !modelo.comparando,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (modelo.comparando) "Comparando…" else "Comparar")
        }
        if (modelo.comparando) {
            LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
        }
        modelo.aviso?.let { texto ->
            Card(modifier = Modifier.fillMaxWidth()) {
                Text(texto, modifier = Modifier.padding(16.dp))
            }
        }
        Avisos(modelo)
        modelo.delegado?.let { modo ->
            Text(
                when (modo) {
                    "gpu" -> "Inferencia local en la GPU."
                    "mixto" -> "Parte de la inferencia local pasó a la CPU con 4 hilos."
                    else -> "La GPU no respondió; la inferencia local usó la CPU con 4 hilos."
                },
                style = MaterialTheme.typography.bodySmall,
            )
        }
        modelo.resultados.forEach { resultado ->
            TarjetaResultado(resultado)
        }
        if (!modelo.comparando && foto != null && candidatos.isNotEmpty()) {
            Text("Guardar en el historial", style = MaterialTheme.typography.titleMedium)
            Text("Elige la especie que vas a registrar en el trabajo de campo.")
            FlowRow(
                horizontalArrangement = Arrangement.spacedBy(8.dp),
                verticalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                candidatos.forEach { etiqueta ->
                    val clave = claveCientifica(etiqueta.cientifico)
                    FilterChip(
                        selected = clave == claveActual,
                        onClick = { especieClave = clave },
                        label = { Text(titulo(etiqueta.comun)) },
                    )
                }
            }
            OutlinedTextField(
                value = nota,
                onValueChange = { nota = it },
                modifier = Modifier.fillMaxWidth(),
                label = { Text("Nota de campo") },
            )
            avisoCampo?.let { texto ->
                Text(texto)
            }
            Button(
                onClick = {
                    val elegida = candidatos.find { claveCientifica(it.cientifico) == claveActual }
                    val imagen = foto
                    if (elegida != null && imagen != null) {
                        val modelos = modelo.resultados
                            .filter { it.top.isNotEmpty() }
                            .joinToString(", ") { it.ficha.nombre }
                        alGuardar(imagen, elegida, modelos, nota)
                        nota = ""
                    }
                },
                enabled = !guardando && candidatos.isNotEmpty(),
                modifier = Modifier.fillMaxWidth(),
            ) {
                Text(if (guardando) "Guardando…" else "Guardar análisis")
            }
        }
        Spacer(Modifier.height(12.dp))
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun Chips(modelo: PlantasViewModel, vocab: String) {
    FlowRow(
        horizontalArrangement = Arrangement.spacedBy(8.dp),
        verticalArrangement = Arrangement.spacedBy(8.dp),
    ) {
        modelo.catalogo.filter { it.vocab == vocab }.forEach { ficha ->
            val extra = if (ficha.remoto) " · PC" else ""
            FilterChip(
                selected = ficha.id in modelo.seleccion,
                onClick = { modelo.alternar(ficha.id) },
                enabled = !modelo.comparando && (ficha.remoto || ficha.archivoTflite != null),
                label = { Text("${ficha.nombre} · ${porcentaje(ficha.accuracy)}$extra") },
            )
        }
    }
}

@Composable
private fun Avisos(modelo: PlantasViewModel) {
    val herbify = modelo.acuerdoHerbify
    val colombia = modelo.acuerdoColombia
    val nombre = modelo.acuerdoNombre
    if (herbify == null && colombia == null && nombre != true) return
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        if (herbify == true) {
            Banner("Los modelos de Herbify coinciden en la especie.")
        } else if (herbify == false) {
            Banner("Los modelos de Herbify no coinciden entre sí.")
        }
        if (colombia == true) {
            Banner("Los modelos de Colombia coinciden en la especie.")
        } else if (colombia == false) {
            Banner("Los modelos de Colombia no coinciden entre sí.")
        }
        if (nombre == true) {
            Banner("Herbify y Colombia coinciden en el nombre científico.")
        }
    }
}

@Composable
private fun Banner(texto: String) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Text(texto, modifier = Modifier.padding(16.dp), style = MaterialTheme.typography.titleSmall)
    }
}

@Composable
private fun TarjetaResultado(resultado: Resultado) {
    Card(modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Text(resultado.ficha.nombre, style = MaterialTheme.typography.titleMedium)
            Text(resultado.ficha.dataset, style = MaterialTheme.typography.labelLarge)
            val error = resultado.error
            if (error != null) {
                Text(error)
            } else {
                resultado.top.forEach { etiqueta ->
                    Text(titulo(etiqueta.comun), style = MaterialTheme.typography.bodyLarge)
                    Text(etiqueta.cientifico, fontStyle = FontStyle.Italic)
                    Text(porcentaje(etiqueta.prob.toDouble()))
                    Barra(etiqueta.prob)
                }
            }
        }
    }
}

@Composable
private fun Barra(prob: Float) {
    Box(
        Modifier
            .fillMaxWidth()
            .height(8.dp)
            .clip(RoundedCornerShape(4.dp))
            .background(MaterialTheme.colorScheme.surfaceVariant),
    ) {
        Box(
            Modifier
                .fillMaxWidth(prob.coerceIn(0f, 1f))
                .fillMaxHeight()
                .background(MaterialTheme.colorScheme.primary),
        )
    }
}

@Composable
private fun VistaCamara(
    activa: Boolean,
    alCaptura: (ImageCapture) -> Unit,
    alError: (String) -> Unit,
) {
    val contexto = LocalContext.current
    val ciclo = LocalLifecycleOwner.current
    val aviso = rememberUpdatedState(alCaptura)
    val error = rememberUpdatedState(alError)
    val vista = remember {
        PreviewView(contexto).apply {
            scaleType = PreviewView.ScaleType.FILL_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    DisposableEffect(activa, ciclo) {
        if (!activa) {
            return@DisposableEffect onDispose { }
        }
        val futuro = ProcessCameraProvider.getInstance(contexto)
        val ejecutor = ContextCompat.getMainExecutor(contexto)
        val tarea = Runnable {
            runCatching {
                val proveedor = futuro.get()
                val preview = Preview.Builder().build().also { it.surfaceProvider = vista.surfaceProvider }
                val captura = ImageCapture.Builder()
                    .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                    .build()
                proveedor.unbindAll()
                proveedor.bindToLifecycle(ciclo, CameraSelector.DEFAULT_BACK_CAMERA, preview, captura)
                aviso.value(captura)
            }.onFailure { fallo ->
                error.value(fallo.message ?: "No se pudo abrir la cámara")
            }
        }
        futuro.addListener(tarea, ejecutor)
        onDispose {
            if (futuro.isDone) {
                runCatching { futuro.get().unbindAll() }
            }
        }
    }
    AndroidView(
        factory = { vista },
        modifier = Modifier
            .fillMaxWidth()
            .height(320.dp)
            .clip(RoundedCornerShape(16.dp)),
    )
}

private fun tomarFoto(
    contexto: android.content.Context,
    captura: ImageCapture,
    ejecutor: Executor,
    alFoto: (File) -> Unit,
    alError: (String) -> Unit,
) {
    val archivo = File(contexto.cacheDir, "captura.jpg")
    val opciones = ImageCapture.OutputFileOptions.Builder(archivo).build()
    captura.takePicture(
        opciones,
        ejecutor,
        object : ImageCapture.OnImageSavedCallback {
            override fun onImageSaved(outputFileResults: ImageCapture.OutputFileResults) {
                alFoto(archivo)
            }

            override fun onError(exception: ImageCaptureException) {
                alError(exception.message ?: "No se pudo tomar la foto")
            }
        },
    )
}

private fun porcentaje(valor: Double): String {
    return String.format(Locale("es"), "%.1f %%", valor * 100.0)
}

private fun candidatosDe(resultados: List<Resultado>): List<Etiqueta> {
    val mejores = linkedMapOf<String, Etiqueta>()
    for (resultado in resultados) {
        for (etiqueta in resultado.top) {
            val clave = claveCientifica(etiqueta.cientifico)
            val previo = mejores[clave]
            if (previo == null || etiqueta.prob > previo.prob) mejores[clave] = etiqueta
        }
    }
    return mejores.values.sortedByDescending { it.prob }
}

private fun titulo(texto: String): String {
    return texto.replaceFirstChar { letra ->
        if (letra.isLowerCase()) letra.titlecase(Locale("es")) else letra.toString()
    }
}
