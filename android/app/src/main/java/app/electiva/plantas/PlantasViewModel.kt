package app.electiva.plantas

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class PlantasViewModel(app: Application) : AndroidViewModel(app) {
    private val ajustes = Ajustes(app)
    private var guardarUrl: Job? = null

    val catalogo: List<Ficha>
    var errorCatalogo by mutableStateOf<String?>(null)
        private set
    var foto by mutableStateOf<Bitmap?>(null)
        private set
    var url by mutableStateOf("")
        private set
    var seleccion by mutableStateOf(setOf<String>())
        private set
    var comparando by mutableStateOf(false)
        private set
    var resultados by mutableStateOf<List<Resultado>>(emptyList())
        private set
    var aviso by mutableStateOf<String?>(null)
        private set
    var delegado by mutableStateOf<String?>(null)
        private set
    var acuerdoHerbify by mutableStateOf<Boolean?>(null)
        private set
    var acuerdoColombia by mutableStateOf<Boolean?>(null)
        private set
    var acuerdoNombre by mutableStateOf<Boolean?>(null)
        private set

    init {
        val cargado = runCatching { Catalogo.leer(app) }
        catalogo = cargado.getOrElse { emptyList() }
        errorCatalogo = cargado.exceptionOrNull()?.message
        seleccion = catalogo.filter { !it.remoto && it.archivoTflite != null }.map { it.id }.toSet()
        viewModelScope.launch {
            url = ajustes.leerUrl()
        }
    }

    fun cambiarUrl(valor: String) {
        url = valor
        guardarUrl?.cancel()
        guardarUrl = viewModelScope.launch {
            delay(300)
            ajustes.guardarUrl(valor)
        }
    }

    fun mostrarAviso(texto: String) {
        aviso = texto
    }

    fun alternar(id: String) {
        if (comparando) return
        val ficha = catalogo.find { it.id == id } ?: return
        if (!ficha.remoto && ficha.archivoTflite == null) return
        seleccion = if (id in seleccion) seleccion - id else seleccion + id
    }

    fun usarUri(uri: Uri) {
        viewModelScope.launch {
            val resultado = runCatching {
                withContext(Dispatchers.IO) { Imagenes.desdeUri(getApplication(), uri) }
            }
            resultado.onSuccess { aplicarFoto(it) }
            resultado.onFailure { aviso = it.message ?: "No se pudo abrir la imagen" }
        }
    }

    fun usarArchivo(archivo: File) {
        viewModelScope.launch {
            val resultado = runCatching {
                withContext(Dispatchers.IO) { Imagenes.desdeArchivo(archivo) }
            }
            resultado.onSuccess { aplicarFoto(it) }
            resultado.onFailure { aviso = it.message ?: "No se pudo leer la foto" }
        }
    }

    fun comparar() {
        val imagen = foto ?: return
        if (seleccion.isEmpty() || comparando) return
        viewModelScope.launch {
            comparando = true
            aviso = null
            resultados = emptyList()
            delegado = null
            acuerdoHerbify = null
            acuerdoColombia = null
            acuerdoNombre = null
            ajustes.guardarUrl(url)
            val elegidos = catalogo.filter { it.id in seleccion }
            val locales = elegidos.filter { !it.remoto && it.archivoTflite != null }
            val remotos = elegidos.filter { it.remoto }
            val acumulado = mutableListOf<Resultado>()
            var gpuPermitida = true
            var algunoGpu = false
            var algunoCpu = false
            val entrada = withContext(Dispatchers.Default) { Imagenes.aTensor(imagen) }
            for (ficha in locales) {
                val (resultado, gpu) = withContext(Dispatchers.Default) {
                    Inferencia.correr(getApplication(), ficha, entrada, gpuPermitida)
                }
                if (gpu) algunoGpu = true else algunoCpu = true
                if (!gpu) gpuPermitida = false
                acumulado += resultado
                resultados = acumulado.toList()
            }
            if (locales.isNotEmpty()) {
                delegado = when {
                    algunoGpu && algunoCpu -> "mixto"
                    algunoGpu -> "gpu"
                    else -> "cpu"
                }
            }
            if (remotos.isNotEmpty()) {
                val base = url.trim()
                if (!base.startsWith("http://") && !base.startsWith("https://")) {
                    aviso = "Falta la dirección del PC. Los resultados de abajo son solo del teléfono."
                    remotos.forEach { acumulado += Resultado(it, emptyList(), "Sin servidor") }
                } else {
                    val consulta = runCatching {
                        val jpeg = withContext(Dispatchers.Default) { Imagenes.aJpeg(imagen) }
                        withContext(Dispatchers.IO) { Servidor.predecir(base, jpeg, remotos) }
                    }
                    consulta.onSuccess { lista ->
                        acumulado += lista
                        if (lista.none { it.top.isNotEmpty() }) {
                            aviso = lista.firstOrNull { !it.error.isNullOrBlank() }?.error
                                ?: "El PC no pudo usar los ResNet. Se muestran los modelos del teléfono."
                        }
                    }
                    consulta.onFailure { error ->
                        val detalle = error.message?.takeIf { it.isNotBlank() } ?: "sin respuesta"
                        aviso = "El PC no respondió ($detalle). Se muestran los modelos del teléfono."
                        remotos.forEach { acumulado += Resultado(it, emptyList(), "Sin respuesta del PC") }
                    }
                }
                resultados = acumulado.toList()
            }
            val acuerdo = Acuerdo.calcular(resultados, catalogo.map { it.id })
            acuerdoHerbify = acuerdo.herbify
            acuerdoColombia = acuerdo.colombia
            acuerdoNombre = acuerdo.nombre
            comparando = false
        }
    }

    private fun aplicarFoto(bitmap: Bitmap) {
        foto = bitmap
        resultados = emptyList()
        aviso = null
        delegado = null
        acuerdoHerbify = null
        acuerdoColombia = null
        acuerdoNombre = null
    }
}
