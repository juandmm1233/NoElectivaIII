package app.electiva.plantas

import android.app.Application
import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

enum class PantallaCampo {
    ANALIZAR,
    HISTORIAL,
    DETALLE,
}

class CampoViewModel(app: Application) : AndroidViewModel(app) {
    private val ajustes = Ajustes(app)
    private val repo = CampoRepositorio(app)

    var sesionLista by mutableStateOf(false)
        private set
    var usuarioId by mutableLongStateOf(0L)
        private set
    var nombre by mutableStateOf("")
        private set
    var pantalla by mutableStateOf(PantallaCampo.ANALIZAR)
        private set
    var historial by mutableStateOf<List<Analisis>>(emptyList())
        private set
    var detalle by mutableStateOf<Analisis?>(null)
        private set
    var ficha by mutableStateOf<FichaMedica?>(null)
        private set
    var ocupado by mutableStateOf(false)
        private set
    var error by mutableStateOf<String?>(null)
        private set
    var aviso by mutableStateOf<String?>(null)
        private set

    private var token = ""

    init {
        viewModelScope.launch {
            val id = ajustes.leerUsuario()
            if (id > 0L) {
                val usuario = repo.usuario(id)
                if (usuario != null) {
                    aplicar(usuario)
                    sincronizar()
                } else {
                    ajustes.guardarUsuario(0L)
                }
            }
            sesionLista = true
        }
    }

    fun crear(nombreCuenta: String, contrasena: String) {
        if (ocupado) return
        viewModelScope.launch {
            ocupado = true
            error = null
            val resultado = runCatching { repo.registrar(nombreCuenta, contrasena) }
            resultado.onSuccess { usuario ->
                ajustes.guardarUsuario(usuario.id)
                aplicar(usuario)
            }
            resultado.onFailure { error = it.message ?: "No se pudo crear la cuenta" }
            ocupado = false
        }
    }

    fun entrar(nombreCuenta: String, contrasena: String) {
        if (ocupado) return
        viewModelScope.launch {
            ocupado = true
            error = null
            val resultado = runCatching { repo.entrar(nombreCuenta, contrasena) }
            resultado.onSuccess { usuario ->
                ajustes.guardarUsuario(usuario.id)
                aplicar(usuario)
                sincronizar()
            }
            resultado.onFailure { error = it.message ?: "No se pudo entrar" }
            ocupado = false
        }
    }

    fun salir() {
        viewModelScope.launch {
            ajustes.guardarUsuario(0L)
            usuarioId = 0L
            nombre = ""
            token = ""
            historial = emptyList()
            detalle = null
            ficha = null
            aviso = null
            pantalla = PantallaCampo.ANALIZAR
        }
    }

    fun verHistorial() {
        pantalla = PantallaCampo.HISTORIAL
        viewModelScope.launch {
            historial = repo.listar(usuarioId)
            sincronizar()
        }
    }

    fun verAnalisis() {
        pantalla = PantallaCampo.ANALIZAR
        detalle = null
    }

    fun abrir(analisis: Analisis) {
        detalle = analisis
        ficha = Fichas.buscar(getApplication(), analisis.cientifico)
        pantalla = PantallaCampo.DETALLE
    }

    fun guardar(
        bitmap: Bitmap,
        etiqueta: Etiqueta,
        modelos: String,
        nota: String,
        url: String,
    ) {
        if (usuarioId == 0L || ocupado) return
        viewModelScope.launch {
            ocupado = true
            aviso = null
            val guardado = runCatching {
                repo.guardar(
                    usuarioId = usuarioId,
                    bitmap = bitmap,
                    cientifico = etiqueta.cientifico,
                    comun = etiqueta.comun,
                    probabilidad = etiqueta.prob,
                    modelos = modelos,
                    nota = nota,
                )
            }
            guardado.onSuccess {
                historial = repo.listar(usuarioId)
                val copia = sincronizar(url)
                aviso = if (copia) {
                    "Análisis guardado y copiado al PC."
                } else {
                    "Análisis guardado en el teléfono. La copia al PC queda pendiente."
                }
            }
            guardado.onFailure { aviso = it.message ?: "No se pudo guardar el análisis" }
            ocupado = false
        }
    }

    private suspend fun aplicar(usuario: Usuario) {
        usuarioId = usuario.id
        nombre = usuario.nombre
        token = usuario.token
        historial = repo.listar(usuario.id)
        pantalla = PantallaCampo.ANALIZAR
        error = null
    }

    private suspend fun sincronizar(urlForzada: String? = null): Boolean {
        if (usuarioId == 0L || token.isBlank()) return false
        val url = (urlForzada ?: ajustes.leerUrl()).trim()
        if (!url.startsWith("http://") && !url.startsWith("https://")) return false
        val empuje = runCatching {
            withContext(Dispatchers.IO) {
                val pendientes = repo.pendientes(usuarioId)
                for (lote in pendientes.chunked(3)) {
                    val pares = lote.map { analisis ->
                        val bytes = File(analisis.rutaFoto).takeIf { it.isFile }?.readBytes() ?: byteArrayOf()
                        analisis to bytes
                    }
                    val recibidos = Servidor.sincronizar(url, nombre, token, pares)
                    recibidos.forEach { repo.marcarSincronizado(it) }
                }
            }
        }
        val bajada = runCatching {
            withContext(Dispatchers.IO) {
                val remotos = Servidor.historial(url, nombre, token)
                for (remoto in remotos) {
                    if (repo.existe(remoto.id) || !remoto.tieneFoto) continue
                    val jpeg = Servidor.foto(url, nombre, token, remoto.id)
                    if (jpeg.isNotEmpty()) repo.guardarRemoto(usuarioId, remoto, jpeg)
                }
            }
        }
        if (empuje.isSuccess || bajada.isSuccess) {
            historial = repo.listar(usuarioId)
        }
        return empuje.isSuccess
    }
}
