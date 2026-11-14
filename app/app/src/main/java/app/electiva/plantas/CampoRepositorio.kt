package app.electiva.plantas

import android.content.Context
import android.graphics.Bitmap
import java.io.File
import java.util.UUID

class CampoRepositorio(contexto: Context) {
    private val app = contexto.applicationContext
    private val dao = CampoDb.obtener(app).dao()
    private val fotos = File(app.filesDir, "analisis").apply { mkdirs() }

    suspend fun registrar(nombre: String, contrasena: String): Usuario {
        val visible = nombre.trim()
        val clave = visible.lowercase()
        if (clave.length < 2) error("El nombre necesita al menos 2 caracteres")
        if (contrasena.length < 4) error("La contraseña necesita al menos 4 caracteres")
        if (dao.usuarioPorNombre(clave) != null) error("Ese nombre ya tiene cuenta en este teléfono")
        val (sal, hash, iteraciones) = Claves.crear(contrasena)
        val id = dao.insertarUsuario(
            Usuario(
                nombre = visible,
                nombreClave = clave,
                sal = sal,
                hash = hash,
                iteraciones = iteraciones,
                token = Claves.token(),
            ),
        )
        return dao.usuario(id) ?: error("No se pudo crear la cuenta")
    }

    suspend fun entrar(nombre: String, contrasena: String): Usuario {
        val usuario = dao.usuarioPorNombre(nombre.trim().lowercase())
            ?: error("No hay una cuenta con ese nombre en este teléfono")
        if (!Claves.coincide(contrasena, usuario.sal, usuario.hash, usuario.iteraciones)) {
            error("La contraseña no coincide")
        }
        return usuario
    }

    suspend fun usuario(id: Long): Usuario? = dao.usuario(id)

    suspend fun listar(usuarioId: Long): List<Analisis> = dao.analisisDe(usuarioId)

    suspend fun guardar(
        usuarioId: Long,
        bitmap: Bitmap,
        cientifico: String,
        comun: String,
        probabilidad: Float,
        modelos: String,
        nota: String,
    ): Analisis {
        val id = UUID.randomUUID().toString()
        val archivo = File(fotos, "$id.jpg")
        archivo.outputStream().use { salida ->
            salida.write(Imagenes.aJpeg(bitmap))
        }
        val analisis = Analisis(
            id = id,
            usuarioId = usuarioId,
            fecha = System.currentTimeMillis(),
            cientifico = cientifico,
            comun = comun,
            probabilidad = probabilidad,
            modelos = modelos,
            nota = nota.trim(),
            rutaFoto = archivo.absolutePath,
            sincronizado = false,
        )
        dao.insertarAnalisis(analisis)
        return analisis
    }

    suspend fun pendientes(usuarioId: Long): List<Analisis> = dao.pendientes(usuarioId)

    suspend fun marcarSincronizado(id: String) = dao.marcarSincronizado(id)

    suspend fun existe(id: String): Boolean = dao.analisis(id) != null

    suspend fun guardarRemoto(usuarioId: Long, remoto: AnalisisRemoto, jpeg: ByteArray) {
        if (dao.analisis(remoto.id) != null) return
        val archivo = File(fotos, "${remoto.id}.jpg")
        archivo.writeBytes(jpeg)
        dao.insertarAnalisis(
            Analisis(
                id = remoto.id,
                usuarioId = usuarioId,
                fecha = remoto.fecha,
                cientifico = remoto.cientifico,
                comun = remoto.comun,
                probabilidad = remoto.probabilidad,
                modelos = remoto.modelos,
                nota = remoto.nota,
                rutaFoto = archivo.absolutePath,
                sincronizado = true,
            ),
        )
    }
}
