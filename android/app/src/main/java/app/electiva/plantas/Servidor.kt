package app.electiva.plantas

import android.util.Base64
import org.json.JSONArray
import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder

data class AnalisisRemoto(
    val id: String,
    val fecha: Long,
    val cientifico: String,
    val comun: String,
    val probabilidad: Float,
    val modelos: String,
    val nota: String,
    val tieneFoto: Boolean,
)

object Servidor {
    fun predecir(urlBase: String, jpeg: ByteArray, fichas: List<Ficha>): List<Resultado> {
        val destino = urlBase.trim().trimEnd('/') + "/api/predecir"
        val limite = "----Plantas${System.currentTimeMillis()}"
        val conexion = (URL(destino).openConnection() as HttpURLConnection).apply {
            requestMethod = "POST"
            doOutput = true
            connectTimeout = 15_000
            readTimeout = 180_000
            setRequestProperty("Content-Type", "multipart/form-data; boundary=$limite")
        }
        try {
            conexion.outputStream.use { salida ->
                salida.write("--$limite\r\n".toByteArray())
                salida.write(
                    "Content-Disposition: form-data; name=\"imagen\"; filename=\"planta.jpg\"\r\n".toByteArray(),
                )
                salida.write("Content-Type: image/jpeg\r\n\r\n".toByteArray())
                salida.write(jpeg)
                salida.write("\r\n".toByteArray())
                val ids = fichas.joinToString(",") { it.id }
                salida.write("--$limite\r\n".toByteArray())
                salida.write("Content-Disposition: form-data; name=\"modelos\"\r\n\r\n".toByteArray())
                salida.write(ids.toByteArray(Charsets.UTF_8))
                salida.write("\r\n".toByteArray())
                salida.write("--$limite--\r\n".toByteArray())
            }
            val codigo = conexion.responseCode
            val cuerpo = (if (codigo in 200..299) conexion.inputStream else conexion.errorStream)
                ?.bufferedReader()
                ?.use { it.readText() }
                .orEmpty()
            if (codigo !in 200..299) {
                val mensaje = runCatching { JSONObject(cuerpo).optString("error") }.getOrNull()
                throw IOException(mensaje?.takeIf { it.isNotBlank() } ?: "El PC respondió $codigo")
            }
            return parsear(cuerpo, fichas)
        } finally {
            conexion.disconnect()
        }
    }

    fun sincronizar(
        urlBase: String,
        usuario: String,
        token: String,
        pendientes: List<Pair<Analisis, ByteArray>>,
    ): List<String> {
        val cuerpo = JSONObject()
        cuerpo.put("usuario", usuario)
        cuerpo.put("token", token)
        val arreglo = JSONArray()
        for ((analisis, jpeg) in pendientes) {
            val item = JSONObject()
            item.put("id", analisis.id)
            item.put("fecha", analisis.fecha)
            item.put("cientifico", analisis.cientifico)
            item.put("comun", analisis.comun)
            item.put("probabilidad", analisis.probabilidad.toDouble())
            item.put("modelos", analisis.modelos)
            item.put("nota", analisis.nota)
            item.put("foto", Base64.encodeToString(jpeg, Base64.NO_WRAP))
            arreglo.put(item)
        }
        cuerpo.put("analisis", arreglo)
        val respuesta = pedir(
            url = urlBase.trim().trimEnd('/') + "/api/campo/sync",
            metodo = "POST",
            json = cuerpo.toString(),
        )
        val recibidos = respuesta.optJSONArray("recibidos") ?: JSONArray()
        return buildList {
            for (indice in 0 until recibidos.length()) add(recibidos.getString(indice))
        }
    }

    fun historial(urlBase: String, usuario: String, token: String): List<AnalisisRemoto> {
        val consulta = "usuario=${enc(usuario)}&token=${enc(token)}"
        val respuesta = pedir(
            url = urlBase.trim().trimEnd('/') + "/api/campo/historial?$consulta",
            metodo = "GET",
            json = null,
        )
        val arreglo = respuesta.optJSONArray("analisis") ?: JSONArray()
        return buildList {
            for (indice in 0 until arreglo.length()) {
                val item = arreglo.getJSONObject(indice)
                add(
                    AnalisisRemoto(
                        id = item.getString("id"),
                        fecha = item.getLong("fecha"),
                        cientifico = item.optString("cientifico"),
                        comun = item.optString("comun"),
                        probabilidad = item.optDouble("probabilidad").toFloat(),
                        modelos = item.optString("modelos"),
                        nota = item.optString("nota"),
                        tieneFoto = item.optBoolean("tiene_foto"),
                    ),
                )
            }
        }
    }

    fun foto(urlBase: String, usuario: String, token: String, id: String): ByteArray {
        val consulta = "usuario=${enc(usuario)}&token=${enc(token)}"
        val destino = urlBase.trim().trimEnd('/') + "/api/campo/foto/${enc(id)}?$consulta"
        val conexion = abrir(destino, "GET")
        try {
            val codigo = conexion.responseCode
            val bytes = (if (codigo in 200..299) conexion.inputStream else conexion.errorStream)
                ?.use { it.readBytes() }
                ?: byteArrayOf()
            if (codigo !in 200..299) {
                throw IOException("El PC no envió la foto ($codigo)")
            }
            return bytes
        } finally {
            conexion.disconnect()
        }
    }

    private fun enc(valor: String): String = URLEncoder.encode(valor, Charsets.UTF_8.name())

    private fun pedir(url: String, metodo: String, json: String?): JSONObject {
        val conexion = abrir(url, metodo)
        try {
            if (json != null) {
                conexion.doOutput = true
                conexion.setRequestProperty("Content-Type", "application/json; charset=utf-8")
                conexion.outputStream.use { it.write(json.toByteArray(Charsets.UTF_8)) }
            }
            val codigo = conexion.responseCode
            val cuerpo = (if (codigo in 200..299) conexion.inputStream else conexion.errorStream)
                ?.bufferedReader()
                ?.use { it.readText() }
                .orEmpty()
            if (codigo !in 200..299) {
                val mensaje = runCatching { JSONObject(cuerpo).optString("error") }.getOrNull()
                throw IOException(mensaje?.takeIf { it.isNotBlank() } ?: "El PC respondió $codigo")
            }
            return JSONObject(cuerpo.ifBlank { "{}" })
        } finally {
            conexion.disconnect()
        }
    }

    private fun abrir(destino: String, metodo: String): HttpURLConnection {
        return (URL(destino).openConnection() as HttpURLConnection).apply {
            requestMethod = metodo
            connectTimeout = 15_000
            readTimeout = 120_000
        }
    }

    private fun parsear(cuerpo: String, fichas: List<Ficha>): List<Resultado> {
        val raiz = JSONObject(cuerpo)
        val modelos = raiz.optJSONObject("modelos")
            ?: throw IOException(raiz.optString("error").ifBlank { "El PC no devolvió modelos" })
        return fichas.map { ficha ->
            val bloque = modelos.optJSONObject(ficha.id)
            if (bloque == null) {
                Resultado(ficha, emptyList(), "El PC no devolvió este modelo")
            } else if (!bloque.optBoolean("disponible", false)) {
                Resultado(ficha, emptyList(), bloque.optString("error", "Este modelo no está disponible"))
            } else {
                val top = bloque.optJSONArray("top")
                val etiquetas = buildList {
                    if (top != null) {
                        for (indice in 0 until top.length()) {
                            val item = top.getJSONObject(indice)
                            add(
                                Etiqueta(
                                    id = item.optString("id"),
                                    cientifico = item.optString("cientifico"),
                                    comun = item.optString("comun"),
                                    prob = item.optDouble("prob").toFloat(),
                                ),
                            )
                        }
                    }
                }
                Resultado(ficha, etiquetas, null)
            }
        }
    }
}
