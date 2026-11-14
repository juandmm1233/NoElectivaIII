package app.electiva.plantas

import org.json.JSONObject
import java.io.IOException
import java.net.HttpURLConnection
import java.net.URL

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
