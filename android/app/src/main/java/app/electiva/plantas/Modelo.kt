package app.electiva.plantas

import android.content.Context
import org.json.JSONObject

data class Ficha(
    val id: String,
    val nombre: String,
    val arquitectura: String,
    val vocab: String,
    val dataset: String,
    val accuracy: Double,
    val archivoTflite: String?,
    val archivoClases: String,
    val remoto: Boolean,
) {
    val grupo: String
        get() = if (vocab == "herbify") "Herbify" else "Colombia"
}

data class Etiqueta(
    val id: String,
    val cientifico: String,
    val comun: String,
    val prob: Float,
)

data class Resultado(
    val ficha: Ficha,
    val top: List<Etiqueta>,
    val error: String?,
)

data class AcuerdoVocab(
    val herbify: Boolean?,
    val colombia: Boolean?,
    val nombre: Boolean?,
)

object Catalogo {
    fun leer(contexto: Context): List<Ficha> {
        val texto = contexto.assets.open("catalogo.json").bufferedReader().use { it.readText() }
        val arreglo = JSONObject(texto).getJSONArray("modelos")
        return buildList {
            for (indice in 0 until arreglo.length()) {
                val objeto = arreglo.getJSONObject(indice)
                val tflite = if (objeto.isNull("archivo_tflite")) {
                    null
                } else {
                    objeto.optString("archivo_tflite").ifBlank { null }
                }
                add(
                    Ficha(
                        id = objeto.getString("id"),
                        nombre = objeto.getString("nombre"),
                        arquitectura = objeto.getString("arquitectura"),
                        vocab = objeto.getString("vocab"),
                        dataset = objeto.getString("dataset"),
                        accuracy = objeto.getDouble("accuracy"),
                        archivoTflite = tflite,
                        archivoClases = objeto.getString("archivo_clases"),
                        remoto = objeto.optBoolean("remoto", tflite == null),
                    ),
                )
            }
        }
    }
}

fun parsearClase(clase: String): Triple<String, String, String> {
    if ("__" in clase) {
        val partes = clase.split("__", limit = 2)
        return Triple(clase, partes[0].replace('_', ' '), partes[1].replace('_', ' '))
    }
    val abre = clase.indexOf('(')
    if (abre >= 0) {
        val cientifico = clase.substring(0, abre).trim()
        val comun = clase.substring(abre)
            .replace("(", " ")
            .replace(")", " ")
            .replace('_', ' ')
            .split(Regex("\\s+"))
            .filter { it.isNotEmpty() }
            .joinToString(" ")
        return Triple(clase, cientifico, comun)
    }
    return Triple(clase, clase, clase)
}

fun claveCientifica(nombre: String): String {
    val base = nombre.substringBefore("(").substringBefore("__")
    return base.replace('_', ' ')
        .lowercase()
        .split(Regex("\\s+"))
        .filter { it.isNotEmpty() }
        .joinToString(" ")
}

object Acuerdo {
    fun calcular(resultados: List<Resultado>, orden: List<String>): AcuerdoVocab {
        val indice = orden.withIndex().associate { it.value to it.index }
        val ordenados = resultados.sortedBy { indice[it.ficha.id] ?: Int.MAX_VALUE }
        val herbify = ordenados.filter { it.ficha.vocab == "herbify" && it.top.isNotEmpty() }
        val regional = ordenados.filter { it.ficha.vocab == "regional" && it.top.isNotEmpty() }
        val acuerdoHerbify = if (herbify.size >= 2) {
            herbify.map { it.top.first().id }.toSet().size == 1
        } else {
            null
        }
        val acuerdoColombia = if (regional.size >= 2) {
            regional.map { it.top.first().id }.toSet().size == 1
        } else {
            null
        }
        val acuerdoNombre = if (herbify.isNotEmpty() && regional.isNotEmpty()) {
            val claves = herbify.map { claveCientifica(it.top.first().cientifico) }.toSet()
            claveCientifica(regional.first().top.first().cientifico) in claves
        } else {
            null
        }
        return AcuerdoVocab(acuerdoHerbify, acuerdoColombia, acuerdoNombre)
    }
}
