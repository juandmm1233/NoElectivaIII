package app.electiva.plantas

import android.content.Context

data class FichaMedica(
    val cientifico: String,
    val comun: String,
    val parte: String,
    val usos: String,
    val precaucion: String,
    val fuente: String,
) {
    val documentada: Boolean
        get() = usos.isNotBlank() && fuente.isNotBlank()
}

object Fichas {
    const val AVISO =
        "Estos datos son usos tradicionales para el estudio de campo. No son un diagnóstico ni una recomendación de uso."

    private val alias = mapOf(
        "aloe barbadensis miller" to "aloe vera",
        "aloe barbadensis" to "aloe vera",
        "rosmarinus officinalis" to "salvia rosmarinus",
        "plectranthus amboinicus" to "coleus amboinicus",
        "chenopodium ambrosioides" to "dysphania ambrosioides",
    )

    private var indice: Map<String, FichaMedica>? = null

    fun buscar(contexto: Context, cientifico: String): FichaMedica? {
        val mapa = indice ?: cargar(contexto).also { indice = it }
        val clave = alias[claveCientifica(cientifico)] ?: claveCientifica(cientifico)
        val ficha = mapa[clave] ?: return null
        return ficha.takeIf { it.documentada }
    }

    private fun cargar(contexto: Context): Map<String, FichaMedica> {
        val texto = contexto.assets.open("fichas_medicinales.csv").bufferedReader().use { it.readText() }
        val filas = leerCsv(texto)
        if (filas.isEmpty()) return emptyMap()
        val encabezado = filas.first()
        return filas.drop(1).mapNotNull { fila ->
            if (fila.size < encabezado.size) return@mapNotNull null
            val datos = encabezado.zip(fila).toMap()
            val cientifico = datos["nombre_cientifico"].orEmpty().trim()
            if (cientifico.isBlank()) return@mapNotNull null
            claveCientifica(cientifico) to FichaMedica(
                cientifico = cientifico,
                comun = datos["nombre_comun"].orEmpty().trim(),
                parte = datos["parte_usada"].orEmpty().trim(),
                usos = datos["usos_tradicionales"].orEmpty().trim(),
                precaucion = datos["contraindicaciones"].orEmpty().trim(),
                fuente = datos["fuente"].orEmpty().trim(),
            )
        }.toMap()
    }

    private fun leerCsv(texto: String): List<List<String>> {
        val filas = mutableListOf<List<String>>()
        val celda = StringBuilder()
        val fila = mutableListOf<String>()
        var comillas = false
        var i = 0
        while (i < texto.length) {
            val caracter = texto[i]
            if (comillas) {
                if (caracter == '"') {
                    if (i + 1 < texto.length && texto[i + 1] == '"') {
                        celda.append('"')
                        i += 1
                    } else {
                        comillas = false
                    }
                } else {
                    celda.append(caracter)
                }
            } else {
                when (caracter) {
                    '"' -> comillas = true
                    ',' -> {
                        fila += celda.toString()
                        celda.clear()
                    }
                    '\n' -> {
                        fila += celda.toString()
                        celda.clear()
                        filas += fila.toList()
                        fila.clear()
                    }
                    '\r' -> Unit
                    else -> celda.append(caracter)
                }
            }
            i += 1
        }
        if (celda.isNotEmpty() || fila.isNotEmpty()) {
            fila += celda.toString()
            filas += fila.toList()
        }
        return filas
    }
}
