package app.electiva.plantas

import android.content.Context
import org.json.JSONArray
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

object Inferencia {
    private val etiquetas = mutableMapOf<String, List<Triple<String, String, String>>>()

    fun correr(
        contexto: Context,
        ficha: Ficha,
        entrada: ByteBuffer,
        permitirGpu: Boolean,
    ): Pair<Resultado, Boolean> {
        val archivo = ficha.archivoTflite ?: return Resultado(ficha, emptyList(), "Este modelo no está en el teléfono") to false
        val clases = etiquetas.getOrPut(ficha.archivoClases) { leerClases(contexto, ficha.archivoClases) }
        if (permitirGpu) {
            try {
                val probs = ejecutar(contexto, archivo, entrada, gpu = true)
                return Resultado(ficha, top(probs, clases), null) to true
            } catch (_: Throwable) {
                entrada.rewind()
            }
        }
        return try {
            val probs = ejecutar(contexto, archivo, entrada, gpu = false)
            Resultado(ficha, top(probs, clases), null) to false
        } catch (error: Throwable) {
            Resultado(ficha, emptyList(), error.message ?: "Falló la inferencia") to false
        }
    }

    private fun ejecutar(contexto: Context, ruta: String, entrada: ByteBuffer, gpu: Boolean): FloatArray {
        val modelo = mapear(contexto, ruta)
        entrada.rewind()
        var delegado: GpuDelegate? = null
        var interprete: Interpreter? = null
        try {
            val opciones = Interpreter.Options()
            if (gpu) {
                delegado = GpuDelegate()
                opciones.addDelegate(delegado)
            } else {
                opciones.numThreads = 4
                opciones.setUseXNNPACK(true)
            }
            interprete = Interpreter(modelo, opciones)
            val forma = interprete.getOutputTensor(0).shape()
            if (forma.size <= 1) {
                val salida = FloatArray(forma.last())
                interprete.run(entrada, salida)
                return salida
            }
            val salida = Array(forma[0]) { FloatArray(forma.last()) }
            interprete.run(entrada, salida)
            return salida[0]
        } finally {
            interprete?.close()
            delegado?.close()
        }
    }

    private fun top(probs: FloatArray, clases: List<Triple<String, String, String>>): List<Etiqueta> {
        if (probs.size != clases.size) {
            error("El modelo devolvió ${probs.size} clases y la lista tiene ${clases.size}")
        }
        return probs.indices
            .sortedByDescending { probs[it] }
            .take(3)
            .map { indice ->
                val (id, cientifico, comun) = clases[indice]
                Etiqueta(id, cientifico, comun, probs[indice])
            }
    }

    private fun leerClases(contexto: Context, archivo: String): List<Triple<String, String, String>> {
        val texto = contexto.assets.open(archivo).bufferedReader().use { it.readText() }
        val arreglo = JSONArray(texto)
        return buildList {
            for (indice in 0 until arreglo.length()) {
                add(parsearClase(arreglo.getString(indice)))
            }
        }
    }

    private fun mapear(contexto: Context, ruta: String): MappedByteBuffer {
        contexto.assets.openFd(ruta).use { descriptor ->
            FileInputStream(descriptor.fileDescriptor).use { entrada ->
                return entrada.channel.map(
                    FileChannel.MapMode.READ_ONLY,
                    descriptor.startOffset,
                    descriptor.declaredLength,
                )
            }
        }
    }
}
