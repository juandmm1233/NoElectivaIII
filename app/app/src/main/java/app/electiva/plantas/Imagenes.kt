package app.electiva.plantas

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import androidx.exifinterface.media.ExifInterface
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.min

object Imagenes {
    fun desdeUri(contexto: Context, uri: Uri): Bitmap {
        val resolver = contexto.contentResolver
        val bitmap = decodificar { resolver.openInputStream(uri) ?: error("No se pudo abrir la imagen") }
        val grados = resolver.openInputStream(uri)?.use { entrada ->
            ExifInterface(entrada).rotationDegrees
        } ?: 0
        return preparar(rotar(bitmap, grados))
    }

    fun desdeArchivo(archivo: File): Bitmap {
        val bitmap = decodificar { archivo.inputStream() }
        val grados = ExifInterface(archivo.absolutePath).rotationDegrees
        return preparar(rotar(bitmap, grados))
    }

    fun aTensor(bitmap: Bitmap): ByteBuffer {
        val escalada = if (bitmap.width == 224 && bitmap.height == 224) {
            bitmap
        } else {
            Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        }
        val pixeles = IntArray(224 * 224)
        escalada.getPixels(pixeles, 0, 224, 0, 0, 224, 224)
        val buffer = ByteBuffer.allocateDirect(224 * 224 * 3 * 4).order(ByteOrder.nativeOrder())
        for (color in pixeles) {
            buffer.putFloat(((color shr 16) and 0xFF).toFloat())
            buffer.putFloat(((color shr 8) and 0xFF).toFloat())
            buffer.putFloat((color and 0xFF).toFloat())
        }
        buffer.rewind()
        return buffer
    }

    fun aJpeg(bitmap: Bitmap): ByteArray {
        val salida = ByteArrayOutputStream()
        val cuadrado = if (bitmap.width > 1024) {
            Bitmap.createScaledBitmap(bitmap, 1024, 1024, true)
        } else {
            bitmap
        }
        cuadrado.compress(Bitmap.CompressFormat.JPEG, 92, salida)
        return salida.toByteArray()
    }

    private fun preparar(bitmap: Bitmap): Bitmap {
        val lado = min(bitmap.width, bitmap.height)
        val recorte = Bitmap.createBitmap(
            bitmap,
            (bitmap.width - lado) / 2,
            (bitmap.height - lado) / 2,
            lado,
            lado,
        )
        if (recorte.width <= 1024) return recorte
        return Bitmap.createScaledBitmap(recorte, 1024, 1024, true)
    }

    private fun rotar(bitmap: Bitmap, grados: Int): Bitmap {
        if (grados == 0) return bitmap
        val matriz = Matrix().apply { postRotate(grados.toFloat()) }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matriz, true)
    }

    private fun decodificar(abrir: () -> InputStream): Bitmap {
        val limites = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        abrir().use { BitmapFactory.decodeStream(it, null, limites) }
        var muestra = 1
        while (limites.outWidth / muestra > 2000 || limites.outHeight / muestra > 2000) {
            muestra *= 2
        }
        val opciones = BitmapFactory.Options().apply {
            inSampleSize = muestra
            inPreferredConfig = Bitmap.Config.ARGB_8888
        }
        return abrir().use { BitmapFactory.decodeStream(it, null, opciones) }
            ?: error("La imagen no se pudo leer")
    }
}
