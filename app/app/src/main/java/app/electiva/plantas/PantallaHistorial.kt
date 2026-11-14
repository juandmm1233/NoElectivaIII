package app.electiva.plantas

import android.graphics.BitmapFactory
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Card
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp
import java.io.File
import java.time.Instant
import java.time.ZoneId
import java.time.format.DateTimeFormatter
import java.util.Locale

private val formatoFecha: DateTimeFormatter =
    DateTimeFormatter.ofPattern("d MMM yyyy, HH:mm", Locale("es")).withZone(ZoneId.systemDefault())

@Composable
fun PantallaHistorial(campo: CampoViewModel) {
    val desplazamiento = rememberScrollState()
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .verticalScroll(desplazamiento)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text("Historial", style = MaterialTheme.typography.headlineLarge, color = MaterialTheme.colorScheme.primary)
        Text(campo.nombre, style = MaterialTheme.typography.bodyLarge)
        OutlinedButton(onClick = campo::verAnalisis) {
            Text("Volver al análisis")
        }
        if (campo.historial.isEmpty()) {
            Text("Todavía no hay plantas guardadas. Después de comparar una foto, elige la especie y pulsa Guardar análisis.")
        }
        campo.historial.forEach { analisis ->
            Card(
                modifier = Modifier
                    .fillMaxWidth()
                    .clickable { campo.abrir(analisis) },
            ) {
                Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                    Text(tituloCampo(analisis.comun), style = MaterialTheme.typography.titleMedium)
                    Text(analisis.cientifico, fontStyle = FontStyle.Italic)
                    Text(formatoFecha.format(Instant.ofEpochMilli(analisis.fecha)))
                    Text(porcentajeCampo(analisis.probabilidad))
                    if (!analisis.sincronizado) {
                        Text("Pendiente de copiar al PC", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}

@Composable
fun PantallaDetalle(campo: CampoViewModel) {
    val analisis = campo.detalle ?: return
    val foto = remember(analisis.rutaFoto) {
        File(analisis.rutaFoto).takeIf { it.isFile }?.let { BitmapFactory.decodeFile(it.absolutePath) }
    }
    val desplazamiento = rememberScrollState()
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .verticalScroll(desplazamiento)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text(tituloCampo(analisis.comun), style = MaterialTheme.typography.headlineLarge, color = MaterialTheme.colorScheme.primary)
        Text(analisis.cientifico, fontStyle = FontStyle.Italic, style = MaterialTheme.typography.titleMedium)
        Text(formatoFecha.format(Instant.ofEpochMilli(analisis.fecha)))
        if (foto != null) {
            Image(
                bitmap = foto.asImageBitmap(),
                contentDescription = "Foto del análisis",
                modifier = Modifier
                    .fillMaxWidth()
                    .height(280.dp)
                    .clip(RoundedCornerShape(16.dp)),
                contentScale = ContentScale.Crop,
            )
        }
        Text("Probabilidad ${porcentajeCampo(analisis.probabilidad)}")
        if (analisis.modelos.isNotBlank()) {
            Text("Modelos: ${analisis.modelos}")
        }
        if (analisis.nota.isNotBlank()) {
            Text("Nota de campo")
            Text(analisis.nota)
        }
        Text(Fichas.AVISO, style = MaterialTheme.typography.bodyMedium)
        val ficha = campo.ficha
        if (ficha == null) {
            Text("Sin ficha documentada para esta especie.")
        } else {
            Text("Parte usada", style = MaterialTheme.typography.titleMedium)
            Text(ficha.parte.ifBlank { "No consta en la fuente." })
            Text("Usos tradicionales", style = MaterialTheme.typography.titleMedium)
            Text(ficha.usos)
            if (ficha.precaucion.isNotBlank()) {
                Text("Precaución", style = MaterialTheme.typography.titleMedium)
                Text(ficha.precaucion)
            }
            Text("Fuente", style = MaterialTheme.typography.titleMedium)
            Text(ficha.fuente)
        }
        Row {
            OutlinedButton(onClick = campo::verHistorial) {
                Text("Volver al historial")
            }
        }
    }
}

private fun tituloCampo(texto: String): String {
    return texto.replaceFirstChar { letra ->
        if (letra.isLowerCase()) letra.titlecase(Locale("es")) else letra.toString()
    }
}

private fun porcentajeCampo(valor: Float): String {
    return String.format(Locale("es"), "%.1f %%", valor * 100.0)
}
