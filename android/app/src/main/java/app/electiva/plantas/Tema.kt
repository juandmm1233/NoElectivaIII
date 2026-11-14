package app.electiva.plantas

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color

private val Verde = Color(0xFF1B4332)
private val Hoja = Color(0xFF2D6A4F)
private val Papel = Color(0xFFF6F3EA)
private val Tinta = Color(0xFF1C2416)
private val Arcilla = Color(0xFF9C6644)

private val colores = lightColorScheme(
    primary = Verde,
    onPrimary = Color.White,
    secondary = Hoja,
    onSecondary = Color.White,
    tertiary = Arcilla,
    background = Papel,
    surface = Color.White,
    onBackground = Tinta,
    onSurface = Tinta,
)

@Composable
fun TemaPlantas(contenido: @Composable () -> Unit) {
    MaterialTheme(colorScheme = colores, content = contenido)
}
