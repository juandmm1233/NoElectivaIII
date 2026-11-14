package app.electiva.plantas

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val campo: CampoViewModel = viewModel()
            val modelo: PlantasViewModel = viewModel()
            TemaPlantas {
                when {
                    !campo.sesionLista -> Box(Modifier.fillMaxSize())
                    campo.usuarioId == 0L -> PantallaAcceso(campo)
                    campo.pantalla == PantallaCampo.HISTORIAL -> PantallaHistorial(campo)
                    campo.pantalla == PantallaCampo.DETALLE -> PantallaDetalle(campo)
                    else -> PantallaPlantas(
                        modelo = modelo,
                        estudiante = campo.nombre,
                        avisoCampo = campo.aviso,
                        guardando = campo.ocupado,
                        alHistorial = campo::verHistorial,
                        alSalir = campo::salir,
                        alGuardar = { bitmap, etiqueta, modelos, nota ->
                            campo.guardar(bitmap, etiqueta, modelos, nota, modelo.url)
                        },
                    )
                }
            }
        }
    }
}
