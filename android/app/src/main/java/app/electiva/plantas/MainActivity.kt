package app.electiva.plantas

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.lifecycle.viewmodel.compose.viewModel

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val modelo: PlantasViewModel = viewModel()
            TemaPlantas {
                PantallaPlantas(modelo)
            }
        }
    }
}
