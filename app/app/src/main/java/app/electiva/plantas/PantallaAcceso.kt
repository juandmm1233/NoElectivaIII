package app.electiva.plantas

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp

@Composable
fun PantallaAcceso(campo: CampoViewModel) {
    var creando by rememberSaveable { mutableStateOf(true) }
    var nombre by rememberSaveable { mutableStateOf("") }
    var contrasena by rememberSaveable { mutableStateOf("") }
    var repetir by rememberSaveable { mutableStateOf("") }
    val desplazamiento = rememberScrollState()

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .imePadding()
            .verticalScroll(desplazamiento)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
    ) {
        Text("Plantas", style = MaterialTheme.typography.headlineLarge, color = MaterialTheme.colorScheme.primary)
        Text(
            if (creando) {
                "Crea una cuenta en este teléfono para guardar las plantas del trabajo de campo."
            } else {
                "Entra con la cuenta que ya creaste en este teléfono."
            },
            style = MaterialTheme.typography.bodyLarge,
        )
        OutlinedTextField(
            value = nombre,
            onValueChange = { nombre = it },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            label = { Text("Nombre") },
        )
        OutlinedTextField(
            value = contrasena,
            onValueChange = { contrasena = it },
            modifier = Modifier.fillMaxWidth(),
            singleLine = true,
            label = { Text("Contraseña") },
            visualTransformation = PasswordVisualTransformation(),
        )
        if (creando) {
            OutlinedTextField(
                value = repetir,
                onValueChange = { repetir = it },
                modifier = Modifier.fillMaxWidth(),
                singleLine = true,
                label = { Text("Repetir contraseña") },
                visualTransformation = PasswordVisualTransformation(),
            )
        }
        campo.error?.let { texto ->
            Text(texto, color = MaterialTheme.colorScheme.error)
        }
        Button(
            onClick = {
                if (creando && contrasena != repetir) return@Button
                if (creando) campo.crear(nombre, contrasena) else campo.entrar(nombre, contrasena)
            },
            enabled = !campo.ocupado && nombre.isNotBlank() && contrasena.isNotBlank(),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (campo.ocupado) "Un momento…" else if (creando) "Crear cuenta" else "Entrar")
        }
        if (creando && contrasena.isNotEmpty() && repetir.isNotEmpty() && contrasena != repetir) {
            Text("Las contraseñas no coinciden.", color = MaterialTheme.colorScheme.error)
        }
        OutlinedButton(
            onClick = { creando = !creando },
            enabled = !campo.ocupado,
            modifier = Modifier.fillMaxWidth(),
        ) {
            Text(if (creando) "Ya tengo cuenta" else "Crear una cuenta")
        }
    }
}
