package app.electiva.plantas

import android.content.Context
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.map

private val Context.ajustes by preferencesDataStore(name = "ajustes")

class Ajustes(context: Context) {
    private val contexto = context.applicationContext
    private val claveUrl = stringPreferencesKey("url_pc")

    suspend fun leerUrl(): String {
        return contexto.ajustes.data.map { it[claveUrl] ?: "" }.first()
    }

    suspend fun guardarUrl(valor: String) {
        contexto.ajustes.edit { it[claveUrl] = valor.trim() }
    }
}
