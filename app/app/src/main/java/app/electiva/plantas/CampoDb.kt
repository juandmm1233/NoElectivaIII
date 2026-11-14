package app.electiva.plantas

import android.content.Context
import androidx.room.Dao
import androidx.room.Database
import androidx.room.Entity
import androidx.room.ForeignKey
import androidx.room.Index
import androidx.room.Insert
import androidx.room.PrimaryKey
import androidx.room.Query
import androidx.room.Room
import androidx.room.RoomDatabase

@Entity(
    tableName = "usuarios",
    indices = [Index(value = ["nombreClave"], unique = true)],
)
data class Usuario(
    @PrimaryKey(autoGenerate = true) val id: Long = 0,
    val nombre: String,
    val nombreClave: String,
    val sal: String,
    val hash: String,
    val iteraciones: Int,
    val token: String,
)

@Entity(
    tableName = "analisis",
    foreignKeys = [
        ForeignKey(
            entity = Usuario::class,
            parentColumns = ["id"],
            childColumns = ["usuarioId"],
            onDelete = ForeignKey.CASCADE,
        ),
    ],
    indices = [Index("usuarioId")],
)
data class Analisis(
    @PrimaryKey val id: String,
    val usuarioId: Long,
    val fecha: Long,
    val cientifico: String,
    val comun: String,
    val probabilidad: Float,
    val modelos: String,
    val nota: String,
    val rutaFoto: String,
    val sincronizado: Boolean,
)

@Dao
interface CampoDao {
    @Insert
    suspend fun insertarUsuario(usuario: Usuario): Long

    @Query("SELECT * FROM usuarios WHERE id = :id")
    suspend fun usuario(id: Long): Usuario?

    @Query("SELECT * FROM usuarios WHERE nombreClave = :nombreClave")
    suspend fun usuarioPorNombre(nombreClave: String): Usuario?

    @Insert
    suspend fun insertarAnalisis(analisis: Analisis)

    @Query("SELECT * FROM analisis WHERE usuarioId = :usuarioId ORDER BY fecha DESC")
    suspend fun analisisDe(usuarioId: Long): List<Analisis>

    @Query("SELECT * FROM analisis WHERE id = :id")
    suspend fun analisis(id: String): Analisis?

    @Query("SELECT * FROM analisis WHERE usuarioId = :usuarioId AND sincronizado = 0 ORDER BY fecha")
    suspend fun pendientes(usuarioId: Long): List<Analisis>

    @Query("UPDATE analisis SET sincronizado = 1 WHERE id = :id")
    suspend fun marcarSincronizado(id: String)
}

@Database(entities = [Usuario::class, Analisis::class], version = 1, exportSchema = false)
abstract class CampoDb : RoomDatabase() {
    abstract fun dao(): CampoDao

    companion object {
        @Volatile
        private var instancia: CampoDb? = null

        fun obtener(contexto: Context): CampoDb {
            return instancia ?: synchronized(this) {
                instancia ?: Room.databaseBuilder(
                    contexto.applicationContext,
                    CampoDb::class.java,
                    "campo.db",
                ).build().also { instancia = it }
            }
        }
    }
}
