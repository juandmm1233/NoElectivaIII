package app.electiva.plantas

import java.security.MessageDigest
import java.security.SecureRandom
import javax.crypto.SecretKeyFactory
import javax.crypto.spec.PBEKeySpec

object Claves {
    private const val ITERACIONES = 120_000
    private val azar = SecureRandom()

    fun crear(contrasena: String): Triple<String, String, Int> {
        val sal = ByteArray(16)
        azar.nextBytes(sal)
        val hash = derivar(contrasena, sal, ITERACIONES)
        return Triple(hex(sal), hex(hash), ITERACIONES)
    }

    fun coincide(contrasena: String, salHex: String, hashHex: String, iteraciones: Int): Boolean {
        val calculado = derivar(contrasena, bytes(salHex), iteraciones)
        return MessageDigest.isEqual(calculado, bytes(hashHex))
    }

    fun token(): String {
        val bytes = ByteArray(32)
        azar.nextBytes(bytes)
        return hex(bytes)
    }

    private fun derivar(contrasena: String, sal: ByteArray, iteraciones: Int): ByteArray {
        val spec = PBEKeySpec(contrasena.toCharArray(), sal, iteraciones, 256)
        return SecretKeyFactory.getInstance("PBKDF2WithHmacSHA256").generateSecret(spec).encoded
    }

    private fun hex(bytes: ByteArray): String {
        return bytes.joinToString("") { "%02x".format(it) }
    }

    private fun bytes(hexadecimal: String): ByteArray {
        return hexadecimal.chunked(2).map { it.toInt(16).toByte() }.toByteArray()
    }
}
