"""Copia del historial de campo que envía el teléfono."""
from __future__ import annotations

import hashlib
import hmac
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_file

WEB = Path(__file__).resolve().parent
RAIZ = WEB.parent
DATOS = WEB / "datos"
FOTOS = DATOS / "fotos"
SQLITE = DATOS / "campo.sqlite"
FICHAS = RAIZ / "fichas_medicinales.csv"
AVISO = (
    "Estos datos son usos tradicionales para el estudio de campo. "
    "No son un diagnóstico ni una recomendación de uso."
)


def registrar(app: Flask) -> None:
    @app.post("/api/campo/sync")
    def campo_sync():
        cuerpo = request.get_json(silent=True) or {}
        estudiante = _estudiante(cuerpo.get("usuario"), cuerpo.get("token"), crear=True)
        if isinstance(estudiante, tuple):
            return estudiante
        recibidos = []
        with _conexion() as con:
            for item in cuerpo.get("analisis") or []:
                identificador = str(item.get("id") or "").strip()
                if not identificador:
                    continue
                foto = _guardar_foto(identificador, item.get("foto"))
                con.execute(
                    """
                    INSERT OR IGNORE INTO analisis
                    (id, estudiante_id, fecha, cientifico, comun, probabilidad, modelos, nota, foto)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        identificador,
                        estudiante,
                        int(item.get("fecha") or 0),
                        str(item.get("cientifico") or ""),
                        str(item.get("comun") or ""),
                        float(item.get("probabilidad") or 0),
                        str(item.get("modelos") or ""),
                        str(item.get("nota") or ""),
                        foto,
                    ),
                )
                recibidos.append(identificador)
        return jsonify({"ok": True, "recibidos": recibidos})

    @app.get("/api/campo/historial")
    def campo_historial():
        estudiante = _estudiante(request.args.get("usuario"), request.args.get("token"), crear=False)
        if isinstance(estudiante, tuple):
            return estudiante
        with _conexion() as con:
            filas = con.execute(
                """
                SELECT id, fecha, cientifico, comun, probabilidad, modelos, nota, foto
                FROM analisis WHERE estudiante_id = ? ORDER BY fecha DESC
                """,
                (estudiante,),
            ).fetchall()
        return jsonify(
            {
                "ok": True,
                "analisis": [
                    {
                        "id": fila["id"],
                        "fecha": fila["fecha"],
                        "cientifico": fila["cientifico"],
                        "comun": fila["comun"],
                        "probabilidad": fila["probabilidad"],
                        "modelos": fila["modelos"],
                        "nota": fila["nota"],
                        "tiene_foto": bool(fila["foto"]),
                    }
                    for fila in filas
                ],
            }
        )

    @app.get("/api/campo/foto/<identificador>")
    def campo_foto_api(identificador: str):
        estudiante = _estudiante(request.args.get("usuario"), request.args.get("token"), crear=False)
        if isinstance(estudiante, tuple):
            return estudiante
        ruta = _foto_de(estudiante, identificador)
        if ruta is None:
            return jsonify({"ok": False, "error": "No está la foto"}), 404
        return send_file(ruta, mimetype="image/jpeg")

    @app.get("/campo")
    def campo_pagina():
        with _conexion() as con:
            filas = con.execute(
                """
                SELECT a.id, a.fecha, a.cientifico, a.comun, a.probabilidad, a.modelos, a.nota, a.foto,
                       e.nombre AS estudiante
                FROM analisis a
                JOIN estudiantes e ON e.id = a.estudiante_id
                ORDER BY a.fecha DESC
                """
            ).fetchall()
        return Response(_html(filas), mimetype="text/html")

    @app.get("/campo/foto/<identificador>")
    def campo_foto_pagina(identificador: str):
        ruta = FOTOS / f"{_nombre_seguro(identificador)}.jpg"
        if not ruta.is_file():
            return "No está la foto", 404
        return send_file(ruta, mimetype="image/jpeg")


def _estudiante(nombre, token, crear: bool):
    visible = str(nombre or "").strip()
    secreto = str(token or "").strip()
    if len(visible) < 2 or len(secreto) < 16:
        return jsonify({"ok": False, "error": "Faltan el nombre o el token"}), 400
    clave = visible.casefold()
    resumen = hashlib.sha256(secreto.encode("utf-8")).hexdigest()
    with _conexion() as con:
        fila = con.execute(
            "SELECT id, token_hash FROM estudiantes WHERE nombre_clave = ?",
            (clave,),
        ).fetchone()
        if fila is None:
            if not crear:
                return jsonify({"ok": False, "error": "Esa cuenta no está en el PC"}), 401
            cursor = con.execute(
                "INSERT INTO estudiantes (nombre, nombre_clave, token_hash) VALUES (?, ?, ?)",
                (visible, clave, resumen),
            )
            return int(cursor.lastrowid)
        if not hmac.compare_digest(fila["token_hash"], resumen):
            return jsonify({"ok": False, "error": "El token no coincide con esa cuenta"}), 401
        return int(fila["id"])


def _foto_de(estudiante: int, identificador: str) -> Path | None:
    with _conexion() as con:
        fila = con.execute(
            "SELECT foto FROM analisis WHERE id = ? AND estudiante_id = ?",
            (identificador, estudiante),
        ).fetchone()
    if fila is None or not fila["foto"]:
        return None
    ruta = Path(fila["foto"])
    return ruta if ruta.is_file() else None


def _guardar_foto(identificador: str, foto_b64) -> str | None:
    if not foto_b64:
        return None
    import base64

    try:
        datos = base64.b64decode(str(foto_b64), validate=False)
    except Exception:
        return None
    if not datos:
        return None
    FOTOS.mkdir(parents=True, exist_ok=True)
    ruta = FOTOS / f"{_nombre_seguro(identificador)}.jpg"
    ruta.write_bytes(datos)
    return str(ruta)


def _nombre_seguro(identificador: str) -> str:
    limpio = "".join(caracter for caracter in identificador if caracter.isalnum() or caracter in "-_")
    return limpio or "foto"


@contextmanager
def _conexion():
    DATOS.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(SQLITE)
    con.row_factory = sqlite3.Row
    try:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS estudiantes (
                id INTEGER PRIMARY KEY,
                nombre TEXT NOT NULL,
                nombre_clave TEXT NOT NULL UNIQUE,
                token_hash TEXT NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS analisis (
                id TEXT PRIMARY KEY,
                estudiante_id INTEGER NOT NULL,
                fecha INTEGER NOT NULL,
                cientifico TEXT NOT NULL,
                comun TEXT NOT NULL,
                probabilidad REAL NOT NULL,
                modelos TEXT NOT NULL,
                nota TEXT NOT NULL,
                foto TEXT,
                FOREIGN KEY (estudiante_id) REFERENCES estudiantes(id)
            )
            """
        )
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def _fichas() -> dict[str, dict[str, str]]:
    if not FICHAS.is_file():
        return {}
    import csv

    indice = {}
    with FICHAS.open(encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            cientifico = (fila.get("nombre_cientifico") or "").strip()
            usos = (fila.get("usos_tradicionales") or "").strip()
            fuente = (fila.get("fuente") or "").strip()
            if not cientifico or not usos or not fuente:
                continue
            if fuente.startswith("Herbify"):
                continue
            indice[_clave(cientifico)] = fila
    return indice


def _clave(nombre: str) -> str:
    base = nombre.split("(")[0].split("__")[0]
    return " ".join(base.replace("_", " ").lower().split())


def _html(filas) -> str:
    fichas = _fichas()
    bloques = []
    for fila in filas:
        ficha = fichas.get(_clave(fila["cientifico"]))
        if ficha:
            medico = (
                f"<p><strong>Parte usada.</strong> {_esc(ficha.get('parte_usada') or 'No consta en la fuente.')}</p>"
                f"<p><strong>Usos tradicionales.</strong> {_esc(ficha.get('usos_tradicionales') or '')}</p>"
                f"<p><strong>Precaución.</strong> {_esc(ficha.get('contraindicaciones') or 'No consta en la fuente.')}</p>"
                f"<p><strong>Fuente.</strong> {_esc(ficha.get('fuente') or '')}</p>"
            )
        else:
            medico = "<p>Sin ficha documentada para esta especie.</p>"
        foto = ""
        if fila["foto"]:
            foto = f'<img src="/campo/foto/{_esc(fila["id"])}" alt="Foto del análisis">'
        nota = f"<p>{_esc(fila['nota'])}</p>" if fila["nota"] else ""
        bloques.append(
            f"""
            <article>
              {foto}
              <h2>{_esc(fila['comun'])}</h2>
              <p class="cientifico">{_esc(fila['cientifico'])}</p>
              <p>{_esc(fila['estudiante'])} · {float(fila['probabilidad']) * 100:.1f} %</p>
              <p>{_esc(fila['modelos'])}</p>
              {nota}
              <p class="aviso">{_esc(AVISO)}</p>
              {medico}
            </article>
            """
        )
    cuerpo = "\n".join(bloques) or "<p>Todavía no hay análisis copiados desde un teléfono.</p>"
    return f"""<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Historial de campo</title>
  <style>
    body {{ font-family: Georgia, serif; margin: 0; background: #f4f1ea; color: #1d2a22; }}
    header, main {{ max-width: 880px; margin: 0 auto; padding: 24px; }}
    a {{ color: #1f6b45; }}
    article {{ background: white; border-radius: 16px; padding: 16px; margin: 16px 0; }}
    img {{ width: 100%; max-height: 360px; object-fit: cover; border-radius: 12px; }}
    .cientifico {{ font-style: italic; }}
    .aviso {{ background: #eef6f0; padding: 12px; border-radius: 12px; }}
  </style>
</head>
<body>
  <header>
    <p><a href="/">Volver al comparador</a></p>
    <h1>Historial de campo</h1>
  </header>
  <main>{cuerpo}</main>
</body>
</html>"""


def _esc(valor) -> str:
    texto = str(valor or "")
    return (
        texto.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
