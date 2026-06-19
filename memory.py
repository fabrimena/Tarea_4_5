"""
memory.py
---------
Memoria histórica persistente para el agente académico.
Usa SQLite para almacenar conversaciones entre sesiones.

Tablas:
  sessions   → cada sesión de chat (id, fecha inicio, resumen)
  messages   → cada mensaje (sesión, rol, contenido, timestamp)
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path

DB_PATH = "./memory.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Crea las tablas si no existen."""
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at  TEXT NOT NULL,
            summary     TEXT DEFAULT ''
        );

        CREATE TABLE IF NOT EXISTS messages (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id  INTEGER NOT NULL,
            role        TEXT NOT NULL,
            content     TEXT NOT NULL,
            timestamp   TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        );
    """)
    conn.commit()
    conn.close()


def create_session() -> int:
    """Crea una nueva sesión y devuelve su id."""
    conn = _get_conn()
    cur = conn.execute(
        "INSERT INTO sessions (started_at) VALUES (?)",
        (datetime.now().isoformat(),)
    )
    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return session_id


def save_message(session_id: int, role: str, content: str):
    """Guarda un mensaje en la sesión actual."""
    conn = _get_conn()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def save_session_summary(session_id: int, summary: str):
    """Guarda el resumen de una sesión al terminar."""
    conn = _get_conn()
    conn.execute(
        "UPDATE sessions SET summary = ? WHERE id = ?",
        (summary, session_id)
    )
    conn.commit()
    conn.close()


def get_session_messages(session_id: int) -> list[dict]:
    """Devuelve todos los mensajes de una sesión."""
    conn = _get_conn()
    rows = conn.execute(
        "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def get_user_messages(exclude_session_id: int = None, limit: int = 30) -> list[dict]:
    """Devuelve los mensajes del usuario de sesiones anteriores."""
    conn = _get_conn()
    if exclude_session_id:
        rows = conn.execute(
            """
            SELECT m.content, m.timestamp, s.id as session_id
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE m.role = 'user' AND m.session_id != ?
            ORDER BY m.id DESC
            LIMIT ?
            """,
            (exclude_session_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT m.content, m.timestamp, s.id as session_id
            FROM messages m
            JOIN sessions s ON m.session_id = s.id
            WHERE m.role = 'user'
            ORDER BY m.id DESC
            LIMIT ?
            """,
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

def search_history(query: str, limit: int = 5) -> list[dict]:
    """
    Busca en el historial de mensajes por coincidencia de texto.
    Devuelve los mensajes más recientes que contengan el query.
    """
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT m.role, m.content, m.timestamp, s.id as session_id, s.started_at
        FROM messages m
        JOIN sessions s ON m.session_id = s.id
        WHERE m.content LIKE ?
        ORDER BY m.id DESC
        LIMIT ?
        """,
        (f"%{query}%", limit)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_recent_sessions(limit: int = 5) -> list[dict]:
    """Devuelve las sesiones más recientes con sus mensajes."""
    conn = _get_conn()
    sessions = conn.execute(
        "SELECT id, started_at, summary FROM sessions ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()

    result = []
    for s in sessions:
        messages = conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id",
            (s["id"],)
        ).fetchall()
        result.append({
            "session_id": s["id"],
            "started_at": s["started_at"],
            "summary": s["summary"],
            "messages": [dict(m) for m in messages]
        })

    conn.close()
    return result


def get_all_messages_today() -> list[dict]:
    """Devuelve todos los mensajes del día de hoy."""
    today = datetime.now().strftime("%Y-%m-%d")
    conn = _get_conn()
    rows = conn.execute(
        """
        SELECT m.role, m.content, m.timestamp, s.id as session_id
        FROM messages m
        JOIN sessions s ON m.session_id = s.id
        WHERE m.timestamp LIKE ?
        ORDER BY m.id
        """,
        (f"{today}%",)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

init_db()