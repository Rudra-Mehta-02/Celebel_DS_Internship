"""
Memory store — dual-backend: PostgreSQL (production) + SQLite (fallback).

Stores:
  - chat_history: per-user conversation turns
  - user_memory: durable facts with contradiction resolution
"""

from __future__ import annotations

import abc
import json
import logging
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

from src.config import get_settings

logger = logging.getLogger(__name__)


class BaseMemoryStore(abc.ABC):
    """Abstract memory store interface."""

    @abc.abstractmethod
    def add_message(self, user_id: str, role: str, content: str, session_id: str = "") -> None: ...

    @abc.abstractmethod
    def get_history(self, user_id: str, limit: int = 10) -> list[dict]: ...

    @abc.abstractmethod
    def add_fact(self, user_id: str, fact: str, category: str = "general", confidence: float = 0.8) -> Optional[str]: ...

    @abc.abstractmethod
    def get_facts(self, user_id: str, limit: int = 50) -> list[dict]: ...

    @abc.abstractmethod
    def deactivate_fact(self, fact_id: str, superseded_by: str = "") -> None: ...

    @abc.abstractmethod
    def delete_fact(self, fact_id: str) -> None: ...

    @abc.abstractmethod
    def clear_user(self, user_id: str) -> None: ...


# ── SQLite Backend ───────────────────────────────────────────

class SQLiteMemoryStore(BaseMemoryStore):
    """SQLite-backed memory store — zero-config local development."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or get_settings().sqlite_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.info("✅ SQLite memory store: %s", self.db_path)

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                session_id TEXT DEFAULT '',
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id, created_at);

            CREATE TABLE IF NOT EXISTS user_memory (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                confidence REAL DEFAULT 0.8,
                active INTEGER DEFAULT 1,
                superseded_by TEXT DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_memory_user ON user_memory(user_id, active);
            CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_unique ON user_memory(user_id, fact);
        """)
        conn.commit()
        conn.close()

    def add_message(self, user_id: str, role: str, content: str, session_id: str = "") -> None:
        conn = self._get_conn()
        now = time.time()
        conn.execute(
            "INSERT INTO chat_history (id, user_id, role, content, session_id, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (str(uuid.uuid4())[:8], user_id, role, content, session_id, now),
        )
        conn.commit()
        conn.close()

    def get_history(self, user_id: str, limit: int = 10) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT role, content, created_at FROM chat_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        conn.close()
        # Return in chronological order
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def add_fact(self, user_id: str, fact: str, category: str = "general", confidence: float = 0.8) -> Optional[str]:
        conn = self._get_conn()
        fact_id = str(uuid.uuid4())[:8]
        now = time.time()
        try:
            conn.execute(
                "INSERT INTO user_memory (id, user_id, fact, category, confidence, active, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 1, ?, ?)",
                (fact_id, user_id, fact, category, confidence, now, now),
            )
            conn.commit()
            conn.close()
            return fact_id
        except sqlite3.IntegrityError:
            # Duplicate fact — ignore
            conn.close()
            return None

    def get_facts(self, user_id: str, limit: int = 50) -> list[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, fact, category, confidence, created_at FROM user_memory "
            "WHERE user_id = ? AND active = 1 ORDER BY created_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        conn.close()
        return [
            {"id": r["id"], "fact": r["fact"], "category": r["category"],
             "confidence": r["confidence"], "created_at": r["created_at"]}
            for r in rows
        ]

    def deactivate_fact(self, fact_id: str, superseded_by: str = "") -> None:
        conn = self._get_conn()
        conn.execute(
            "UPDATE user_memory SET active = 0, superseded_by = ?, updated_at = ? WHERE id = ?",
            (superseded_by, time.time(), fact_id),
        )
        conn.commit()
        conn.close()

    def delete_fact(self, fact_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM user_memory WHERE id = ?", (fact_id,))
        conn.commit()
        conn.close()

    def clear_user(self, user_id: str) -> None:
        conn = self._get_conn()
        conn.execute("DELETE FROM chat_history WHERE user_id = ?", (user_id,))
        conn.execute("DELETE FROM user_memory WHERE user_id = ?", (user_id,))
        conn.commit()
        conn.close()


# ── PostgreSQL Backend ───────────────────────────────────────

class PostgresMemoryStore(BaseMemoryStore):
    """PostgreSQL-backed memory store — production-grade."""

    def __init__(self, dsn: str):
        import psycopg2
        self._dsn = dsn
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = True
        self._init_db()
        logger.info("✅ PostgreSQL memory store connected")

    def _init_db(self) -> None:
        cur = self._conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_history (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                session_id TEXT DEFAULT '',
                created_at DOUBLE PRECISION NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chat_user ON chat_history(user_id, created_at);
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS user_memory (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                fact TEXT NOT NULL,
                category TEXT DEFAULT 'general',
                confidence DOUBLE PRECISION DEFAULT 0.8,
                active BOOLEAN DEFAULT TRUE,
                superseded_by TEXT DEFAULT '',
                created_at DOUBLE PRECISION NOT NULL,
                updated_at DOUBLE PRECISION NOT NULL,
                UNIQUE(user_id, fact)
            );
            CREATE INDEX IF NOT EXISTS idx_memory_user ON user_memory(user_id, active);
        """)
        cur.close()

    def add_message(self, user_id: str, role: str, content: str, session_id: str = "") -> None:
        cur = self._conn.cursor()
        cur.execute(
            "INSERT INTO chat_history (id, user_id, role, content, session_id, created_at) VALUES (%s, %s, %s, %s, %s, %s)",
            (str(uuid.uuid4())[:8], user_id, role, content, session_id, time.time()),
        )
        cur.close()

    def get_history(self, user_id: str, limit: int = 10) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT role, content, created_at FROM chat_history WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
        rows = cur.fetchall()
        cur.close()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]

    def add_fact(self, user_id: str, fact: str, category: str = "general", confidence: float = 0.8) -> Optional[str]:
        cur = self._conn.cursor()
        fact_id = str(uuid.uuid4())[:8]
        now = time.time()
        try:
            cur.execute(
                "INSERT INTO user_memory (id, user_id, fact, category, confidence, active, created_at, updated_at) "
                "VALUES (%s, %s, %s, %s, %s, TRUE, %s, %s) ON CONFLICT (user_id, fact) DO NOTHING",
                (fact_id, user_id, fact, category, confidence, now, now),
            )
            cur.close()
            return fact_id
        except Exception:
            cur.close()
            return None

    def get_facts(self, user_id: str, limit: int = 50) -> list[dict]:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT id, fact, category, confidence, created_at FROM user_memory "
            "WHERE user_id = %s AND active = TRUE ORDER BY created_at DESC LIMIT %s",
            (user_id, limit),
        )
        rows = cur.fetchall()
        cur.close()
        return [
            {"id": r[0], "fact": r[1], "category": r[2], "confidence": r[3], "created_at": r[4]}
            for r in rows
        ]

    def deactivate_fact(self, fact_id: str, superseded_by: str = "") -> None:
        cur = self._conn.cursor()
        cur.execute(
            "UPDATE user_memory SET active = FALSE, superseded_by = %s, updated_at = %s WHERE id = %s",
            (superseded_by, time.time(), fact_id),
        )
        cur.close()

    def delete_fact(self, fact_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM user_memory WHERE id = %s", (fact_id,))
        cur.close()

    def clear_user(self, user_id: str) -> None:
        cur = self._conn.cursor()
        cur.execute("DELETE FROM chat_history WHERE user_id = %s", (user_id,))
        cur.execute("DELETE FROM user_memory WHERE user_id = %s", (user_id,))
        cur.close()


# ── Factory ──────────────────────────────────────────────────
_store: Optional[BaseMemoryStore] = None


def get_memory_store() -> BaseMemoryStore:
    """Return the memory store — PostgreSQL if configured, else SQLite."""
    global _store
    if _store is None:
        settings = get_settings()
        if settings.has_postgres:
            try:
                _store = PostgresMemoryStore(settings.postgres_dsn)
            except Exception as e:
                logger.warning("PostgreSQL connection failed: %s — using SQLite", e)
                _store = SQLiteMemoryStore()
        else:
            _store = SQLiteMemoryStore()
    return _store
