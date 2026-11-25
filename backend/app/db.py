import json
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    system_prompt TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    metadata TEXT,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages (session_id, id);
"""


def resolve_sqlite_path(database_url: str) -> str:
    """Turn a sqlite URL into a filesystem path.

    ``sqlite:///./data/app.db`` is relative. ``sqlite:////data/app.db`` is absolute.
    """
    prefix = "sqlite:"
    if not database_url.startswith(prefix):
        raise ValueError("DATABASE_URL must be a sqlite URL")
    rest = database_url[len(prefix) :]
    if rest.startswith("////"):
        path = rest[3:]
    elif rest.startswith("///"):
        path = rest[3:]
    else:
        raise ValueError(
            "DATABASE_URL must look like sqlite:///./data/app.db or sqlite:////data/app.db"
        )
    if not path:
        raise ValueError("DATABASE_URL is missing a file path")
    return path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, database_url: str) -> None:
        self.path = resolve_sqlite_path(database_url)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, check_same_thread=False)
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def migrate(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(SCHEMA)

    def ping(self) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("SELECT 1")

    def get_system_prompt(self, session_id: str) -> str | None:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT system_prompt FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return str(row["system_prompt"])

    def list_messages(self, session_id: str) -> list[dict[str, Any]] | None:
        """Return history oldest-first, or None when the session has no rows."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, metadata
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        if not rows:
            return None
        messages: list[dict[str, Any]] = []
        for row in rows:
            item: dict[str, Any] = {"role": row["role"], "content": row["content"]}
            if row["role"] == "assistant":
                raw = row["metadata"]
                item["metadata"] = json.loads(raw) if raw else {}
            messages.append(item)
        return messages

    def conversation(self, session_id: str) -> list[dict[str, str]]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def save_turn(
        self,
        session_id: str,
        system_prompt: str,
        user_content: str,
        assistant_content: str,
        metadata: dict[str, Any],
    ) -> None:
        created_at = utc_now()
        metadata_json = json.dumps(metadata, ensure_ascii=False, separators=(",", ":"))
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, system_prompt, created_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET system_prompt = excluded.system_prompt
                """,
                (session_id, system_prompt, created_at),
            )
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'user', ?, NULL, ?)
                """,
                (session_id, user_content, created_at),
            )
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata, created_at)
                VALUES (?, 'assistant', ?, ?, ?)
                """,
                (session_id, assistant_content, metadata_json, created_at),
            )
