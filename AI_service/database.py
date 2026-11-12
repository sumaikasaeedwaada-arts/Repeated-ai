import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

from AI_service import settings


logger = logging.getLogger(__name__)
MAX_AI_CACHE_ENTRIES = 10000


SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    email TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS conversations_user_updated
    ON conversations(user_id, updated_at DESC);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content TEXT NOT NULL,
    citations_json TEXT NOT NULL DEFAULT '[]',
    request_fingerprint TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_conversation_created
    ON messages(conversation_id, created_at, id);
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename TEXT NOT NULL,
    media_type TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('ready')),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_user_created
    ON documents(user_id, created_at DESC);
CREATE TABLE IF NOT EXISTS notifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    message TEXT NOT NULL,
    type TEXT NOT NULL,
    related_entity_id TEXT,
    related_entity_type TEXT,
    is_read INTEGER NOT NULL DEFAULT 0 CHECK(is_read IN (0, 1)),
    created_at TEXT NOT NULL,
    dedupe_key TEXT,
    UNIQUE(user_id, dedupe_key)
);
CREATE INDEX IF NOT EXISTS notifications_user_created
    ON notifications(user_id, created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS notifications_user_read_created
    ON notifications(user_id, is_read, created_at DESC, id DESC);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    content TEXT NOT NULL,
    embedding_json TEXT NOT NULL,
    UNIQUE(document_id, chunk_index)
);
CREATE INDEX IF NOT EXISTS chunks_document
    ON chunks(document_id, chunk_index);
CREATE TABLE IF NOT EXISTS vector_index_state (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    embedding_model TEXT NOT NULL,
    embedding_dimensions INTEGER,
    similarity_metric TEXT NOT NULL CHECK(similarity_metric = 'cosine')
);
CREATE TABLE IF NOT EXISTS rate_limit_buckets (
    bucket_key TEXT PRIMARY KEY,
    window_started REAL NOT NULL,
    request_count INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS ai_cache_entries (
    user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    cache_type TEXT NOT NULL,
    cache_key TEXT NOT NULL,
    value_json TEXT,
    expires_at REAL NOT NULL DEFAULT 0,
    lease_token TEXT,
    lease_until REAL NOT NULL DEFAULT 0,
    PRIMARY KEY(user_id, cache_type, cache_key)
);
CREATE INDEX IF NOT EXISTS ai_cache_entries_expiry
    ON ai_cache_entries(expires_at);
"""


def database_path() -> str:
    return os.getenv("DATABASE_PATH", settings.DATABASE_PATH)


@contextmanager
def connect():
    path = database_path()
    if path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 10000")
    if path != ":memory:":
        connection.execute("PRAGMA journal_mode = WAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def initialize_database() -> None:
    with connect() as connection:
        connection.executescript(SCHEMA)
        message_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(messages)")
        }
        if "request_fingerprint" not in message_columns:
            connection.execute(
                "ALTER TABLE messages ADD COLUMN request_fingerprint TEXT"
            )
        connection.execute(
            """
            INSERT OR IGNORE INTO vector_index_state
                (id, embedding_model, embedding_dimensions, similarity_metric)
            VALUES (1, ?, NULL, 'cosine')
            """,
            (settings.GEMINI_EMBEDDING_MODEL,)
        )


def make_cache_key(namespace: str, payload: object) -> str:
    encoded = json.dumps(
        [namespace, payload],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def get_cached_value(
    user_id: int,
    cache_type: str,
    cache_key: str
) -> object | None:
    now = time.time()
    with connect() as connection:
        row = connection.execute(
            """
            SELECT value_json, expires_at
            FROM ai_cache_entries
            WHERE user_id = ? AND cache_type = ? AND cache_key = ?
            """,
            (user_id, cache_type, cache_key)
        ).fetchone()
        if row is None or row["value_json"] is None or row["expires_at"] <= now:
            return None
        try:
            return json.loads(row["value_json"])
        except json.JSONDecodeError:
            logger.warning("Discarding malformed cached AI value.")
            connection.execute(
                """
                DELETE FROM ai_cache_entries
                WHERE user_id = ? AND cache_type = ? AND cache_key = ?
                  AND lease_token IS NULL
                """,
                (user_id, cache_type, cache_key)
            )
            return None


def claim_cache_entry(
    user_id: int,
    cache_type: str,
    cache_key: str,
    lease_seconds: float
) -> str | None:
    now = time.time()
    lease_token = uuid.uuid4().hex
    lease_until = now + lease_seconds
    with connect() as connection:
        connection.execute("BEGIN IMMEDIATE")
        inserted = connection.execute(
            """
            INSERT OR IGNORE INTO ai_cache_entries
                (user_id, cache_type, cache_key, lease_token, lease_until)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, cache_type, cache_key, lease_token, lease_until)
        )
        if inserted.rowcount:
            return lease_token
        updated = connection.execute(
            """
            UPDATE ai_cache_entries
            SET value_json = NULL, expires_at = 0,
                lease_token = ?, lease_until = ?
            WHERE user_id = ? AND cache_type = ? AND cache_key = ?
              AND expires_at <= ? AND lease_until <= ?
            """,
            (
                lease_token,
                lease_until,
                user_id,
                cache_type,
                cache_key,
                now,
                now
            )
        )
        return lease_token if updated.rowcount else None


def store_claimed_value(
    user_id: int,
    cache_type: str,
    cache_key: str,
    lease_token: str,
    value: object,
    ttl_seconds: float
) -> bool:
    now = time.time()
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    with connect() as connection:
        stored = connection.execute(
            """
            UPDATE ai_cache_entries
            SET value_json = ?, expires_at = ?, lease_token = NULL,
                lease_until = 0
            WHERE user_id = ? AND cache_type = ? AND cache_key = ?
              AND lease_token = ?
            """,
            (
                encoded,
                now + ttl_seconds,
                user_id,
                cache_type,
                cache_key,
                lease_token
            )
        )
        _prune_ai_cache(connection, now)
        return stored.rowcount > 0


def release_cache_claim(
    user_id: int,
    cache_type: str,
    cache_key: str,
    lease_token: str
) -> None:
    with connect() as connection:
        connection.execute(
            """
            DELETE FROM ai_cache_entries
            WHERE user_id = ? AND cache_type = ? AND cache_key = ?
              AND lease_token = ?
            """,
            (user_id, cache_type, cache_key, lease_token)
        )


def delete_cached_value(
    user_id: int,
    cache_type: str,
    cache_key: str
) -> None:
    with connect() as connection:
        connection.execute(
            """
            DELETE FROM ai_cache_entries
            WHERE user_id = ? AND cache_type = ? AND cache_key = ?
              AND lease_token IS NULL
            """,
            (user_id, cache_type, cache_key)
        )


def set_cached_value(
    user_id: int,
    cache_type: str,
    cache_key: str,
    value: object,
    ttl_seconds: float
) -> None:
    if ttl_seconds <= 0:
        return
    now = time.time()
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
    with connect() as connection:
        connection.execute(
            """
            INSERT INTO ai_cache_entries
                (user_id, cache_type, cache_key, value_json, expires_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, cache_type, cache_key) DO UPDATE SET
                value_json = excluded.value_json,
                expires_at = excluded.expires_at,
                lease_token = NULL,
                lease_until = 0
            """,
            (user_id, cache_type, cache_key, encoded, now + ttl_seconds)
        )
        _prune_ai_cache(connection, now)


def _prune_ai_cache(connection: sqlite3.Connection, now: float) -> None:
    connection.execute(
        "DELETE FROM ai_cache_entries WHERE expires_at <= ? AND lease_until <= ?",
        (now, now)
    )
    excess = connection.execute(
        "SELECT COUNT(*) - ? AS excess FROM ai_cache_entries",
        (MAX_AI_CACHE_ENTRIES,)
    ).fetchone()["excess"]
    if excess > 0:
        connection.execute(
            """
            DELETE FROM ai_cache_entries
            WHERE rowid IN (
                SELECT rowid FROM ai_cache_entries
                ORDER BY expires_at ASC, rowid ASC LIMIT ?
            )
            """,
            (excess,)
        )


def row_dict(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


def decode_citations(value: str) -> list[dict]:
    citations = json.loads(value)
    return citations if isinstance(citations, list) else []
