"""Conversation Registry Database for ConstitutionGPT.

Maintains conversation thread metadata (ID, title, user_id, creation, and update timestamps)
in SQLite for Streamlit sidebar navigation and session persistence.

Each conversation is scoped to a user_id so different Auth0 accounts
have completely isolated chat histories.
"""

from __future__ import annotations

from contextlib import contextmanager
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from config.settings import get_settings

logger = logging.getLogger(__name__)

_GUEST_USER_ID = "advocate_guest"


class ConversationDB:
    """Thread metadata registry backed by SQLite — scoped per user_id."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or get_settings().conversations_db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Create a thread-safe connection to SQLite registry and cleanly close."""
        conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False, timeout=10.0
        )
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Initialize conversations registry schema with user_id support."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    thread_id  TEXT PRIMARY KEY,
                    title      TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    engine_mode TEXT DEFAULT 'both',
                    user_id    TEXT DEFAULT 'advocate_guest'
                )
                """
            )
            # Live-migrate older schemas that are missing columns
            existing_cols = {
                row[1]
                for row in conn.execute("PRAGMA table_info(conversations)").fetchall()
            }
            if "engine_mode" not in existing_cols:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN engine_mode TEXT DEFAULT 'both'"
                )
            if "user_id" not in existing_cols:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN user_id TEXT DEFAULT 'advocate_guest'"
                )
                # Create index for fast per-user lookup
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)"
            )
            conn.commit()

    # ------------------------------------------------------------------
    # CRUD — all operations are scoped to user_id
    # ------------------------------------------------------------------

    def create_conversation(
        self,
        thread_id: str,
        title: Optional[str] = None,
        engine_mode: Optional[str] = "both",
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Register a new conversation thread owned by user_id."""
        now = datetime.now(timezone.utc).isoformat()
        clean_title = (title or "New Conversation").strip()
        clean_engine = engine_mode or "both"
        uid = (user_id or _GUEST_USER_ID).strip()
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO conversations
                    (thread_id, title, created_at, updated_at, engine_mode, user_id)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (thread_id, clean_title, now, now, clean_engine, uid),
            )
            conn.commit()

        return {
            "thread_id": thread_id,
            "title": clean_title,
            "created_at": now,
            "updated_at": now,
            "engine_mode": clean_engine,
            "user_id": uid,
        }

    def list_conversations(
        self, user_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List conversation threads for a specific user, ordered by updated_at DESC."""
        uid = (user_id or _GUEST_USER_ID).strip()
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT thread_id, title, created_at, updated_at, engine_mode, user_id
                FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC
                """,
                (uid,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_conversation(
        self, thread_id: str, user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """Retrieve metadata for a specific conversation thread, optionally validating ownership."""
        with self._get_connection() as conn:
            if user_id:
                cursor = conn.execute(
                    """
                    SELECT thread_id, title, created_at, updated_at, engine_mode, user_id
                    FROM conversations
                    WHERE thread_id = ? AND user_id = ?
                    """,
                    (thread_id, user_id.strip()),
                )
            else:
                cursor = conn.execute(
                    """
                    SELECT thread_id, title, created_at, updated_at, engine_mode, user_id
                    FROM conversations
                    WHERE thread_id = ?
                    """,
                    (thread_id,),
                )
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_users(self) -> List[str]:
        """Return all distinct user IDs that have conversations in the database."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT DISTINCT user_id
                FROM conversations
                WHERE user_id IS NOT NULL AND user_id != ''
                ORDER BY updated_at DESC
                """
            )
            return [row[0] for row in cursor.fetchall()]

    def update_conversation(
        self,
        thread_id: str,
        title: Optional[str] = None,
        engine_mode: Optional[str] = None,
    ) -> bool:
        """Update last modified timestamp, and optionally title or engine_mode."""
        now = datetime.now(timezone.utc).isoformat()
        with self._get_connection() as conn:
            if title is not None and engine_mode is not None:
                cursor = conn.execute(
                    """
                    UPDATE conversations
                    SET title = ?, engine_mode = ?, updated_at = ?
                    WHERE thread_id = ?
                    """,
                    (title.strip(), engine_mode, now, thread_id),
                )
            elif title is not None:
                cursor = conn.execute(
                    """
                    UPDATE conversations
                    SET title = ?, updated_at = ?
                    WHERE thread_id = ?
                    """,
                    (title.strip(), now, thread_id),
                )
            elif engine_mode is not None:
                cursor = conn.execute(
                    """
                    UPDATE conversations
                    SET engine_mode = ?, updated_at = ?
                    WHERE thread_id = ?
                    """,
                    (engine_mode, now, thread_id),
                )
            else:
                cursor = conn.execute(
                    """
                    UPDATE conversations
                    SET updated_at = ?
                    WHERE thread_id = ?
                    """,
                    (now, thread_id),
                )
            conn.commit()
            return cursor.rowcount > 0

    def delete_conversation(self, thread_id: str) -> bool:
        """Remove a conversation record from the registry."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE thread_id = ?", (thread_id,)
            )
            conn.commit()
            return cursor.rowcount > 0

    def delete_all_for_user(self, user_id: str) -> int:
        """Delete every conversation belonging to a user. Returns count deleted."""
        uid = (user_id or _GUEST_USER_ID).strip()
        with self._get_connection() as conn:
            cursor = conn.execute(
                "DELETE FROM conversations WHERE user_id = ?", (uid,)
            )
            conn.commit()
            return cursor.rowcount


def generate_title_from_query(query: str, max_length: int = 32) -> str:
    """Generate a clean conversation title deterministically from user's first query."""
    clean = " ".join(query.strip().split())
    if len(clean) <= max_length:
        return clean
    return clean[:max_length].rstrip() + "..."


_conv_db_instance: Optional[ConversationDB] = None


def get_conversation_db() -> ConversationDB:
    """Singleton getter for ConversationDB."""
    global _conv_db_instance
    if _conv_db_instance is None:
        _conv_db_instance = ConversationDB()
    return _conv_db_instance
