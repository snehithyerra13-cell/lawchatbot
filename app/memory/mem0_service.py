"""Mem0 Self-Improving Memory Service for ConstitutionGPT.

Provides personalized constitutional memory across sessions:
- Extracts and preserves user preferences, research domain topics, and facts.
- Supports both Mem0 Cloud Platform (via MemoryClient) and local persistent SQLite fallback.
- Injects retrieved memories into Gemini LLM synthesis and LangGraph graph context.
"""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import sqlite3
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from config.settings import get_settings

logger = logging.getLogger(__name__)


class LocalMem0Storage:
    """Local SQLite-backed persistent memory store replicating Mem0 API contract."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._init_db()

    @contextlib.contextmanager
    def _connection(self):
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS mem0_memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    memory TEXT NOT NULL,
                    categories TEXT,
                    metadata_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem0_user ON mem0_memories(user_id)"
            )
            conn.commit()

    def add(
        self,
        messages: Union[str, List[Dict[str, str]]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Add one or more memory entries for a user."""
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        metadata_str = json.dumps(metadata or {})
        results: List[Dict[str, Any]] = []

        if isinstance(messages, str):
            mem_texts = [messages.strip()]
        elif isinstance(messages, list):
            # Extract content from message turns
            mem_texts = []
            for item in messages:
                if isinstance(item, dict):
                    content = item.get("content", "").strip()
                    if content and item.get("role") == "user":
                        mem_texts.append(f"User noted: {content}")
                    elif content:
                        mem_texts.append(content)
                elif isinstance(item, str):
                    mem_texts.append(item.strip())
        else:
            mem_texts = [str(messages)]

        with self._connection() as conn:
            for text in mem_texts:
                if not text:
                    continue
                mem_id = f"mem_{uuid.uuid4().hex[:12]}"
                conn.execute(
                    """
                    INSERT INTO mem0_memories (id, user_id, memory, categories, metadata_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (mem_id, user_id, text, json.dumps(["general"]), metadata_str, now, now),
                )
                results.append({
                    "id": mem_id,
                    "memory": text,
                    "user_id": user_id,
                    "metadata": metadata or {},
                    "created_at": now,
                    "event": "ADD",
                })
            conn.commit()

        return results

    def get_all(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all stored memories for a user."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, user_id, memory, metadata_json, created_at, updated_at
                FROM mem0_memories
                WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (user_id, limit),
            )
            rows = cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "user_id": r["user_id"],
                    "memory": r["memory"],
                    "metadata": json.loads(r["metadata_json"] or "{}"),
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
                for r in rows
            ]

    def search(self, query: str, user_id: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search user memories via keyword and relevance scoring."""
        all_memories = self.get_all(user_id=user_id, limit=200)
        if not all_memories or not query:
            return all_memories[:limit]

        q_terms = [t.lower() for t in query.split() if len(t) >= 2]
        scored: List[tuple[int, Dict[str, Any]]] = []

        for m in all_memories:
            text = m["memory"].lower()
            score = 0
            for term in q_terms:
                if term in text:
                    score += 2
            if score > 0:
                scored.append((score, m))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [item[1] for item in scored[:limit]]
        if not results:
            # Fallback: return most recent memories if no exact keyword hit
            results = all_memories[:limit]
        return results

    def delete(self, memory_id: str) -> bool:
        """Delete a single memory by ID."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM mem0_memories WHERE id = ?", (memory_id,))
            conn.commit()
            return cursor.rowcount > 0

    def delete_all(self, user_id: str) -> bool:
        """Delete all memories for a user."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM mem0_memories WHERE user_id = ?", (user_id,))
            conn.commit()
            return cursor.rowcount > 0

    def count(self, user_id: Optional[str] = None) -> int:
        """Count memories in database."""
        with self._connection() as conn:
            if user_id:
                cursor = conn.execute(
                    "SELECT COUNT(*) FROM mem0_memories WHERE user_id = ?", (user_id,)
                )
            else:
                cursor = conn.execute("SELECT COUNT(*) FROM mem0_memories")
            return cursor.fetchone()[0]


class Mem0Service:
    """High-level Mem0 Memory Service supporting Cloud & Local Storage."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        settings = get_settings()
        self.api_key = (api_key if api_key is not None else settings.mem0_api_key).strip()
        self.local_storage = LocalMem0Storage(settings.mem0_db_path)
        self.cloud_client = None
        self._cloud_available = False

        if self.api_key:
            self._init_cloud_client()

    def _init_cloud_client(self) -> None:
        """Attempt to initialize official Mem0 MemoryClient."""
        try:
            from mem0 import MemoryClient

            self.cloud_client = MemoryClient(api_key=self.api_key)
            self._cloud_available = True
            logger.info("Mem0 Cloud Client connected successfully.")
        except Exception as e:
            logger.warning("Mem0 Cloud Client initialization failed, using local engine: %s", e)
            self.cloud_client = None
            self._cloud_available = False

    def is_cloud_active(self) -> bool:
        """Check if Mem0 Cloud Client is authenticated and active."""
        return self._cloud_available and self.cloud_client is not None

    def add_memory(
        self,
        content: Union[str, List[Dict[str, str]]],
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Add user memory either to Mem0 Cloud or local SQLite store."""
        uid = (user_id or "default_user").strip()
        if self.is_cloud_active():
            try:
                res = self.cloud_client.add(messages=content, user_id=uid, metadata=metadata or {})
                # Format Mem0 cloud return
                if isinstance(res, list):
                    return res
                if isinstance(res, dict) and "results" in res:
                    return res["results"]
                return [res] if isinstance(res, dict) else []
            except Exception as e:
                logger.warning("Mem0 cloud add failed, falling back to local: %s", e)

        return self.local_storage.add(messages=content, user_id=uid, metadata=metadata)

    def search_memories(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories matching query for specified user."""
        uid = (user_id or "default_user").strip()
        if self.is_cloud_active():
            try:
                try:
                    res = self.cloud_client.search(query=query, filters={"user_id": uid}, limit=limit)
                except Exception:
                    res = self.cloud_client.search(query=query, user_id=uid, limit=limit)
                if isinstance(res, list):
                    return res
                if isinstance(res, dict) and "results" in res:
                    return res["results"]
            except Exception as e:
                logger.warning("Mem0 cloud search failed, falling back to local: %s", e)

        return self.local_storage.search(query=query, user_id=uid, limit=limit)

    def get_all_memories(self, user_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve all memories for the given user."""
        uid = (user_id or "default_user").strip()
        if self.is_cloud_active():
            try:
                try:
                    res = self.cloud_client.get_all(filters={"user_id": uid}, limit=limit)
                except Exception:
                    res = self.cloud_client.get_all(user_id=uid, limit=limit)
                if isinstance(res, list):
                    return res
                if isinstance(res, dict) and "results" in res:
                    return res["results"]
            except Exception as e:
                logger.warning("Mem0 cloud get_all failed, falling back to local: %s", e)

        return self.local_storage.get_all(user_id=uid, limit=limit)

    def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory by its ID."""
        if self.is_cloud_active():
            try:
                self.cloud_client.delete(memory_id=memory_id)
                return True
            except Exception as e:
                logger.warning("Mem0 cloud delete failed: %s", e)

        return self.local_storage.delete(memory_id=memory_id)

    def clear_user_memories(self, user_id: str) -> bool:
        """Clear all memories for a specific user."""
        uid = (user_id or "default_user").strip()
        if self.is_cloud_active():
            try:
                self.cloud_client.delete_all(user_id=uid)
            except Exception as e:
                logger.warning("Mem0 cloud delete_all failed: %s", e)

        return self.local_storage.delete_all(user_id=uid)

    def extract_and_save_conversation_memory(
        self,
        user_id: str,
        question: str,
        answer: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Extract key constitutional focus and domain preferences to persist in Mem0."""
        uid = (user_id or "default_user").strip()
        meta = metadata or {}
        art = meta.get("article_number")
        topic = meta.get("topic")

        facts = []
        if art:
            facts.append(f"Researched Article {art} of the Constitution of India.")
        if topic:
            facts.append(f"Interested in constitutional topic: {topic}.")

        q_lower = question.lower()
        if "advocate" in q_lower or "counsel" in q_lower or "petition" in q_lower or "writ" in q_lower:
            facts.append("Identifies as legal practitioner / working on litigation or writ petitions.")
        elif "student" in q_lower or "exam" in q_lower or "upsc" in q_lower or "judiciary" in q_lower:
            facts.append("Preparing for competitive legal examinations / judicial services.")
        elif "simple" in q_lower or "explain" in q_lower or "layman" in q_lower:
            facts.append("Prefers clear, accessible, plain-English explanations.")

        for fact in facts:
            self.add_memory(content=fact, user_id=uid, metadata=meta)

    def get_status(self, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Return diagnostic status of Mem0 engine."""
        uid = user_id or "default_user"
        total_user = len(self.get_all_memories(uid))
        return {
            "mode": "Mem0 Cloud Platform" if self.is_cloud_active() else "Mem0 Local Engine (SQLite)",
            "is_cloud": self.is_cloud_active(),
            "user_id": uid,
            "total_memories": total_user,
        }


_mem0_instance: Optional[Mem0Service] = None


def get_mem0_service() -> Mem0Service:
    """Singleton getter for Mem0Service."""
    global _mem0_instance
    if _mem0_instance is None:
        _mem0_instance = Mem0Service()
    return _mem0_instance
