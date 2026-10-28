"""SQLite checkpointer integration for LangGraph.

Provides persistent state storage, thread isolation, and checkpoint management
using langgraph.checkpoint.sqlite.SqliteSaver.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

from langgraph.checkpoint.sqlite import SqliteSaver

from config.settings import get_settings

logger = logging.getLogger(__name__)

_checkpointer_instance: Optional[SqliteSaver] = None
_sqlite_connection: Optional[sqlite3.Connection] = None


def get_sqlite_checkpointer(
    db_path: Optional[Path] = None,
) -> SqliteSaver:
    """Retrieve or initialize the persistent SQLite checkpointer."""
    global _checkpointer_instance, _sqlite_connection

    target_path = db_path or get_settings().checkpoints_db_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if _checkpointer_instance is None or _sqlite_connection is None:
        logger.info(
            "Initializing SQLite checkpointer at %s", target_path
        )
        # Use check_same_thread=False to support Streamlit rerun threads
        _sqlite_connection = sqlite3.connect(
            str(target_path),
            check_same_thread=False,
            timeout=30.0,
        )
        _checkpointer_instance = SqliteSaver(_sqlite_connection)
        _checkpointer_instance.setup()

    return _checkpointer_instance


def delete_thread_checkpoints(
    thread_id: str, db_path: Optional[Path] = None
) -> bool:
    """Delete all persisted checkpoints for a thread from SQLite checkpointer."""
    target_path = db_path or get_settings().checkpoints_db_path
    success = False

    # 1. Attempt official checkpointer delete_thread
    try:
        checkpointer = get_sqlite_checkpointer(target_path)
        if hasattr(checkpointer, "delete_thread"):
            try:
                checkpointer.delete_thread(thread_id)
                success = True
            except Exception as e:
                logger.debug("checkpointer.delete_thread exception: %s", e)
    except Exception as e:
        logger.debug("get_sqlite_checkpointer exception: %s", e)

    # 2. Direct SQL cleanup on checkpoints and writes tables
    try:
        conn = sqlite3.connect(str(target_path), timeout=10.0)
        try:
            conn.execute("DELETE FROM checkpoints WHERE thread_id = ?", (thread_id,))
            conn.execute("DELETE FROM writes WHERE thread_id = ?", (thread_id,))
            conn.commit()
            success = True
            logger.info("Direct SQL deleted checkpoints for thread %s", thread_id)
        finally:
            conn.close()
    except Exception as exc:
        logger.warning("Failed to direct-delete checkpoints for thread %s: %s", thread_id, exc)

    return success


def discard_thread_interrupted_turn(
    thread_id: str, db_path: Optional[Path] = None
) -> bool:
    """Roll back uncompleted checkpoints from an interrupted turn in SQLite."""
    target_path = db_path or get_settings().checkpoints_db_path
    try:
        from app.graph.workflow import get_compiled_graph, get_global_store

        checkpointer = get_sqlite_checkpointer(target_path)
        store = get_global_store()
        graph = get_compiled_graph(checkpointer=checkpointer, store=store)
        cfg = {"configurable": {"thread_id": thread_id}}

        uncompleted_ids = []
        for snap in graph.get_state_history(cfg):
            if snap.next:
                uncompleted_ids.append(snap.config["configurable"]["checkpoint_id"])
            else:
                break

        if uncompleted_ids:
            conn = sqlite3.connect(str(target_path), timeout=10.0)
            try:
                for cid in uncompleted_ids:
                    conn.execute(
                        "DELETE FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ?",
                        (thread_id, cid),
                    )
                    conn.execute(
                        "DELETE FROM writes WHERE thread_id = ? AND checkpoint_id = ?",
                        (thread_id, cid),
                    )
                conn.commit()
                logger.info(
                    "Discarded %d uncompleted checkpoints for thread %s",
                    len(uncompleted_ids),
                    thread_id,
                )
            finally:
                conn.close()
            return True
    except Exception as e:
        logger.warning(
            "Error discarding uncompleted checkpoints for thread %s: %s",
            thread_id,
            e,
        )
    return False

