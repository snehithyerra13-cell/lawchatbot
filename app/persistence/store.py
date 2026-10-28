"""LangGraph Store implementation for Cross-Thread Long-Term Memory.

Stores shared memory across conversation threads:
1. Constitutional Bookmarks (starred articles, notes)
2. User Profile Preferences (explanation tone, legal detail level)
3. Cross-thread Constitutional Research History
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langgraph.store.memory import InMemoryStore
from config.settings import get_settings

logger = logging.getLogger(__name__)

# File path to persist cross-thread store data across process restarts
STORE_FILE = Path(get_settings().data_dir) / "store.json"

_global_store: Optional[InMemoryStore] = None


def get_global_store() -> InMemoryStore:
    """Retrieve or initialize the global LangGraph Store instance."""
    global _global_store
    if _global_store is None:
        _global_store = InMemoryStore()
        load_store_from_disk(_global_store)
    return _global_store


def sanitize_store_namespace(label: str) -> str:
    """Sanitize namespace label to satisfy LangGraph Store rules.

    LangGraph forbids namespace labels that contain periods ('.'). We also
    replace any other characters that could cause issues (@, /, spaces) so
    that email addresses, Google sub-claims, and arbitrary user IDs are all
    safely converted to valid labels.
    """
    import re
    clean = (label or "user_default").strip()
    # Replace dots, @, slashes, spaces and any remaining non-alphanum/-_ chars
    clean = re.sub(r"[.\s@/|]+", "_", clean)
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", clean)
    return clean or "user_default"


def save_store_to_disk(store: Optional[InMemoryStore] = None) -> None:
    """Serialize the store items to disk for cross-session persistence."""
    st = store or get_global_store()
    try:
        data: Dict[str, Any] = {"bookmarks": {}, "profile": {}, "topics": {}}

        namespaces = st.list_namespaces()
        for ns in namespaces:
            if not ns or len(ns) < 2:
                continue
            ns_type, uid = ns[0], sanitize_store_namespace(ns[1])
            if ns_type == "bookmarks":
                if uid not in data["bookmarks"]:
                    data["bookmarks"][uid] = {}
                for item in st.search(ns):
                    data["bookmarks"][uid][item.key] = item.value
            elif ns_type == "profile":
                if uid not in data["profile"]:
                    data["profile"][uid] = {}
                for item in st.search(ns):
                    data["profile"][uid][item.key] = item.value
            elif ns_type == "topics":
                if uid not in data["topics"]:
                    data["topics"][uid] = {}
                for item in st.search(ns):
                    data["topics"][uid][item.key] = item.value

        STORE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(STORE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not persist store to disk: {e}")


def load_store_from_disk(store: InMemoryStore) -> None:
    """Load persisted store items from disk if available."""
    if not STORE_FILE.exists():
        return
    try:
        with open(STORE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Bookmarks loading (supports per-user dict and legacy flat dict)
        bookmarks_data = data.get("bookmarks", {})
        for key, val in bookmarks_data.items():
            if isinstance(val, dict) and "article_number" in val:
                # Legacy format: key="art_...", val={article_number, ...}
                store.put(("bookmarks", "user_default"), key, val)
            elif isinstance(val, dict):
                # Multi-user format: key=user_id, val={art_key: bookmark_dict}
                uid = sanitize_store_namespace(key)
                for art_key, art_val in val.items():
                    store.put(("bookmarks", uid), art_key, art_val)

        # Profile preferences loading
        profile_data = data.get("profile", {})
        for key, val in profile_data.items():
            if key == "preferences" and isinstance(val, dict):
                # Legacy format: data["profile"]["preferences"]
                store.put(("profile", "user_default"), "preferences", val)
            elif isinstance(val, dict):
                # Multi-user format: data["profile"][user_id]
                uid = sanitize_store_namespace(key)
                for pref_key, pref_val in val.items():
                    store.put(("profile", uid), pref_key, pref_val)

        # Topics loading
        topics_data = data.get("topics", {})
        for key, val in topics_data.items():
            if isinstance(val, dict):
                for t_key, t_val in val.items():
                    store.put(("topics", sanitize_store_namespace(key)), t_key, t_val)
            else:
                store.put(("topics", "global"), key, val)
    except Exception as e:
        logger.warning(f"Could not load store from disk: {e}")


# Helper functions for Streamlit UI and Graph Nodes
def bookmark_article(
    article_number: str,
    title: str,
    part: str = "",
    notes: str = "",
    user_id: str = "user_default",
) -> None:
    """Save an article to cross-thread persistent bookmarks for a specific user."""
    st = get_global_store()
    key = f"art_{article_number}"
    uid = sanitize_store_namespace(user_id)
    st.put(
        ("bookmarks", uid),
        key,
        {
            "article_number": article_number,
            "title": title,
            "part": part,
            "notes": notes,
        },
    )
    save_store_to_disk(st)


def remove_bookmark(article_number: str, user_id: str = "user_default") -> None:
    """Remove an article from cross-thread bookmarks for a specific user."""
    st = get_global_store()
    key = f"art_{article_number}"
    uid = sanitize_store_namespace(user_id)
    st.delete(("bookmarks", uid), key)
    save_store_to_disk(st)


def list_bookmarks(user_id: str = "user_default") -> List[Dict[str, Any]]:
    """List all saved constitutional bookmarks for a specific user."""
    st = get_global_store()
    uid = sanitize_store_namespace(user_id)
    items = st.search(("bookmarks", uid))
    return [it.value for it in items]


def is_article_bookmarked(article_number: str, user_id: str = "user_default") -> bool:
    """Check if an article is currently bookmarked for a specific user."""
    st = get_global_store()
    key = f"art_{article_number}"
    uid = sanitize_store_namespace(user_id)
    item = st.get(("bookmarks", uid), key)
    return item is not None


def get_user_preferences(user_id: str = "user_default") -> Dict[str, Any]:
    """Retrieve user research preferences from store for a specific user."""
    st = get_global_store()
    uid = sanitize_store_namespace(user_id)
    item = st.get(("profile", uid), "preferences")
    if item:
        return item.value
    return {"tone": "Comprehensive", "detail_level": "Comprehensive"}


def update_user_preferences(
    prefs: Dict[str, Any], user_id: str = "user_default"
) -> None:
    """Update user research preferences in store for a specific user."""
    st = get_global_store()
    uid = sanitize_store_namespace(user_id)
    st.put(("profile", uid), "preferences", prefs)
    save_store_to_disk(st)
