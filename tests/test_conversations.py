"""Tests for Conversation Registry SQLite database."""

import tempfile
from pathlib import Path
import uuid
import pytest

from app.persistence.conversation_db import (
    ConversationDB,
    generate_title_from_query,
)


def test_conversation_db_crud():
    """Verify conversation registry CRUD operations."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        registry = ConversationDB(db_path=db_path)

        thread_1 = str(uuid.uuid4())
        thread_2 = str(uuid.uuid4())

        # 1. Create
        c1 = registry.create_conversation(thread_1, "Article 21 Consultation")
        assert c1["title"] == "Article 21 Consultation"
        assert c1["thread_id"] == thread_1

        c2 = registry.create_conversation(thread_2, "Article 14 Inquiry")
        assert c2["thread_id"] == thread_2

        # 2. List
        convs = registry.list_conversations()
        assert len(convs) == 2
        thread_ids = [c["thread_id"] for c in convs]
        assert thread_1 in thread_ids
        assert thread_2 in thread_ids

        # 3. Get
        retrieved = registry.get_conversation(thread_1)
        assert retrieved is not None
        assert retrieved["title"] == "Article 21 Consultation"
        assert retrieved.get("engine_mode") == "both"

        # 4. Update
        updated = registry.update_conversation(thread_1, title="Renamed Consultation", engine_mode="rule_based")
        assert updated is True
        retrieved_again = registry.get_conversation(thread_1)
        assert retrieved_again["title"] == "Renamed Consultation"
        assert retrieved_again["engine_mode"] == "rule_based"

        # 5. Delete
        deleted = registry.delete_conversation(thread_2)
        assert deleted is True
        remaining = registry.list_conversations()
        assert len(remaining) == 1
        assert remaining[0]["thread_id"] == thread_1
    finally:
        if db_path.exists():
            db_path.unlink()


def test_generate_title_from_query():
    """Verify deterministic title truncation without LLM."""
    short_q = "What is Article 21?"
    assert generate_title_from_query(short_q) == "What is Article 21?"

    long_q = "Please explain Article 19 regarding freedom of speech and expression in great detail for a law exam"
    title = generate_title_from_query(long_q, max_length=25)
    assert len(title) <= 28
    assert title.endswith("...")


def test_multi_user_isolation():
    """Verify that different users have completely isolated conversations and storage."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

    try:
        registry = ConversationDB(db_path=db_path)

        user_alice = "user_alice"
        user_bob = "user_bob"

        t_alice_1 = str(uuid.uuid4())
        t_alice_2 = str(uuid.uuid4())
        t_bob_1 = str(uuid.uuid4())

        # Alice creates two conversations
        registry.create_conversation(t_alice_1, "Alice Chat 1", user_id=user_alice)
        registry.create_conversation(t_alice_2, "Alice Chat 2", user_id=user_alice)

        # Bob creates one conversation
        registry.create_conversation(t_bob_1, "Bob Chat 1", user_id=user_bob)

        # Alice only sees her 2 conversations
        alice_convs = registry.list_conversations(user_id=user_alice)
        assert len(alice_convs) == 2
        alice_ids = [c["thread_id"] for c in alice_convs]
        assert t_alice_1 in alice_ids
        assert t_alice_2 in alice_ids
        assert t_bob_1 not in alice_ids

        # Bob only sees his 1 conversation
        bob_convs = registry.list_conversations(user_id=user_bob)
        assert len(bob_convs) == 1
        assert bob_convs[0]["thread_id"] == t_bob_1
        assert bob_convs[0]["title"] == "Bob Chat 1"

        # Ownership validation
        assert registry.get_conversation(t_alice_1, user_id=user_alice) is not None
        assert registry.get_conversation(t_alice_1, user_id=user_bob) is None
        assert registry.get_conversation(t_bob_1, user_id=user_alice) is None
        assert registry.get_conversation(t_bob_1, user_id=user_bob) is not None

        # List all users
        users = registry.list_users()
        assert user_alice in users
        assert user_bob in users

        # Alice deletes all her conversations; Bob's remains untouched
        registry.delete_all_for_user(user_alice)
        assert len(registry.list_conversations(user_id=user_alice)) == 0
        assert len(registry.list_conversations(user_id=user_bob)) == 1
    finally:
        if db_path.exists():
            db_path.unlink()


def test_multi_user_store_isolation():
    """Verify that bookmarks and user preferences are isolated per user in Store, even with emails containing periods."""
    from app.persistence.store import (
        bookmark_article,
        is_article_bookmarked,
        list_bookmarks,
        remove_bookmark,
        get_user_preferences,
        update_user_preferences,
    )

    # Test with email addresses containing periods (e.g. snehithyerra13@gmail.com)
    u1 = "snehithyerra13@gmail.com"
    u2 = "lawyer.kumar@delhi.bar.in"

    # User 1 bookmarks Article 21
    bookmark_article("21", "Protection of life and personal liberty", user_id=u1)
    # User 2 bookmarks Article 14
    bookmark_article("14", "Equality before law", user_id=u2)

    # Check isolation
    assert is_article_bookmarked("21", user_id=u1) is True
    assert is_article_bookmarked("21", user_id=u2) is False

    assert is_article_bookmarked("14", user_id=u1) is False
    assert is_article_bookmarked("14", user_id=u2) is True

    u1_bms = [b["article_number"] for b in list_bookmarks(user_id=u1)]
    u2_bms = [b["article_number"] for b in list_bookmarks(user_id=u2)]
    assert "21" in u1_bms and "14" not in u1_bms
    assert "14" in u2_bms and "21" not in u2_bms

    # Preferences isolation
    update_user_preferences({"tone": "Academic"}, user_id=u1)
    update_user_preferences({"tone": "Plain English"}, user_id=u2)

    assert get_user_preferences(user_id=u1)["tone"] == "Academic"
    assert get_user_preferences(user_id=u2)["tone"] == "Plain English"

    # Cleanup
    remove_bookmark("21", user_id=u1)
    remove_bookmark("14", user_id=u2)
