"""Tests for Auth0 Authentication and Mem0 Memory Services in ConstitutionGPT."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.auth.auth0_service import Auth0Service, get_auth0_service
from app.graph.workflow import run_constitution_chat
from app.memory.mem0_service import LocalMem0Storage, Mem0Service, get_mem0_service
from app.retrieval.gemini_service import generate_gemini_constitutional_response


def test_auth0_service_defaults_and_demo():
    """Verify default Auth0 state and demo persona generation."""
    service = Auth0Service(domain="", client_id="")
    assert not service.is_configured
    assert service.get_authorize_url() == ""

    # Test demo personas
    advocate = Auth0Service.get_demo_user("advocate")
    assert advocate["user_id"] == "auth0|demo_advocate_priya"
    assert advocate["name"] == "Advocate Priya Sharma"
    assert advocate["is_authenticated"] is True
    assert advocate["auth_provider"] == "demo"

    scholar = Auth0Service.get_demo_user("scholar")
    assert scholar["name"] == "Prof. Vikram Sen"
    assert scholar["is_authenticated"] is True


def test_auth0_authorize_url_generation():
    """Verify Auth0 Universal Login URL is constructed correctly."""
    service = Auth0Service(
        domain="legal-tenant.us.auth0.com",
        client_id="test_client_id_123",
        client_secret="test_secret",
        callback_url="http://localhost:8501",
    )
    assert service.is_configured
    url = service.get_authorize_url(state="xyz_state")
    assert "https://legal-tenant.us.auth0.com/authorize" in url
    assert "client_id=test_client_id_123" in url
    assert "response_type=code" in url
    assert "redirect_uri=http%3A%2F%2Flocalhost%3A8501" in url
    assert "state=xyz_state" in url


def test_auth0_profile_normalization():
    """Verify user profile normalization from Auth0 payload."""
    raw = {
        "sub": "auth0|65f123456789",
        "name": "Dr. B. R. Ambedkar",
        "email": "ambedkar@cadindia.clpr.org.in",
        "nickname": "Babasaheb",
        "picture": "https://example.com/pic.png",
    }
    normalized = Auth0Service.normalize_user_profile(raw)
    assert normalized["user_id"] == "auth0|65f123456789"
    assert normalized["name"] == "Dr. B. R. Ambedkar"
    assert normalized["email"] == "ambedkar@cadindia.clpr.org.in"
    assert normalized["is_authenticated"] is True
    assert normalized["auth_provider"] == "auth0"


def test_mem0_local_storage_crud():
    """Verify Mem0 Local SQLite storage CRUD operations."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_memories.db"
        storage = LocalMem0Storage(db_path)

        user_id = "test_user_42"

        # 1. Add memories
        res1 = storage.add(
            messages="Focusing on Article 21 and the right to privacy.",
            user_id=user_id,
            metadata={"domain": "fundamental_rights"},
        )
        assert len(res1) == 1
        assert res1[0]["memory"] == "Focusing on Article 21 and the right to privacy."

        storage.add(
            messages="Practicing as an advocate in the Supreme Court.",
            user_id=user_id,
            metadata={"role": "advocate"},
        )

        # 2. Get all memories
        all_mems = storage.get_all(user_id)
        assert len(all_mems) == 2

        # 3. Search memories
        privacy_results = storage.search(query="privacy Article 21", user_id=user_id)
        assert len(privacy_results) > 0
        assert any("Article 21" in m["memory"] for m in privacy_results)

        # 4. Delete individual memory
        mem_id_to_del = all_mems[0]["id"]
        deleted = storage.delete(mem_id_to_del)
        assert deleted is True
        remaining = storage.get_all(user_id)
        assert len(remaining) == 1

        # 5. Clear all memories for user
        cleared = storage.delete_all(user_id)
        assert cleared is True
        assert len(storage.get_all(user_id)) == 0


def test_mem0_service_operations():
    """Verify high-level Mem0Service methods."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_mem0_service.db"
        service = Mem0Service(api_key="")  # Empty key triggers local storage
        service.local_storage = LocalMem0Storage(db_path)

        user_id = "advocate_test_user"

        # Add memory
        added = service.add_memory(
            content="Client has a habeas corpus petition under Article 32.",
            user_id=user_id,
        )
        assert len(added) >= 1

        # Search memory
        matches = service.search_memories(query="habeas corpus", user_id=user_id)
        assert len(matches) >= 1
        assert "habeas corpus" in matches[0]["memory"]

        # Status check
        status = service.get_status(user_id)
        assert status["total_memories"] == 1
        assert status["user_id"] == user_id


def test_mem0_auto_extraction():
    """Verify Mem0 auto-extraction of conversation turn."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_auto_extract.db"
        service = Mem0Service(api_key="")
        service.local_storage = LocalMem0Storage(db_path)

        user_id = "student_user"
        service.extract_and_save_conversation_memory(
            user_id=user_id,
            question="I am an advocate preparing a writ petition under Article 32.",
            answer="Article 32 provides remedies for enforcement of rights...",
            metadata={"article_number": "32", "topic": "Constitutional Remedies"},
        )

        mems = service.get_all_memories(user_id)
        mem_texts = [m["memory"] for m in mems]
        assert any("Article 32" in t for t in mem_texts)
        assert any("Constitutional Remedies" in t for t in mem_texts)
        assert any("advocate" in t.lower() or "litigation" in t.lower() for t in mem_texts)


def test_gemini_service_with_mem0_memories():
    """Verify Gemini service properly formats Mem0 memories into the prompt."""
    with patch("app.retrieval.gemini_service.get_http_session") as mock_http:
        mock_session = MagicMock()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {"content": {"parts": [{"text": "Article 21 provides protection of life and personal liberty."}]}}
            ]
        }
        mock_session.post.return_value = mock_resp
        mock_http.return_value = mock_session

        memories = [
            {"memory": "User is a senior advocate specializing in Article 21 jurisprudence."},
            {"memory": "User prefers concise case law citations."},
        ]

        ans = generate_gemini_constitutional_response(
            question="Explain the scope of Article 21",
            retrieved_docs=[
                {
                    "article_number": "21",
                    "article_title": "Protection of life and personal liberty",
                    "part": "PART III",
                    "page_start": 39,
                    "constitutional_text": "No person shall be deprived of his life or personal liberty except according to procedure established by law.",
                }
            ],
            conversation_history=[],
            user_memories=memories,
            api_key="mock_key",
        )

        assert "Article 21" in ans
        # Ensure post was called with prompt containing the memory section
        call_args = mock_session.post.call_args
        prompt_sent = call_args[1]["json"]["contents"][0]["parts"][0]["text"]
        assert "MEM0 PERSISTENT USER MEMORIES" in prompt_sent
        assert "senior advocate specializing in Article 21" in prompt_sent


def test_workflow_execution_with_user_id():
    """Verify LangGraph run_constitution_chat runs end-to-end with user_id and Mem0."""
    result = run_constitution_chat(
        question="What is Article 14?",
        thread_id="test_mem0_thread_999",
        selected_engine="rule_based",
        user_id="advocate_user_test",
    )

    assert result is not None
    assert "answer" in result
    assert "Article 14" in result["answer"]
    assert result.get("user_id") == "advocate_user_test"
