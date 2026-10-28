"""Tests for LLM Direct Domain Boundary Decision.

Verifies that user questions go directly to the LLM, which decides whether the query is
related to the Constitution of India domain or not.
If not domain-related (e.g. 'who is modi', 'who is chandra shekar rao'):
  - Asks the user to ask domain-related questions only.
  - Gives NO extra information (no unrelated constitutional articles or chapters).
If domain-related (e.g. 'what is article 21'):
  - Gives accurate, PDF-grounded constitutional answer.
"""

from langgraph.checkpoint.memory import MemorySaver

from app.graph.workflow import get_global_store, run_constitution_chat


def test_out_of_domain_who_is_modi():
    """Verify that 'who is modi' is identified as out-of-domain with no extra info."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = "test-out-of-domain-modi"

    result = run_constitution_chat(
        question="who is modi",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=False,
    )

    answer = result.get("answer", "")
    gemini_answer = result.get("gemini_answer", "")
    resp_type = result.get("response_type")

    # Must be classified as out_of_scope
    assert resp_type == "out_of_scope"

    # Must prompt user to ask questions based on domain only
    assert "Please ask questions based on the Constitution of India domain only." in answer
    if gemini_answer:
        assert "Please ask questions based on the Constitution of India domain only." in gemini_answer

    # Must give NO extra information (must NOT dump Article 52, Article 74, or Council of Ministers)
    assert "Article 52" not in answer
    assert "Article 74" not in answer
    assert "Council of Ministers" not in answer
    assert "Parliament" not in answer


def test_out_of_domain_who_is_chandra_shekar_rao():
    """Verify that 'who is chandra shekar rao' is identified as out-of-domain with no extra info."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = "test-out-of-domain-kcr"

    result = run_constitution_chat(
        question="who is chandra shekar rao",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=False,
    )

    answer = result.get("answer", "")
    resp_type = result.get("response_type")

    assert resp_type == "out_of_scope"
    assert "Please ask questions based on the Constitution of India domain only." in answer
    assert "Article 79" not in answer
    assert "Article 80" not in answer


def test_out_of_domain_name_the_president_of_india():
    """Verify that 'name the president of india' asks for a person and is rejected with domain restriction."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = "test-name-president"

    result = run_constitution_chat(
        question="name the president of india",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=False,
    )

    answer = result.get("answer", "")
    resp_type = result.get("response_type")

    assert resp_type == "out_of_scope"
    assert "Please ask questions based on the Constitution of India domain only." in answer
    # Must NOT dump articles 52-78 or The Union Executive
    assert "Articles 52 to 78" not in answer
    assert "The Union Executive — President" not in answer


def test_out_of_domain_who_is_the_president():
    """Verify that 'who is the president' asks for a person and is rejected with domain restriction."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = "test-who-is-president"

    result = run_constitution_chat(
        question="who is the president",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=False,
    )

    answer = result.get("answer", "")
    resp_type = result.get("response_type")

    assert resp_type == "out_of_scope"
    assert "Please ask questions based on the Constitution of India domain only." in answer


def test_in_domain_article_21():
    """Verify that 'what is article 21' is correctly identified as domain-related and answered."""
    saver = MemorySaver()
    store = get_global_store()
    thread_id = "test-in-domain-art21"

    result = run_constitution_chat(
        question="what is article 21",
        thread_id=thread_id,
        checkpointer=saver,
        store=store,
        enable_interrupt=False,
    )

    answer = result.get("answer", "")
    resp_type = result.get("response_type")

    assert resp_type == "article"
    assert "Article 21" in answer
    assert "Protection of life and personal liberty" in answer or "personal liberty" in answer
