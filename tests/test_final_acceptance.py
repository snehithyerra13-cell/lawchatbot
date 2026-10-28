"""Final Acceptance Test — Section 17 Workflow Verification.

Verifies:
1. Ask 'What is Article 21?' -> Comes from loaded PDF.
2. Ask 'Explain it simply.' -> Uses correct article from conversation history.
3. Create a new conversation -> Ask about Article 14.
4. Switch back to Article 21 conversation -> History remains intact.
5. Simulate app restart (re-instantiate checkpointer & graph) -> Reopen original conversation.
6. Confirm persisted graph state remains available.
"""

import uuid
import pytest
from langchain_core.messages import HumanMessage

from app.graph.workflow import build_constitution_graph, run_constitution_chat
from app.persistence.checkpointer import get_sqlite_checkpointer
from app.persistence.conversation_db import get_conversation_db


def test_section_17_full_acceptance_workflow():
    """Verify Section 17 full multi-turn conversation and persistence lifecycle."""
    conv_db = get_conversation_db()
    checkpointer = get_sqlite_checkpointer()

    # Step 1: Create Conversation A
    thread_a = str(uuid.uuid4())
    conv_db.create_conversation(thread_a, "Article 21 Thread")

    # Step 2 & 3: Ask 'What is Article 21?' and confirm it comes from loaded PDF
    res1 = run_constitution_chat("What is Article 21?", thread_a, checkpointer=checkpointer)
    assert res1["response_type"] == "article"
    assert "Article 21" in res1["answer"]
    assert "constitution.pdf" in res1["answer"] or "Page 42" in res1["answer"]
    assert "Protection of life and personal liberty" in res1["answer"]
    assert len(res1["messages"]) == 2

    # Step 4 & 5: Ask 'Explain it simply.' and confirm follow-up uses Article 21
    res2 = run_constitution_chat("Explain it simply.", thread_a, checkpointer=checkpointer)
    assert res2["response_type"] == "article"
    assert "Article 21" in res2["answer"]
    assert "In Plain English" in res2["answer"] or "Plain English" in res2["answer"]
    assert len(res2["messages"]) == 4

    # Step 6 & 7: Create a new conversation B and ask about Article 14
    thread_b = str(uuid.uuid4())
    conv_db.create_conversation(thread_b, "Article 14 Thread")
    res_b1 = run_constitution_chat("Tell me about Article 14.", thread_b, checkpointer=checkpointer)
    assert res_b1["response_type"] == "article"
    assert "Article 14" in res_b1["answer"]
    assert "Equality before law" in res_b1["answer"]
    assert len(res_b1["messages"]) == 2

    # Step 8 & 9: Switch back to Article 21 conversation and confirm history remains intact
    builder = build_constitution_graph()
    graph = builder.compile(checkpointer=checkpointer)
    state_a = graph.get_state({"configurable": {"thread_id": thread_a}})
    assert state_a is not None
    messages_a = state_a.values.get("messages", [])
    assert len(messages_a) == 4
    assert "Article 21" in messages_a[1].content
    assert "Article 21" in messages_a[3].content

    # Confirm Thread B state is isolated and has only 2 messages
    state_b = graph.get_state({"configurable": {"thread_id": thread_b}})
    assert state_b is not None
    messages_b = state_b.values.get("messages", [])
    assert len(messages_b) == 2
    assert "Article 14" in messages_b[1].content

    # Step 10 & 11: Simulate restarting Streamlit / app by creating a new graph instance
    # with a fresh connection to the existing database file
    fresh_checkpointer = get_sqlite_checkpointer()
    restarted_graph = build_constitution_graph().compile(checkpointer=fresh_checkpointer)

    # Step 12: Reopen original conversation and confirm persisted state remains available
    recovered_state_a = restarted_graph.get_state({"configurable": {"thread_id": thread_a}})
    assert recovered_state_a is not None
    recovered_msgs_a = recovered_state_a.values.get("messages", [])
    assert len(recovered_msgs_a) == 4
    assert "Article 21" in recovered_msgs_a[1].content
    assert "Article 21" in recovered_msgs_a[3].content

    # Clean up test threads
    conv_db.delete_conversation(thread_a)
    conv_db.delete_conversation(thread_b)
