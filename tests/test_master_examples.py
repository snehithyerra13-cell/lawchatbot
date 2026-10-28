"""End-to-end test verifying all 10 canonical example questions from the specification."""

import uuid
import pytest
from app.graph.workflow import run_constitution_chat


def test_ten_canonical_spec_questions():
    """Verify all 10 questions specified in project requirements in a continuous conversation."""
    thread_id = str(uuid.uuid4())

    # 1. What is Article 21?
    r1 = run_constitution_chat("What is Article 21?", thread_id)
    assert r1["response_type"] == "article"
    assert "Article 21" in r1["answer"]
    assert "Protection of life and personal liberty" in r1["answer"]
    assert len(r1["messages"]) == 2

    # 2. Explain Article 14 in simple English.
    r2 = run_constitution_chat("Explain Article 14 in simple English.", thread_id)
    assert r2["response_type"] == "article"
    assert "Article 14" in r2["answer"]
    assert "Equality before law" in r2["answer"]
    assert len(r2["messages"]) == 4

    # 3. What does Article 32 say?
    r3 = run_constitution_chat("What does Article 32 say?", thread_id)
    assert r3["response_type"] == "article"
    assert "Article 32" in r3["answer"]
    assert "Remedies" in r3["answer"]
    assert len(r3["messages"]) == 6

    # 4. What are the Fundamental Rights?
    r4 = run_constitution_chat("What are the Fundamental Rights?", thread_id)
    assert r4["response_type"] == "topic"
    assert "Fundamental Rights" in r4["answer"]
    assert "Part III" in r4["answer"]
    assert len(r4["messages"]) == 8

    # 5. Explain Article 19.
    r5 = run_constitution_chat("Explain Article 19.", thread_id)
    assert r5["response_type"] == "article"
    assert "Article 19" in r5["answer"]
    assert "freedom of speech" in r5["answer"].lower()
    assert len(r5["messages"]) == 10

    # 6. Compare Article 14 and Article 21.
    r6 = run_constitution_chat("Compare Article 14 and Article 21.", thread_id)
    assert r6["response_type"] == "comparison"
    assert "Article 14" in r6["answer"] and "Article 21" in r6["answer"]
    assert "Golden Triangle" in r6["answer"]
    assert len(r6["messages"]) == 12

    # 7. Explain the article we discussed earlier.
    # Last mentioned in comparison is Article 14/21, history resolves to 14/21
    r7 = run_constitution_chat("Explain the article we discussed earlier.", thread_id)
    assert r7["response_type"] == "article"
    assert ("Article 14" in r7["answer"] or "Article 21" in r7["answer"])
    assert len(r7["messages"]) == 14

    # 8. Give an example of Article 21.
    r8 = run_constitution_chat("Give an example of Article 21.", thread_id)
    assert r8["response_type"] == "article"
    assert "Article 21" in r8["answer"]
    assert "Scenario" in r8["answer"] or "Privacy" in r8["answer"] or "Example" in r8["answer"]
    assert len(r8["messages"]) == 16

    # 9. Which article did I ask about previously?
    r9 = run_constitution_chat("Which article did I ask about previously?", thread_id)
    assert r9["response_type"] == "article"
    assert "Article 21" in r9["answer"]
    assert len(r9["messages"]) == 18

    # 10. Explain it in simpler words.
    r10 = run_constitution_chat("Explain it in simpler words.", thread_id)
    assert r10["response_type"] == "article"
    assert "Article 21" in r10["answer"]
    assert "In Plain English" in r10["answer"] or "Explanation" in r10["answer"]
    assert len(r10["messages"]) == 20
