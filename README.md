# ConstitutionGPT — Indian Constitution Assistant

A complete, deterministic legal question-answering application for the **Constitution of India**, built with **Streamlit**, **LangGraph Graph API**, and **SQLite Checkpointer**.

> **100% Local & Rule-Based:**
> No LLMs • No Generative AI • No Model Embeddings • No Paid APIs • No React • No External Network Calls

---

## 🏛️ Project Overview

ConstitutionGPT answers constitutional inquiries by directly querying the official, loaded Constitution of India PDF (`95ef471035ee0b5e4e5b1cac3d3584c1.pdf`). It features:

- **Exact Article Lookups:** Full verbatim constitutional text, parts, and page numbers for Articles 1 to 395 and lettered articles (e.g. 21A, 51A).
- **Plain English Explanations:** Clear, student-friendly legal explanations grounded in statutory wording.
- **Constitutional Comparisons:** Side-by-side comparative analysis (e.g., Article 14 vs Article 21, the Golden Triangle doctrine).
- **Contextual Follow-ups:** Conversation memory resolving pronouns ("Explain it simply", "Give an example", "Which article did I ask about previously?").
- **Thematic Topics:** Fundamental Rights, Directive Principles, Fundamental Duties, President & Parliament, Supreme Court & High Courts, Emergency Provisions, and Schedules.
- **Honest Uncertainty:** Explicitly reports when information is missing rather than inventing facts.
- **Session Persistence:** Retains full conversation state and history across Streamlit reruns and app restarts via `langgraph-checkpoint-sqlite`.

---

## 🛠️ Technology Stack

- **Python 3.11+ / 3.13**
- **Streamlit:** Complete frontend and application entry point
- **LangGraph:** `StateGraph`, `START`, `END`, conditional routing, and graph execution
- **LangChain Core:** `BaseMessage`, `HumanMessage`, `AIMessage`, and `add_messages` reducer
- **SQLite & `langgraph-checkpoint-sqlite`:** Persistent state storage and conversation registry
- **`pypdf`:** PDF text extraction with page preservation and SHA-256 verification
- **Pytest:** Comprehensive automated test suite

---

## 📁 Project Structure

```text
lawchatbot/
├── streamlit_app.py             # Streamlit application entry point
├── requirements.txt             # Python package dependencies
├── README.md                    # This documentation file
├── .gitignore                   # Ignored files for version control
├── config/
│   ├── __init__.py
│   └── settings.py              # Central settings and file path definitions
├── app/
│   ├── __init__.py
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── state.py             # ConstitutionState TypedDict
│   │   ├── nodes.py             # 8 deterministic graph nodes
│   │   ├── routing.py           # Conditional routing functions
│   │   └── workflow.py          # StateGraph assembly & compilation
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py     # PDF validation, hash, and text extraction
│   │   ├── article_parser.py    # Article parser & curated repositories
│   │   └── knowledge_base.py    # Multi-field scoring and search
│   └── persistence/
│       ├── __init__.py
│       ├── checkpointer.py      # SQLite checkpointer (SqliteSaver)
│       └── conversation_db.py   # Conversation registry database
├── data/
│   ├── constitution.pdf         # Local copy of the verified PDF
│   ├── extracted_text.json      # Extracted text per page (402 pages)
│   ├── articles.json            # Parsed articles & constitutional text
│   ├── metadata.json            # Hash, page counts, and extraction stats
│   ├── checkpoints.db           # Persistent LangGraph state checkpoints
│   └── conversations.db         # Conversation registry for sidebar
├── tests/
│   ├── __init__.py
│   ├── test_pdf_processor.py    # PDF existence, extraction, OCR tests
│   ├── test_retrieval.py        # Lookup, normalization, scoring tests
│   ├── test_graph.py            # Nodes, routing, accumulation tests
│   ├── test_checkpointer.py     # Persistence & thread isolation tests
│   ├── test_conversations.py    # Conversation CRUD tests
│   └── test_master_examples.py  # 10 canonical user queries test
└── docs/
    ├── LANGGRAPH_GUIDE.md       # Beginner guide mapping concepts to code
    └── ARCHITECTURE.md          # Architectural diagrams and data flow
```

---

## 🚀 Getting Started (Windows PowerShell)

### 1. Prerequisites

Ensure Python 3.11+ is installed. The Constitution PDF should be present at:
```text
C:\Users\Lenovo\Downloads\95ef471035ee0b5e4e5b1cac3d3584c1.pdf
```

### 2. Setup Virtual Environment (Optional but Recommended)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Run Automated Test Suite

```powershell
python -m pytest -v
```

All 23 unit and integration tests should pass:
```text
============================= 23 passed in 1.50s ==============================
```

### 5. Launch the Streamlit Application

```powershell
streamlit run streamlit_app.py
```

The application will open in your default browser at `http://localhost:8501`.

---

## 💡 Example Questions to Try

1. **"What is Article 21?"**
   - Retrieves verbatim constitutional text from Page 42, Part III, plain English explanation, and privacy doctrines.
2. **"Explain Article 14 in simple English."**
   - Breaks down equality before the law, equal protection, and non-arbitrariness.
3. **"What does Article 32 say?"**
   - Details constitutional remedies, the 5 prerogative writs, and Dr. B.R. Ambedkar's "Heart and Soul" doctrine.
4. **"What are the Fundamental Rights?"**
   - Provides a comprehensive overview of Part III (Articles 12-35) with key categories.
5. **"Compare Article 14 and Article 21."**
   - Shows a structured side-by-side comparison table and explains the *Golden Triangle* doctrine.
6. **"Explain it simply."** (Follow-up)
   - Resolves context from prior discussion and simplifies the explanation.
7. **"Give an example of Article 21."**
   - Illustrates practical applications like the Right to Privacy (*Puttaswamy*) and humane treatment.
8. **"Which article did I ask about previously?"**
   - Inspects conversation history in the active thread and identifies the earlier article.
9. **"What is Article 51A?"**
   - Details Fundamental Duties of citizens under Part IVA.
10. **"What is the stock price of Apple?"**
    - Correctly triggers the uncertainty handler, explaining that the question is outside the loaded Constitution PDF.

---

## 🔄 Conversation Persistence & Deletion

- **Multi-thread Isolation:** Clicking **"➕ New Chat"** in the sidebar generates a new UUID thread. You can freely switch between chats; their states remain isolated.
- **Restart Resilience:** If you restart Streamlit or your computer, your conversations and messages remain intact in `data/checkpoints.db` and `data/conversations.db`.
- **Complete Deletion:** Deleting a chat thread removes the entry from the conversation list AND purges all state checkpoints from the SQLite checkpointer database.
