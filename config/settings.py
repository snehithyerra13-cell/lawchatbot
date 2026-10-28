"""Application settings and path configurations for ConstitutionGPT."""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Load .env file automatically if present (so credentials survive restarts)
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(_env_path, override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on OS environment variables

# Base project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Default PDF location requested by user
DEFAULT_SOURCE_PDF = Path(
    os.getenv(
        "CONSTITUTION_PDF_PATH",
        r"C:\Users\Lenovo\Downloads\95ef471035ee0b5e4e5b1cac3d3584c1.pdf",
    )
)

DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class Settings:
    """Settings configuration for ConstitutionGPT."""

    app_name: str = "ConstitutionGPT — Indian Constitution Assistant"
    app_version: str = "1.0.0"
    app_subtitle: str = "Rule-based Constitutional Retrieval & Reasoning Engine (No LLM)"

    # Base Directories
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR

    # PDF Paths
    source_pdf_path: Path = DEFAULT_SOURCE_PDF
    local_pdf_path: Path = DATA_DIR / "constitution.pdf"

    # Ingestion Artifacts
    extracted_text_path: Path = DATA_DIR / "extracted_text.json"
    articles_path: Path = DATA_DIR / "articles.json"
    metadata_path: Path = DATA_DIR / "metadata.json"

    # RAG Datasets & Evaluation Paths
    rag_dir: Path = DATA_DIR / "rag"
    source_dir: Path = DATA_DIR / "source"
    processed_dir: Path = DATA_DIR / "processed"
    processed_chunks_path: Path = DATA_DIR / "processed" / "constitution_chunks.jsonl"
    ingestion_manifest_path: Path = DATA_DIR / "processed" / "ingestion_manifest.json"
    evaluation_dir: Path = DATA_DIR / "evaluation"
    golden_eval_dir: Path = DATA_DIR / "evaluation" / "golden"
    normal_eval_dir: Path = DATA_DIR / "evaluation" / "normal"
    indexes_dir: Path = DATA_DIR / "indexes"
    reports_dir: Path = PROJECT_ROOT / "reports"
    normal_retrieval_dataset_path: Path = DATA_DIR / "rag" / "normal_retrieval_dataset.jsonl"
    golden_evaluation_dataset_path: Path = (
        PROJECT_ROOT / "tests" / "evaluation" / "golden_reranking_dataset.jsonl"
    )
    evaluation_reports_dir: Path = PROJECT_ROOT / "reports" / "rag"

    # Persistence Paths
    checkpoints_db_path: Path = DATA_DIR / "checkpoints.db"
    conversations_db_path: Path = DATA_DIR / "conversations.db"

    # Retrieval Tuning
    min_query_length: int = 2
    max_search_results: int = 5
    summary_char_limit: int = 1500

    # Gemini AI Configuration
    gemini_api_key: str = field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model: str = "gemini-flash-latest"

    # Mem0 Intelligent Memory Configuration
    mem0_api_key: str = field(
        default_factory=lambda: os.getenv("MEM0_API_KEY", "")
    )
    mem0_db_path: Path = DATA_DIR / "mem0_memories.db"
    mem0_default_user_id: str = field(
        default_factory=lambda: os.getenv("MEM0_DEFAULT_USER_ID", "advocate_guest")
    )

    # Auth0 Authentication Configuration
    auth0_domain: str = field(
        default_factory=lambda: os.getenv("AUTH0_DOMAIN", "")
    )
    auth0_client_id: str = field(
        default_factory=lambda: os.getenv("AUTH0_CLIENT_ID", "")
    )
    auth0_client_secret: str = field(
        default_factory=lambda: os.getenv("AUTH0_CLIENT_SECRET", "")
    )
    auth0_callback_url: str = field(
        default_factory=lambda: os.getenv("AUTH0_CALLBACK_URL", "http://localhost:8501")
    )
    auth0_audience: str = field(
        default_factory=lambda: os.getenv("AUTH0_AUDIENCE", "")
    )

    def ensure_directories(self) -> None:
        """Ensure all required project directories exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.rag_dir.mkdir(parents=True, exist_ok=True)
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.golden_eval_dir.mkdir(parents=True, exist_ok=True)
        self.normal_eval_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.golden_evaluation_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.evaluation_reports_dir.mkdir(parents=True, exist_ok=True)


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Retrieve singleton settings instance."""
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
        _settings_instance.ensure_directories()
    return _settings_instance
