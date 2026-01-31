from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Keys
    openai_api_key: str = Field(..., description="OpenAI API key for embeddings")

    # Paths
    project_root: Path = Field(
        default_factory=lambda: Path(__file__).parent.parent,
    )
    dataset_root: Path | None = Field(default=None)

    @field_validator("dataset_root", mode="before")
    @classmethod
    def detect_dataset_root(cls, v, info):
        if v is not None:
            return Path(v)
        project_root = info.data.get("project_root", Path(__file__).parent.parent)
        local = project_root / "Dataset"
        if local.exists():
            return local
        sibling = project_root.parent / "test-tecnico-ai-engineer" / "Dataset"
        if sibling.exists():
            return sibling
        return local

    # Vector Store
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333)
    qdrant_collection_name: str = Field(default="menu_dishes")

    # Embedding
    embedding_model: str = Field(default="text-embedding-3-small")
    embedding_dimensions: int = Field(default=1536)
    embedding_name: str = Field(default="openai")
    embedding_batch_size: int = Field(default=50, ge=1, le=2048)

    # LLM
    llm_model: str = Field(default="gpt-4o-mini")
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    # Retrieval
    retrieval_top_k: int = Field(default=20, ge=1, le=100)

    # Output
    output_dir: Path | None = Field(default=None)

    @field_validator("output_dir", mode="before")
    @classmethod
    def set_output_dir(cls, v, info):
        if v is not None:
            path = Path(v)
        else:
            project_root = info.data.get("project_root", Path(__file__).parent.parent)
            path = project_root / "outputs"
        path.mkdir(exist_ok=True, parents=True)
        return path

    # Computed paths
    @property
    def menu_dir(self) -> Path:
        return self.dataset_root / "knowledge_base" / "menu"

    @property
    def misc_dir(self) -> Path:
        return self.dataset_root / "knowledge_base" / "misc"

    @property
    def codice_galattico_dir(self) -> Path:
        return self.dataset_root / "knowledge_base" / "codice_galattico"

    @property
    def blogpost_dir(self) -> Path:
        return self.dataset_root / "knowledge_base" / "blogpost"

    @property
    def dish_mapping_path(self) -> Path:
        return self.dataset_root / "ground_truth" / "dish_mapping.json"

    @property
    def ground_truth_csv_path(self) -> Path:
        return self.dataset_root / "ground_truth" / "ground_truth_mapped.csv"

    @property
    def questions_csv_path(self) -> Path:
        return self.dataset_root / "domande.csv"

    @property
    def distances_csv_path(self) -> Path:
        return self.misc_dir / "Distanze.csv"

    @property
    def manuale_path(self) -> Path:
        return self.misc_dir / "Manuale di Cucina.pdf"


settings = Settings()
