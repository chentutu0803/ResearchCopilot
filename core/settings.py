from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["Settings", "get_settings"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="ResearchCopilot", alias="APP_NAME")
    app_env: str = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    report_language: str = Field(default="中文", alias="REPORT_LANGUAGE")

    deepseek_api_key: str = Field(default="", alias="DEEPSEEK_API_KEY")
    model_name: str = Field(default="deepseek-chat", alias="MODEL_NAME")
    deepseek_base_url: str = Field(
        default="", alias="DEEPSEEK_BASE_URL"
    )
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    max_subquestions: int = Field(default=0, alias="MAX_SUBQUESTIONS")
    max_search_rounds: int = Field(default=5, alias="MAX_SEARCH_ROUNDS")
    max_review_rounds: int = Field(default=2, alias="MAX_REVIEW_ROUNDS")
    max_replan_cycles: int = Field(default=1, alias="MAX_REPLAN_CYCLES")
    top_k_bm25: int = Field(default=8, alias="TOP_K_BM25")
    top_k_dense: int = Field(default=8, alias="TOP_K_DENSE")
    top_k_rerank: int = Field(default=5, alias="TOP_K_RERANK")
    kb_backend: str = Field(default="qdrant", alias="KB_BACKEND")
    qdrant_path: str = Field(default="./data/qdrant", alias="QDRANT_PATH")
    qdrant_collection: str = Field(
        default="researchcopilot_kb", alias="QDRANT_COLLECTION"
    )
    qdrant_url: str = Field(default="", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_cloud_inference: bool = Field(
        default=False, alias="QDRANT_CLOUD_INFERENCE"
    )
    qdrant_bm25_collection: str = Field(
        default="researchcopilot_kb_bm25", alias="QDRANT_BM25_COLLECTION"
    )
    qdrant_bm25_vector_name: str = Field(
        default="text-bm25", alias="QDRANT_BM25_VECTOR_NAME"
    )
    qdrant_bm25_tokenizer: str = Field(
        default="multilingual", alias="QDRANT_BM25_TOKENIZER"
    )
    qdrant_bm25_language: str = Field(
        default="", alias="QDRANT_BM25_LANGUAGE"
    )
    qdrant_bm25_ascii_folding: bool = Field(
        default=False, alias="QDRANT_BM25_ASCII_FOLDING"
    )
    embedding_backend: str = Field(
        default="sentence_transformer", alias="EMBEDDING_BACKEND"
    )
    embedding_model_name: str = Field(
        default="BAAI/bge-small-zh-v1.5", alias="EMBEDDING_MODEL_NAME"
    )
    hf_endpoint: str = Field(default="https://hf-mirror.com", alias="HF_ENDPOINT")
    llm_timeout_seconds: float = Field(default=60.0, alias="LLM_TIMEOUT_SECONDS")
    llm_max_retries: int = Field(default=2, alias="LLM_MAX_RETRIES")
    web_search_timeout_seconds: float = Field(
        default=10.0, alias="WEB_SEARCH_TIMEOUT_SECONDS"
    )
    web_search_retries: int = Field(default=2, alias="WEB_SEARCH_RETRIES")
    web_max_results: int = Field(default=5, alias="WEB_MAX_RESULTS")
    web_fetch_pages: int = Field(default=1, alias="WEB_FETCH_PAGES")
    playwright_module_path: str = Field(
        default="/usr/lib/node_modules/playwright",
        alias="PLAYWRIGHT_MODULE_PATH",
    )
    playwright_executable_path: str = Field(
        default="/root/.cache/ms-playwright/chromium-1208/chrome-linux64/chrome",
        alias="PLAYWRIGHT_EXECUTABLE_PATH",
    )
    playwright_cdp_endpoint: str = Field(
        default="http://127.0.0.1:9222",
        alias="PLAYWRIGHT_CDP_ENDPOINT",
    )
    playwright_headless: bool = Field(default=False, alias="PLAYWRIGHT_HEADLESS")
    browser_user_data_dir: str = Field(
        default="./data/browser-profile",
        alias="BROWSER_USER_DATA_DIR",
    )
    browser_remote_debug_port: int = Field(
        default=9222,
        alias="BROWSER_REMOTE_DEBUG_PORT",
    )

    trace_save_dir: str = Field(default="./experiments/results", alias="TRACE_SAVE_DIR")
    data_dir: str = Field(default="./data", alias="DATA_DIR")

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parent.parent

    @property
    def trace_path(self) -> Path:
        return (self.project_root / self.trace_save_dir).resolve()

    @property
    def data_path(self) -> Path:
        return (self.project_root / self.data_dir).resolve()

    @property
    def qdrant_path_resolved(self) -> Path:
        raw_path = Path(self.qdrant_path)
        if raw_path.is_absolute():
            return raw_path
        return (self.project_root / raw_path).resolve()

    @property
    def browser_user_data_path_resolved(self) -> Path:
        raw_path = Path(self.browser_user_data_dir)
        if raw_path.is_absolute():
            return raw_path
        return (self.project_root / raw_path).resolve()

    @property
    def has_llm_credentials(self) -> bool:
        return bool(self.deepseek_api_key.strip())

    @property
    def use_qdrant_official_bm25(self) -> bool:
        return bool(
            self.qdrant_url.strip()
            and self.qdrant_api_key.strip()
            and self.qdrant_cloud_inference
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
