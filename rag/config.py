from dataclasses import dataclass
import os

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class RAGSettings:
    # model provider: openai | groq | ollama
    llm_provider: str = os.getenv("LLM_PROVIDER", "groq")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    llm_model: str = os.getenv("LLM_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    llm_temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))

    # embeddings provider: openai | huggingface
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "openai")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

    # vector store: chroma | faiss
    vector_store: str = os.getenv("VECTOR_STORE", "chroma")
    vector_store_dir: str = os.getenv("VECTOR_STORE_DIR", "data/vector_store")

    # data inputs (MySQL via SQLAlchemy; must match Django DATABASE_URL)
    database_url: str = os.getenv("DATABASE_URL", "mysql+pymysql://root:@127.0.0.1:3306/smartphone")
    text_files_dir: str = os.getenv("TEXT_FILES_DIR", "data/text_files")
    max_chunk_size: int = int(os.getenv("MAX_CHUNK_SIZE", "900"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))

    # retrieval and ranking
    top_k: int = int(os.getenv("TOP_K", "6"))
    keyword_top_k: int = int(os.getenv("KEYWORD_TOP_K", "8"))
    vector_top_k: int = int(os.getenv("VECTOR_TOP_K", "8"))
    rerank_top_n: int = int(os.getenv("RERANK_TOP_N", "6"))
    reranker_type: str = os.getenv("RERANKER_TYPE", "cross_encoder")  # cross_encoder | llm | none
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    multi_query_count: int = int(os.getenv("MULTI_QUERY_COUNT", "3"))

    # compression and memory
    max_context_chars: int = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "10"))

    # observability and behavior
    debug_mode: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    log_file: str = os.getenv("RAG_LOG_FILE", "data/logs/rag_pipeline.log")
    response_cache_ttl_seconds: int = int(os.getenv("RESPONSE_CACHE_TTL_SECONDS", "900"))
    enable_tools: bool = os.getenv("ENABLE_TOOLS", "true").lower() == "true"


settings = RAGSettings()
