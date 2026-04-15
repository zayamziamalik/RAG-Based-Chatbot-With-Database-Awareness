from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_core.documents import Document

from rag.config import settings
from rag.loaders import load_all_documents
from utils.llm_factory import get_embeddings


class HybridRetriever:
    """Hybrid retriever combining vector search and BM25 keyword retrieval."""

    def __init__(self) -> None:
        self.documents: List[Document] = []
        self.keyword_retriever = None
        self.vector_store = None
        self._initialized = False

    def refresh(self) -> None:
        self.documents = load_all_documents()
        if not self.documents:
            self.keyword_retriever = None
            self.vector_store = None
            self._initialized = True
            return

        self.keyword_retriever = BM25Retriever.from_documents(self.documents)
        self.keyword_retriever.k = settings.keyword_top_k

        embeddings = get_embeddings()
        store_type = settings.vector_store.lower()
        vector_dir = Path(settings.vector_store_dir)
        vector_dir.mkdir(parents=True, exist_ok=True)

        if store_type == "chroma":
            self.vector_store = Chroma(
                collection_name="rag_docs",
                embedding_function=embeddings,
                persist_directory=str(vector_dir / "chroma"),
            )
            self.vector_store.reset_collection()
            self.vector_store.add_documents(self.documents)
        elif store_type == "faiss":
            self.vector_store = FAISS.from_documents(self.documents, embeddings)
            self.vector_store.save_local(str(vector_dir / "faiss_index"))
        else:
            raise ValueError(f"Unsupported vector store: {settings.vector_store}")
        self._initialized = True

    def _ensure_initialized(self) -> None:
        if not self._initialized:
            self.refresh()

    def _vector_search(self, query: str) -> List[Document]:
        if self.vector_store is None:
            return []
        if settings.vector_store.lower() == "chroma":
            return self.vector_store.similarity_search(query, k=settings.vector_top_k)
        return self.vector_store.similarity_search(query, k=settings.vector_top_k)

    def retrieve(self, queries: List[str]) -> List[Document]:
        self._ensure_initialized()
        if not self.documents:
            return []

        scored: Dict[str, Dict[str, object]] = defaultdict(
            lambda: {"score": 0.0, "doc": None}
        )
        for q in queries:
            for rank, doc in enumerate(self._vector_search(q), start=1):
                key = f"{doc.metadata.get('source_name','?')}::{hash(doc.page_content)}"
                scored[key]["doc"] = doc
                scored[key]["score"] = float(scored[key]["score"]) + 1.0 / rank

            if self.keyword_retriever is not None:
                for rank, doc in enumerate(self.keyword_retriever.invoke(q), start=1):
                    key = f"{doc.metadata.get('source_name','?')}::{hash(doc.page_content)}"
                    scored[key]["doc"] = doc
                    scored[key]["score"] = float(scored[key]["score"]) + 1.0 / rank

        ranked = sorted(
            [x for x in scored.values() if x["doc"] is not None],
            key=lambda x: float(x["score"]),
            reverse=True,
        )
        return [x["doc"] for x in ranked[: settings.top_k * 3]]
