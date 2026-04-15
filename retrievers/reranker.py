from __future__ import annotations

from typing import List

from langchain_core.documents import Document

from rag.config import settings
from utils.llm_factory import get_chat_model


class DocumentReranker:
    def rerank(self, query: str, docs: List[Document]) -> List[Document]:
        if not docs:
            return []
        mode = settings.reranker_type.lower()
        if mode == "none":
            return docs[: settings.rerank_top_n]
        if mode == "llm":
            return self._llm_rerank(query, docs)
        return self._cross_encoder_rerank(query, docs)

    def _cross_encoder_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        try:
            from sentence_transformers import CrossEncoder

            model = CrossEncoder(settings.reranker_model)
            pairs = [(query, d.page_content) for d in docs]
            scores = model.predict(pairs)
            ranked_idx = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
            return [docs[i] for i in ranked_idx[: settings.rerank_top_n]]
        except Exception:
            # Degrade gracefully if model is unavailable in the environment.
            return docs[: settings.rerank_top_n]

    def _llm_rerank(self, query: str, docs: List[Document]) -> List[Document]:
        llm = get_chat_model()
        scored = []
        for d in docs:
            prompt = (
                "Score relevance from 0 to 100.\n"
                f"Query: {query}\n"
                f"Document: {d.page_content[:1200]}\n"
                "Return only the integer score."
            )
            try:
                raw = llm.invoke(prompt).content.strip()
                score = int("".join([c for c in raw if c.isdigit()]) or "0")
            except Exception:
                score = 0
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[: settings.rerank_top_n]]
