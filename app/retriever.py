from __future__ import annotations

from dataclasses import dataclass
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.loaders import DocumentChunk, load_all_chunks


@dataclass
class RetrievedContext:
    chunk: DocumentChunk
    score: float


class HybridRetriever:
    def __init__(self) -> None:
        self.chunks: List[DocumentChunk] = []
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = None

    def refresh(self) -> None:
        self.chunks = load_all_chunks()
        if not self.chunks:
            self.doc_matrix = None
            return
        corpus = [c.content for c in self.chunks]
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int | None = None) -> List[RetrievedContext]:
        if top_k is None:
            top_k = settings.top_k

        if not self.chunks or self.doc_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.doc_matrix)[0]
        ranked_ids = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        best_ids = ranked_ids[:top_k]
        return [
            RetrievedContext(chunk=self.chunks[i], score=float(scores[i])) for i in best_ids
        ]
