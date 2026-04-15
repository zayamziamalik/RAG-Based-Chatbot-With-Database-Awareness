from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.documents import Document


@dataclass
class SourceChunk:
    content: str
    source_type: str
    source_name: str
    metadata: Dict[str, Any]

    def to_document(self) -> Document:
        meta = dict(self.metadata)
        meta["source_type"] = self.source_type
        meta["source_name"] = self.source_name
        return Document(page_content=self.content, metadata=meta)


@dataclass
class QueryLog:
    user_query: str
    rewritten_queries: List[str]
    retrieved_count: int
    reranked_count: int
    answer: str
