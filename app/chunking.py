from __future__ import annotations

from typing import List


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    clean = " ".join(text.split())
    if not clean:
        return []
    if len(clean) <= chunk_size:
        return [clean]

    chunks: List[str] = []
    start = 0
    step = max(1, chunk_size - overlap)
    while start < len(clean):
        end = start + chunk_size
        chunks.append(clean[start:end])
        start += step
    return chunks
