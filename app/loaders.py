from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, List

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine

from app.chunking import chunk_text
from app.config import settings


@dataclass
class DocumentChunk:
    content: str
    source_type: str
    source_name: str
    metadata: Dict[str, Any]


SUPPORTED_EXTENSIONS = {
    ".txt",
    ".md",
    ".csv",
    ".log",
    ".pdf",
    ".docx",
    ".json",
    ".html",
    ".htm",
    ".xml",
    ".yaml",
    ".yml",
}


def _normalize_text(content: str) -> str:
    return re.sub(r"\s+", " ", content or "").strip()


def _read_plain_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json(path: Path) -> str:
    text_content = _read_plain_text(path)
    try:
        data = json.loads(text_content)
        return json.dumps(data, ensure_ascii=True, indent=2)
    except Exception:
        return text_content


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""

    try:
        reader = PdfReader(str(path))
        pages = [(page.extract_text() or "") for page in reader.pages]
        return "\n".join(pages)
    except Exception:
        return ""


def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except Exception:
        return ""

    try:
        document = Document(str(path))
        return "\n".join([p.text for p in document.paragraphs])
    except Exception:
        return ""


def _read_file_content(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".md", ".csv", ".log", ".html", ".htm", ".xml", ".yaml", ".yml"}:
        return _read_plain_text(path)
    if suffix == ".json":
        return _read_json(path)
    if suffix == ".pdf":
        return _read_pdf(path)
    if suffix == ".docx":
        return _read_docx(path)
    return ""


def load_text_file_chunks() -> List[DocumentChunk]:
    base_path = Path(settings.text_files_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    chunks: List[DocumentChunk] = []
    for path in sorted(base_path.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        content = _normalize_text(_read_file_content(path))
        if not content:
            continue
        for idx, piece in enumerate(
            chunk_text(content, settings.max_chunk_size, settings.chunk_overlap)
        ):
            chunks.append(
                DocumentChunk(
                    content=piece,
                    source_type="text_file",
                    source_name=str(path),
                    metadata={"chunk_index": idx},
                )
            )
    return chunks


def _safe_engine(database_url: str) -> Engine | None:
    try:
        return create_engine(database_url)
    except Exception:
        return None


def load_database_chunks() -> List[DocumentChunk]:
    engine = _safe_engine(settings.database_url)
    if engine is None:
        return []

    chunks: List[DocumentChunk] = []
    inspector = inspect(engine)
    table_names = inspector.get_table_names()

    for table_name in table_names:
        query = text(f"SELECT * FROM {table_name}")
        try:
            with engine.connect() as conn:
                rows = conn.execute(query).mappings().all()
        except Exception:
            continue

        for row_idx, row in enumerate(rows):
            row_text = " | ".join(
                [f"{col}={value}" for col, value in row.items()]
            )
            row_chunks = chunk_text(
                f"Table: {table_name}. Row: {row_text}",
                settings.max_chunk_size,
                settings.chunk_overlap,
            )
            for chunk_idx, piece in enumerate(row_chunks):
                chunks.append(
                    DocumentChunk(
                        content=piece,
                        source_type="database",
                        source_name=table_name,
                        metadata={"row_index": row_idx, "chunk_index": chunk_idx},
                    )
                )
    return chunks


def load_all_chunks() -> List[DocumentChunk]:
    return load_text_file_chunks() + load_database_chunks()
