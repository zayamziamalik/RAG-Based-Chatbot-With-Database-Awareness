from __future__ import annotations

import json
from pathlib import Path
import re
from typing import List

from langchain_core.documents import Document
from sqlalchemy import create_engine, inspect, text

from app.chunking import chunk_text
from rag.config import settings
from rag.schemas import SourceChunk


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


def _normalize(content: str) -> str:
    return re.sub(r"\s+", " ", content or "").strip()


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def _read_docx(path: Path) -> str:
    try:
        from docx import Document as DocxDocument

        doc = DocxDocument(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def _read_path(path: Path) -> str:
    suffix = path.suffix.lower()
    try:
        if suffix in {".txt", ".md", ".csv", ".log", ".html", ".htm", ".xml", ".yaml", ".yml"}:
            return path.read_text(encoding="utf-8", errors="ignore")
        if suffix == ".json":
            raw = path.read_text(encoding="utf-8", errors="ignore")
            try:
                parsed = json.loads(raw)
                return json.dumps(parsed, ensure_ascii=True, indent=2)
            except Exception:
                return raw
        if suffix == ".pdf":
            return _read_pdf(path)
        if suffix == ".docx":
            return _read_docx(path)
    except Exception:
        return ""
    return ""


def load_file_documents() -> List[Document]:
    base = Path(settings.text_files_dir)
    base.mkdir(parents=True, exist_ok=True)
    docs: List[Document] = []

    for path in sorted(base.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        content = _normalize(_read_path(path))
        if not content:
            continue
        for i, chunk in enumerate(
            chunk_text(content, chunk_size=settings.max_chunk_size, overlap=settings.chunk_overlap)
        ):
            source = SourceChunk(
                content=chunk,
                source_type="file",
                source_name=str(path),
                metadata={"chunk_index": i},
            )
            docs.append(source.to_document())
    return docs


def load_database_documents() -> List[Document]:
    docs: List[Document] = []
    try:
        engine = create_engine(settings.database_url)
        inspector = inspect(engine)
        tables = inspector.get_table_names()
    except Exception:
        return docs

    for table in tables:
        try:
            with engine.connect() as conn:
                rows = conn.execute(text(f"SELECT * FROM {table}")).mappings().all()
        except Exception:
            continue
        for row_idx, row in enumerate(rows):
            content = f"Table: {table}. " + " | ".join([f"{k}={v}" for k, v in row.items()])
            for chunk_idx, chunk in enumerate(
                chunk_text(content, chunk_size=settings.max_chunk_size, overlap=settings.chunk_overlap)
            ):
                source = SourceChunk(
                    content=chunk,
                    source_type="database",
                    source_name=table,
                    metadata={"row_index": row_idx, "chunk_index": chunk_idx},
                )
                docs.append(source.to_document())
    return docs


def load_all_documents() -> List[Document]:
    return load_file_documents() + load_database_documents()
