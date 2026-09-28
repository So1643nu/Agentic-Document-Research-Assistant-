"""Load documents (.txt, .md, .pdf) and split them into overlapping chunks.

The chunker is a simple recursive character splitter: it tries to break on
paragraph boundaries first, then sentences, then falls back to a hard
character cut. That's not as clever as a token-aware splitter, but it's
dependency-free, predictable, and good enough for the corpus sizes this
project targets.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    id: str
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            # Some PDFs have malformed pages; skip rather than fail the
            # whole ingest run.
            continue
    return "\n".join(pages)


def load_document_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return _read_txt(path)
    if ext == ".pdf":
        return _read_pdf(path)
    raise ValueError(f"Unsupported file type: {ext}")


def split_text(
    text: str,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[str]:
    """Recursive-ish splitter: paragraphs -> sentences -> hard cut."""
    text = re.sub(r"\r\n", "\n", text).strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: List[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # Paragraph itself too big -> split on sentences.
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if len(current) + len(sent) + 1 <= chunk_size:
                    current += (" " if current else "") + sent
                else:
                    flush()
                    if len(sent) > chunk_size:
                        # Even a single sentence is too big -> hard cut.
                        for i in range(0, len(sent), chunk_size - chunk_overlap):
                            chunks.append(sent[i : i + chunk_size].strip())
                        current = ""
                    else:
                        current = sent
            continue

        if len(current) + len(para) + 2 <= chunk_size:
            current += ("\n\n" if current else "") + para
        else:
            flush()
            current = para

    flush()

    # Apply a light overlap by prepending the tail of the previous chunk.
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append((tail + " " + chunks[i]).strip())
        chunks = overlapped

    return chunks


def ingest_directory(
    directory: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Chunk]:
    directory = Path(directory)
    chunks: List[Chunk] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text = load_document_text(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[ingest] skipping {path}: {exc}")
            continue
        pieces = split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for idx, piece in enumerate(pieces):
            chunks.append(
                Chunk(
                    id=f"{path.name}::{idx}",
                    text=piece,
                    source=path.name,
                    metadata={"chunk_index": idx, "path": str(path)},
                )
            )
    return chunks


def ingest_uploaded_files(
    files: Iterable[tuple[str, bytes]],
    tmp_dir: str | Path,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Chunk]:
    """Write (filename, bytes) pairs to tmp_dir then ingest that directory.

    Used by the Streamlit UI for uploaded files, which arrive as in-memory
    bytes rather than paths on disk.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    for filename, data in files:
        (tmp_dir / filename).write_bytes(data)
    return ingest_directory(tmp_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
