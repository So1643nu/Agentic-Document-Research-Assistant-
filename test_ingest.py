import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import ingest_directory, split_text

SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "data" / "sample_docs"


def test_split_text_short_returns_single_chunk():
    text = "This is a short piece of text."
    chunks = split_text(text, chunk_size=800, chunk_overlap=100)
    assert chunks == [text]


def test_split_text_empty_returns_empty_list():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_split_text_long_produces_multiple_chunks():
    paragraph = "Sentence number {}. " * 1
    text = "\n\n".join(f"Paragraph {i}. " + ("word " * 50) for i in range(20))
    chunks = split_text(text, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    for c in chunks:
        # allow slight overrun due to overlap prefix
        assert len(c) <= 300 + 60


def test_ingest_directory_reads_sample_docs():
    chunks = ingest_directory(SAMPLE_DOCS_DIR, chunk_size=500, chunk_overlap=80)
    assert len(chunks) > 0
    sources = {c.source for c in chunks}
    assert "onboarding_guide.txt" in sources
    assert "pricing_sheet.txt" in sources
    for c in chunks:
        assert c.text.strip() != ""
        assert c.id.startswith(c.source)
