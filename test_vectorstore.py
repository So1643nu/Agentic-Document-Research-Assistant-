import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ingest import ingest_directory
from src.vectorstore import TfidfStore, build_store

SAMPLE_DOCS_DIR = Path(__file__).parent.parent / "data" / "sample_docs"


def test_empty_store_search_returns_empty():
    store = TfidfStore()
    assert store.is_empty()
    assert store.search("anything") == []


def test_build_store_default_is_tfidf():
    store = build_store(use_embeddings=False)
    assert isinstance(store, TfidfStore)


def test_search_returns_relevant_results():
    store = TfidfStore()
    chunks = ingest_directory(SAMPLE_DOCS_DIR, chunk_size=500, chunk_overlap=80)
    store.add_documents(chunks)
    assert not store.is_empty()

    results = store.search("probation period 90 days", k=3)
    assert len(results) > 0
    assert any("onboarding_guide.txt" == r.source for r in results)


def test_search_pricing_query_hits_pricing_doc():
    store = TfidfStore()
    chunks = ingest_directory(SAMPLE_DOCS_DIR, chunk_size=500, chunk_overlap=80)
    store.add_documents(chunks)

    results = store.search("Pro tier pricing per seat", k=3)
    assert len(results) > 0
    assert any("pricing_sheet.txt" == r.source for r in results)
