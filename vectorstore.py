"""Retrieval backends.

Two implementations behind one small interface:

- TfidfStore: scikit-learn TF-IDF + cosine similarity. No downloads,
  fully offline, deterministic. This is the default.
- EmbeddingStore: sentence-transformers + FAISS. Better semantic recall,
  but requires downloading an embedding model on first use. Opt-in via
  Settings.use_embeddings.

Both expose the same methods (add_documents / search) so the agent and UI
code never need to know which one is active.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .ingest import Chunk


@dataclass
class SearchResult:
    text: str
    source: str
    score: float
    metadata: dict


class VectorStore(Protocol):
    def add_documents(self, chunks: List[Chunk]) -> None: ...
    def search(self, query: str, k: int = 4) -> List[SearchResult]: ...
    def is_empty(self) -> bool: ...


class TfidfStore:
    """Default, dependency-light retrieval backend."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._vectorizer: TfidfVectorizer | None = None
        self._matrix = None

    def add_documents(self, chunks: List[Chunk]) -> None:
        self._chunks.extend(chunks)
        corpus = [c.text for c in self._chunks]
        if not corpus:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=1,
        )
        self._matrix = self._vectorizer.fit_transform(corpus)

    def is_empty(self) -> bool:
        return not self._chunks

    def search(self, query: str, k: int = 4) -> List[SearchResult]:
        if self.is_empty() or self._vectorizer is None:
            return []
        query_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self._matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:k]
        results = []
        for idx in top_idx:
            score = float(sims[idx])
            if score <= 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                SearchResult(
                    text=chunk.text,
                    source=chunk.source,
                    score=score,
                    metadata=chunk.metadata,
                )
            )
        return results


class EmbeddingStore:
    """Optional semantic backend (sentence-transformers + FAISS).

    Imports are done lazily inside __init__ so that simply having
    USE_EMBEDDINGS=false never requires these optional packages to be
    installed.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        try:
            import faiss  # noqa: F401
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional path
            raise ImportError(
                "USE_EMBEDDINGS=true requires 'sentence-transformers' and "
                "'faiss-cpu'. Install them or set USE_EMBEDDINGS=false."
            ) from exc

        self._faiss = __import__("faiss")
        self._model = SentenceTransformer(model_name)
        self._index = None
        self._chunks: List[Chunk] = []

    def add_documents(self, chunks: List[Chunk]) -> None:
        if not chunks:
            return
        self._chunks.extend(chunks)
        embeddings = self._model.encode(
            [c.text for c in self._chunks], normalize_embeddings=True
        )
        dim = embeddings.shape[1]
        self._index = self._faiss.IndexFlatIP(dim)
        self._index.add(np.array(embeddings, dtype="float32"))

    def is_empty(self) -> bool:
        return not self._chunks

    def search(self, query: str, k: int = 4) -> List[SearchResult]:
        if self.is_empty() or self._index is None:
            return []
        query_emb = self._model.encode([query], normalize_embeddings=True)
        scores, indices = self._index.search(np.array(query_emb, dtype="float32"), k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self._chunks[idx]
            results.append(
                SearchResult(
                    text=chunk.text,
                    source=chunk.source,
                    score=float(score),
                    metadata=chunk.metadata,
                )
            )
        return results


def build_store(use_embeddings: bool = False) -> VectorStore:
    if use_embeddings:
        return EmbeddingStore()
    return TfidfStore()
