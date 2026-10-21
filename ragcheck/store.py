"""An in-memory vector store, deliberately small.

Exact cosine search over every chunk. No index, no approximation, no
dependency. On corpora of the size people actually evaluate against, exact
search is fast enough and removes a variable: when recall is bad you know it
is the chunking or the embedder, not an index parameter you forgot about.

Swap this for pgvector or Qdrant behind the same two methods once you are
past evaluation and into production.
"""

from __future__ import annotations

from collections.abc import Sequence

from .chunk import Chunk
from .embed import Embedder, cosine


class VectorStore:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def add(self, chunks: Sequence[Chunk]) -> None:
        if not chunks:
            return
        vectors = self.embedder.embed([c.text for c in chunks])
        if len(vectors) != len(chunks):
            raise ValueError(
                f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks"
            )
        self.chunks.extend(chunks)
        self.vectors.extend(vectors)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Top k by cosine. Ties break on chunk id so results are reproducible
        rather than dependent on insertion order."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        if not self.chunks:
            return []
        qvec = self.embedder.embed([query])[0]
        scored = [
            (chunk, cosine(qvec, vec))
            for chunk, vec in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:k]

    def __len__(self) -> int:
        return len(self.chunks)
