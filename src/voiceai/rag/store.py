"""Vector store.

An in-memory exact-cosine store, plus the seam a FAISS/Qdrant backend plugs
into. Exact search is the right default at this scale for a reason worth
stating: an ANN index trades recall for speed, and at a few thousand chunks the
speed is not needed while the lost recall is invisible until it costs you an
answer. Exact search is also the *oracle* any approximate backend can be tested
against — same queries, same ordering, or the backend is wrong.

Latency note: this store is queried inside the voice response budget. Brute
force over N vectors is O(N·d); at 384 dimensions the crossover where an index
starts paying for itself is in the tens of thousands of chunks. Measured in
``voiceai.bench_pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage plus the provenance needed to cite it."""

    id: str
    text: str
    source: str
    section: str = ""

    def citation(self) -> str:
        return f"{self.source}, {self.section}" if self.section else self.source


@dataclass(frozen=True, slots=True)
class Hit:
    chunk: Chunk
    score: float


class VectorStore(Protocol):
    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None: ...

    def search(self, query: Sequence[float], k: int) -> list[Hit]: ...

    def __len__(self) -> int: ...


@dataclass(slots=True)
class InMemoryStore:
    """Exact cosine similarity over unit vectors.

    Vectors are assumed normalised by the embedder, so cosine reduces to a dot
    product. That assumption is asserted on insert rather than trusted: an
    unnormalised vector silently inflates its own score against everything and
    dominates every result list, which looks like a relevance bug and is
    actually an arithmetic one.
    """

    _chunks: list[Chunk] = field(default_factory=list)
    _vectors: list[list[float]] = field(default_factory=list)

    def add(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        for chunk, vec in zip(chunks, vectors):
            norm_sq = sum(v * v for v in vec)
            if norm_sq > 0 and abs(norm_sq - 1.0) > 1e-3:
                raise ValueError(
                    f"vector for {chunk.id!r} is not unit length "
                    f"(norm^2={norm_sq:.4f}); normalise in the embedder"
                )
            self._chunks.append(chunk)
            self._vectors.append(list(vec))

    def search(self, query: Sequence[float], k: int) -> list[Hit]:
        if not self._vectors:
            return []
        if all(v == 0.0 for v in query):
            # An empty query has no direction. Returning the first k rows would
            # look like a working search and be pure noise.
            return []
        scored: list[Hit] = []
        for chunk, vec in zip(self._chunks, self._vectors):
            scored.append(Hit(chunk=chunk, score=sum(a * b for a, b in zip(query, vec))))
        scored.sort(key=lambda h: h.score, reverse=True)
        return scored[:k]

    def __len__(self) -> int:
        return len(self._chunks)

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
