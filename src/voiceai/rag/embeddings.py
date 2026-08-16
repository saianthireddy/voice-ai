"""Embeddings.

Two implementations behind one interface. The default is a hashing embedder:
deterministic, offline, no model download, ~microseconds per document. It is
**not semantic** — it sees token overlap and nothing else — and that limitation
is stated here rather than discovered later when retrieval quality disappoints.

Why ship a weak embedder as the default: in a voice pipeline the embedding call
sits inside the response-latency budget, and a project that cannot run without a
GPU cannot be benchmarked by whoever clones it. The hashing embedder makes the
whole system runnable and measurable anywhere; OpenAI slots in behind the same
interface when quality matters more than portability.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, Sequence

_TOKEN = re.compile(r"[a-z0-9]+")


def stem(token: str) -> str:
    """Crude suffix stripping. Shared by *both* retrieval paths, deliberately.

    Without it, "refund policy" fails against "Refunds are available" — the
    tokens are different strings, so lexical coverage scores low *and* the
    hashing embedder puts them in different buckets. The failure is silent:
    retrieval doesn't error, it declines, and the bug reads as "the corpus
    doesn't cover that".

    Stemming only one side is worse than stemming neither. If the embedder
    hashes ``refunds`` while coverage compares ``refund``, the two halves of the
    hybrid score disagree about what the words are, and the fusion quietly
    degrades in a way no single test would catch.

    Not a real stemmer — no Porter, no lemmatisation. It over-stems some words
    and under-stems others, applied identically everywhere so mismatches cancel.
    """
    for suffix in ("ies", "es", "s", "ing", "ed"):
        if len(token) > len(suffix) + 2 and token.endswith(suffix):
            base = token[: -len(suffix)]
            return base + "y" if suffix == "ies" else base
    return token


def tokenize(text: str) -> list[str]:
    return [stem(t) for t in _TOKEN.findall(text.lower())]


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def name(self) -> str: ...


class HashingEmbedder:
    """Bag-of-hashed-tokens projected onto a fixed-size unit vector.

    Deterministic across processes and machines: the hash is BLAKE2b with a
    fixed digest size, not Python's ``hash()``, which is salted per process and
    would make stored vectors meaningless after a restart. That bug is easy to
    ship and hard to notice — retrieval simply gets quietly worse.
    """

    __slots__ = ("_dim",)

    def __init__(self, dimensions: int = 384) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be positive")
        self._dim = dimensions

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "hashing"

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dim

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self._dim
            for tok in tokenize(text):
                vec[self._bucket(tok)] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            # A zero vector has undefined direction; cosine against it is
            # meaningless, so an empty document must not silently become a
            # valid-looking point at the origin.
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


class OpenAIEmbedder:
    """Hosted embeddings. Written, unverified — no API key in CI."""

    def __init__(self, model: str = "text-embedding-3-small", dimensions: int = 1536) -> None:
        self._model = model
        self._dim = dimensions

    @property
    def dimensions(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return "openai"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("openai is not installed. `pip install openai`.") from exc
        client = OpenAI()
        resp = client.embeddings.create(model=self._model, input=list(texts))
        return [d.embedding for d in resp.data]
