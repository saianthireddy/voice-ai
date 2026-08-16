"""Retrieval and the relevance floor.

Two things happen here. Retrieval finds candidates; the *floor* decides whether
any of them are good enough to speak from.

The floor matters more in voice than in chat. A text assistant that answers
weakly gives the user a paragraph they can skim and dismiss in a second. A voice
assistant that answers weakly spends fifteen seconds saying it out loud while
the user waits, and the only way to stop it is to interrupt. Wrong answers cost
more when they are spoken, so the gate is tuned to refuse more readily than a
text system would.

Scoring is hybrid: dense cosine fused with lexical coverage. Dense alone smears
identifiers — a part number or an error code becomes "something technical" —
which is exactly the query type people speak to a support agent. Lexical alone
misses paraphrase, which is exactly how people speak when they do not know the
jargon. Neither is optional.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .embeddings import Embedder, stem, tokenize
from .store import Chunk, Hit, VectorStore

# Words that appear in every question and so carry no topical signal. Counting
# them in coverage is how a refusal gate quietly stops working: "what is the X
# of Y" overlaps with any passage on earth via what/is/the/of.
#
# The list is stemmed to match, and that is not cosmetic. ``tokenize`` stems, so
# an unstemmed list silently stops matching the moment stemming is introduced:
# "does" becomes "doe", fails the stopword check, and is counted as a topical
# word. Coverage then divides by a larger denominator and every score drifts
# down — a refusal gate that grew stricter for no stated reason.
_STOPWORDS = frozenset(
    stem(w)
    for w in """a an and are as at be by can do does for from has have how i in is it of
    on or that the this to was what when where which who why will with you your
    me my our we they them their""".split()
)


def content_words(text: str) -> set[str]:
    """Topical words only. ``tokenize`` already stems, so both retrieval paths
    agree on what a word is."""
    return {t for t in tokenize(text) if t not in _STOPWORDS and len(t) > 1}


def lexical_coverage(query: str, passage: str) -> float:
    """Fraction of the query's content words present in the passage."""
    q = content_words(query)
    if not q:
        return 0.0
    p = content_words(passage)
    return len(q & p) / len(q)


@dataclass(frozen=True, slots=True)
class Retrieval:
    hits: list[Hit]
    top_score: float
    grounded: bool
    reason: str = ""

    @property
    def context(self) -> str:
        return "\n\n".join(f"[{h.chunk.citation()}] {h.chunk.text}" for h in self.hits)

    @property
    def citations(self) -> list[str]:
        return [h.chunk.citation() for h in self.hits]


class Retriever:
    """Hybrid retrieval with an explicit, tunable relevance floor.

    ``min_relevance`` is the single number that decides answer-vs-refuse, and it
    comes from the sweep in ``voiceai.eval``, not from taste. The first guess
    was 0.25; the labelled set showed that cost 25% recall (9/12 instead of
    11/12) while preventing no false answers at all — a stricter gate that
    bought nothing and lost real answers. 0.20 is the highest floor that still
    reaches peak recall with zero false answers on that set.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        *,
        k: int = 3,
        candidate_k: int = 10,
        min_relevance: float = 0.20,
        dense_weight: float = 0.5,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._k = k
        self._candidate_k = candidate_k
        self._floor = min_relevance
        self._w = dense_weight

    def index(self, chunks: Sequence[Chunk]) -> int:
        vectors = self._embedder.embed([c.text for c in chunks])
        self._store.add(chunks, vectors)
        return len(chunks)

    def retrieve(self, query: str) -> Retrieval:
        if not query.strip():
            return Retrieval([], 0.0, False, "empty query")

        qvec = self._embedder.embed([query])[0]
        candidates = self._store.search(qvec, self._candidate_k)
        if not candidates:
            return Retrieval([], 0.0, False, "nothing indexed")

        # Fuse dense similarity with lexical coverage. Coverage is computed on
        # content words only; see _STOPWORDS above for why.
        fused: list[Hit] = []
        for hit in candidates:
            cov = lexical_coverage(query, hit.chunk.text)
            # If no topical word matches at all, the passage is not about this
            # question regardless of what cosine says. Hard zero, not a small
            # number, so it cannot creep over the floor by accumulation.
            score = 0.0 if cov == 0.0 else self._w * hit.score + (1 - self._w) * cov
            fused.append(Hit(chunk=hit.chunk, score=score))

        fused.sort(key=lambda h: h.score, reverse=True)
        top = fused[0].score if fused else 0.0
        if top < self._floor:
            return Retrieval([], top, False, f"top score {top:.2f} below floor {self._floor:.2f}")

        # Only passages that actually cleared the floor are returned. Taking a
        # flat top-k would attach citations to passages that contributed
        # nothing — the answer would still be right, and the sources beneath it
        # would be a lie. A citation nobody checks is exactly the kind of claim
        # this project refuses to make.
        kept = [h for h in fused[: self._k] if h.score >= self._floor]
        return Retrieval(kept, top, True)

    @property
    def size(self) -> int:
        return len(self._store)

    @property
    def floor(self) -> float:
        return self._floor
