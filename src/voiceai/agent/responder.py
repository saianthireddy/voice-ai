"""The agent: route → retrieve → generate, streaming tokens to TTS.

This is the component the voice session talks to, and it satisfies the same
``Responder`` protocol the scripted stub does — so the entire duplex/barge-in
test suite runs against the real RAG agent without modification.

Two decisions specific to *spoken* output:

**Citations are not read aloud.** "According to handbook.docx, Password reset
section" is unbearable in speech. Sources are attached to the turn as metadata
for the transcript and the API response; the spoken text stays clean. The
grounding is still verifiable, just not narrated.

**Refusal is short.** In text, a long apology is skimmable. In speech it is ten
seconds of the user waiting to talk. "I don't have that in my documents" and
stop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol

from ..rag.retriever import Retrieval, Retriever
from .router import Intent, route


class LLM(Protocol):
    def stream(self, question: str, context: str) -> AsyncIterator[str]: ...

    @property
    def name(self) -> str: ...


class ExtractiveLLM:
    """Selects sentences from the retrieved context. Cannot hallucinate.

    It also cannot paraphrase, reconcile two sources, or handle a question whose
    answer is spread across passages. That ceiling is the price of a generator
    that provably never invents anything, and it makes the offline path
    trustworthy enough to benchmark the *retrieval* in isolation: if the answer
    is wrong, retrieval brought the wrong passage — the generator had no chance
    to be creative about it.
    """

    @property
    def name(self) -> str:
        return "extractive"

    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        from ..rag.retriever import content_words

        q = content_words(question)
        best, best_overlap = "", 0.0
        for block in context.split("\n\n"):
            body = block.split("] ", 1)[-1]
            for sentence in _sentences(body):
                overlap = len(q & content_words(sentence)) / len(q) if q else 0.0
                if overlap > best_overlap:
                    best, best_overlap = sentence, overlap
        reply = best or "I don't have that in my documents."
        for word in reply.split():
            yield word + " "


def _sentences(text: str) -> list[str]:
    out, current = [], []
    for token in text.replace("\n", " ").split(" "):
        current.append(token)
        if token.endswith((".", "!", "?")):
            out.append(" ".join(current).strip())
            current = []
    if current:
        out.append(" ".join(current).strip())
    return [s for s in out if s]


class OpenAILLM:
    """Hosted generation. Written, unverified — no API key in CI."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self._model = model

    @property
    def name(self) -> str:
        return "openai"

    async def stream(self, question: str, context: str) -> AsyncIterator[str]:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("openai is not installed. `pip install openai`.") from exc
        client = AsyncOpenAI()
        stream = await client.chat.completions.create(
            model=self._model,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer only from the provided context. If the context "
                        "does not contain the answer, say you don't have it. "
                        "Answer in one or two spoken sentences. Do not read out "
                        "citations or file names."
                    ),
                },
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
        )
        async for part in stream:
            delta = part.choices[0].delta.content
            if delta:
                yield delta


@dataclass(slots=True)
class TurnResult:
    """What happened on a turn, for the transcript and the API."""

    intent: str
    grounded: bool
    citations: list[str] = field(default_factory=list)
    top_score: float = 0.0
    reason: str = ""


class RAGAgent:
    """Route, retrieve, then speak — or decline.

    Satisfies the ``Responder`` protocol, so it drops straight into the duplex
    session and inherits barge-in for free: interrupting the agent mid-answer
    abandons the token stream exactly as it abandons the scripted one.
    """

    REFUSAL = "I don't have that in my documents."
    GREETING = "Hello. What would you like to know?"
    ACK = "Okay."

    def __init__(self, retriever: Retriever, llm: LLM | None = None) -> None:
        self._retriever = retriever
        self._llm = llm or ExtractiveLLM()
        self.last: TurnResult | None = None

    async def respond(self, utterance: str) -> AsyncIterator[str]:
        decision = route(utterance)

        if decision.intent is Intent.GREETING:
            self.last = TurnResult(intent=decision.intent.value, grounded=True)
            yield self.GREETING
            return

        if decision.intent in (Intent.SMALLTALK, Intent.COMMAND):
            self.last = TurnResult(intent=decision.intent.value, grounded=True)
            yield self.ACK
            return

        retrieval: Retrieval = self._retriever.retrieve(utterance)
        if not retrieval.grounded:
            self.last = TurnResult(
                intent=decision.intent.value,
                grounded=False,
                top_score=retrieval.top_score,
                reason=retrieval.reason,
            )
            yield self.REFUSAL
            return

        self.last = TurnResult(
            intent=decision.intent.value,
            grounded=True,
            citations=retrieval.citations,
            top_score=retrieval.top_score,
        )
        async for token in self._llm.stream(utterance, retrieval.context):
            yield token
