"""Provider seams.

Three interfaces, deliberately narrow. Everything swappable lives behind one of
them: the offline stack (webrtcvad + faster-whisper + Piper) and a hosted
speech-to-speech service both satisfy the same contracts, which is what makes
the two benchmarkable against each other rather than merely both "supported".

The contracts are written in terms of *frames in, events out*. No provider is
allowed to block the audio loop; anything slow returns a handle and reports
completion later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import AsyncIterator, Protocol, runtime_checkable

from .audio import Frame


class SpeechState:
    """VAD verdict for a single frame."""

    SILENCE = "silence"
    SPEECH = "speech"


@runtime_checkable
class VAD(Protocol):
    """Voice activity detection, frame by frame.

    Must be pure with respect to the frame sequence: feeding the same frames in
    the same order always yields the same decisions. Stateful smoothing is fine
    (and expected); nondeterminism is not, because the latency harness depends
    on replayability.
    """

    def accepts(self, frame: Frame) -> str:
        """Return SpeechState for this frame."""
        ...

    def reset(self) -> None:
        """Clear smoothing state between turns."""
        ...


@dataclass(frozen=True, slots=True)
class Transcript:
    text: str
    is_final: bool
    # Frame index of the last audio that contributed. Lets the caller attribute
    # latency to a position in the stream rather than to wall-clock time.
    upto_frame: int


@runtime_checkable
class STT(Protocol):
    """Streaming speech-to-text.

    Partial results are expected and encouraged: the conversation loop can start
    planning a response before the user finishes, which is where a chunk of the
    time-to-first-audio budget is won back.
    """

    async def push(self, frame: Frame) -> Transcript | None: ...

    async def finish(self) -> Transcript: ...

    def reset(self) -> None: ...


@runtime_checkable
class TTS(Protocol):
    """Streaming text-to-speech.

    ``synthesize`` yields audio frames as they are produced. The consumer may
    stop iterating at any moment — that abandonment *is* the interrupt path, so
    implementations must treat generator close as a hard cancel and release
    resources immediately rather than finishing the current sentence.
    """

    def synthesize(self, text: str) -> AsyncIterator[Frame]: ...

    @property
    def name(self) -> str: ...


@runtime_checkable
class Responder(Protocol):
    """Turns a user utterance into reply text, token by token."""

    def respond(self, utterance: str) -> AsyncIterator[str]: ...
