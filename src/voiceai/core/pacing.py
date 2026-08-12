"""Playback pacing.

A speaker consumes audio at exactly one frame per frame-duration; it cannot be
handed a whole reply at once. Modelling that constraint is not test scaffolding,
it is the thing that makes barge-in possible at all: if the session drains the
synthesiser as fast as it will yield, the entire reply is committed to the
output buffer before the user opens their mouth, and "stopping" means draining a
buffer that already contains three seconds of speech.

So playback awaits a Pacer between frames. In production that is the audio
device applying backpressure. In tests it is lockstep with the input stream,
which is what lets a virtual clock produce exact latency numbers.
"""

from __future__ import annotations

import asyncio
from typing import Protocol

from .audio import FRAME_MS


class Pacer(Protocol):
    async def tick(self) -> None:
        """Return when the output device is ready for one more frame."""
        ...


class RealtimePacer:
    """Wall-clock pacing, one frame per frame-duration."""

    __slots__ = ("_period",)

    def __init__(self, frame_ms: float = FRAME_MS) -> None:
        self._period = frame_ms / 1000.0

    async def tick(self) -> None:
        await asyncio.sleep(self._period)


class LockstepPacer:
    """Emits one output frame per input frame.

    The driver calls :meth:`release` once per input frame; playback blocks until
    then. This mirrors real duplex timing (input and output both run at frame
    rate) without depending on wall-clock sleeps, so the suite is fast and
    deterministic.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = asyncio.Event()

    async def tick(self) -> None:
        await self._event.wait()
        self._event.clear()

    def release(self) -> None:
        self._event.set()


class UnpacedPacer:
    """No backpressure. Useful only for measuring raw synthesis throughput."""

    __slots__ = ()

    async def tick(self) -> None:
        await asyncio.sleep(0)
