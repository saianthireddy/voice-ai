"""Time sources.

Every latency number this project publishes comes from a Clock, never from
``time.monotonic()`` called inline. That indirection is the whole reason the
benchmark is reproducible: tests inject a VirtualClock driven by the audio
frame counter, so a measured barge-in latency is a deterministic function of
the fixture, not of how busy the machine was that afternoon.
"""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float:
        """Seconds, monotonic. Origin is arbitrary; only deltas mean anything."""
        ...


class SystemClock:
    """Wall-clock monotonic time, for live runs."""

    __slots__ = ()

    def now(self) -> float:
        return time.monotonic()


class VirtualClock:
    """Audio-driven clock.

    Advances only when frames are consumed, so 20ms of audio is exactly 20ms
    regardless of how long the CPU took to process it. This is what makes the
    latency assertions in the test suite meaningful rather than flaky: they
    measure the *algorithm's* reaction time in audio-time, not the runtime's.

    Real-world latency adds the compute cost on top; ``voiceai.bench_pipeline``
    measures that separately against SystemClock and both are reported.

    Time accumulates in integer nanoseconds, not floats. Adding 0.02 repeatedly
    drifts, and an assertion of "<= 100ms" then fails against a measurement of
    100.00000000000003: a bug in the ruler, reported as a bug in the thing being
    measured. Integers keep the arithmetic exact and the failures honest.
    """

    __slots__ = ("_ns",)

    _NS = 1_000_000_000

    def __init__(self, start: float = 0.0) -> None:
        self._ns = int(round(start * self._NS))

    def now(self) -> float:
        return self._ns / self._NS

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("clocks do not run backwards")
        self._ns += int(round(seconds * self._NS))
