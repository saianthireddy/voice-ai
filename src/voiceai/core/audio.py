"""Audio framing primitives.

One frame is the atomic unit of everything downstream: VAD decisions, latency
accounting, and interrupt granularity. Frame size is therefore a latency floor
you cannot get under - a 20ms frame means barge-in can never be detected faster
than 20ms plus whatever confirmation the VAD requires.
"""

from __future__ import annotations

from dataclasses import dataclass

SAMPLE_RATE = 16_000
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 320
BYTES_PER_SAMPLE = 2  # int16 PCM


@dataclass(frozen=True, slots=True)
class Frame:
    """A fixed-duration slice of 16-bit mono PCM.

    ``index`` is the position in the stream. Latency is measured in frames and
    converted to milliseconds exactly once, at the boundary, so nothing
    accumulates float drift over a long conversation.
    """

    index: int
    pcm: bytes

    def __post_init__(self) -> None:
        expected = FRAME_SAMPLES * BYTES_PER_SAMPLE
        if len(self.pcm) != expected:
            raise ValueError(
                f"frame {self.index}: expected {expected} bytes, got {len(self.pcm)}"
            )

    @property
    def start_ms(self) -> float:
        return self.index * FRAME_MS

    @property
    def end_ms(self) -> float:
        return (self.index + 1) * FRAME_MS


def frames_to_ms(n_frames: int) -> float:
    return n_frames * FRAME_MS


def ms_to_frames(ms: float) -> int:
    """Round up: a budget of 100ms must not be satisfied by 5.9 frames."""
    return int(-(-ms // FRAME_MS))
