"""Energy VAD with hangover smoothing, plus a barge-in gate.

Two separate concerns that are easy to conflate and expensive to get wrong:

*Is there speech in this frame?* - a cheap energy test. Deliberately not a
neural VAD by default: the offline stack must run anywhere, and the barge-in
decision is dominated by the confirmation logic below, not by the frame test.

*Should we interrupt the agent?* - a different question entirely. A single loud
frame is a cough, a door, or the agent's own voice leaking through the mic.
Interrupting on one frame produces an agent that flinches at everything; waiting
too long produces one that talks over you. That tradeoff is the ``onset_frames``
knob, and it is the single most consequential number in this repo.
"""

from __future__ import annotations

import array
import math

from .audio import Frame
from .protocols import SpeechState


def frame_rms(frame: Frame) -> float:
    """Root-mean-square amplitude, normalised to 0..1."""
    samples = array.array("h")
    samples.frombytes(frame.pcm)
    if not samples:
        return 0.0
    acc = 0
    for s in samples:
        acc += s * s
    return math.sqrt(acc / len(samples)) / 32768.0


class EnergyVAD:
    """Threshold VAD with asymmetric hangover.

    Onset is fast (we want to notice speech starting) and release is slow (we do
    not want to declare silence during the natural gaps inside a sentence). The
    asymmetry is the point; a symmetric VAD chops words in half.
    """

    __slots__ = ("_threshold", "_release_frames", "_silence_run")

    def __init__(self, threshold: float = 0.02, release_frames: int = 12) -> None:
        self._threshold = threshold
        self._release_frames = release_frames
        self._silence_run = release_frames

    def accepts(self, frame: Frame) -> str:
        if frame_rms(frame) >= self._threshold:
            self._silence_run = 0
            return SpeechState.SPEECH
        self._silence_run += 1
        if self._silence_run < self._release_frames:
            return SpeechState.SPEECH
        return SpeechState.SILENCE

    def reset(self) -> None:
        self._silence_run = self._release_frames


class BargeInDetector:
    """Decides when user speech should cancel agent playback.

    Requires ``onset_frames`` *consecutive* speech frames before firing. At the
    default 20ms framing, 6 frames = 120ms of confirmation.

    ``echo_guard`` exists because the microphone hears the speaker. Without
    acoustic echo cancellation, the agent's own output re-enters as "user
    speech" and it interrupts itself within a frame or two - the single most
    common way a naive full-duplex loop fails. When no AEC is available, this
    gate suppresses barge-in unless the input meaningfully exceeds the level the
    agent is currently emitting. It is a crude proxy for AEC and is documented
    as such rather than sold as one.
    """

    __slots__ = ("_onset_frames", "_run", "_echo_guard", "_margin")

    def __init__(
        self,
        onset_frames: int = 6,
        echo_guard: bool = True,
        margin_db: float = 6.0,
    ) -> None:
        if onset_frames < 1:
            raise ValueError("onset_frames must be >= 1")
        self._onset_frames = onset_frames
        self._run = 0
        self._echo_guard = echo_guard
        self._margin = 10 ** (margin_db / 20.0)

    def observe(self, frame: Frame, agent_output_rms: float = 0.0) -> bool:
        """Feed one input frame. Returns True on the frame that triggers barge-in.

        ``agent_output_rms`` is the level of what the agent is playing right now;
        pass 0.0 when it is silent or when real AEC has already removed it.
        """
        level = frame_rms(frame)
        speaking = level >= 0.02

        if speaking and self._echo_guard and agent_output_rms > 0.0:
            speaking = level > agent_output_rms * self._margin

        if not speaking:
            self._run = 0
            return False

        self._run += 1
        if self._run == self._onset_frames:
            return True
        return False

    def reset(self) -> None:
        self._run = 0

    @property
    def onset_frames(self) -> int:
        return self._onset_frames
