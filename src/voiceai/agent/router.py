"""Intent routing.

Four handlers, chosen before retrieval runs. The point is not classification for
its own sake — it is to avoid paying for retrieval on turns that cannot benefit
from it. "Hello" does not need a vector search, and in a voice pipeline that
saved lookup is a measurable slice of the response budget.

The router is rule-based on purpose. A learned classifier here would add model
load time to the critical path to decide something keyword rules get right
almost always, and when it was wrong it would be wrong invisibly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Intent(str, Enum):
    GREETING = "greeting"
    SMALLTALK = "smalltalk"
    QUESTION = "question"      # needs retrieval
    COMMAND = "command"        # ends the call, repeats, etc.


@dataclass(frozen=True, slots=True)
class Route:
    intent: Intent
    confidence: float
    needs_retrieval: bool


_GREETINGS = frozenset("hi hello hey morning afternoon evening".split())
_COMMANDS = frozenset("stop cancel repeat again goodbye bye quit exit".split())
_QUESTION_WORDS = frozenset("what when where which who why how can does do is are".split())
_SMALLTALK = frozenset(
    ["how are you", "thank you", "thanks", "nice", "okay", "ok", "sure"]
)


def route(utterance: str) -> Route:
    text = utterance.strip().lower()
    if not text:
        return Route(Intent.SMALLTALK, 0.0, needs_retrieval=False)

    words = text.split()
    first = words[0].strip("?.,!")

    # Commands are checked first: "stop" must stop, even if the rest of the
    # sentence looks like a question.
    if any(w.strip("?.,!") in _COMMANDS for w in words):
        return Route(Intent.COMMAND, 0.9, needs_retrieval=False)

    if first in _GREETINGS and len(words) <= 3:
        return Route(Intent.GREETING, 0.9, needs_retrieval=False)

    if any(phrase in text for phrase in _SMALLTALK) and len(words) <= 5:
        return Route(Intent.SMALLTALK, 0.7, needs_retrieval=False)

    # A question mark or an interrogative opener is a strong signal; anything
    # else substantial is treated as a question too, because the cost of a
    # needless retrieval is far lower than the cost of refusing to look.
    if text.endswith("?") or first in _QUESTION_WORDS:
        return Route(Intent.QUESTION, 0.85, needs_retrieval=True)

    return Route(Intent.QUESTION, 0.5, needs_retrieval=True)
