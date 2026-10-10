"""Full-context (no memory system) baseline.

There is no memory module at all: every observed dialogue turn is kept in a
plain buffer and the whole transcript is replayed to the agent LLM as a
multi-turn conversation at query time — the standing instruction as the system
message, one role-tagged message per turn, and the probe question as the final
user message. Forgetting requests therefore reach the model as conversational
moves addressed to it rather than as quoted text inside a reading passage, so
this baseline measures whether a long-context LLM honors unlearning requests
on its own. The system instruction stays neutral and mirrors the other
baselines' answer prompt; it does not instruct the model to comply with
forgetting requests.

The transcript is passed through verbatim. Nothing is dropped, reordered or
summarized, so the baseline reflects exactly what the backbone's context
window can hold; a backbone too small for the longest instances fails loudly
at the API boundary instead of silently scoring on a mutilated dialogue.
"""

from __future__ import annotations

from typing import Any

from bemula.baselines.base import MemoryAdapter, TurnProgress
from bemula.llm import BaseLLM
from bemula.schemas import Turn

_SYSTEM_PROMPT = (
    "You are a personal assistant with access to the full conversation "
    "history with the user. Answer the user's question using ONLY the "
    "conversation history. If the answer is not in the history, say you "
    "don't have that information. Keep the answer short."
)


class FullContextAdapter(MemoryAdapter):
    """No memory system: the whole dialogue transcript is the context."""

    name = "fullcontext"

    def __init__(self, llm: BaseLLM) -> None:
        self.llm = llm
        self._turns: list[Turn] = []
        self._progress = TurnProgress("FullContext")

    def begin_instance(self, instance: Any) -> None:
        self._progress.begin_instance(instance)

    def end_instance(self) -> None:
        self._progress.end_instance()

    def begin_session(self, session: Any) -> None:
        self._progress.begin_session(session)

    def end_variant(self) -> None:
        self._progress.end_variant()

    def reset(self) -> None:
        self._turns = []
        self._progress.start_variant()

    def observe(self, turn: Turn) -> None:
        self._turns.append(turn)
        self._progress.update()

    def query(self, question: str) -> str:
        if not self._turns:
            return "I don't have any relevant information stored."
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
        messages.extend(
            {"role": turn.speaker, "content": turn.text} for turn in self._turns
        )
        messages.append({"role": "user", "content": question})
        return self.llm.chat(
            messages,
            temperature=0.0,
            max_tokens=512,
            task="fullcontext_answer",
        ).strip()
