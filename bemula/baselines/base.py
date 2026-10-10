"""Memory adapter interface for baselines.

Every memory system under evaluation is wrapped in a `MemoryAdapter` so the
eval runner can drive it uniformly: reset -> observe dialogue turns -> probe
with questions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tqdm import tqdm

from bemula.schemas import Turn


class TurnProgress:
    """Opt-in turn-level progress bar for an adapter's observe phase.

    Driven from the adapter's lifecycle hooks: `begin_instance` arms it,
    `start_variant` (called from the adapter's `reset()`) opens a bar for the
    variant about to be observed, `begin_session`/`update` advance it, and
    `end_variant`/`end_instance` close it. Probes that observe extra context
    turns after `end_variant` (relearning items) are not counted.
    """

    def __init__(self, label: str, enabled: bool = True) -> None:
        self._label = label
        self._enabled = enabled
        self._instance: Any = None
        self._variant_index = 0
        self._bar: tqdm | None = None
        self._num_sessions = 0
        self._session_no = 0
        self._session_turns = 0
        self._turn_in_session = 0

    def begin_instance(self, instance: Any) -> None:
        self._instance = instance
        self._variant_index = 0

    def end_instance(self) -> None:
        self.close()
        self._instance = None
        self._variant_index = 0

    def start_variant(self) -> None:
        """Open the bar for the variant whose observe phase starts now."""
        self.close()
        if not self._enabled or self._instance is None:
            return
        if self._variant_index == 0:
            sessions = self._instance.control_sessions
            variant = "control"
        else:
            sessions = self._instance.sessions
            variant = "forget"
        self._num_sessions = len(sessions)
        total_turns = sum(len(s.turns) for s in sessions)
        self._bar = tqdm(
            total=total_turns,
            desc=f"{self._label} observe [{variant}]",
            unit="turn",
            leave=False,
        )
        self._variant_index += 1

    def begin_session(self, session: Any) -> None:
        if self._bar is None:
            return
        self._session_no += 1
        self._turn_in_session = 0
        self._session_turns = len(session.turns)

    def update(self) -> None:
        if self._bar is None:
            return
        self._turn_in_session += 1
        self._bar.set_postfix(
            session=f"{self._session_no}/{self._num_sessions}",
            turn_in_session=f"{self._turn_in_session}/{self._session_turns}",
            refresh=False,
        )
        self._bar.update(1)

    def end_variant(self) -> None:
        self.close()

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()
            self._bar = None
        self._session_no = 0
        self._session_turns = 0
        self._turn_in_session = 0


class MemoryAdapter(ABC):
    """Uniform interface between the Bemula eval runner and a memory system.

    Contract:
    - `reset()` must wipe ALL user state; instances are evaluated in isolation.
    - `observe(turn)` is called once per dialogue turn, in order, for both the
      control variant and the forget variant.
    - `query(question)` answers from the current memory state. Probe QA must
      NOT be written back into memory (`evaluation.probe_writeback: false` in
      configs/default.yaml) so probes cannot contaminate each other.
    """

    name: str = "abstract"

    @abstractmethod
    def reset(self) -> None:
        """Clear all stored memory; called before each dialogue variant."""

    @abstractmethod
    def observe(self, turn: Turn) -> None:
        """Write one dialogue turn (user or assistant) into memory."""

    @abstractmethod
    def query(self, question: str) -> str:
        """Answer a probe question from memory; never writes to memory."""

    def begin_instance(self, instance: Any) -> None:
        """Optional lifecycle hook: called before an instance's first reset."""

    def end_instance(self) -> None:
        """Optional lifecycle hook: called after an instance is fully evaluated."""

    def begin_session(self, session: Any) -> None:
        """Optional lifecycle hook: called before each session's turns are observed."""

    def end_variant(self) -> None:
        """Optional lifecycle hook: called after a dialogue variant is observed."""
