"""Core data models for Bemula.

All benchmark data is stored as JSONL and validated with these Pydantic models.
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, Field


class FactCategory(str, Enum):
    """The 12 atomic-fact categories.

    ``event`` covers "what the user did" facts (trips, races, purchases,
    moves, celebrations, workshops — usually time-anchored), so the
    benchmark's temporal and activity probes have a dedicated category.
    """
    ADDRESS = "address"
    HEALTH = "health"
    PREFERENCE = "preference"
    CAREER = "career"
    EDUCATION = "education"
    RELATIONSHIP = "relationship"
    FINANCE = "finance"
    ROUTINE = "routine"
    EVENT = "event"      # things the user did / experienced (time-anchored)
    HOBBY = "hobby"      # leisure pursuits and hobbies
    SOCIAL = "social"    # friendships, clubs, community ties
    OTHER = "other"


class Sensitivity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ForgetType(str, Enum):
    """The 6 forget-request types."""

    F1 = "F1"  # factual: forget one isolated atomic fact
    F2 = "F2"  # compositional: forget several unrelated facts at once
    F3 = "F3"  # semantic: forget everything under a topic/category
    F4 = "F4"  # inferential: forget information derived from atomic facts
    F5 = "F5"  # temporal: forget memories within a time scope
    F6 = "F6"  # replacement: replace an old fact with a new one


class RequestStyle(str, Enum):
    """The 5 natural-language styles of a forget request."""

    DIRECT = "direct"
    POLITE = "polite"
    INDIRECT = "indirect"
    IMPERATIVE = "imperative"
    IMPLICIT = "implicit"


class TimeScopeMode(str, Enum):
    """F5 distinguishes dialogue time from event time."""

    DIALOGUE_TIME = "dialogue_time"
    EVENT_TIME = "event_time"


class Suite(str, Enum):
    """The 5 test suites."""

    RETENTION = "retention"
    EFFICACY = "efficacy"
    PRESERVATION = "preservation"
    RELEARNING = "relearning"
    ATTACK = "attack"


class AttackType(str, Enum):
    """The 5 attack types of the Attack suite.

    Direct questioning is intentionally absent: it duplicates the Efficacy
    suite's probes, so the suite only covers adversarial framings.
    """

    INDIRECT = "indirect"
    ROLEPLAY = "roleplay"
    JAILBREAK = "jailbreak"
    ENUMERATION = "enumeration"
    MULTIHOP = "multihop"


class Grading(str, Enum):
    EXACT = "exact"
    LLM_JUDGE = "llm_judge"


class AtomicFact(BaseModel):
    """An atomic fact in a user profile."""

    fact_id: str
    category: FactCategory
    subject: str
    predicate: str
    value: str
    surface_form: str  # natural-language phrasing used in dialogues
    event_time: str | None = None  # when the event happened (ISO-ish, e.g. "2026-01")
    sensitivity: Sensitivity = Sensitivity.LOW
    derivations: list[str] = Field(default_factory=list)  # fact_ids derivable from this fact
    source_facts: list[str] = Field(default_factory=list)  # set if this fact is itself derived

    @property
    def is_derived(self) -> bool:
        return bool(self.source_facts)


class UserProfile(BaseModel):
    """A synthetic user profile."""

    profile_id: str
    persona_summary: str
    demographics: dict[str, str] = Field(default_factory=dict)
    facts: list[AtomicFact] = Field(default_factory=list)

    def fact_map(self) -> dict[str, AtomicFact]:
        return {f.fact_id: f for f in self.facts}

    def get_fact(self, fact_id: str) -> AtomicFact:
        return self.fact_map()[fact_id]


class Turn(BaseModel):
    speaker: str  # "user" | "assistant"
    text: str
    timestamp: str
    linked_fact_ids: list[str] = Field(default_factory=list)


class Session(BaseModel):
    session_id: str
    theme: str
    scenario: str = ""  # user goal driving this session (dialogue planning)
    turns: list[Turn] = Field(default_factory=list)


class RequestPosition(BaseModel):
    session_index: int
    turn_index: int


class TimeScope(BaseModel):
    """F5 time scope; mode distinguishes dialogue time from event time."""

    mode: TimeScopeMode
    start: str  # event_time string ("2026-01") or session index ("0"), by mode
    end: str


class ForgetRequest(BaseModel):
    """The forget request embedded in a dialogue."""

    request_id: str
    forget_type: ForgetType
    style: RequestStyle
    position: RequestPosition
    target_fact_ids: list[str]
    request_text: str
    mention_count: int = 1  # how often the targets are mentioned before the request
    time_scope: TimeScope | None = None  # F5 only
    replacement: str | None = None  # F6 only: the new value


class Difficulty(BaseModel):
    """Difficulty dimension values recorded per instance.

    Request-level dimensions (style, position, mention count) live on each
    ForgetRequest; the instance records only the aggregate values.
    """

    length_bucket: str  # short | medium | long | xlong
    num_sessions: int
    num_turns: int
    num_requests: int  # forget requests in this instance (1-8)
    distractor_count: int  # unrelated facts mentioned in the instance


class DialogueInstance(BaseModel):
    """A dialogue instance with its paired control variant."""

    instance_id: str
    profile_id: str
    source: str = "synthetic"  # all data is synthetic (multi-agent pipeline)
    # 1-8 forget requests, ordered by position ascending; types may differ.
    forget_requests: list[ForgetRequest] = Field(min_length=1)
    difficulty: Difficulty
    sessions: list[Session]
    # Identical except each request/ack exchange -> a neutral filler exchange.
    control_sessions: list[Session]


class TestItem(BaseModel):
    """A single test question."""

    __test__ = False  # not a pytest test class

    item_id: str
    instance_id: str
    suite: Suite
    # Links retention/efficacy/relearning/attack items to their forget request;
    # preservation items are instance-level (None).
    request_id: str | None = None
    probe_type: str = "direct"  # direct | derived | new_value | ...
    attack_type: AttackType | None = None  # Attack suite only
    question: str
    gold_answer: str
    grading: Grading
    rubric: str | None = None  # required for llm_judge items
    related_fact_ids: list[str] = Field(default_factory=list)
    hops: int = 0  # 0 direct retrieval, 1 single-step, 2 multi-hop
    # Relearning items only: a short natural dialogue that states the new
    # information; the evaluator observes it before asking the question.
    context_turns: list[Turn] = Field(default_factory=list)


T = TypeVar("T", bound=BaseModel)


def dump_jsonl(items: list[BaseModel], path: str | Path) -> None:
    """Write a list of Pydantic models to a JSONL file (overwrites)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")


def append_jsonl(items: list[BaseModel], path: str | Path) -> None:
    """Append Pydantic models to a JSONL file (prompt persistence).

    Each item is written and flushed immediately, so a crash after a
    successfully generated item never loses already-persisted work.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.model_dump(mode="json"), ensure_ascii=False) + "\n")
            f.flush()


def jsonl_ids(path: str | Path, key: str) -> set[str]:
    """IDs of the records already present in a JSONL file (resumable runs).

    Returns the value of field `key` for every parseable line. Malformed
    lines are ignored so a truncated tail (crash mid-write) does not block
    resuming.
    """
    ids: set[str] = set()
    path = Path(path)
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line).get(key)
            except (json.JSONDecodeError, AttributeError):
                continue
            if isinstance(value, str):
                ids.add(value)
    return ids


def load_jsonl(path: str | Path, model: type[T]) -> list[T]:
    """Load a JSONL file into a list of validated Pydantic models."""
    out: list[T] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(model.model_validate(json.loads(line)))
    return out
