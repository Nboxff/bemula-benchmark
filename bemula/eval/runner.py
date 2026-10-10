"""Control-pair evaluation runner.

Protocol per dialogue instance:
1. `adapter.reset()`; observe `control_sessions` turn by turn; run Retention
   plus the control-variant calibration probes (preservation and attack on
   intact memory).
2. `adapter.reset()`; observe `sessions` (forget variant); run Efficacy,
   Preservation, Attack, Relearning — in that order. Attack is probed before
   Relearning so the adversarial-extraction suite runs on the post-unlearning
   state, before any relearning short dialogue re-introduces same-type facts.

Probe answers are never written back to memory (adapter contract), so probes
cannot contaminate each other.

Both Preservation and Attack are measured on the forget variant *and* on the
intact control variant: a system that never stored an unrelated fact cannot be
credited with preserving it, just as an attack that cannot extract the target
from intact memory proves nothing about robustness.
"""

from __future__ import annotations

from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Any

from pydantic import BaseModel
from tqdm import tqdm

from bemula.baselines.base import MemoryAdapter
from bemula.eval.judge import Judge
from bemula.schemas import (
    AttackType,
    DialogueInstance,
    ForgetType,
    Grading,
    Session,
    Suite,
    TestItem,
)

# Suites run on the forget variant, in protocol order. Attack precedes
# Relearning: the robustness suite must measure the memory right after
# forgetting, not after relearning has re-seeded same-type values.
FORGET_SUITE_ORDER = (
    Suite.EFFICACY,
    Suite.PRESERVATION,
    Suite.ATTACK,
    Suite.RELEARNING,
)


class ItemRecord(BaseModel):
    """One probed item with the system's answer and its verdict."""

    item_id: str
    instance_id: str
    suite: Suite
    attack_type: AttackType | None = None
    # Forget type of the item's request (None for instance-level preservation
    # items, which are not tied to a single request).
    forget_type: ForgetType | None = None
    request_id: str | None = None
    grading: Grading
    question: str
    gold_answer: str
    answer: str
    correct: bool
    leakage: bool | None = None  # attack items only
    # Which dialogue variant this probe ran on. Retention always runs on the
    # control variant; attack items run on BOTH (the control run calibrates
    # whether the attack can extract the target when memory is intact).
    variant: str = "forget"  # "control" | "forget"
    # Pairing metadata for the per-item efficacy condition:
    # an efficacy item shares (request_id, probe_type, fact) with a retention
    # item run on the control variant.
    probe_type: str = "direct"
    related_fact_ids: list[str] = []
    # Instance length bucket (short | medium | long | xlong), copied from the
    # instance's difficulty so records can be stratified without a data join.
    # None only for records written before this field existed.
    length_bucket: str | None = None


def _observe(adapter: MemoryAdapter, sessions: Iterable[Session]) -> None:
    for session in sessions:
        adapter.begin_session(session)
        for turn in session.turns:
            adapter.observe(turn)
    adapter.end_variant()


def _probe(
    adapter: MemoryAdapter,
    judge: Judge,
    instance: DialogueInstance,
    item: TestItem,
    variant: str = "forget",
) -> ItemRecord:
    # Relearning items carry a short dialogue stating the new information;
    # it is observed right before the question so the system must have learned
    # it post-unlearning to answer.
    for turn in item.context_turns:
        adapter.observe(turn)
    answer = adapter.query(item.question)
    verdict = judge.judge(item, answer)
    # Stratification is request-level: look up the item's forget type via its
    # request_id; instance-level items fall back to None.
    request_types = {r.request_id: r.forget_type for r in instance.forget_requests}
    forget_type = request_types.get(item.request_id) if item.request_id else None
    return ItemRecord(
        item_id=item.item_id,
        instance_id=instance.instance_id,
        suite=item.suite,
        attack_type=item.attack_type,
        forget_type=forget_type,
        request_id=item.request_id,
        grading=item.grading,
        question=item.question,
        gold_answer=item.gold_answer,
        answer=answer,
        correct=verdict.correct,
        leakage=verdict.leakage,
        variant=variant,
        probe_type=item.probe_type,
        related_fact_ids=item.related_fact_ids,
        length_bucket=instance.difficulty.length_bucket,
    )


def run_instance(
    adapter: MemoryAdapter,
    instance: DialogueInstance,
    items: list[TestItem],
    judge: Judge,
) -> list[ItemRecord]:
    """Run the full control-pair protocol for one dialogue instance."""
    adapter.begin_instance(instance)
    by_suite: dict[Suite, list[TestItem]] = {}
    for item in items:
        by_suite.setdefault(item.suite, []).append(item)

    records: list[ItemRecord] = []

    # Step 1: control variant -> retention probes plus the calibration probes
    # for preservation (the fact must be retrievable when memory is intact) and
    # for attack (an attack must be able to extract the target when memory is
    # intact).
    adapter.reset()
    _observe(adapter, instance.control_sessions)
    for item in by_suite.get(Suite.RETENTION, []):
        records.append(_probe(adapter, judge, instance, item, variant="control"))
    for item in by_suite.get(Suite.PRESERVATION, []):
        records.append(_probe(adapter, judge, instance, item, variant="control"))
    for item in by_suite.get(Suite.ATTACK, []):
        records.append(_probe(adapter, judge, instance, item, variant="control"))

    # Step 2: forget variant -> the remaining suites, in protocol order.
    adapter.reset()
    _observe(adapter, instance.sessions)
    for suite in FORGET_SUITE_ORDER:
        for item in by_suite.get(suite, []):
            records.append(_probe(adapter, judge, instance, item))
    adapter.end_instance()
    return records


def run_evaluation(
    adapter: MemoryAdapter,
    instances: list[DialogueInstance],
    items_by_instance: dict[str, list[TestItem]],
    judge: Judge,
) -> list[ItemRecord]:
    """Run the protocol over all instances and collect item records."""
    records: list[ItemRecord] = []
    for instance in instances:
        records.extend(
            run_instance(adapter, instance, items_by_instance.get(instance.instance_id, []), judge)
        )
    return records


def run_evaluation_parallel(
    adapters: list[MemoryAdapter],
    instances: list[DialogueInstance],
    items_by_instance: dict[str, list[TestItem]],
    judge: Judge,
    on_instance: Any = None,
) -> list[ItemRecord]:
    """Instance-level parallel evaluation with one dedicated adapter per worker.

    Instances are independent (each run_instance starts with reset()), so
    workers draw an adapter from a pool, run a full instance, and return it.
    Adapters are constructed by the caller (sequentially) so heavyweight
    init (local vector stores, migrations) does not race. `on_instance` is
    invoked with each finished instance's records (for checkpointing).
    """
    pool_queue: Queue[MemoryAdapter] = Queue()
    for adapter in adapters:
        pool_queue.put(adapter)

    def work(instance: DialogueInstance) -> list[ItemRecord]:
        adapter = pool_queue.get()
        try:
            return run_instance(
                adapter, instance, items_by_instance.get(instance.instance_id, []), judge
            )
        finally:
            pool_queue.put(adapter)

    records: list[ItemRecord] = []
    with ThreadPoolExecutor(max_workers=len(adapters)) as pool:
        futures = [pool.submit(work, inst) for inst in instances]
        with tqdm(total=len(instances), desc="eval", unit="inst") as bar:
            for fut in as_completed(futures):
                recs = fut.result()
                records.extend(recs)
                if on_instance is not None:
                    on_instance(recs)
                bar.update(1)
    return records
