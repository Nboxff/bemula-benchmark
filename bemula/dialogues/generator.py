from __future__ import annotations

import copy
import random
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

from tqdm import tqdm

from bemula.dialogues.agents import (
    AssistantAgent,
    UserSimulator,
    sanitize_chat_text,
    verify_session,
)
from bemula.dialogues.injection import (
    _humanize_month,
    inject_exchange,
    render_request_text,
)
from bemula.dialogues.planner import (
    _GENERIC_TOKENS,
    THEME_FACT_AFFINITY,
    InstancePlan,
    RequestPlan,
    plan_instances,
)
from bemula.llm import BaseLLM, parse_json
from bemula.schemas import (
    AtomicFact,
    DialogueInstance,
    Difficulty,
    ForgetRequest,
    ForgetType,
    RequestPosition,
    Session,
    TimeScopeMode,
    Turn,
    UserProfile,
)

# Every forget request, regardless of style, must unambiguously require that the
# information is no longer retained; styles differ in tone and wording, never in
# intent strength. A generated request that leaves removal optional is a failed
# generation (checked by an LLM intent verifier in _generate_request_exchange).
_INTENT_PROMPT = """A user sent this message to an AI assistant with memory:

"{request}"

Does the message unambiguously require the assistant to no longer retain \
{targets}? The requirement may be expressed in any tone (polite, commanding, \
hinted through a described situation) and need not use words like "delete" — \
but a reasonable reader must conclude the assistant is expected to not keep \
the information. If removal is merely hinted as a preference, or the message \
only remarks that the information is outdated, answer no.

Respond with JSON only: {{"clear": true | false}}."""


# Bounded turn-level retries for a user turn that drops a fact's stated year;
# the violation is fed back into the prompt, and only a persistent violation
# fails the instance (a one-off slip must not discard a whole simulation).
_TEMPORAL_RETRIES = 3


def generate_instances(
    profile: UserProfile,
    llm: BaseLLM,
    config: dict[str, Any],
    on_instance: Callable[[DialogueInstance], None] | None = None,
    profile_index: int = 0,
) -> list[DialogueInstance]:
    """Plan and realize all dialogue instances for one profile.

    Instances are independent, so they are realized concurrently
    (instance-level thread pool; each instance's simulation is sequential
    internally). Each instance gets its own deterministically seeded RNG so
    results do not depend on scheduling.

    Failure policy: each instance is retried up to
    `generation.max_retries` times; an instance that still fails is skipped
    with a warning instead of being emitted in a degraded form. `on_instance`
    (when given) is invoked from the main thread as each instance completes
    successfully, so callers can persist it immediately.
    """
    retries = int(config.get("generation", {}).get("max_retries", 3))
    plans = None
    for attempt in range(retries):
        try:
            plans = plan_instances(profile, llm, config, profile_index=profile_index)
            break
        except Exception as e:  # noqa: BLE001 - retried below
            if attempt == retries - 1:
                raise
            print(f"  {profile.profile_id}: planning attempt {attempt + 1} failed "
                  f"({e}); retrying")
    assert plans is not None
    seed = config.get("seed", 42)
    workers = int(config.get("generation", {}).get("max_workers", 1))

    def build(i: int, plan: InstancePlan) -> tuple[int, DialogueInstance | None]:
        instance_id = f"{profile.profile_id}:inst{i}"
        rng = random.Random(f"gen:{seed}:{profile.profile_id}:{instance_id}")
        for attempt in range(retries):
            try:
                return i, realize_instance(plan, profile, llm, rng, config, instance_id)
            except Exception as e:  # noqa: BLE001 - retried below
                if attempt == retries - 1:
                    print(f"  warning: {instance_id} failed after {retries} attempts "
                          f"({e}); skipping")
                    return i, None
                print(f"  {instance_id} attempt {attempt + 1} failed ({e}); retrying")
        raise AssertionError("unreachable")

    instances: list[DialogueInstance | None] = [None] * len(plans)
    with tqdm(total=len(plans), desc=f"dialogues:{profile.profile_id}", unit="inst") as bar:
        if workers <= 1:
            for i, plan in enumerate(plans):
                i, inst = build(i, plan)
                instances[i] = inst
                if inst is not None and on_instance is not None:
                    on_instance(inst)
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(build, i, plan) for i, plan in enumerate(plans)]
                for fut in as_completed(futures):
                    i, inst = fut.result()
                    instances[i] = inst
                    if inst is not None and on_instance is not None:
                        on_instance(inst)
                    bar.update(1)
    return [inst for inst in instances if inst is not None]


def realize_instance(
    plan: InstancePlan,
    profile: UserProfile,
    llm: BaseLLM,
    rng: random.Random,
    config: dict[str, Any],
    instance_id: str,
) -> DialogueInstance:
    """Realize a single dialogue instance from its plan.

    Requests are injected in ascending position order; because each request
    lives in its own session segment, earlier insertions never shift later
    requests' positions.
    """
    fmap = profile.fact_map()
    timeline = _Timeline(rng, plan.num_sessions)

    sessions = _simulate_sessions(plan, profile, llm, rng, config, instance_id,
                                  timeline)
    control_sessions = copy.deepcopy(sessions)
    request_texts = []
    positions = []
    for j, req in enumerate(plan.requests):
        request_text, ack_text = _generate_request_exchange(
            req, plan, profile, sessions, llm, config
        )
        filler_text, filler_ack_text = _generate_filler_exchange(
            request_text, plan, profile, llm, config, j
        )
        req_ts = timeline.ts(req.request_session,
                             len(sessions[req.request_session].turns), rng)
        turn_index = inject_exchange(
            sessions, control_sessions, req, request_text, ack_text,
            filler_text, filler_ack_text, req_ts,
            follow_fact_ids=_f4_source_ids(plan, profile, req),
        )
        request_texts.append(request_text)
        positions.append((req.request_session, turn_index))
    # Neither side may restate a target after its request: a later mention
    # would legitimately re-store the fact and break the efficacy semantics.
    _scrub_post_request_leaks(sessions, control_sessions, plan, profile, positions,
                              llm, config)

    # Injected pairs are stamped from the pre-injection session length, which
    # is out of order once inserted mid-session; re-stamp both variants with
    # the same timeline so timestamps are monotonic and stay equal across
    # variants.
    _restamp_timestamps(sessions, timeline, rng)
    _restamp_timestamps(control_sessions, timeline, rng)

    # mention_count records what actually happened, not the planning target:
    # the user simulator may surface a target more or fewer times than
    # scheduled, so stratified analysis by target mention count must use the
    # realized values.
    actual_counts = _actual_mention_counts(plan, positions, sessions)

    requests = [
        ForgetRequest(
            request_id=f"{instance_id}:req{j}",
            forget_type=req.forget_type,
            style=req.style,
            position=RequestPosition(session_index=positions[j][0], turn_index=positions[j][1]),
            target_fact_ids=req.target_fact_ids,
            request_text=request_texts[j],
            mention_count=actual_counts[j],
            time_scope=req.time_scope,
            replacement=req.replacement,
        )
        for j, req in enumerate(plan.requests)
    ]
    # Distractor density reflects facts actually mentioned, not the planned pool.
    mentioned = {fid for s in sessions for t in s.turns for fid in t.linked_fact_ids}
    distractor_count = len(set(plan.distractor_fact_ids) & mentioned)
    difficulty = Difficulty(
        length_bucket=plan.length_bucket,
        num_sessions=len(sessions),
        num_turns=sum(len(s.turns) for s in sessions),
        num_requests=len(plan.requests),
        distractor_count=distractor_count,
    )
    # Touch fmap so plan targets are validated early.
    for req in plan.requests:
        for tid in req.target_fact_ids:
            fmap[tid]
    return DialogueInstance(
        instance_id=instance_id,
        profile_id=profile.profile_id,
        forget_requests=requests,
        difficulty=difficulty,
        sessions=sessions,
        control_sessions=control_sessions,
    )


_BASE_DATE = datetime(2026, 1, 5, 9, 0, 0)


class _Timeline:
    """Per-instance timestamp schedule with realistic cadence.

    Timestamps must look like real chat cadence rather than a machine-flavored
    schedule (consecutive days, fixed per-turn gaps would stretch a 130-turn
    session over 15 hours):
    - the instance starts on a rng-picked date within ~5 weeks of _BASE_DATE;
    - sessions are 1-5 days apart, at a plausible hour of the day;
    - turns inside a session are 20-120 seconds apart, so even a 200-turn
      xlong session spans a few hours at most.
    """

    def __init__(self, rng: random.Random, num_sessions: int) -> None:
        self._session_starts: list[datetime] = []
        t = _BASE_DATE + timedelta(days=rng.randint(0, 35), hours=rng.randint(0, 14))
        for _ in range(num_sessions):
            self._session_starts.append(t)
            t += timedelta(days=rng.randint(1, 5), hours=rng.randint(0, 23),
                           minutes=rng.randint(0, 59))
        self._gaps: list[list[int]] = [[] for _ in range(num_sessions)]

    def ts(self, session_index: int, turn_index: int, rng: random.Random) -> str:
        gaps = self._gaps[session_index]
        while len(gaps) <= turn_index:
            gaps.append(rng.randint(20, 120))
        dt = self._session_starts[session_index] + timedelta(seconds=sum(gaps[: turn_index + 1]))
        return dt.isoformat() + "Z"


def _f4_source_ids(plan: InstancePlan, profile: UserProfile,
                   req: RequestPlan) -> set[str]:
    """Fact IDs an F4 request must land after: its target AND its sources.

    F4 semantics: the request forgets an inference; its source facts must
    have been stated before the request so the target is genuinely inferable
    from the dialogue. For other request types this is just the empty set
    (targets are handled by inject_exchange itself).
    """
    if req.forget_type != ForgetType.F4:
        return set()
    fmap = profile.fact_map()
    out: set[str] = set()
    for tid in req.target_fact_ids:
        if tid in fmap:
            out.update(fmap[tid].source_facts)
    return out


def _actual_mention_counts(
    plan: InstancePlan,
    positions: list[tuple[int, int]],
    sessions: list[Session],
) -> list[int]:
    """Realized per-request mention counts.

    For each request, count the turns before its injected position that
    mention each target fact (via linked_fact_ids), then take the median over
    the request's targets. The planner's `mention_count` is a *schedule*; the
    realized dialogue may mention a target more or fewer times (the LLM user
    simulator decides how to surface facts), so the metadata records what
    actually happened. Only the forget targets count (F4 source facts are
    premises, not targets).
    """
    from statistics import median

    counts: list[int] = []
    for j, req in enumerate(plan.requests):
        s_idx, t_idx = positions[j]
        per_target: list[int] = []
        for tid in req.target_fact_ids:
            c = 0
            for s in sessions[:s_idx]:
                c += sum(1 for t in s.turns if tid in t.linked_fact_ids)
            c += sum(
                1 for t in sessions[s_idx].turns[:t_idx]
                if tid in t.linked_fact_ids
            )
            per_target.append(c)
        counts.append(int(median(per_target)) if per_target else 1)
    return counts


def _restamp_timestamps(sessions: list[Session], timeline: _Timeline,
                        rng: random.Random) -> None:
    """Re-stamp every turn by (session, index) so timestamps stay monotonic.

    Injected request/ack pairs are stamped from the pre-injection session
    length, which is out of order once inserted mid-session; re-stamping both
    variants with the same timeline restores ordering and variant equality.
    """
    for si, session in enumerate(sessions):
        for ti, turn in enumerate(session.turns):
            turn.timestamp = timeline.ts(si, ti, rng)


# ---------------------------------------------------------------------------
# Shared mention distribution
# ---------------------------------------------------------------------------


def _fact_capacity(min_turns: int) -> int:
    """Max facts woven into one session. Capped so the user simulator's fact
    checklist stays small even for xlong sessions (100-200 turns)."""
    return max(2, min(min_turns // 2, 8))


def _f5_time_anchors(plan: InstancePlan, fmap: dict[str, AtomicFact]) -> dict[str, str]:
    """fact_id -> humanized month for F5 event_time targets.

    An F5 event_time request ("delete everything from September 2019") is only
    actionable if the dialogue itself ties the target fact to that month, so
    the mention surface is augmented with the time anchor.
    """
    anchors: dict[str, str] = {}
    for req in plan.requests:
        if (
            req.forget_type == ForgetType.F5
            and req.time_scope
            and req.time_scope.mode == TimeScopeMode.EVENT_TIME
        ):
            for tid in req.target_fact_ids:
                if tid in fmap and fmap[tid].event_time:
                    anchors[tid] = _humanize_month(fmap[tid].event_time)
    return anchors


def _anchored_surface(fact: AtomicFact, anchors: dict[str, str]) -> str:
    """Surface form with the time anchor appended, when the fact is an F5 target."""
    if fact.fact_id in anchors:
        return f"{fact.surface_form.rstrip('.')} (this has been the case since {anchors[fact.fact_id]})."
    return fact.surface_form


def _distribute_mentions(
    plan: InstancePlan,
    profile: UserProfile,
    capacity: int = 4,
) -> list[list[str]]:
    """Assign fact IDs to sessions: each request's targets only in sessions up
    to that request's position (repeated per its mention_count), distractors
    spread over all sessions.

    Mentions are scheduled round-robin (round 0 covers every target before any
    target is repeated), and trimming never drops a fact's only mention — every
    forget target is guaranteed to surface at least once before its request
    (capacity truncation must not silently drop targets, e.g. F3 category
    targets, leaving their probes unanswerable).
    `capacity` stays a soft cap: a session may exceed it when that is the only
    way to keep every target's first mention.

    Slot choice prefers sessions whose theme can host the fact's category
    (THEME_FACT_AFFINITY), so facts surface in scenarios where they read as
    relevant background; unmatched facts keep the plain round-robin order.
    """
    n = plan.num_sessions
    fmap = profile.fact_map()
    mentions: list[list[str]] = [[] for _ in range(n)]
    session_cats = [
        set(THEME_FACT_AFFINITY.get(plan.themes[i], ())) if i < len(plan.themes) else set()
        for i in range(n)
    ]

    def prefer_matched(slots: list[int], fact_id: str) -> list[int]:
        """Stable-partition legal slots: theme-matched sessions first."""
        fact = fmap.get(fact_id)
        if fact is None:
            return slots
        return (
            [s for s in slots if fact.category in session_cats[s]]
            + [s for s in slots if fact.category not in session_cats[s]]
        )
    for req in plan.requests:
        if req.time_scope and req.time_scope.mode == TimeScopeMode.DIALOGUE_TIME:
            # F5 dialogue_time: targets may only be mentioned inside the
            # requested session range (planner guarantees it ends at the
            # request's session, so the request still follows every mention).
            try:
                a, b = int(req.time_scope.start), int(req.time_scope.end)
            except ValueError:
                a, b = 0, req.request_session
            pre = list(range(max(0, a), min(b, req.request_session) + 1)) or [0]
        else:
            pre = list(range(0, req.request_session + 1))  # targets only before the request
        mention_ids = list(req.target_fact_ids)
        if req.forget_type == ForgetType.F4:
            # F4 targets are derived facts: they are never scheduled into the
            # dialogue — the user never states the conclusion itself, it is
            # only inferable from its sources. Only the source facts surface
            # before the request, so the inference is grounded.
            mention_ids = []
            for tid in req.target_fact_ids:
                if tid in fmap:
                    mention_ids.extend(
                        s for s in fmap[tid].source_facts if s not in mention_ids
                    )
        elif not (req.time_scope and req.time_scope.mode == TimeScopeMode.DIALOGUE_TIME):
            # Non-F4: when a target is itself a source of a derived fact
            # (e.g. home address -> commute, the derived_source_quota picks
            # exactly such targets), the derived probes are answerable only
            # if ALL premises were stated, so schedule the derived facts'
            # other sources before the request as well; without them the
            # derived probes are unanswerable even on the control variant.
            # F5 dialogue_time is exempt: its window is being forgotten, so injecting co-premises
            # there would blur the target boundary; the question-side filter
            # skips such probes instead.
            for tid in req.target_fact_ids:
                if tid not in fmap:
                    continue
                for did in fmap[tid].derivations:
                    if did in fmap:
                        mention_ids.extend(
                            s for s in fmap[did].source_facts if s not in mention_ids
                        )
        for k in range(req.mention_count):
            for j, tid in enumerate(mention_ids):
                slots = prefer_matched(pre, tid)
                mentions[slots[(k + j) % len(slots)]].append(tid)
    for k, did in enumerate(plan.distractor_fact_ids):
        order = prefer_matched(list(range(n)), did)
        order = order[k % n:] + order[: k % n]
        for s in order:
            if len(mentions[s]) < capacity:
                mentions[s].append(did)
                break

    def trim(slot: list[str]) -> list[str]:
        """Drop repeats first; never drop a fact's only scheduled mention."""
        if len(slot) <= capacity:
            return slot
        firsts, repeats = [], []
        seen: set[str] = set()
        for fid in slot:
            (firsts if fid not in seen else repeats).append(fid)
            seen.add(fid)
        if len(firsts) >= capacity:
            return firsts
        return firsts + repeats[: capacity - len(firsts)]

    return [trim(slot) for slot in mentions]


# ---------------------------------------------------------------------------
# Multi-agent turn-by-turn simulation
# ---------------------------------------------------------------------------

_REQUEST_PROMPT = """The following is part of an ongoing chat between a user and an \
AI assistant with memory. Write the user's next message: a request to forget \
personal information, in the style "{style}".

Style guide:
- direct: plainly ask to delete/forget it
- polite: courteous, hedged, appreciative, but still clearly asking for removal
- indirect: describe a concern or situation instead of giving a command, but \
end with an unmistakable expectation that the information is no longer kept
- imperative: firm, commanding, non-negotiable
- implicit: avoid delete/forget/remove vocabulary entirely, but state the \
desired end state — that the assistant must not keep the information anymore

Every style must unambiguously require that the information is no longer \
retained. The styles differ only in tone and wording, never in whether \
removal is requested.

The request must convey this meaning (do not copy it verbatim): {skeleton}

Important: if that meaning includes a new value replacing an old one (e.g. \
"changed from X to Y"), the request MUST explicitly state the new value Y — \
never leave it implied or refer to it vaguely.

Write it the way a real person types in a chat app: casual wording, no em dashes, \
no formal essay-style phrasing.

Recent conversation context:
{context}

Return ONLY a JSON object:
{{"user_request": "<the user's message, 1-3 sentences>", \
"assistant_ack": "<the assistant's brief confirmation that the information has been \
forgotten/deleted; it must NOT restate the sensitive value>"}}"""

_FILLER_PROMPT = """Write one short, casual small-talk exchange between a user and \
an AI assistant about the topic "{topic}". It must not touch any personal facts \
about the user (no addresses, health, finance, career, education, relationship, \
preference, or routine details). The user's message should be roughly {n_chars} \
characters long. Keep both turns in casual chat tone: no em dashes, no polished \
essay-style phrasing.

Return ONLY a JSON object:
{{"user_message": "...", "assistant_reply": "<brief friendly response>"}}"""

_FILLER_TOPICS = [
    "a documentary the user watched", "the weather this week", "a new coffee shop",
    "weekend hiking plans in general", "a book the user is reading",
    "a recipe the user tried", "a podcast episode", "gardening tips",
    "a museum exhibition", "an interesting news story about space",
]

_SCRUB_PROMPT = """Rewrite the following {speaker} turn from an ongoing chat so that it \
does NOT state or imply this piece of information about the user: {surface}

The rewritten text MUST NOT contain the exact string "{value}" or any close \
variant of it — drop or replace that mention entirely, even if it seems \
on-topic. Keep the rest of the turn's meaning and the conversational flow. \
If the turn is entirely about that information, replace it with a brief \
on-topic remark that moves the conversation forward. Return only the \
rewritten text.

Turn: {text}"""

_GROUNDING_PROMPT = """A user in a chat simulation was asked to convey this personal fact: {surface}

The user's message: {text}

Does the message actually state the fact's core information (the who/what/\
where/when of it), so that a reader would learn it from the message alone? \
Mere allusion (a name, a place, a partial detail without the substance) does \
NOT count. Return ONLY JSON: {{"grounded": true/false}}"""


def _grounding_confirmed(llm: BaseLLM, fact: AtomicFact, text: str) -> bool:
    """LLM adjudicator for self-reported tags the `_fact_surfaced` heuristic
    misses. Raises on LLM failure: grounding decides whether retention probes
    are answerable, so an undecidable tag must not silently pass or fail."""
    raw = parse_json(llm.chat(
        [{"role": "user", "content": _GROUNDING_PROMPT.format(
            surface=fact.surface_form, text=text)}],
        temperature=0.0,
        max_tokens=256,
        task="grounding_check",
    ))
    return bool(raw.get("grounded", False))


def _simulate_sessions(
    plan: InstancePlan,
    profile: UserProfile,
    llm: BaseLLM,
    rng: random.Random,
    config: dict[str, Any],
    instance_id: str,
    timeline: _Timeline,
) -> list[Session]:
    """Generate sessions via multi-agent turn-by-turn simulation."""
    fmap = profile.fact_map()
    min_t = plan.turns_per_session[0]
    capacity = _fact_capacity(min_t)
    mentions = _distribute_mentions(plan, profile, capacity=capacity)
    anchors = _f5_time_anchors(plan, fmap)

    def with_anchor(fact: AtomicFact) -> AtomicFact:
        """Give the user simulator the time-anchored surface for F5 targets."""
        if fact.fact_id not in anchors:
            return fact
        return fact.model_copy(
            update={"surface_form": _anchored_surface(fact, anchors)}
        )

    sessions: list[Session] = []
    critic = config.get("generation", {}).get("critic", False)
    for i in tqdm(range(plan.num_sessions), desc=f"sessions:{instance_id}",
                  unit="session", leave=False):
        facts = {fid: with_anchor(fmap[fid]) for fid in mentions[i] if fid in fmap}
        turns = _simulate_session(plan, profile, facts, llm, rng, config, instance_id, i,
                                  timeline)
        if critic:
            issues = verify_session(
                llm, profile.persona_summary, [(t.speaker, t.text) for t in turns],
                dated_facts=[f.surface_form for f in facts.values() if f.event_time],
            )
            if issues["issues"]:
                print(f"  verifier issues for {instance_id} s{i}: {issues['issues']}")
                # Bounded regeneration: re-run the director loop once and keep
                # the second attempt regardless.
                turns = _simulate_session(plan, profile, facts, llm, rng, config,
                                          instance_id, i, timeline)
                print(f"  regenerated {instance_id} s{i} after verifier issues")
        sessions.append(Session(
            session_id=f"{instance_id}:s{i}",
            theme=plan.themes[i],
            scenario=plan.scenarios[i] if i < len(plan.scenarios) else "",
            turns=turns,
        ))
    return sessions


def _simulate_session(
    plan: InstancePlan,
    profile: UserProfile,
    facts: dict[str, AtomicFact],
    llm: BaseLLM,
    rng: random.Random,
    config: dict[str, Any],
    instance_id: str,
    session_index: int,
    timeline: _Timeline,
) -> list[Turn]:
    """Director loop for one session: user simulator and assistant alternate.

    The director tracks the per-session fact checklist; once past 60% of the
    target length it nudges the user simulator to weave in remaining facts.
    The assistant sees only a sliding window of the current session (no
    persistent memory).
    """
    gen_cfg = config.get("generation", {})
    temperature = gen_cfg.get("temperature", 0.9)
    window_turns = gen_cfg.get("window_turns", 10)
    min_t, max_t = plan.turns_per_session
    target_turns = rng.randint(min_t, max_t)
    hard_cap = max_t + 4  # headroom for nudged facts

    user_sim = UserSimulator(llm, temperature=temperature)
    assistant = AssistantAgent(llm, temperature=max(0.0, temperature - 0.2))
    scenario = plan.scenarios[session_index] if session_index < len(plan.scenarios) else ""

    remaining: dict[str, str] = {fid: f.surface_form for fid, f in facts.items()}
    revealed: dict[str, str] = {}
    turns: list[Turn] = []

    def produce_turn(feedback: str | None = None,
                     closing: bool = False) -> tuple[str, set[str], set[str], set[str]]:
        """One user turn: its grounded tagged fact set, the self-reported tags
        that failed grounding, and the already-revealed facts it echoes.

        A fact counts as conveyed only when the text itself surfaces it: the
        simulator's self-report is not trusted on its own (it tags facts the
        message merely alludes to, e.g. "at Seton" for a works-as-nurse-at-Seton
        fact, leaving retention probes unanswerable). Grounding is the
        `_fact_surfaced` heuristic, with an LLM adjudicator for self-reported
        tags the heuristic misses: the heuristic checks surface/value wording,
        but a faithful paraphrase can share too few content words with either
        (e.g. value "monthly to a 403(b) retirement plan" vs. text "I put a
        bit of each paycheck into my 403(b)"), and wrongly rejecting those
        leaves the fact in `remaining` so the nudge re-pushes it every turn —
        the fact-echo loop this check exists to prevent.
        """
        turn_month = _humanize_month(timeline.ts(session_index, len(turns), rng)[:7])
        text, self_reported = user_sim.next_turn(
            profile.persona_summary, scenario, remaining, history, nudge,
            current_month=turn_month, feedback=feedback, closing=closing,
            revealed_facts=revealed,
        )
        tags = {fid for fid in remaining if _fact_surfaced(facts[fid], [text])}
        ungrounded = (set(self_reported) & set(remaining)) - tags
        confirmed = {
            fid for fid in ungrounded
            if _grounding_confirmed(llm, facts[fid], text)
        }
        tags |= confirmed
        ungrounded -= confirmed
        echo = {fid for fid in revealed if _fact_surfaced(facts[fid], [text])}
        return text, tags, ungrounded, echo

    while len(turns) < hard_cap:
        history = [(t.speaker, t.text) for t in turns]
        at_target = len(turns) >= target_turns
        closing = at_target and not remaining
        nudge = None
        if remaining and len(turns) >= int(target_turns * 0.6):
            fid = next(iter(remaining))
            nudge = (fid, remaining[fid])
        # next_turn / reply raise on LLM failure: the caller's instance-level
        # retry regenerates the instance instead of keeping a truncated
        # session or substituting a canned reply.
        #
        # Temporal backstop: a turn that surfaces a fact whose value carries a
        # stated year must keep that year (never "just got married" for a 2024
        # wedding in a 2026 conversation). Retry the turn with the violation
        # fed back into the prompt; only a persistent violation fails the
        # instance, so a one-off slip does not discard a whole simulation.
        # Departure backstop: before the session is done, a user turn that ends
        # the conversation (a goodbye / "I'll catch you later") is rejected so
        # the simulator keeps the chat productive instead of looping on a
        # sign-off.
        # Grounding backstop: a self-reported tag whose value the text does not
        # actually state is rejected and fed back; if the violation persists the
        # tag is dropped and the fact stays in `remaining` for a later turn.
        # Echo backstop: a fact already revealed this session must not be
        # brought up again (scheduled repetition only ever spans sessions, so
        # an in-session re-mention is always an unplanned history echo). Retry
        # with feedback; a persistent echo fails the instance like a temporal
        # violation, instead of keeping a session with a looping user.
        feedback = None
        conflicting: list[AtomicFact] = []
        ungrounded: set[str] = set()
        echo: set[str] = set()
        for _ in range(_TEMPORAL_RETRIES):
            text, tags, ungrounded, echo = produce_turn(feedback, closing=closing)
            # Temporal check only constrains grounding-level reveals (tags):
            # an allusion too thin to ground the tag ("training for the
            # half-marathon", no year) is natural chat and must not be forced
            # to carry the date.
            conflicting = [
                f for f in (facts[fid] for fid in tags)
                if _turn_drops_stated_year(text, f)
            ]
            if conflicting:
                feedback = (
                    "Your previous message was rejected: it talks about "
                    + ", ".join(f'"{f.value}"' for f in conflicting)
                    + " but drops the stated year. Rewrite it keeping the exact "
                      "year (and month, if stated) of every dated fact."
                )
                continue
            if echo:
                feedback = (
                    "Your previous message was rejected: you already shared "
                    + ", ".join(f'"{revealed[fid]}"' for fid in sorted(echo))
                    + " earlier in this conversation. Do not bring it up again, "
                      "not even in different words; move the conversation "
                      "forward instead."
                )
                continue
            if not closing and _is_farewell(text) and (remaining or not at_target):
                feedback = (
                    "Your previous message ends the conversation, but the "
                    "conversation is not over yet. Do not say goodbye or "
                    "announce that you are leaving. Keep the conversation "
                    "going: continue the task, ask a follow-up question, or "
                    "bring up something new."
                )
                continue
            if ungrounded:
                feedback = (
                    "Your previous message was rejected: you tagged "
                    + ", ".join(f'"{remaining[fid]}"' for fid in sorted(ungrounded))
                    + " as conveyed, but the message never actually states the "
                      "fact's value. Rewrite it so the stated values appear "
                      "exactly, or leave the fact for a later message."
                )
                continue
            break
        if conflicting:
            raise RuntimeError(
                f"user turn dropped the stated year of {[f.value for f in conflicting]}"
            )
        if echo:
            raise RuntimeError(
                f"user turn keeps echoing revealed facts {[revealed[f] for f in sorted(echo)]}"
            )
        if ungrounded:
            print(f"  warning: {instance_id} s{session_index} dropped ungrounded tags "
                  f"{sorted(ungrounded)}")
        turns.append(Turn(
            speaker="user", text=text,
            timestamp=timeline.ts(session_index, len(turns), rng),
            linked_fact_ids=sorted(tags),
        ))
        for fid in tags:
            revealed[fid] = remaining.pop(fid)

        reply = assistant.reply([(t.speaker, t.text) for t in turns[-window_turns:]])
        turns.append(Turn(
            speaker="assistant", text=reply,
            timestamp=timeline.ts(session_index, len(turns), rng),
        ))
        if len(turns) >= target_turns and not remaining:
            break

    if remaining:
        print(f"  warning: {instance_id} s{session_index} missed facts {sorted(remaining)}")
    return turns


_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


_FAREWELL_RE = re.compile(
    r"\b(?:catch you (?:later|then|after)|"
    r"(?:talk|see) you (?:later|after|then|next week)|"
    r"be in touch|"
    r"(?:get going|gotta run|better be going|heading off|head off|"
    r"head(?:ing)? out|be out the door|be right back|take care|goodbye)|"
    r"(?:i'll|i will) (?:let you know|shoot you|send you|catch you|"
    r"talk to you|be in touch|see you|get back|be out|be right back))\b",
    re.I,
)


def _is_farewell(text: str) -> bool:
    """True when a user turn clearly ends the conversation (a departure/sign-off).

    Used as a deterministic backstop against the sign-off mode collapse where
    the simulator, having nothing left to say, keeps re-announcing that it is
    leaving. Only unambiguous closing phrases count, so an ordinary "I'll set
    up the reminders tonight" plan never trips it.
    """
    return bool(_FAREWELL_RE.search(text))


# Words too generic to evidence that a turn is talking about a dated fact:
# "since"/"half"/"reading" appear everywhere in ordinary chat, so a turn
# containing them without the fact's year must not be judged as dropping it.
_TEMPORAL_GENERIC = _GENERIC_TOKENS | frozenset({
    "since", "half", "reading", "back", "just", "year", "years", "ago",
})


def _turn_drops_stated_year(text: str, fact: AtomicFact) -> bool:
    """True when a turn reveals a dated fact but omits its stated year.

    Fires only on a grounding-level mention: the turn must hit at least two
    distinctive content tokens of the fact (exact token matching with
    morphological variants; generic words excluded). The previous
    substring/single-word trigger was a systematic false-positive channel —
    any turn containing "since" without "2021" was judged as dropping the
    year of "reading glasses since 2021", and the retry feedback could not
    correct a violation that never happened, killing whole instances.
    """
    year = _YEAR_RE.search(fact.value)
    if not year:
        return False
    if year.group(0) in text:
        return False
    words = {
        w
        for w in re.findall(
            r"[a-z0-9]+",
            f"{fact.predicate} {fact.value} {fact.surface_form}".lower(),
        )
        if len(w) >= 4 and not w.isdigit() and w not in _TEMPORAL_GENERIC
    }
    tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    hits = sum(bool(_word_variants(w) & tokens) for w in words)
    return hits >= 2


def _word_variants(w: str) -> set[str]:
    """Crude morphological variants so inflections match: the simulator often
    paraphrases a value word ("documentaries" -> "a documentary")."""
    vs = {w}
    if w.endswith("ies") and len(w) > 4:
        vs.add(w[:-3] + "y")
    if w.endswith("es"):
        vs.add(w[:-2])
        vs.add(w[:-1])
    elif w.endswith("s") and not w.endswith("ss"):
        vs.add(w[:-1])
    else:
        vs.add(w + "s")
    return vs


def _content_words(text: str) -> set[str]:
    return {
        w
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if len(w) >= 4 and not w.isdigit()
    }


def _overlap_ratio(words: set[str], tokens: set[str]) -> float:
    if not words:
        return 0.0
    hits = sum(bool(_word_variants(w) & tokens) for w in words)
    return hits / len(words)


def _fact_surfaced(fact: AtomicFact, user_texts: list[str]) -> bool:
    """Heuristic grounding check: the fact's value or surface form appears
    verbatim, as a distinctive fragment, or via most of its content words.

    Both the value and the surface form are tried: the simulator is shown the
    surface form and often copies it nearly verbatim while paraphrasing away
    the value's own wording (e.g. surface "I put a bit of each paycheck into
    my 403(b)" for value "monthly to a 403(b) retirement plan"), so checking
    the value alone systematically misses genuine mentions.
    """
    joined = " ".join(user_texts).lower()
    for surface in {fact.value, fact.surface_form}:
        s = surface.lower().strip()
        if s and s in joined:
            return True
        # Accept long distinctive fragments (e.g. street name without number).
        for frag in re.split(r"[,;/] ", surface):
            frag = frag.lower().strip()
            if len(frag) >= 8 and frag in joined:
                return True
    tokens = set(re.findall(r"[a-z0-9]+", joined))
    # Accept paraphrases: most content words of the value + predicate, or of
    # the surface form, are present (morphological variants count).
    return max(
        _overlap_ratio(_content_words(f"{fact.predicate} {fact.value}"), tokens),
        _overlap_ratio(_content_words(fact.surface_form), tokens),
    ) >= 0.6


def _scrub_post_request_leaks(
    sessions: list[Session],
    control_sessions: list[Session],
    plan: InstancePlan,
    profile: UserProfile,
    positions: list[tuple[int, int]],
    llm: BaseLLM,
    config: dict[str, Any],
) -> None:
    """Rewrite turns after each forget request that restate one of its targets.

    Scanning starts right after the request's injected pair and covers all
    later turns (including later sessions); injected request/ack/filler turns
    of every request are skipped. The same edit is applied to both variants so
    they stay identical outside the injected exchanges.
    """
    fmap = profile.fact_map()
    injected = {(s, t) for s, t in positions} | {(s, t + 1) for s, t in positions}
    temperature = config.get("generation", {}).get("temperature", 0.9)
    for req, (start_session, turn_index) in zip(plan.requests, positions, strict=True):
        target_ids = list(req.target_fact_ids)
        targets = [fmap[tid] for tid in target_ids if tid in fmap]
        if not targets:
            continue
        for si in range(start_session, len(sessions)):
            start = turn_index + 2 if si == start_session else 0
            for ti in range(start, len(sessions[si].turns)):
                if (si, ti) in injected:
                    continue
                turn = sessions[si].turns[ti]
                # A turn may mention several targets of the same request: each
                # one gets its own rewrite, and later facts are checked against
                # the already-rewritten text (a rewrite can incidentally drop
                # a fact it was not aimed at).
                for fact in targets:
                    if not _fact_surfaced(fact, [turn.text]):
                        continue
                    # Up to two rewrite attempts; the prompt bans the literal
                    # value, but verify anyway and retry once if it survives.
                    # A turn that still leaks after the attempts (or a rewrite
                    # call that keeps failing) fails the instance: the
                    # instance-level retry regenerates it instead of shipping
                    # a post-request leak.
                    out = ""
                    last_err: Exception | None = None
                    for attempt in range(2):
                        try:
                            out = llm.chat(
                                [{"role": "user", "content": _SCRUB_PROMPT.format(
                                    speaker=turn.speaker, surface=fact.surface_form,
                                    value=fact.value, text=turn.text)}],
                                temperature=temperature,
                                max_tokens=1024,
                                task=f"scrub:{si}:{ti}",
                            ).strip()
                        except Exception as e:  # noqa: BLE001 - retried below
                            out = ""
                            last_err = e
                        if not out or not _fact_surfaced(fact, [out]):
                            break
                    if not out:
                        raise RuntimeError(
                            f"scrub rewrite failed at s{si} turn {ti} "
                            f"(last error: {last_err})"
                        )
                    if turn.speaker == "user":
                        out = sanitize_chat_text(out)
                    if _fact_surfaced(fact, [out]):
                        raise RuntimeError(
                            f"scrub at s{si} turn {ti} still contains target value "
                            f"{fact.value!r} after retries"
                        )
                    sessions[si].turns[ti].text = out
                    control_sessions[si].turns[ti].text = out
                    print(f"  scrubbed post-request target mention at s{si} turn {ti}")


def _generate_request_exchange(
    req: RequestPlan,
    plan: InstancePlan,
    profile: UserProfile,
    base_sessions: list[Session],
    llm: BaseLLM,
    config: dict[str, Any],
) -> tuple[str, str]:
    """Generate the in-context forget request and the assistant acknowledgment.

    Raises on LLM failure or on output that drops required content (the F6
    replacement value, the F5 time scope, or — for any request — an ambiguous
    removal intent per the LLM intent verifier): the instance is retried and
    finally skipped instead of falling back to the template skeleton or a
    canned ack.
    """
    gen_cfg = config.get("generation", {})
    skeleton = render_request_text(req, profile)
    session = base_sessions[req.request_session]
    context = "\n".join(f"{t.speaker}: {t.text}" for t in session.turns[-6:])
    raw = llm.chat_json(
        [{"role": "user", "content": _REQUEST_PROMPT.format(
            style=req.style.value, skeleton=skeleton, context=context)}],
        temperature=gen_cfg.get("temperature", 0.9),
        task=f"request:{plan.index}:{req.request_session}",
    )
    user_req = sanitize_chat_text(str(raw.get("user_request", "")))
    ack = str(raw.get("assistant_ack", "")).strip()
    if not (user_req and ack):
        raise ValueError(
            f"request exchange generation returned empty output "
            f"(request {req.forget_type}/{req.style.value})"
        )
    if req.forget_type == ForgetType.F6 and req.replacement:
        # The new value must be stated in the request itself, otherwise the
        # memory system cannot learn it and the new_value probe is
        # unanswerable. A request that drops it is a failed generation.
        if req.replacement.lower() not in user_req.lower():
            raise ValueError("F6 request dropped the replacement value")
    if req.forget_type == ForgetType.F5 and req.time_scope:
        # The request must keep the time scope anchor, whatever the style.
        scope = req.time_scope
        if scope.mode == TimeScopeMode.EVENT_TIME:
            has_anchor = _humanize_month(scope.start).lower() in user_req.lower()
        else:
            # Dialogue-time scope: both session indices must appear as
            # standalone tokens — a bare substring check would pass on any
            # number containing the digit (e.g. "2020" for session "0").
            has_anchor = all(
                re.search(rf"\b{re.escape(v)}\b", user_req)
                for v in {scope.start, scope.end}
            )
        if not has_anchor:
            raise ValueError("F5 request dropped the time scope")
    # Removal intent must be unambiguous in every style. A paraphrase that
    # only remarks the information is outdated is not executable; keyword
    # lists cannot judge implicit style, so an LLM verifies the intent.
    verdict = llm.chat_json(
        [{"role": "user", "content": _INTENT_PROMPT.format(
            request=user_req, targets=skeleton)}],
        temperature=0.0,
        task=f"request_intent:{plan.index}:{req.request_session}",
    )
    if not bool(verdict.get("clear")):
        raise ValueError(
            f"request lacks an unambiguous removal intent "
            f"(request {req.forget_type}/{req.style.value})"
        )
    return user_req, ack


def _generate_filler_exchange(
    request_text: str,
    plan: InstancePlan,
    profile: UserProfile,
    llm: BaseLLM,
    config: dict[str, Any],
    request_index: int,
) -> tuple[str, str]:
    """Generate a varied neutral small-talk exchange for the control variant.

    Raises on LLM failure or empty output: the control filler is part of the
    instance, so a failed generation retries the instance instead of
    substituting the fixed template filler.
    """
    gen_cfg = config.get("generation", {})
    rng = random.Random(f"filler:{profile.profile_id}:{plan.index}:{request_index}")
    topic = rng.choice(_FILLER_TOPICS)
    raw = llm.chat_json(
        [{"role": "user", "content": _FILLER_PROMPT.format(
            topic=topic, n_chars=len(request_text))}],
        temperature=gen_cfg.get("temperature", 0.9),
        task=f"filler:{plan.index}:{request_index}",
    )
    user_msg = sanitize_chat_text(str(raw.get("user_message", "")))
    reply = str(raw.get("assistant_reply", "")).strip()
    if not (user_msg and reply):
        raise ValueError(
            f"filler exchange generation returned empty output "
            f"(request {request_index})"
        )
    return user_msg, reply
