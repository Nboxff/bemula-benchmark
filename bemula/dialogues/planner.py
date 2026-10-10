"""Dialogue planning.

Samples the mentioned-fact set (per-request forget targets + distractors),
plans themed sessions, and fixes each forget request's type, target, position,
and style. Difficulty dimension values are recorded per instance (aggregates)
and per request (style/position/mention count).
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from bemula.llm import BaseLLM
from bemula.schemas import (
    AtomicFact,
    FactCategory,
    ForgetType,
    RequestStyle,
    TimeScope,
    TimeScopeMode,
    UserProfile,
)

# Session scenarios: (theme, user goal). In the LLM path each session is driven by
# a concrete goal the user brings to the assistant; facts surface as background
# information needed for the task instead of being recited.
#
# 50 scenarios: the pool matches the longest
# session count (xlong: up to 50 sessions), so an instance never reuses a
# scenario internally, and every theme appears far less often across the
# formal-scale dataset. Themes deliberately span the everyday domains where
# users most often disclose personal information to chat assistants (health,
# finance, relationships, career, education, hobbies), so each of the 12 fact
# categories can surface naturally as task background.
SESSION_SCENARIOS: list[tuple[str, str]] = [
    ("home renovation",
     "The user wants the assistant's help planning and budgeting a home renovation project."),
    ("medical check-up",
     "The user is preparing for a medical appointment and wants help organizing what to "
     "discuss with the doctor."),
    ("job change",
     "The user is considering a job change and asks the assistant for advice on the "
     "decision and on negotiating."),
    ("travel planning",
     "The user asks the assistant to help plan an upcoming trip (itinerary, bookings, "
     "packing)."),
    ("weekend routine",
     "The user wants suggestions for organizing their weekend and free time better."),
    ("finance review",
     "The user asks the assistant to help review their household finances and savings "
     "plans."),
    ("school reunion",
     "The user is getting ready for a school-related event (reunion, application, "
     "alumni activity) and wants help preparing."),
    ("pet care",
     "The user asks the assistant for advice on caring for or training a pet, and other "
     "pet-related plans."),
    ("relationship advice",
     "The user asks the assistant for advice about a personal relationship, such as "
     "dating or a conflict with their partner."),
    ("family gathering",
     "The user is organizing a family gathering or celebration and wants planning help."),
    ("moving logistics",
     "The user is dealing with logistics around housing (moving, mail, utilities) and "
     "asks for help."),
    ("software side project",
     "The user is working on a personal software project (a script, an app, a website) "
     "and asks the assistant for coding help, debugging, or design advice."),
    ("fitness planning",
     "The user wants the assistant to help build or adjust an exercise and fitness plan."),
    ("gift shopping",
     "The user asks the assistant for gift ideas and help choosing presents for family "
     "or friends."),
    ("learning plan",
     "The user wants to pick up a new skill or subject and asks for a study plan and "
     "resource recommendations."),
    ("college coursework",
     "The user is a student and asks the assistant for help planning coursework, "
     "assignments, and exam preparation."),
    ("home cooking",
     "The user asks for meal planning, weekly meal-prep ideas, recipes, or help "
     "organizing a cooking routine."),
    ("retirement planning",
     "The user is planning for retirement and asks the assistant for help projecting "
     "savings, setting contribution levels, and reviewing pension options."),
    ("car maintenance",
     "The user asks the assistant for help scheduling and budgeting routine car "
     "maintenance and repairs."),
    ("insurance review",
     "The user wants to review their insurance coverage (home, health, auto, life) and "
     "asks the assistant for help comparing policies and costs."),
    ("freelance tax filing",
     "The user has freelance income and asks the assistant for help organizing expenses, "
     "deductions, and filing taxes."),
    ("visa application",
     "The user is preparing a visa or immigration application and asks the assistant "
     "for help organizing documents and requirements."),
    ("home security setup",
     "The user is setting up a home security system and asks the assistant for advice on "
     "devices, installation, and budgeting."),
    ("gardening",
     "The user asks the assistant for help planning a garden or caring for indoor "
     "plants: what to plant, when, watering, light, and troubleshooting."),
    ("DIY craft project",
     "The user is working on a DIY craft or woodworking project and asks the assistant "
     "for help with materials, steps, and tools."),
    ("music practice plan",
     "The user wants a structured practice plan for learning or improving on a musical "
     "instrument."),
    ("hiking trip planning",
     "The user asks the assistant to help plan a hiking or camping trip: route, gear, "
     "food, and preparation."),
    ("volunteering coordination",
     "The user is organizing a volunteering activity and asks the assistant for help "
     "coordinating people, logistics, and schedules."),
    ("anniversary celebration",
     "The user is planning an anniversary celebration and asks the assistant for ideas, "
     "budgeting, and arrangements."),
    ("kids' school project",
     "The user asks the assistant for help with a child's school project: topic, "
     "research, and presentation."),
    ("elderly parent care",
     "The user is coordinating care for an aging parent and asks the assistant for help "
     "with appointments, logistics, and records."),
    ("digital declutter",
     "The user wants to organize and declutter their digital files, photos, and email, "
     "and asks the assistant for a plan."),
    ("new device setup",
     "The user just got a new device and asks the assistant for help with setup, data "
     "migration, and configuration."),
    ("language exchange",
     "The user is learning a language and asks the assistant for a study plan and "
     "conversation practice tips."),
    ("community class enrollment",
     "The user wants to pick a community class or workshop and asks the assistant for "
     "help choosing and enrolling."),
    ("emergency preparedness",
     "The user wants to build a household emergency kit and plan, and asks the assistant "
     "for a checklist."),
    ("career networking",
     "The user wants to expand their professional network and asks the assistant for "
     "outreach and event-planning help."),
    ("resume update",
     "The user is updating their resume and portfolio and asks the assistant for help "
     "structuring and wording them."),
    ("personal statement writing",
     "The user is writing a personal statement or application essay and asks the "
     "assistant for help drafting and revising it."),
    ("salary negotiation",
     "The user is preparing to negotiate a salary and asks the assistant for a strategy "
     "and talking points."),
    ("home office setup",
     "The user is setting up a home office and asks the assistant for help with the "
     "desk, equipment, and a daily routine."),
    ("photography outing",
     "The user is planning a photography outing and asks the assistant for location, "
     "timing, and gear tips."),
    ("game night hosting",
     "The user is hosting a game night and asks the assistant for game picks, snacks, "
     "and planning help."),
    ("charity fundraiser",
     "The user is organizing a charity fundraiser and asks the assistant for help with "
     "planning, promotion, and budget."),
    ("housewarming party",
     "The user is planning a housewarming party and asks the assistant for help with the "
     "guest list, food, and logistics."),
    ("energy bill reduction",
     "The user wants to lower their home energy bills and asks the assistant for a "
     "practical audit and plan."),
    ("sleep improvement",
     "The user wants to improve their sleep and asks the assistant for help building a "
     "bedtime routine and better habits."),
    ("mental health check-in",
     "The user has been feeling stressed or low and asks the assistant for help "
     "building coping routines and healthier habits."),
    ("medication management",
     "The user wants to organize their medications and asks the assistant for help with "
     "schedules and reminders."),
    ("marathon training",
     "The user signed up for a marathon and asks the assistant for a training plan, "
     "nutrition, and gear advice."),
]

# Theme list used for Session.theme metadata.
SESSION_THEMES = [theme for theme, _ in SESSION_SCENARIOS]

# Which fact categories each scenario theme can host naturally. Mention
# scheduling prefers sessions whose theme fits the fact's category, so facts
# surface as relevant background instead of being forced into unrelated
# scenarios (e.g. a cooking diploma in a device-setup chat).
THEME_FACT_AFFINITY: dict[str, tuple[FactCategory, ...]] = {
    "home renovation": (FactCategory.ADDRESS, FactCategory.FINANCE,
                        FactCategory.PREFERENCE, FactCategory.EVENT),
    "medical check-up": (FactCategory.HEALTH, FactCategory.ROUTINE),
    "job change": (FactCategory.CAREER, FactCategory.FINANCE,
                   FactCategory.EDUCATION, FactCategory.EVENT),
    "travel planning": (FactCategory.EVENT, FactCategory.PREFERENCE,
                        FactCategory.HOBBY),
    "weekend routine": (FactCategory.ROUTINE, FactCategory.HOBBY,
                        FactCategory.PREFERENCE),
    "finance review": (FactCategory.FINANCE, FactCategory.CAREER),
    "school reunion": (FactCategory.EDUCATION, FactCategory.SOCIAL,
                       FactCategory.RELATIONSHIP, FactCategory.EVENT),
    "pet care": (FactCategory.ROUTINE, FactCategory.HOBBY,
                 FactCategory.PREFERENCE),
    "relationship advice": (FactCategory.RELATIONSHIP, FactCategory.SOCIAL),
    "family gathering": (FactCategory.RELATIONSHIP, FactCategory.EVENT,
                         FactCategory.SOCIAL),
    "moving logistics": (FactCategory.ADDRESS, FactCategory.EVENT,
                         FactCategory.FINANCE),
    "software side project": (FactCategory.CAREER, FactCategory.HOBBY,
                              FactCategory.EDUCATION),
    "fitness planning": (FactCategory.HEALTH, FactCategory.ROUTINE,
                         FactCategory.HOBBY),
    "gift shopping": (FactCategory.RELATIONSHIP, FactCategory.SOCIAL,
                      FactCategory.PREFERENCE, FactCategory.FINANCE),
    "learning plan": (FactCategory.EDUCATION, FactCategory.HOBBY,
                      FactCategory.ROUTINE),
    "college coursework": (FactCategory.EDUCATION, FactCategory.ROUTINE),
    "home cooking": (FactCategory.HOBBY, FactCategory.PREFERENCE,
                     FactCategory.EDUCATION, FactCategory.ROUTINE),
    "retirement planning": (FactCategory.FINANCE, FactCategory.CAREER),
    "car maintenance": (FactCategory.ROUTINE, FactCategory.FINANCE,
                        FactCategory.EVENT),
    "insurance review": (FactCategory.FINANCE, FactCategory.HEALTH),
    "freelance tax filing": (FactCategory.FINANCE, FactCategory.CAREER),
    "visa application": (FactCategory.ADDRESS, FactCategory.EVENT,
                         FactCategory.EDUCATION),
    "home security setup": (FactCategory.ADDRESS, FactCategory.PREFERENCE),
    "gardening": (FactCategory.HOBBY, FactCategory.ROUTINE),
    "DIY craft project": (FactCategory.HOBBY, FactCategory.PREFERENCE),
    "music practice plan": (FactCategory.HOBBY, FactCategory.ROUTINE,
                            FactCategory.EDUCATION),
    "hiking trip planning": (FactCategory.HOBBY, FactCategory.EVENT,
                             FactCategory.HEALTH),
    "volunteering coordination": (FactCategory.SOCIAL, FactCategory.EVENT,
                                  FactCategory.ROUTINE),
    "anniversary celebration": (FactCategory.RELATIONSHIP, FactCategory.EVENT),
    "kids' school project": (FactCategory.RELATIONSHIP, FactCategory.EDUCATION,
                             FactCategory.ROUTINE),
    "elderly parent care": (FactCategory.RELATIONSHIP, FactCategory.HEALTH,
                            FactCategory.ROUTINE),
    "digital declutter": (FactCategory.ROUTINE, FactCategory.PREFERENCE),
    "new device setup": (FactCategory.PREFERENCE, FactCategory.ROUTINE),
    "language exchange": (FactCategory.EDUCATION, FactCategory.SOCIAL,
                          FactCategory.HOBBY),
    "community class enrollment": (FactCategory.HOBBY, FactCategory.EDUCATION,
                                   FactCategory.SOCIAL),
    "emergency preparedness": (FactCategory.ADDRESS, FactCategory.HEALTH,
                               FactCategory.FINANCE),
    "career networking": (FactCategory.CAREER, FactCategory.SOCIAL),
    "resume update": (FactCategory.CAREER, FactCategory.EDUCATION),
    "personal statement writing": (FactCategory.EDUCATION, FactCategory.CAREER),
    "salary negotiation": (FactCategory.CAREER, FactCategory.FINANCE),
    "home office setup": (FactCategory.CAREER, FactCategory.ADDRESS,
                          FactCategory.PREFERENCE),
    "photography outing": (FactCategory.HOBBY, FactCategory.EVENT),
    "game night hosting": (FactCategory.SOCIAL, FactCategory.HOBBY,
                           FactCategory.PREFERENCE),
    "charity fundraiser": (FactCategory.SOCIAL, FactCategory.EVENT,
                           FactCategory.FINANCE),
    "housewarming party": (FactCategory.ADDRESS, FactCategory.SOCIAL,
                           FactCategory.EVENT),
    "energy bill reduction": (FactCategory.FINANCE, FactCategory.ADDRESS,
                              FactCategory.ROUTINE),
    "sleep improvement": (FactCategory.HEALTH, FactCategory.ROUTINE),
    "mental health check-in": (FactCategory.HEALTH, FactCategory.ROUTINE),
    "medication management": (FactCategory.HEALTH, FactCategory.ROUTINE),
    "marathon training": (FactCategory.HEALTH, FactCategory.HOBBY,
                          FactCategory.EVENT, FactCategory.ROUTINE),
}

assert set(THEME_FACT_AFFINITY) == set(SESSION_THEMES), (
    "every scenario theme must have a fact-category affinity entry"
)

_LENGTH_BUCKETS = ["short", "medium", "long"]
_STYLES = list(RequestStyle)

# Bounded retries for LLM-generated F6 replacement values before failing.
_REPLACEMENT_RETRIES = 3

_REPLACEMENT_PROMPT = """The user profile contains this fact: {surface}

The user has changed their mind about "{predicate}" (category: {category}) and \
wants to replace the old value. The new value must fill the SAME slot as the \
old one: if the old value is a course name, give a course name; if it is a \
food, give a food; if it is a place, give a place. It must be a short, \
concrete noun phrase — never a full sentence, never a preference statement, \
never a description of how the user feels.
Return ONLY JSON: {{"new_value": "..."}}"""

# Leading tokens that mark a full sentence rather than a value. A replacement
# value must be a noun phrase ("Cantonese", "an online machine learning
# certificate"), never a sentence ("I prefer hands-on projects over formal
# courses...") — a sentence-shaped replacement does not fill the old value's
# slot, so the F6 new_value probe (whose gold answer is the replacement) would
# be unanswerable.
_SENTENCE_OPENERS = (
    "i ", "i'd ", "i'll ", "i'm ", "i've ", "he ", "she ", "we ", "they ",
    "it ", "the ", "my ", "me ", "people ", "you ", "that ", "this ", "there ",
)
_SENTENCE_END_PUNCT = (".", "!", "?")


def is_clean_value(text: str | None) -> bool:
    """True when `text` is a usable F6 replacement *value* (not a sentence).

    Rejects empty/over-long strings, sentence-shaped outputs (leading personal
    pronouns or sentence-final punctuation), placeholder-ish strings, and
    anything containing the old placeholder bug pattern ("new {predicate}").
    """
    if not text or not text.strip():
        return False
    stripped = text.strip()
    if len(stripped) > 60:
        return False
    if any(c in stripped for c in "[]{}<>"):
        return False
    if stripped.lower().startswith("new "):
        return False
    lower = stripped.lower()
    if lower.startswith(_SENTENCE_OPENERS):
        return False
    if stripped.endswith(_SENTENCE_END_PUNCT):
        return False
    if any(p + " " in stripped for p in _SENTENCE_END_PUNCT):
        return False
    return True


def _sample_replacement(fact: AtomicFact, llm: BaseLLM) -> str:
    """Natural replacement value for an F6 target (never "new {predicate}").

    Asks the model, retrying (bounded) until it returns a clean *value* (see
    `is_clean_value`); a sentence-shaped replacement ("I prefer hands-on
    projects...") does not fill the old value's slot and would make the
    new_value probe unanswerable, so any unusable output is
    treated as a failure that is retried and finally raised — the caller skips
    the instance instead of silently substituting canned content.
    """
    old = (fact.value or "").strip().lower()
    last_err: Exception | None = None
    for _ in range(_REPLACEMENT_RETRIES):
        try:
            raw = llm.chat_json(
                [{"role": "user", "content": _REPLACEMENT_PROMPT.format(
                    surface=fact.surface_form, predicate=fact.predicate,
                    category=fact.category.value)}],
                temperature=0.7,
                task="f6_replacement",
            )
            out = str(raw.get("new_value", "")).strip()
            if (
                is_clean_value(out)
                and out.lower() != old
                and not any(c in out for c in "[]{}<>")
            ):
                return out
            last_err = ValueError(f"unusable replacement value {out!r}")
        except Exception as e:  # noqa: BLE001 - retried below
            last_err = e
    raise RuntimeError(
        f"F6 replacement generation failed for {fact.predicate!r}: {last_err}"
    )


@dataclass
class RequestPlan:
    """Everything needed to inject one forget request into an instance."""

    forget_type: ForgetType
    style: RequestStyle
    position_third: str
    mention_count: int
    target_fact_ids: list[str] = field(default_factory=list)
    request_session: int = 0
    time_scope: TimeScope | None = None
    replacement: str | None = None


@dataclass
class InstancePlan:
    """Everything needed to realize one dialogue instance."""

    index: int
    length_bucket: str  # short | medium | long | xlong
    num_sessions: int
    turns_per_session: tuple[int, int]  # per-bucket range
    themes: list[str]
    scenarios: list[str]  # per-session user goals (LLM path)
    distractor_fact_ids: list[str]
    requests: list[RequestPlan] = field(default_factory=list)  # position ascending


def plan_instances(
    profile: UserProfile,
    llm: BaseLLM,
    config: dict[str, Any],
    profile_index: int = 0,
) -> list[InstancePlan]:
    """Plan the instances for one profile.

    Each instance carries 1-8 forget requests (sampled per length bucket).
    The first request's type rotates over F1-F6 by the
    instance's run-global index (`profile_index * instances_per_profile + i`),
    so coverage stays balanced even when every profile has a single instance;
    additional requests are sampled by the configured weights. Style and
    mention-count rotation are keyed by the same run-global index for the
    same reason (a per-profile index is always 0 when profiles have a single
    instance, which would pin every first request to one style).
    """
    rng = random.Random(f"{config.get('seed', 42)}:{profile.profile_id}")
    n = config.get("scale", {}).get("instances_per_profile", 3)
    weights = config.get("forget_type_weights", {})
    types = list(ForgetType)

    def first_type(i: int) -> ForgetType:
        return types[(profile_index * n + i) % len(types)]

    def extra_type() -> ForgetType:
        w = [weights.get(t.value, 1.0) for t in types]
        return rng.choices(types, weights=w, k=1)[0]

    diff_cfg = config.get("difficulty", {})
    buckets = diff_cfg.get("sessions_buckets", {"short": [2, 3], "medium": [4, 5], "long": [6, 8]})
    # turns_per_session is either one global [lo, hi] range applied to every
    # bucket, or a per-bucket mapping (default.yaml uses per-bucket ranges).
    turns_cfg = diff_cfg.get("turns_per_session", [8, 16])

    def turns_for(bucket: str) -> tuple[int, int]:
        rng_turns = turns_cfg.get(bucket) if isinstance(turns_cfg, dict) else turns_cfg
        return (int(rng_turns[0]), int(rng_turns[1]))

    mention_buckets = diff_cfg.get("mention_counts", [1, 3, 5])
    requests_per = diff_cfg.get(
        "requests_per_instance",
        {"short": [1, 1], "medium": [2, 2], "long": [3, 3], "xlong": [4, 4]},
    )
    # Bucket mix: when `bucket_weights` is configured,
    # each instance's length bucket is sampled by weight (xlong included);
    # otherwise fall back to round-robin over short/medium/long plus a legacy
    # `xlong.fraction` override.
    bucket_weights = diff_cfg.get("bucket_weights")
    xlong_cfg = diff_cfg.get("xlong", {})
    xlong_fraction = xlong_cfg.get("fraction", 0.0)

    # Fraction of non-F4 requests whose target must be a *source* of a derived
    # fact, keeping derived-residue probes and 2-hop multihop attacks covered.
    derived_quota = float(diff_cfg.get("derived_source_quota", 0.0))

    plans = []
    for i in range(n):
        if bucket_weights:
            names = list(bucket_weights)
            bucket = rng.choices(names, weights=[bucket_weights[b] for b in names], k=1)[0]
            lo, hi = buckets[bucket]
            plan_turns = turns_for(bucket)
        elif rng.random() < xlong_fraction:
            bucket = "xlong"
            lo, hi = xlong_cfg.get("sessions", [40, 60])
            plan_turns = tuple(xlong_cfg.get("turns_per_session", [100, 200]))
        else:
            bucket = _LENGTH_BUCKETS[i % len(_LENGTH_BUCKETS)]
            lo, hi = buckets[bucket]
            plan_turns = turns_for(bucket)
        num_sessions = rng.randint(lo, hi)
        num_requests = rng.randint(*requests_per.get(bucket, [1, 1]))
        # Each request needs its own segment and the last request must leave at
        # least one session after it (Relearning suite).
        num_requests = max(1, min(num_requests, num_sessions - 1))

        # Scenarios cycle with a shuffled pool as a defensive fallback in case
        # an instance ever has more sessions than distinct scenarios (current
        # xlong maximum of 50 sessions exactly matches the pool size).
        pool = SESSION_SCENARIOS.copy()
        rng.shuffle(pool)
        picked = [pool[j % len(pool)] for j in range(num_sessions)]
        themes = [t for t, _ in picked]
        scenarios = [s for _, s in picked]

        plan = InstancePlan(
            index=i,
            length_bucket=bucket,
            num_sessions=num_sessions,
            turns_per_session=(int(plan_turns[0]), int(plan_turns[1])),
            themes=themes,
            scenarios=scenarios,
            distractor_fact_ids=[],
        )
        _plan_requests(plan, profile, rng, first_type(i), extra_type, num_requests,
                       mention_buckets, llm, derived_quota,
                       run_index=profile_index * n + i)
        _select_distractors(plan, profile, rng, diff_cfg)
        plans.append(plan)
    return plans


def _plan_requests(
    plan: InstancePlan,
    profile: UserProfile,
    rng: random.Random,
    first_type: ForgetType,
    extra_type: Callable[[], ForgetType],
    num_requests: int,
    mention_buckets: list[int],
    llm: BaseLLM,
    derived_quota: float = 0.0,
    run_index: int = 0,
) -> None:
    """Sample per-request type/style/mentions/position and pick targets.

    The session axis is partitioned into k contiguous segments (k = number of
    requests) and each request's session is sampled inside its own segment, so
    positions ascend and the last request stays at or before num_sessions - 2.
    Targets never overlap across requests (a fact already targeted by an
    earlier request, or derived from one, is excluded).

    Style and mention-count buckets rotate over the run-global instance index
    (`run_index`), not the per-profile one, so single-instance profiles still
    spread both dimensions instead of pinning every first request to
    direct/1.

    `derived_quota` (fraction in [0, 1]): each non-F4 request independently
    gets a target that is a *source* of a derived fact with this probability,
    so the derived-residue probes and 2-hop multihop attacks are guaranteed
    coverage without correlating the choice with the request's position.
    F4 (targets the derived fact itself), F3 (whole category) and F5
    (event-time constraint) are exempt.
    """
    n = plan.num_sessions
    used: set[str] = set()
    fmap = profile.fact_map()
    for j in range(num_requests):
        seg_lo = j * (n - 1) // num_requests
        seg_hi = (j + 1) * (n - 1) // num_requests - 1
        request_session = rng.randint(seg_lo, seg_hi)
        if request_session < n / 3:
            third = "front"
        elif request_session >= 2 * n / 3:
            third = "back"
        else:
            third = "middle"
        req = RequestPlan(
            forget_type=first_type if j == 0 else extra_type(),
            style=_STYLES[(run_index + j) % len(_STYLES)],
            position_third=third,
            mention_count=mention_buckets[(run_index + j) % len(mention_buckets)],
            request_session=request_session,
        )
        want_derived_source = (
            derived_quota > 0
            and req.forget_type != ForgetType.F4
            and rng.random() < derived_quota
        )
        _select_targets(req, profile, rng, used, run_index + j, llm,
                        want_derived_source=want_derived_source)
        # Facts targeted by this request (and their derivations) are off-limits
        # for later requests of the same instance.
        for tid in req.target_fact_ids:
            used.add(tid)
            if tid in fmap:
                used.update(fmap[tid].derivations)
        plan.requests.append(req)


def _by_predicate(profile: UserProfile, predicate: str) -> AtomicFact | None:
    for f in profile.facts:
        if f.predicate == predicate:
            return f
    return None


# Generic predicate words that say nothing about entanglement: "favorite"
# alone must not link "favorite fruit: apples" to "favorite cuisine: Italian".
_GENERIC_TOKENS = frozenset({
    "favorite", "favourite", "prefers", "likes", "loves", "enjoys",
    "usually", "currently", "works", "does", "goes",
})


def _value_tokens(value: str) -> set[str]:
    """Distinctive lowercase tokens of a fact value (proper nouns, places...).
    Pure-digit tokens (mostly years) and generic predicate words are excluded:
    a shared year or a shared "favorite" alone is weak evidence of
    entanglement and would over-exclude unrelated facts."""
    return {t for t in re.findall(r"[a-z0-9]+", value.lower())
            if len(t) >= 4 and not t.isdigit() and t not in _GENERIC_TOKENS}


def _fact_tokens(fact: AtomicFact) -> set[str]:
    """Distinctive tokens of both the value and the surface form: a fact whose
    surface literally narrates another fact's event (e.g. "moved into the
    South End apartment ... right after selling the Newton house") is
    entangled with it even when the values share no wording."""
    return _value_tokens(fact.value) | _value_tokens(fact.surface_form)


def _facts_overlap(a: AtomicFact, b: AtomicFact) -> bool:
    """True when two facts share any distinctive token.

    Used to keep semantically entangled facts out of the distractor pool: e.g.
    forgetting "lives in Austin, Texas" while "B.S. from University of Texas at
    Austin" stays as a distractor would leak the city through a legitimate,
    unrelated memory.
    """
    return bool(_fact_tokens(a) & _fact_tokens(b))


def _select_targets(
    req: RequestPlan,
    profile: UserProfile,
    rng: random.Random,
    excluded: set[str],
    seed_index: int,
    llm: BaseLLM,
    want_derived_source: bool = False,
) -> None:
    """Pick forget-target facts per forget type.

    `excluded` holds fact ids already targeted by earlier requests of the same
    instance (plus their derivations); targets never overlap across requests.

    `want_derived_source` (derived-source quota): when set, the
    target is chosen among atomic facts that *source* a derived fact (e.g. a
    home address that derives a commute). Targeting such facts activates the
    derived-residue probes and the 2-hop multihop attack path, so the
    benchmark's cascade-deletion claim stays measurably
    covered instead of depending on sampling luck.

    Degradations stay semantically coherent with the rendered request:
    - F4 without any derived fact falls back to F1
      by switching the request type, so rendering, metadata and questions all
      agree.
    - F5 without event_time facts falls back to dialogue_time with a real
      session range; the request scope and the targets are consistent.
    """
    facts = profile.facts
    atomic = [f for f in facts if not f.is_derived and f.fact_id not in excluded]
    derived = [f for f in facts if f.is_derived and f.fact_id not in excluded]
    # Atomic facts that derive at least one other fact. These are the targets
    # that make derived probes and 2-hop multihop attacks possible.
    derived_sources = [f for f in atomic if f.derivations]

    if req.forget_type == ForgetType.F1:
        # One isolated atomic fact; prefer a high-sensitivity one when present.
        if want_derived_source and derived_sources:
            candidates = (
                [f for f in derived_sources if f.sensitivity.value == "high"]
                or derived_sources
            )
        else:
            candidates = [f for f in atomic if f.sensitivity.value == "high"] or atomic
        req.target_fact_ids = [rng.choice(candidates).fact_id]

    elif req.forget_type == ForgetType.F2:
        # Several unrelated facts (distinct categories).
        rng.shuffle(atomic)
        picked: list[AtomicFact] = []
        if want_derived_source and derived_sources:
            # Start with a derived source, then complete with a distinct category.
            src_candidates = [
                f for f in derived_sources
                if f.category not in {p.category for p in picked}
            ]
            if src_candidates:
                picked.append(rng.choice(src_candidates))
        for f in atomic:
            if f.category not in {p.category for p in picked}:
                picked.append(f)
            if len(picked) == 2:
                break
        req.target_fact_ids = [f.fact_id for f in picked]

    elif req.forget_type == ForgetType.F3:
        # Everything under one category (prefer one with >= 2 atomic facts).
        by_cat: dict[FactCategory, list[AtomicFact]] = {}
        for f in atomic:
            by_cat.setdefault(f.category, []).append(f)
        multi = [c for c, fs in by_cat.items() if len(fs) >= 2]
        cat = rng.choice(multi or list(by_cat))
        req.target_fact_ids = [f.fact_id for f in by_cat[cat]]

    elif req.forget_type == ForgetType.F4:
        # A derived fact. Only the derived fact itself must be forgotten; its
        # source facts are not evaluated (they are excluded from preservation
        # probes instead — see _select_distractors / _mentioned_distractors).
        if derived:
            target = rng.choice(derived)
            req.target_fact_ids = [target.fact_id]
        else:
            # No derived facts in this profile: degrade to F1 behavior with a
            # coherent F1 request (not an F4-styled request about an atomic
            # fact, which would be semantically wrong).
            req.forget_type = ForgetType.F1
            candidates = [f for f in atomic if f.sensitivity.value == "high"] or atomic
            req.target_fact_ids = [rng.choice(candidates).fact_id]

    elif req.forget_type == ForgetType.F5:
        # Facts whose event_time falls inside a time scope; without timed facts,
        # fall back to a dialogue_time scope over real sessions, with targets
        # guaranteed to be mentioned only inside that scope (see
        # _distribute_mentions in generator.py).
        timed = [f for f in atomic if f.event_time]
        if timed:
            month = rng.choice([f.event_time for f in timed if f.event_time])
            targets = [f for f in timed if f.event_time == month]
            req.target_fact_ids = [f.fact_id for f in targets]
            req.time_scope = TimeScope(mode=TimeScopeMode.EVENT_TIME, start=month, end=month)
        else:
            span = rng.randint(1, 2)
            a = max(0, req.request_session - span + 1)
            pool = atomic.copy()
            rng.shuffle(pool)
            n_targets = rng.randint(1, min(2, len(pool)))
            req.target_fact_ids = [f.fact_id for f in pool[:n_targets]]
            req.time_scope = TimeScope(
                mode=TimeScopeMode.DIALOGUE_TIME,
                start=str(a),
                end=str(req.request_session),
            )

    elif req.forget_type == ForgetType.F6:
        # Replace an old value with a new one; prefer a preference fact (the
        # replacement value is always LLM-generated, `_sample_replacement`).
        prefs = [f for f in atomic if f.category == FactCategory.PREFERENCE]
        if want_derived_source:
            # A preference that is also a derived source (e.g. a spicy-cuisine
            # favorite that derives "likes spicy food"): covers the derived
            # residue path for F6 targets too.
            src_prefs = [f for f in prefs if f.derivations]
            if src_prefs:
                target = rng.choice(src_prefs)
                req.target_fact_ids = [target.fact_id]
                req.replacement = _sample_replacement(target, llm)
                return
        target = rng.choice(prefs or atomic)
        req.target_fact_ids = [target.fact_id]
        req.replacement = _sample_replacement(target, llm)


def _select_distractors(
    plan: InstancePlan,
    profile: UserProfile,
    rng: random.Random,
    diff_cfg: dict[str, Any],
) -> None:
    """Sample unrelated distractor facts (distractor density).

    Facts semantically entangled with a target (sharing a distinctive value
    token, e.g. "University of Texas at Austin" vs. target "Austin, Texas")
    are excluded: keeping them as distractors would leak the target through a
    legitimate retained memory.
    """
    lo, hi = diff_cfg.get("distractor_facts", [10, 60])
    excluded = {tid for req in plan.requests for tid in req.target_fact_ids}
    # Also exclude facts derived from targets (they leak the target).
    fmap = profile.fact_map()
    for tid in list(excluded):
        excluded.update(fmap[tid].derivations)
    # F4 source facts: an F4 request only asks to forget the derived
    # conclusion, so its sources are neither forget targets nor valid
    # preservation probes (the benchmark does not require them to survive —
    # probing them for preservation would over-penalize a system that
    # cascade-deleted them).
    for req in plan.requests:
        if req.forget_type == ForgetType.F4:
            for tid in req.target_fact_ids:
                if tid in fmap:
                    excluded.update(fmap[tid].source_facts)
    targets = [fmap[tid] for tid in excluded if tid in fmap]
    pool = [
        f.fact_id
        for f in profile.facts
        if f.fact_id not in excluded
        and not any(_facts_overlap(f, t) for t in targets)
    ]
    rng.shuffle(pool)
    want = min(rng.randint(lo, hi), len(pool))
    plan.distractor_fact_ids = sorted(pool[:want])
