from __future__ import annotations

from bemula.dialogues.planner import RequestPlan
from bemula.schemas import (
    ForgetType,
    RequestStyle,
    Session,
    TimeScopeMode,
    Turn,
    UserProfile,
)

# Skeleton templates per forget type x style (style dimension).
# Slots: {targets} (NL list of target descriptions), {category}, {scope},
# {old}, {new}.
REQUEST_TEMPLATES: dict[ForgetType, dict[RequestStyle, str]] = {
    ForgetType.F1: {
        RequestStyle.DIRECT: "Please forget this information about me: {targets}.",
        RequestStyle.POLITE: "If it's not too much trouble, could you please stop "
        "remembering this: {targets}? I'd really appreciate it.",
        RequestStyle.INDIRECT: "I recently got into trouble because some personal "
        "details of mine leaked — {targets}. I need to know you don't have them "
        "on record anymore.",
        RequestStyle.IMPERATIVE: "Delete this right now, it's private: {targets}. "
        "Do not keep it anywhere.",
        RequestStyle.IMPLICIT: "By the way, {targets} — well, that's ancient history "
        "now, and I don't want you holding onto it anymore.",
    },
    ForgetType.F2: {
        RequestStyle.DIRECT: "Please delete all of the following: {targets}.",
        RequestStyle.POLITE: "Sorry to bother you, but would you mind removing a few "
        "things for me: {targets}? Thank you so much.",
        RequestStyle.INDIRECT: "I've been thinking about my privacy lately. Things "
        "like {targets} — I need every one of them gone from your records.",
        RequestStyle.IMPERATIVE: "Remove all of these immediately: {targets}. "
        "This is not up for discussion.",
        RequestStyle.IMPLICIT: "Funny how things change — {targets}... none of that "
        "applies to my life anymore, and I don't want you keeping any of it on me.",
    },
    ForgetType.F3: {
        RequestStyle.DIRECT: "Please do not store any of my {category} information. "
        "Delete everything you have in that category.",
        RequestStyle.POLITE: "This is a bit uncomfortable for me, but could you please "
        "erase all my {category} related information? I'd be grateful.",
        RequestStyle.INDIRECT: "A friend had their {category} data misused by an app. "
        "It made me realize I can't have mine sitting in your records either — "
        "it all has to go.",
        RequestStyle.IMPERATIVE: "Wipe every piece of my {category} information now. "
        "All of it, nothing stays.",
        RequestStyle.IMPLICIT: "I've decided to keep my {category} matters completely "
        "to myself from now on — I don't want you to hold onto any of it.",
    },
    ForgetType.F4: {
        RequestStyle.DIRECT: "Please forget this conclusion about me: {targets}.",
        RequestStyle.POLITE: "Could I ask you a favor? Please let go of this: "
        "{targets}. Thank you for understanding.",
        RequestStyle.INDIRECT: "It's odd what can be pieced together about a person. "
        "For instance {targets} — I need that conclusion gone from your records.",
        RequestStyle.IMPERATIVE: "Forget this inference immediately: {targets}.",
        RequestStyle.IMPLICIT: "You know, connecting the dots about me like {targets} "
        "feels unnecessary — I don't want you keeping that inference about me.",
    },
    ForgetType.F5: {
        RequestStyle.DIRECT: "Please delete all memories {scope}.",
        RequestStyle.POLITE: "Would you mind clearing out everything {scope}? "
        "I'd really appreciate the fresh start.",
        RequestStyle.INDIRECT: "That period {scope} was a rough patch for me. "
        "I need nothing from it left on record.",
        RequestStyle.IMPERATIVE: "Erase every memory {scope}, right now. "
        "Leave nothing behind.",
        RequestStyle.IMPLICIT: "I've completely moved on from that time {scope} — "
        "I don't want anything from it kept, those days might as well have never "
        "happened.",
    },
    ForgetType.F6: {
        RequestStyle.DIRECT: "My {predicate} has changed from {old} to {new}. "
        "Please forget the old value and only remember {new}.",
        RequestStyle.POLITE: "A small update, if you don't mind: my {predicate} is "
        "now {new} instead of {old}. Could you make sure only the new one is kept?",
        RequestStyle.INDIRECT: "People change, you know. {old} used to be my answer "
        "for {predicate}, but these days it's {new} — I need the old answer gone "
        "from your records.",
        RequestStyle.IMPERATIVE: "Update this now: my {predicate} is {new}, not "
        "{old}. The old value must be gone completely.",
        RequestStyle.IMPLICIT: "It's funny — I can't even remember the last time "
        "{old} was true for me. It's {new} now, and I don't want {old} being what "
        "you remember for my {predicate}.",
    },
}

_MONTH_NAMES = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]


def _humanize_month(iso_month: str) -> str:
    """Render an ISO month ("2020-08") the way a person would say it."""
    try:
        year, month = iso_month.split("-")
        return f"{_MONTH_NAMES[int(month) - 1]} {year}"
    except (ValueError, IndexError):
        return iso_month

def render_request_text(req: RequestPlan, profile: UserProfile) -> str:
    """Render the forget-request text from the style template (skeleton).

    The skeleton is used as the semantic anchor for the LLM request
    generation prompt (`_generate_request_exchange`); the LLM produces the
    final wording.
    """
    fmap = profile.fact_map()
    targets = [fmap[tid] for tid in req.target_fact_ids if tid in fmap]
    target_desc = " and ".join(t.surface_form.rstrip(".") for t in targets)
    style_templates = REQUEST_TEMPLATES[req.forget_type]
    template = style_templates[req.style]

    if req.forget_type == ForgetType.F3:
        category = targets[0].category.value if targets else "personal"
        return template.format(category=category)
    if req.forget_type == ForgetType.F4:
        return template.format(targets=target_desc)
    if req.forget_type == ForgetType.F5:
        scope = req.time_scope
        if scope and scope.mode == TimeScopeMode.EVENT_TIME:
            start, end = _humanize_month(scope.start), _humanize_month(scope.end)
            scope_text = (
                f"about events from {start}"
                if start == end
                else f"about events from {start} to {end}"
            )
        else:
            scope_text = f"from our sessions {scope.start} through {scope.end}" if scope else ""
        return template.format(scope=scope_text)
    if req.forget_type == ForgetType.F6:
        old = targets[0].value if targets else ""
        return template.format(predicate=targets[0].predicate if targets else "preference",
                               old=old, new=req.replacement or "")
    return template.format(targets=target_desc)


def inject_exchange(
    forget_sessions: list[Session],
    control_sessions: list[Session],
    req: RequestPlan,
    request_text: str,
    ack_text: str,
    filler_text: str,
    filler_ack_text: str,
    timestamp: str,
    follow_fact_ids: set[str] | None = None,
) -> int:
    """Insert a request/acknowledgment pair (LLM path) at the planned position.

    Both variants are updated in place: the forget variant gets (user request,
    assistant ack); the control variant gets (user filler, assistant
    filler-ack) at the same position. Everything else is identical. Returns
    the inserted request turn index (identical in both variants).

    `follow_fact_ids` extends the set of facts whose mentions the request must
    follow: F4 requests must land after their *source* facts as well as their
    target (the premises must be stated before the request so the derived
    target is genuinely inferable from the dialogue).
    """
    session_idx = req.request_session
    turns = forget_sessions[session_idx].turns
    # Planned midpoint, but the request must follow EVERY mention of a target
    # fact (a later mention would legitimately re-store the fact after the
    # request and break the efficacy semantics).
    target_ids = set(req.target_fact_ids) | (follow_fact_ids or set())
    last_target = max(
        (i for i, t in enumerate(turns) if target_ids & set(t.linked_fact_ids)),
        default=-1,
    )
    turn_index = max(len(turns) // 2, last_target + 1)
    # Insert at an even boundary so speaker alternation is preserved.
    if turn_index % 2 == 1:
        turn_index += 1
    turn_index = min(turn_index, len(turns))

    forget_pair = [
        Turn(speaker="user", text=request_text, timestamp=timestamp),
        Turn(speaker="assistant", text=ack_text, timestamp=timestamp),
    ]
    control_pair = [
        Turn(speaker="user", text=filler_text, timestamp=timestamp),
        Turn(speaker="assistant", text=filler_ack_text, timestamp=timestamp),
    ]
    forget_sessions[session_idx].turns[turn_index:turn_index] = forget_pair
    control_sessions[session_idx].turns[turn_index:turn_index] = control_pair
    return turn_index
