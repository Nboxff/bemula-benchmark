"""LLM agents for turn-by-turn dialogue simulation.

Two agents simulate the conversation one turn at a time:

- UserSimulator: holds the persona, the session scenario, and the checklist of
  facts it may reveal this session; produces each user turn.
- AssistantAgent: a helpful assistant which sees a sliding window of the 
  current session's recent turns.

The director loop that drives them lives in bemula/dialogues/generator.py.
"""

from __future__ import annotations

import re
from typing import Any

from bemula.llm import BaseLLM, parse_json

_USER_SIM_PROMPT = """You are role-playing a real user chatting with a helpful AI assistant.

Your persona: {persona}
{current_block}Your goal for this conversation: {scenario}

Personal background facts you may reveal in this conversation (reveal each at most \
once, only where it genuinely helps the conversation; use your own words but keep \
the stated values exact; never recite them as a list):
{facts_block}

Conversation so far:
{history}
{nudge_block}
{revealed_block}{feedback_block}{closure_block}Write your next message as the user: 1-3 sentences, driven by your goal above. \
Do not mention any personal facts beyond the list above.

Temporal consistency rules (important: your messages must never contradict a \
fact's stated date):
- When a fact carries an explicit date or year, keep that year (and month) exactly \
as stated in your message — never drop the year, never move the event to a \
different year.
- Only describe an event as happening "just", "recently", or "lately" if its \
stated date is within the last 6 months of today; for older events give the \
actual date or a relative phrase consistent with it (e.g. "about a year and a \
half ago", not "just").

How to write (this matters: real people do not text like an AI):
- casual chat style: contractions, simple everyday words, an occasional fragment \
or run-on is fine
- NEVER use em dashes or semicolons; plain commas and periods only
- no formal connectors ("furthermore", "that being said", "it's worth noting"), \
no polished essay-style sentences, no lists, no excessive politeness
- vary your openings: do NOT start with "Yeah", "Hey", or any word you already \
opened recent messages with (recent openers: {recent_openers}); most messages \
should skip the acknowledgment and go straight to your point
- never end the conversation or announce that you are leaving (e.g. "I'll catch \
you later", "see you next week", "ok I'll get going", "thanks for the chat") more \
than once across the conversation, and only on your final message: if you already \
signed off, say something new instead of repeating it
Return ONLY JSON: {{"text": "<your message>", "fact_ids": [<ids of listed facts you \
just conveyed, or empty>]}}, choosing fact ids from: {fact_id_list}. Tag a fact \
only if your message actually states its value; alluding to it (a name, a place, \
a partial detail) without the value does not count."""

_NUDGE = (
    "\nImportant: before this conversation ends, naturally work the following fact "
    "into your next message (only if it fits the topic; keep the value exact): {surface}\n"
)

_REVEALED_BLOCK = (
    "\nYou already shared these facts earlier in this conversation; do NOT bring "
    "them up again, not even in different words:\n{surfaces}\n"
)

_KEEP_GOING = (
    "\nYou have already revealed the background facts scheduled for this conversation, "
    "but the conversation is still going. Keep it productive: ask the assistant a "
    "practical follow-up, raise a new related topic from your own life, or explore a "
    "new angle. Keep it casual and 1-3 sentences. Do NOT end the conversation or say "
    "goodbye.\n"
)

_CLOSING_NUDGE = (
    "\nThis is your final message: wrap up the conversation naturally in 1-2 sentences. "
    "Do not start a new topic or ask a new question.\n"
)

_ASSISTANT_PROMPT = """You are a helpful AI assistant in an ongoing chat with a user. \
Respond to the user's latest message with substantive, specific help (not generic \
acknowledgments). Ask a short follow-up when useful. Keep it to 1-4 sentences of \
natural chat style. Acknowledge personal details briefly; never repeat the user's \
personal facts back verbatim or build running jokes on them.

Recent conversation:
{window}

Write the assistant's next reply. Return only the reply text."""

_VERIFIER_PROMPT = """You are checking a generated chat session between a user and an \
AI assistant for quality.

User persona: {persona}
{dated_facts_block}Session transcript:
{transcript}

Check three things:
1. Consistency: does anything the user says contradict the persona or earlier turns?
2. Naturalness: any robotic turns (fact-listing, mechanical repetition, "as an AI"), \
or user messages that read like polished AI prose (em dashes, formal connectors, \
essay-style sentences)?
3. Temporal consistency: does the user describe any event's timing in a way that \
contradicts its stated date (e.g. saying "just got married" or "recently" when the \
stated wedding date is over a year before the conversation, or dropping/relocating \
the stated year)?
Return ONLY JSON: {{"consistent": true/false, "natural": true/false, \
"issues": ["<brief issue descriptions>"]}}"""


_DASH_RUN = re.compile(r"\s*[—–―]\s*")


def sanitize_chat_text(text: str) -> str:
    """Deterministic backstop against AI-flavored typography in user-facing text.

    LLMs sprinkle em dashes everywhere; real chat users basically never type
    them. Replace dash interruptions with plain punctuation and clean up the
    artifacts (doubled commas, comma-before-period, stray spaces).
    """
    text = _DASH_RUN.sub(", ", text)
    text = re.sub(r",\s*([.!?])", r"\1", text)  # ", ." -> "."
    text = re.sub(r",(\s*,)+", ",", text)  # collapse duplicated commas
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"^,\s*", "", text)  # dash at the very start left a leading comma
    return text.strip()


def _first_word(text: str) -> str:
    m = re.match(r"[A-Za-z']+", text.strip())
    return m.group(0).lower() if m else ""


def _drop_stale_opener(text: str, recent_openers: list[str]) -> str:
    """Deterministic backstop for opener mode collapse.

    If the model still opens with a word the user already opened 2+ of the
    last 3 messages with, strip the leading acknowledgment ("Yeah, ..." ->
    "...") and re-capitalize. Only fires on a repeated opener, so genuine
    one-off "Yeah, ..." turns survive.
    """
    if not text or len(recent_openers) < 2:
        return text
    word = _first_word(text)
    if word and recent_openers.count(word) >= 2:
        rest = re.sub(r"^[A-Za-z']+[,!]?\s+", "", text, count=1)
        if rest:
            return rest[0].upper() + rest[1:]
    return text


class UserSimulator:
    """Simulates the user turn by turn, with a per-session fact checklist."""

    def __init__(self, llm: BaseLLM, temperature: float = 0.9) -> None:
        self.llm = llm
        self.temperature = temperature

    def next_turn(
        self,
        persona: str,
        scenario: str,
        remaining_facts: dict[str, str],
        history: list[tuple[str, str]],
        nudge: tuple[str, str] | None = None,
        current_month: str | None = None,
        feedback: str | None = None,
        closing: bool = False,
        revealed_facts: dict[str, str] | None = None,
    ) -> tuple[str, list[str]]:
        """Produce the next user message and the fact IDs it conveys.

        `remaining_facts` maps fact_id -> surface form (facts not yet revealed).
        `nudge` is an optional (fact_id, surface) the director wants revealed soon.
        `current_month` is the conversation's current date (humanized, e.g.
        "February 2026"); the simulator must phrase dated facts consistently
        with it (keep the stated year; "just"/"recently" only for events within
        ~6 months of now).
        `feedback` is a rejection reason for the previous attempt (e.g. a
        dropped stated year); the retry must fix exactly that problem.
        `closing` marks the session's final turn: the user wraps up rather than
        continuing the topic.
        `revealed_facts` maps fact_id -> surface form for facts already
        revealed earlier in this session; the prompt bans re-mentioning them
        so a fact cannot echo turn after turn once history contains it.

        Raises on any LLM failure or empty output: the caller (instance-level
        retry) re-runs the whole instance instead of silently keeping a
        truncated session; no silent degradation.
        """
        facts_block = (
            "\n".join(f'- id "{fid}": {surface}' for fid, surface in remaining_facts.items())
            or "- (no more background facts scheduled)"
        )
        if closing:
            closure_block = _CLOSING_NUDGE
        elif not remaining_facts:
            closure_block = _KEEP_GOING
        else:
            closure_block = ""
        history_text = (
            "\n".join(f"{speaker}: {text}" for speaker, text in history) or "(just started)"
        )
        nudge_block = _NUDGE.format(surface=nudge[1]) if nudge else ""
        revealed_block = (
            _REVEALED_BLOCK.format(
                surfaces="\n".join(f"- {s}" for s in revealed_facts.values())
            )
            if revealed_facts
            else ""
        )
        feedback_block = f"\n{feedback}\n" if feedback else ""
        current_block = f"Today is {current_month}.\n" if current_month else ""
        # Opening words of the user's last few messages, so the prompt can ban
        # repeating them (mode collapse otherwise: the model latches onto
        # openers like "Yeah").
        recent_openers = [
            _first_word(text) for speaker, text in history if speaker == "user"
        ][-3:]
        prompt = _USER_SIM_PROMPT.format(
            persona=persona,
            current_block=current_block,
            scenario=scenario,
            facts_block=facts_block,
            history=history_text,
            nudge_block=nudge_block,
            revealed_block=revealed_block,
            feedback_block=feedback_block,
            closure_block=closure_block,
            fact_id_list=list(remaining_facts),
            recent_openers=", ".join(f'"{w}"' for w in recent_openers) or "(none yet)",
        )
        raw = parse_json(self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=2048,
            task="user_sim",
        ))
        text = sanitize_chat_text(str(raw.get("text", "")))
        text = _drop_stale_opener(text, recent_openers)
        if not text:
            raise ValueError("user simulator returned an empty turn")
        fact_ids = [f for f in raw.get("fact_ids", []) if f in remaining_facts]
        return text, fact_ids


class AssistantAgent:
    """Simulates the assistant; sees only a sliding window of the current session."""

    def __init__(self, llm: BaseLLM, temperature: float = 0.7) -> None:
        self.llm = llm
        self.temperature = temperature

    def reply(self, window: list[tuple[str, str]]) -> str:
        """Produce the assistant's reply from the recent turns only.

        Raises on LLM failure or empty output; the director must not
        substitute a canned reply.
        """
        window_text = "\n".join(f"{speaker}: {text}" for speaker, text in window)
        out = self.llm.chat(
            [{"role": "user", "content": _ASSISTANT_PROMPT.format(window=window_text)}],
            temperature=self.temperature,
            max_tokens=1024,
            task="assistant_sim",
        ).strip()
        if not out:
            raise ValueError("assistant agent returned an empty reply")
        return out


def verify_session(
    llm: BaseLLM,
    persona: str,
    transcript: list[tuple[str, str]],
    dated_facts: list[str] | None = None,
) -> dict[str, Any]:
    """Per-session consistency/naturalness/temporal check.

    `dated_facts` lists the session's dated fact surfaces (facts with an
    event_time) so the verifier can judge whether the user's phrasing
    contradicts any stated date. Returns {"consistent": bool, "natural": bool,
    "issues": [...]}. Raises on LLM failure: a session we cannot verify must
    not silently pass.
    """
    text = "\n".join(f"{s}: {t}" for s, t in transcript)
    dated_block = (
        "Known dated facts:\n" + "\n".join(f"- {f}" for f in dated_facts) + "\n"
        if dated_facts
        else ""
    )
    raw = parse_json(llm.chat(
        [{"role": "user", "content": _VERIFIER_PROMPT.format(
            persona=persona, dated_facts_block=dated_block, transcript=text)}],
        temperature=0.0,
        max_tokens=2048,
        task="session_verify",
    ))
    return {
        "consistent": bool(raw.get("consistent", True)),
        "natural": bool(raw.get("natural", True)),
        "issues": [str(i) for i in raw.get("issues", [])],
    }
