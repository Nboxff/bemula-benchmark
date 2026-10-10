"""CLI: attack-suite leakage pre-check.

Simulates a perfect memory system and verifies that every attack item would
leak the target in that state — i.e. that the item is answerable and can
distinguish a forgetting system from a non-forgetting one. Non-discriminative
attack items (unanswerable even with perfect memory) are reported and produce
a non-zero exit code.

Usage: uv run python -m scripts.precheck_attacks \
    --config configs/default.yaml \
    --profiles data/profiles/profiles.jsonl \
    --dialogues data/dialogues/dialogues.jsonl \
    --questions data/questions/questions.jsonl [--llm] [--limit N]
"""

from __future__ import annotations

import argparse

from bemula.config import load_config
from bemula.llm import get_llm
from bemula.questions.precheck import precheck_items
from bemula.schemas import TestItem, UserProfile, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Bemula attack-suite leakage pre-check.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--llm", action="store_true",
                    help="let the configured LLM play the perfect memory system "
                         "(default: deterministic surface-form answers)")
    ap.add_argument("--limit", type=int, default=None, help="max number of items to check")
    ap.add_argument("--profiles", default="data/profiles/profiles.jsonl")
    ap.add_argument("--dialogues", default="data/dialogues/dialogues.jsonl")
    ap.add_argument("--questions", default="data/questions/questions.jsonl")
    ap.add_argument("--out", default=None, help="optional report file")
    args = ap.parse_args()

    config = load_config(args.config)
    llm = get_llm(max_retries=config.get("generation", {}).get("max_retries", 3))
    profiles = {p.profile_id: p for p in load_jsonl(args.profiles, UserProfile)}
    items = load_jsonl(args.questions, TestItem)
    if args.limit is not None:
        items = items[: args.limit]

    report = precheck_items(items, profiles, llm if args.llm else None)
    text = report.format()
    print(text)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
            f.write(f"config: {args.config}\nllm_mode: {args.llm}\n")
    # Non-discriminative attack items are data defects: fail loudly.
    if report.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
