"""CLI: coverage audit for the generated dataset.

Verifies that the core design paths are actually *measurable* in the data,
not just present in the code:

- derived-residue probes (probe_type=derived): targets that are sources of a
  derived fact — the cascade-deletion claim;
- 2-hop multihop attacks (hops=2): multi-hop composition over derived facts;
- F5 time-scope modes: event_time and dialogue_time coverage.

Coverage below the thresholds exits non-zero so the audit can gate a release.
F5 dialogue_time is data-dependent (a fallback for profiles without timed
facts), so a zero count is reported but does not fail the audit unless
--require-dialogue-time is passed.

Usage: uv run python -m scripts.audit_coverage \
    --dialogues data/dialogues/dialogues.jsonl --questions data/questions/questions.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter

from bemula.schemas import DialogueInstance, TestItem, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Bemula dataset coverage audit.")
    ap.add_argument("--dialogues", default="data/dialogues/dialogues.jsonl")
    ap.add_argument("--questions", default="data/questions/questions.jsonl")
    ap.add_argument("--min-derived", type=int, default=1,
                    help="minimum number of derived-residue probes")
    ap.add_argument("--min-hops2", type=int, default=1,
                    help="minimum number of hops=2 attack items")
    ap.add_argument("--require-dialogue-time", action="store_true",
                    help="fail when no F5 dialogue_time scope is present")
    args = ap.parse_args()

    instances = load_jsonl(args.dialogues, DialogueInstance)
    items = load_jsonl(args.questions, TestItem)

    derived = [it for it in items if it.probe_type == "derived"]
    hops2 = [it for it in items if it.hops == 2]
    f4 = [r for inst in instances for r in inst.forget_requests
          if r.forget_type.value == "F4"]
    f5 = [r for inst in instances for r in inst.forget_requests
          if r.forget_type.value == "F5"]
    f5_modes = Counter(
        r.time_scope.mode.value for r in f5 if r.time_scope is not None
    )
    attack_types = Counter(
        it.attack_type.value for it in items if it.attack_type is not None
    )

    print(f"instances: {len(instances)} | test items: {len(items)}")
    print(f"derived-residue probes: {len(derived)} (min {args.min_derived})")
    print(f"hops=2 attack items:    {len(hops2)} (min {args.min_hops2})")
    print(f"F4 requests: {len(f4)}")
    print(f"F5 requests: {len(f5)} | modes: {dict(f5_modes)}")
    print(f"attack types covered: {sorted(attack_types)}")

    failures: list[str] = []
    if len(derived) < args.min_derived:
        failures.append(
            f"derived-residue probes {len(derived)} < {args.min_derived}: "
            "cascade deletion is not measurable in this data"
        )
    if len(hops2) < args.min_hops2:
        failures.append(
            f"hops=2 items {len(hops2)} < {args.min_hops2}: multi-hop attacks "
            "are not measurable in this data"
        )
    if args.require_dialogue_time and f5_modes.get("dialogue_time", 0) == 0:
        failures.append("no F5 dialogue_time scope present")
    if len(attack_types) != 5:
        failures.append(f"attack types incomplete: {sorted(attack_types)}")

    if failures:
        print("\nAUDIT FAILED:")
        for msg in failures:
            print(f"  - {msg}")
        raise SystemExit(1)
    print("\nAUDIT PASSED")


if __name__ == "__main__":
    main()
