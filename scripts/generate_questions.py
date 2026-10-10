"""CLI: generate test items for all 5 suites.

Usage: uv run python -m scripts.generate_questions --config configs/default.yaml

Each instance's items are appended to the output file as soon as the
instance's questions are fully generated (per-instance prompt persistence);
rerunning resumes by skipping instances already present in the
output. An instance whose question generation keeps failing (LLM call
failures or quality-gate rejections) is retried up to `generation.max_retries`
times and then skipped with a warning — never emitted with degraded
(canned/template) questions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from bemula.config import load_config
from bemula.llm import get_llm
from bemula.questions.generator import generate_questions
from bemula.schemas import DialogueInstance, UserProfile, append_jsonl, jsonl_ids, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Bemula test items.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None, help="max number of instances to use")
    ap.add_argument("--profiles", default="data/profiles/profiles.jsonl")
    ap.add_argument("--dialogues", default="data/dialogues/dialogues.jsonl")
    ap.add_argument("--out", default="data/questions/questions.jsonl")
    args = ap.parse_args()

    config = load_config(args.config)
    # Profiles/questions use the configured gen_model (deepseek-v4-flash only).
    gen_model = config.get("generation", {}).get("gen_model")
    llm = get_llm(model=gen_model,
                  max_retries=config.get("generation", {}).get("max_retries", 3))
    profiles = {p.profile_id: p for p in load_jsonl(args.profiles, UserProfile)}
    instances = load_jsonl(args.dialogues, DialogueInstance)
    if args.limit is not None:
        instances = instances[: args.limit]

    # Resume: instances whose items are already persisted are not regenerated.
    existing = jsonl_ids(args.out, "instance_id")
    todo = [inst for inst in instances if inst.instance_id not in existing]
    if existing:
        print(f"Resuming: {len(instances) - len(todo)} instance(s) already in {args.out}")

    retries = int(config.get("generation", {}).get("max_retries", 3))

    def build(instance: DialogueInstance) -> list | None:
        """Generate one instance's items; retry (bounded), then skip."""
        for attempt in range(retries):
            try:
                return generate_questions(instance, profiles[instance.profile_id], llm)
            except Exception as e:  # noqa: BLE001 - retried below
                if attempt == retries - 1:
                    print(f"  warning: {instance.instance_id} question generation failed "
                          f"after {retries} attempts ({e}); skipping")
                    return None
                print(f"  {instance.instance_id} attempt {attempt + 1} failed ({e}); retrying")
        raise AssertionError("unreachable")

    workers = int(config.get("generation", {}).get("max_workers", 1))
    generated = 0
    with tqdm(total=len(todo), desc="questions", unit="inst") as bar:
        if workers <= 1:
            for instance in todo:
                items = build(instance)
                if items is not None:
                    append_jsonl(items, args.out)
                    generated += 1
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(build, inst) for inst in todo]
                for fut in as_completed(futures):
                    items = fut.result()
                    if items is not None:
                        append_jsonl(items, args.out)
                        generated += 1
                    bar.update(1)
    print(f"Wrote test items for {generated} new instance(s) to {args.out} "
          f"(total {len(jsonl_ids(args.out, 'instance_id'))} instance(s) covered)")


if __name__ == "__main__":
    main()
