"""CLI: generate dialogue instances with forget requests.

Usage: uv run python -m scripts.generate_dialogues --config configs/default.yaml

Each successfully generated instance is appended to the output file
immediately (per-instance prompt persistence), so a crash never
loses completed work; rerunning resumes by skipping instance ids already
present in the output.
"""

from __future__ import annotations

import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from bemula.config import load_config
from bemula.dialogues.generator import generate_instances
from bemula.llm import get_llm
from bemula.schemas import UserProfile, append_jsonl, jsonl_ids, load_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Bemula dialogue instances.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None, help="max number of profiles to use")
    ap.add_argument("--profiles", default="data/profiles/profiles.jsonl")
    ap.add_argument("--out", default="data/dialogues/dialogues.jsonl")
    args = ap.parse_args()

    config = load_config(args.config)
    # Dialogue simulation uses the configured sim model (deepseek-v4-flash).
    sim_model = config.get("generation", {}).get("sim_model")
    llm = get_llm(model=sim_model,
                  max_retries=config.get("generation", {}).get("max_retries", 3))
    profiles = load_jsonl(args.profiles, UserProfile)
    if args.limit is not None:
        profiles = profiles[: args.limit]

    # Resume: skip profiles whose instances are all already persisted, so a
    # rerun never re-simulates completed instances. Profile indices are taken
    # from the full list to keep the run-global first-request rotation stable.
    existing = jsonl_ids(args.out, "instance_id")
    per_profile = int(config.get("scale", {}).get("instances_per_profile", 1))

    def is_done(profile: UserProfile) -> bool:
        return all(f"{profile.profile_id}:inst{i}" in existing for i in range(per_profile))

    indexed = [(i, p) for i, p in enumerate(profiles) if not is_done(p)]
    if existing:
        print(f"Resuming: {len(existing)} instance(s) already in {args.out}; "
              f"{len(indexed)} of {len(profiles)} profile(s) to go")
    lock = threading.Lock()

    def persist(inst) -> None:
        with lock:
            if inst.instance_id in existing:
                return
            append_jsonl([inst], args.out)
            existing.add(inst.instance_id)

    # Parallelism is at the profile level; the per-profile instance pool is
    # disabled so total concurrency stays at generation.max_workers.
    workers = int(config.get("generation", {}).get("max_workers", 1))
    inner_config = config
    if workers > 1:
        inner_config = dict(config)
        inner_config["generation"] = {**config.get("generation", {}), "max_workers": 1}

    def gen_profile(p_idx: int, profile: UserProfile) -> str | None:
        """Generate all instances for one profile; return profile id on failure."""
        try:
            # generate_instances retries each instance and skips the ones
            # that still fail; persist() is invoked per completed instance.
            # profile_index drives the run-global F1-F6 first-request rotation.
            generate_instances(profile, llm, inner_config, on_instance=persist,
                               profile_index=p_idx)
            return None
        except Exception as e:  # noqa: BLE001 - a failing profile is skipped
            print(f"  warning: profile {profile.profile_id} failed ({e}); skipping")
            return profile.profile_id

    total_before = len(existing)
    skipped = 0
    with tqdm(total=len(indexed), desc="profiles", unit="profile") as bar:
        if workers <= 1:
            for p_idx, profile in indexed:
                if gen_profile(p_idx, profile) is not None:
                    skipped += 1
                bar.update(1)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(gen_profile, p_idx, p) for p_idx, p in indexed]
                for fut in as_completed(futures):
                    if fut.result() is not None:
                        skipped += 1
                    bar.update(1)
    generated = len(existing) - total_before
    print(f"Wrote {generated} new dialogue instance(s) to {args.out} "
          f"(total {len(existing)}, {skipped} profile(s) skipped)")


if __name__ == "__main__":
    main()
