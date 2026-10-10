"""CLI: generate user profiles.

Usage: uv run python -m scripts.generate_profiles --config configs/default.yaml
"""

from __future__ import annotations

import argparse

from bemula.config import load_config
from bemula.llm import get_llm
from bemula.profiles.generator import generate_profiles
from bemula.schemas import dump_jsonl


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate Bemula user profiles.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--limit", type=int, default=None, help="max number of profiles")
    ap.add_argument("--start-index", type=int, default=0,
                    help="first profile index (ids start at profile-{start_index:04d}); "
                         "use a past batch's size for top-up runs so ids never collide")
    ap.add_argument("--out", default="data/profiles/profiles.jsonl")
    args = ap.parse_args()

    config = load_config(args.config)
    # Profiles/questions use the configured gen_model (deepseek-v4-flash only).
    gen_model = config.get("generation", {}).get("gen_model")
    llm = get_llm(model=gen_model,
                  max_retries=config.get("generation", {}).get("max_retries", 3))
    profiles = generate_profiles(llm, config, limit=args.limit, start_index=args.start_index)
    dump_jsonl(profiles, args.out)
    print(f"Wrote {len(profiles)} profiles to {args.out}")


if __name__ == "__main__":
    main()
