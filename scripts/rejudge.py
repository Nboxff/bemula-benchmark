"""CLI: re-judge the stored answers in results dirs with a different judge model.

Re-grading does not re-run the memory systems: items.jsonl already stores each
probed answer, so only the judge is called again and the verdict fields
(correct / leakage) are updated in place. metrics.json, report.md, and
run_meta.json are rewritten from the new verdicts; the original files are
backed up once as *.bak before the first rewrite.

ItemRecord does not store the rubric, so each record is joined back to its
TestItem through item_id using the questions file recorded in run_meta.json.

Usage:
    BEMULA_JUDGE_MODEL=gpt-5.6-luna uv run python -m scripts.rejudge \
        results/rag_full_qwen3_32b results/amem_core_qwen3_32b

    # Dry run on the first N items: judges them but writes nothing.
    BEMULA_JUDGE_MODEL=gpt-5.6-luna uv run python -m scripts.rejudge \
        --limit 30 results/amem_core_qwen3_32b

    # Re-judge only exact-match items (e.g. after switching them to the LLM path).
    BEMULA_JUDGE_MODEL=gpt-5.6-luna uv run python -m scripts.rejudge \
        --grading exact results/rag_full_gpt56_luna
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from bemula.eval.judge import LLMJudge
from bemula.eval.metrics import compute_metrics
from bemula.eval.runner import ItemRecord
from bemula.llm import get_llm
from bemula.schemas import TestItem, load_jsonl
from scripts.run_eval import write_report


def rejudge_dir(out: Path, judge: LLMJudge, workers: int, limit: int | None,
                grading: str | None = None) -> dict:
    meta = json.loads((out / "run_meta.json").read_text(encoding="utf-8"))
    judge.temperature = meta.get("judge_temperature", 0.0)
    items_by_id = {i.item_id: i for i in load_jsonl(meta["questions"], TestItem)}
    records = load_jsonl(out / "items.jsonl", ItemRecord)
    targets = records
    if grading is not None:
        targets = [r for r in targets if items_by_id[r.item_id].grading == grading]
    if limit is not None:
        targets = targets[:limit]

    def work(rec: ItemRecord) -> tuple[bool, bool | None]:
        verdict = judge.judge(items_by_id[rec.item_id], rec.answer)
        return verdict.correct, verdict.leakage

    failures_before = judge.failures
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, rec): rec for rec in targets}
        results = [
            (futures[fut], *fut.result())
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=out.name, unit="item")
        ]

    changed = 0
    for rec, correct, leakage in results:
        if (rec.correct, rec.leakage) != (correct, leakage):
            changed += 1
        rec.correct, rec.leakage = correct, leakage

    summary = {
        "dir": out.name,
        "items": len(targets),
        "changed": changed,
        "judge_failures": judge.failures - failures_before,
    }
    if limit is not None:
        return summary  # dry run: nothing is written

    for name in ("items.jsonl", "metrics.json", "report.md", "run_meta.json"):
        path = out / name
        if path.is_file() and not (out / f"{name}.bak").exists():
            shutil.copy2(path, out / f"{name}.bak")

    with (out / "items.jsonl").open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec.model_dump(mode="json"), ensure_ascii=False) + "\n")

    metrics = compute_metrics(records)
    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(metrics, meta["baseline"], out)

    meta["judge_model"] = os.environ.get("BEMULA_JUDGE_MODEL")
    meta["rejudged_at"] = datetime.now().isoformat(timespec="seconds")
    (out / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    summary["bemula_score"] = round(metrics["bemula_score"], 4)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path, help="results dirs to re-judge")
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent judge calls (default: 16)")
    ap.add_argument("--limit", type=int, default=None,
                    help="judge only the first N items per dir and write nothing")
    ap.add_argument("--grading", default=None, choices=["exact", "llm_judge"],
                    help="re-judge only items with this grading")
    args = ap.parse_args()

    load_dotenv(override=True)
    judge_model = os.environ.get("BEMULA_JUDGE_MODEL")
    if not judge_model:
        raise SystemExit("BEMULA_JUDGE_MODEL is not set")

    judge = LLMJudge(get_llm(model=judge_model), temperature=0.0)
    for out in args.dirs:
        summary = rejudge_dir(out, judge, args.workers, args.limit, args.grading)
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
