"""CLI: run the control-pair evaluation for a memory baseline.

Usage:
    uv run python -m scripts.run_eval --baseline mem0 \
        --dialogues data/quickcheck/dialogues.jsonl \
        --questions data/quickcheck/questions.jsonl \
        --out results/mem0_quickcheck

Writes items.jsonl, metrics.json, and report.md under the out directory.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

from bemula.baselines.base import MemoryAdapter
from bemula.config import load_config
from bemula.eval.judge import Judge, LLMJudge
from bemula.eval.metrics import compute_metrics
from bemula.eval.runner import ItemRecord, run_evaluation_parallel, run_instance
from bemula.llm import BaseLLM, get_llm
from bemula.schemas import DialogueInstance, TestItem, load_jsonl


def _build_baseline(name: str, llm: BaseLLM,
                    embedder_provider: str) -> MemoryAdapter:
    """Baseline registry: map CLI names to adapter factories."""
    if name == "rag":
        from bemula.baselines.rag import RagAdapter, get_embedder

        return RagAdapter(llm, get_embedder())
    if name == "mem0":
        from bemula.baselines.mem0_adapter import Mem0Adapter

        return Mem0Adapter(llm, embedder_provider=embedder_provider)
    if name == "amem":
        from bemula.baselines.amem import AmemAdapter
        from bemula.baselines.rag import get_embedder

        return AmemAdapter(llm, embedder=get_embedder())
    if name == "simplemem":
        from bemula.baselines.rag import get_embedder
        from bemula.baselines.simplemem import SimpleMemAdapter

        return SimpleMemAdapter(llm, embedder=get_embedder())
    if name == "memoryos":
        from bemula.baselines.memoryos import MemoryosAdapter
        from bemula.baselines.rag import get_embedder

        return MemoryosAdapter(llm, embedder=get_embedder())
    if name == "lightmem":
        from bemula.baselines.lightmem import LightmemAdapter
        from bemula.baselines.rag import get_embedder

        return LightmemAdapter(llm, embedder=get_embedder())
    if name == "langmem":
        from bemula.baselines.langmem import LangmemAdapter
        from bemula.baselines.rag import get_embedder

        return LangmemAdapter(llm, embedder=get_embedder())
    if name == "fullcontext":
        from bemula.baselines.fullcontext import FullContextAdapter

        return FullContextAdapter(llm)
    raise ValueError(
        f"unknown baseline: {name!r} (available: rag, mem0, amem, simplemem, "
        "memoryos, lightmem, langmem, fullcontext)"
    )


def _fmt(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}"


def write_report(metrics: dict, baseline: str, out: Path) -> None:
    """Human-readable markdown summary of the metric report."""
    ar = metrics["AR"]
    gated = metrics["gated"]
    lines = [
        f"# Bemula evaluation report: {baseline}",
        "",
        f"- Instances: {metrics['num_instances']}  |  Items: {metrics['num_items']}",
        f"- RA gate: {metrics['ra_gate']['num_passed']}/{metrics['ra_gate']['num_total']} "
        f"instances passed ({metrics['ra_gate']['criterion']})",
        "",
        "| Metric | All instances | RA-gated |",
        "|--------|---------------|----------|",
        f"| RA | {_fmt(metrics['RA'])} | {_fmt(gated['RA'])} |",
        f"| FE (conditioned) | {_fmt(metrics['FE'])} | {_fmt(gated['FE'])} |",
        f"| FE (raw) | {_fmt(metrics['FE_raw'])} | {_fmt(gated['FE_raw'])} |",
        f"| PR (conditioned) | {_fmt(metrics['PR'])} | {_fmt(gated['PR'])} |",
        f"| PR (raw) | {_fmt(metrics['PR_raw'])} | {_fmt(gated['PR_raw'])} |",
        f"| RLA | {_fmt(metrics['RLA'])} | {_fmt(gated['RLA'])} |",
        f"| AR (avg) | {_fmt(ar['average'])} | {_fmt(gated['AR']['average'])} |",
        f"| FA (forgetting ability) | {_fmt(metrics['FA'])} | {_fmt(gated['FA'])} |",
        f"| UG (usability guard) | {_fmt(metrics['UG'])} | {_fmt(gated['UG'])} |",
        f"| **Bemula Score** | {_fmt(metrics['bemula_score'])} "
        f"| {_fmt(gated['bemula_score'])} |",
        "",
        f"Instance Pass Rate: {_fmt(metrics['instance_pass_rate'])}",
        f"Request Pass Rate (RPR, strictest): {_fmt(metrics['request_pass_rate'])}",
        "",
        "## AR by attack type",
        "",
        "| Attack type | AR |",
        "|-------------|----|",
    ]
    for atype, value in ar["by_attack_type"].items():
        lines.append(f"| {atype} | {_fmt(value)} |")
    lines += ["", "## Suite pass rates", "", "| Suite | Pass rate |", "|-------|-----------|"]
    for suite, value in metrics["suite_pass_rates"].items():
        lines.append(f"| {suite} | {_fmt(value)} |")
    lines += ["", "## By forget type", "", "| Type | RA | FE | PR | RLA | AR | Bemula |",
              "|------|----|----|----|-----|----|--------|"]
    for ftype, block in metrics["by_forget_type"].items():
        lines.append(
            f"| {ftype} | {_fmt(block['RA'])} | {_fmt(block['FE'])} | {_fmt(block['PR'])} "
            f"| {_fmt(block['RLA'])} | {_fmt(block['AR']['average'])} "
            f"| {_fmt(block['bemula_score'])} |"
        )
    if metrics.get("by_length_bucket"):
        lines += ["", "## By length bucket", "",
                  "| Bucket | RA | FE | PR | RLA | AR | Bemula |",
                  "|--------|----|----|----|-----|----|--------|"]
        for bucket, block in metrics["by_length_bucket"].items():
            lines.append(
                f"| {bucket} | {_fmt(block['RA'])} | {_fmt(block['FE'])} "
                f"| {_fmt(block['PR'])} | {_fmt(block['RLA'])} "
                f"| {_fmt(block['AR']['average'])} | {_fmt(block['bemula_score'])} |"
            )
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str | None:
    """Best-effort current commit hash; None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_run_meta(args: argparse.Namespace, adapter_name: str, embedder_provider: str,
                   judge_temperature: float, started_at: str, num_instances: int,
                   num_items: int, out: Path) -> None:
    """Record the run's provenance so every metrics.json stays attributable."""
    meta = {
        "baseline": args.baseline,
        "adapter": adapter_name,
        "agent_model": os.environ.get("BEMULA_AGENT_MODEL"),
        "judge_model": os.environ.get("BEMULA_JUDGE_MODEL"),
        "judge_temperature": judge_temperature,
        "thinking": os.environ.get("BEMULA_THINKING", "disabled"),
        "embed_backend": os.environ.get("BEMULA_EMBED_BACKEND", "api"),
        "embedder_provider": embedder_provider,
        "config": args.config,
        "dialogues": args.dialogues,
        "questions": args.questions,
        "limit": args.limit,
        "workers": args.workers,
        "resume": args.resume,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "num_instances": num_instances,
        "num_items": num_items,
    }
    (out / "run_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    started_at = datetime.now().isoformat(timespec="seconds")
    ap = argparse.ArgumentParser(description="Run the Bemula control-pair evaluation.")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--baseline", default="rag",
                    help="baseline name (rag, mem0, amem, simplemem, memoryos, "
                         "lightmem, langmem, fullcontext)")
    ap.add_argument("--dialogues", default="data/dialogues/dialogues.jsonl")
    ap.add_argument("--questions", default="data/questions/questions.jsonl")
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--limit", type=int, default=None, help="max number of instances")
    ap.add_argument("--embedder-provider", default=None,
                    choices=["openai", "huggingface"],
                    help="mem0 embedder backend (default: derived from "
                         "BEMULA_EMBED_BACKEND in .env: api -> openai, "
                         "local -> huggingface)")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel instances; one adapter instance is built per worker")
    ap.add_argument("--resume", action="store_true",
                    help="skip instances already present in the out dir's items.jsonl")
    args = ap.parse_args()

    config = load_config(args.config)
    load_dotenv(override=True)
    judge_temp = config.get("evaluation", {}).get("judge_temperature", 0.0)

    # Embedding backend: api (OpenAI-compatible endpoint) or local
    # (sentence-transformers, weights from hf-mirror.com).
    embed_backend = os.environ.get("BEMULA_EMBED_BACKEND", "api").strip().lower()
    if embed_backend == "local":
        # huggingface_hub reads HF_ENDPOINT at import time; set it before any
        # adapter imports sentence_transformers.
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    embedder_provider = args.embedder_provider
    if embedder_provider is None:
        if embed_backend not in ("api", "local"):
            raise SystemExit(
                f"unknown BEMULA_EMBED_BACKEND: {embed_backend!r} (expected api|local)"
            )
        embedder_provider = {"api": "openai", "local": "huggingface"}[embed_backend]

    judge: Judge = LLMJudge(get_llm(model=os.environ.get("BEMULA_JUDGE_MODEL")),
                            temperature=judge_temp)
    agent_llm = get_llm(model=os.environ.get("BEMULA_AGENT_MODEL"))

    instances = load_jsonl(args.dialogues, DialogueInstance)
    if args.limit is not None:
        instances = instances[: args.limit]
    items = load_jsonl(args.questions, TestItem)
    items_by_instance: dict[str, list[TestItem]] = {}
    for item in items:
        items_by_instance.setdefault(item.instance_id, []).append(item)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items_path = out / "items.jsonl"

    # Checkpointing: each finished instance's records are appended to
    # items.jsonl immediately, so a killed run keeps partial results and
    # --resume can skip completed instances.
    done_ids: set[str] = set()
    if args.resume and items_path.is_file():
        done_ids = {r.instance_id for r in load_jsonl(items_path, ItemRecord)}
        if done_ids:
            print(f"Resuming: skipping {len(done_ids)} completed instance(s)")
    elif items_path.is_file():
        items_path.unlink()  # fresh run: discard stale records
    todo = [i for i in instances if i.instance_id not in done_ids]

    def checkpoint(recs: list[ItemRecord]) -> None:
        with items_path.open("a", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r.model_dump(mode="json"), ensure_ascii=False) + "\n")

    if args.workers > 1:
        # Adapters are built sequentially (heavyweight local-store init must
        # not race), then instances are evaluated concurrently.
        adapters = [
            _build_baseline(args.baseline, agent_llm, embedder_provider)
            for _ in range(args.workers)
        ]
        adapter_name = adapters[0].name
        run_evaluation_parallel(adapters, todo, items_by_instance, judge,
                                on_instance=checkpoint)
    else:
        adapter = _build_baseline(args.baseline, agent_llm, embedder_provider)
        adapter_name = adapter.name
        for inst in tqdm(todo, desc="eval", unit="inst"):
            checkpoint(run_instance(adapter, inst,
                                    items_by_instance.get(inst.instance_id, []), judge))

    records = load_jsonl(items_path, ItemRecord)
    metrics = compute_metrics(records)

    (out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_report(metrics, adapter_name, out)
    write_run_meta(args, adapter_name, embedder_provider, judge_temp, started_at,
                   metrics["num_instances"], metrics["num_items"], out)
    print(f"Wrote {len(records)} item records, metrics, and report to {out}")
    print(f"Bemula Score: {_fmt(metrics['bemula_score'])} "
          f"(RA-gated: {_fmt(metrics['gated']['bemula_score'])}), "
          f"RA: {_fmt(metrics['RA'])}")


if __name__ == "__main__":
    main()
