# Bemula

A Benchmark for Memory Unlearning in LLM Agents. Bemula tests whether agents can remove specific information from long-term memory, retain unrelated facts, and learn new ones after a forget request.

Projrct page and online leaderboard: [https://nboxff.github.io/bemula-leaderboard](https://nboxff.github.io/bemula-leaderboard)

## Setup

```bash
uv sync                   # core dependencies (Python >= 3.10)
cp .env.example .env      # fill in OPENAI_BASE_URL / OPENAI_API_KEY / BEMULA_*_MODEL
```

All LLM calls (generation, judging, baselines) go through the single
OpenAI-compatible endpoint configured in `.env`.

## Run evaluation

Basic form:

```bash
uv run python -m scripts.run_eval \
    --baseline simplemem \
    --dialogues data/core/dialogues.jsonl \
    --questions data/core/questions.jsonl \
    --out results/simplemem_core \
    --workers 8 \
    --resume
```

Available baselines: `rag`, `mem0`, `amem`, `simplemem`, `lightmem`, `langmem`, `fullcontext`.

Key options:

- `--limit N`: evaluate only the first N instances (smoke runs).
- `--workers N`: evaluate N instances in parallel (one adapter per worker).
- `--resume`: skip instances already recorded in the out dir's `items.jsonl`.

Outputs under `--out`: `items.jsonl` (per-item records), `metrics.json`,
`report.md`, `run_meta.json`.

### Baseline-specific requirements

- `amem` / `simplemem` / `memoryos` / `lightmem` load from their official
  repository clones; set `BEMULA_AMEM_PATH` / `BEMULA_SIMPLEMEM_PATH` /
  `BEMULA_MEMORYOS_PATH` / `BEMULA_LIGHTMEM_PATH` in `.env` (fallback: a
  sibling directory like `../SimpleMem`).
- `simplemem` needs extra packages passed via `--with` (see the production
  example below); `lancedb` must stay below 0.37.
- Alternatively install everything once: `uv sync --extra baselines`.

### Production run (background, logged, resumable)

```bash
mkdir -p logs results/simplemem_core_deepseek

BEMULA_THINKING=disabled \
PYTHONUNBUFFERED=1 \
setsid nohup uv run --frozen \
  --with "lancedb==0.25.3" \
  --with "pyarrow==17.0.0" \
  --with pylance \
  --with dateparser \
  --with tantivy \
  python -m scripts.run_eval \
  --config configs/default.yaml \
  --baseline simplemem \
  --dialogues data/core/dialogues.jsonl \
  --questions data/core/questions.jsonl \
  --out results/simplemem_core_deepseek \
  --workers 8 \
  --resume \
  > logs/simplemem_core_deepseek.log 2>&1 < /dev/null &

echo "pid=$!"
```

## Data generation

Only needed when regenerating the benchmark data (ready-made splits live in
`data/core`, `data/full`, `data/quickcheck`, `data/smallcheck`):

```bash
# 1. User profiles
uv run python -m scripts.generate_profiles --config configs/default.yaml \
    --out data/profiles.jsonl

# 2. Dialogues with embedded forget requests (+ paired control variants)
uv run python -m scripts.generate_dialogues --config configs/default.yaml \
    --profiles data/profiles.jsonl --out data/dialogues.jsonl

# 3. Test items
uv run python -m scripts.generate_questions --config configs/default.yaml \
    --profiles data/profiles.jsonl --dialogues data/dialogues.jsonl \
    --out data/questions.jsonl
```

Every generation script accepts `--limit N` for small trial runs.


## Citation

```tex
@article{xie2026can,
  title={Can agents forget when asked? A benchmark for memory unlearning in LLM agents},
  author={Xie, Bofan and Wang, Chaokun and Ge, Guanchen and Wang, Songyao and Niu, Fang and Zhang, Hang and Wang, Wei},
  journal={arXiv preprint arXiv:2610.18052},
  year={2026}
}
```
