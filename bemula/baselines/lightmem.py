"""LightMem baseline adapter.

Wraps the official LightMem repository (https://github.com/zjunlp/LightMem).
LightMem runs a three-stage pipeline: an LLMLingua-2 pre-compressor shortens the
incoming messages (sensory memory), an attention-based topic segmenter cuts them
into coherent segments (short-term memory), and an LLM then extracts structured
memory entries that are embedded into a vector store (long-term memory). At query
time the entries closest to the question are retrieved and the benchmark's agent
LLM answers from them, so the answer side stays uniform across baselines.

Deviations from the stock LongMemEval script, all deliberate:

- The internal LLM is pinned to the OpenAI-compatible endpoint from .env
  (BEMULA_AGENT_MODEL via resolve_endpoint), and the default local
  SentenceTransformer embedder is replaced by a shim backed by the shared
  embedder from `rag.get_embedder` (API endpoint or local model, per
  BEMULA_EMBED_BACKEND). Both are the conventions the other wrappers follow.
- Pre-compression and topic segmentation run on CPU from the same LLMLingua-2
  checkpoint. The two stages do not share one loaded model: the segmenter needs
  per-layer attention maps, which the SDPA kernel does not return, so it loads
  its own copy with eager attention.
- The script forces segmentation and extraction only on the last turn of a
  dialogue; this adapter forces them on the last turn of every variant, which is
  the same point under Bemula's observe-then-probe protocol.
- Long-term memory is insert-only: `update` stays at LightMem's "offline"
  default, and the optional merge queue
  (construct_update_queue_all_entries / offline_update_all_entries) is never
  triggered, matching the stock LongMemEval script.
"""

from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from bemula.baselines.base import MemoryAdapter, TurnProgress
from bemula.baselines.serving import pin_client_serving
from bemula.llm import BaseLLM, resolve_endpoint
from bemula.schemas import Turn

_REPO_URL = "https://github.com/zjunlp/LightMem"
_INSTALL_HINT = (
    "LightmemAdapter requires the official LightMem repository, which is not "
    "available.\nClone it and point BEMULA_LIGHTMEM_PATH at the clone:\n"
    f"    git clone {_REPO_URL}\n"
    "    export BEMULA_LIGHTMEM_PATH=/path/to/LightMem\n"
    "Install llmlingua into the baselines extra environment and make the "
    "LLMLingua-2 weights reachable (HF_ENDPOINT=https://hf-mirror.com), then "
    "re-run."
)

_DEFAULT_COMPRESSOR_MODEL = (
    "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
def _lightmem_retrieve(memory: Any, question: str, limit: int) -> Any:
    """memory.retrieve with retry, same failure mode as LightmemAdapter._add."""
    return memory.retrieve(question, limit=limit)

_ANSWER_PROMPT = (
    "You are a personal assistant with access to remembered conversation "
    "history. Answer the user's question using ONLY the context below. "
    "If the answer is not in the context, say you don't have that "
    "information. Keep the answer short.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)

# LightMem rejects a message whose `time_stamp` is empty, while `Turn.timestamp`
# is a required-but-possibly-empty string. Relearning context turns in particular
# carry the new information without a stated time, so they inherit the most
# recent timestamp of the variant; only a variant with no timestamps at all
# falls back to this sentinel.
_FALLBACK_TIMESTAMP = "1970-01-01T00:00:00Z"

# LightMem parses session times as "2023/05/20 (Sat) 00:44" and only falls back
# to `datetime.fromisoformat`, which rejects the trailing "Z" Bemula timestamps
# carry until Python 3.11. Rendering the native format keeps the weekday the
# segmenter relies on.
_LIGHTMEM_TIME_FORMAT = "%Y/%m/%d (%a) %H:%M:%S"


def _lightmem_timestamp(raw: str) -> str:
    """Convert a Bemula ISO timestamp into the format LightMem expects."""
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return raw
    return parsed.strftime(_LIGHTMEM_TIME_FORMAT)


def _find_repo_path() -> Path:
    """Locate the LightMem clone.

    ``BEMULA_LIGHTMEM_PATH`` takes precedence.  When it is not set, also try a
    sibling ``LightMem`` directory because this repository is developed
    alongside the LightMem clone in the same workspace.
    """
    env_path = os.environ.get("BEMULA_LIGHTMEM_PATH")
    if env_path:
        return Path(env_path)

    sibling = Path(__file__).resolve().parents[3] / "LightMem"
    if sibling.is_dir():
        return sibling

    return Path("LightMem")


class _SharedEmbedder:
    """Adapt the benchmark's shared embedder to LightMem's text_embedder API.

    LightMem's embedder is called with a single string (query and segment
    embeddings) or with a list of strings, and returns plain lists. The shared
    embedders take a list and return a float32 matrix.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder
        self.total_calls = 0
        self.total_tokens: int | None = None

    def embed(self, text: str | list[str]) -> list[Any]:
        self.total_calls += 1
        if isinstance(text, str):
            return self._embedder.embed([text])[0].tolist()
        return [vector.tolist() for vector in self._embedder.embed(list(text))]

    def get_stats(self) -> dict[str, Any]:
        return {"total_calls": self.total_calls, "total_tokens": self.total_tokens}


def _build_config(
    *,
    work_dir: Path,
    dims: int,
    agent_model: str,
    base_url: str | None,
    api_key: str,
    embed_model: str,
    compressor_model: str,
) -> dict[str, Any]:
    """Assemble the LightMem configuration for the standard pipeline.

    Long-term memory is insert-only (`update: "offline"` with neither update
    trigger set), matching the stock LongMemEval script.
    """
    return {
        "pre_compress": True,
        "pre_compressor": {
            "model_name": "llmlingua-2",
            "configs": {
                "llmlingua_config": {
                    "model_name": compressor_model,
                    "device_map": "cpu",
                    "use_llmlingua2": True,
                },
            },
        },
        "topic_segment": True,
        # The segmenter reads per-layer attention maps to place topic
        # boundaries, which the SDPA kernel does not return; it therefore loads
        # its own copy of the backbone with eager attention instead of sharing
        # the compressor's model.
        "precomp_topic_shared": False,
        "topic_segmenter": {
            "model_name": "llmlingua-2",
            "configs": {
                "model_name": compressor_model,
                "model_config": {"attn_implementation": "eager"},
            },
        },
        "messages_use": "user_only",
        "metadata_generate": True,
        "text_summary": True,
        "memory_manager": {
            "model_name": "openai",
            "configs": {
                "model": agent_model,
                "api_key": api_key,
                "openai_base_url": base_url,
                "max_tokens": 16000,
            },
        },
        "index_strategy": "embedding",
        # LightMemory builds this embedder during construction; it is
        # immediately replaced by the shared-embedder shim, so the endpoint
        # here only has to be constructible.
        "text_embedder": {
            "model_name": "openai",
            "configs": {
                "model": embed_model,
                "api_key": api_key,
                "openai_base_url": base_url,
                "embedding_dims": dims,
            },
        },
        "retrieve_strategy": "embedding",
        "embedding_retriever": {
            "model_name": "qdrant",
            "configs": {
                "collection_name": "bemula_lightmem",
                "embedding_model_dims": dims,
                "path": str(work_dir / "qdrant"),
            },
        },
        "update": "offline",
        "history_db_path": str(work_dir / "history.db"),
        "kv_cache_path": str(work_dir / "kv_cache.db"),
    }


class LightmemAdapter(MemoryAdapter):
    """LightMem three-stage compressed memory with embedding retrieval."""

    name = "lightmem"

    def __init__(
        self,
        llm: BaseLLM,
        repo_path: str | Path | None = None,
        embedder: Any | None = None,
        work_dir: str | Path | None = None,
        retrieve_limit: int = 10,
        compressor_model: str | None = None,
    ) -> None:
        path = Path(repo_path) if repo_path is not None else _find_repo_path()
        if not path.is_dir():
            raise RuntimeError(_INSTALL_HINT)
        # LightMem is a src-layout package; the clone root itself is not on the
        # import path.
        src = path / "src"
        import_root = src if (src / "lightmem").is_dir() else path
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

        # huggingface_hub reads HF_ENDPOINT at import time; the weights are
        # downloaded on first use and huggingface.co is not always reachable.
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        try:
            from lightmem.factory.memory_manager.openai import OpenaiManager
            from lightmem.memory.lightmem import LightMemory
        except ImportError as exc:
            raise RuntimeError(
                f"Could not import LightMem from {path}.\n{_INSTALL_HINT}"
            ) from exc

        if embedder is None:
            from bemula.baselines.rag import get_embedder

            embedder = get_embedder()
        load_dotenv(override=True)
        agent_model = os.environ.get("BEMULA_AGENT_MODEL", "deepseek-v4-flash")
        base_url, api_key = resolve_endpoint(agent_model)
        self.llm = llm
        self._retrieve_limit = retrieve_limit

        self._work_dir = Path(work_dir) if work_dir else Path(
            tempfile.mkdtemp(prefix="bemula-lightmem-")
        )
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # LightMem builds its own OpenAI client for extraction and summarizing,
        # so its calls need the benchmark's serving settings re-applied.
        pin_client_serving(OpenaiManager, agent_model)

        self._text_embedder = _SharedEmbedder(embedder)
        dims = len(embedder.embed(["dimension probe"])[0])

        config = _build_config(
            work_dir=self._work_dir,
            dims=dims,
            agent_model=agent_model,
            base_url=base_url,
            api_key=api_key,
            embed_model=os.environ.get("BEMULA_EMBED_MODEL", "text-embedding-3-small"),
            compressor_model=compressor_model or _DEFAULT_COMPRESSOR_MODEL,
        )
        # Qdrant's local store probes the SQLite build options with a
        # table-valued pragma that older system SQLite builds (e.g. CentOS 7)
        # do not support. The same-thread check is redundant here because each
        # adapter instance is driven by a single worker thread. qdrant-client
        # ships with the LightMem runtime, not with the core dependencies, so
        # skip the patch when it is absent.
        try:
            from qdrant_client.local.persistence import CollectionPersistence
        except ImportError:
            pass
        else:
            CollectionPersistence.CHECK_SAME_THREAD = False
        self._memory = LightMemory.from_config(config)
        self._memory.text_embedder = self._text_embedder

        self._pending: dict[str, str] | None = None
        self._last_timestamp = _FALLBACK_TIMESTAMP
        self._progress = TurnProgress("LightMem")

    def begin_instance(self, instance: Any) -> None:
        self._progress.begin_instance(instance)

    def end_instance(self) -> None:
        self._progress.end_instance()

    def begin_session(self, session: Any) -> None:
        self._progress.begin_session(session)

    def reset(self) -> None:
        # The Qdrant collection holds every memory entry; the sensory and
        # short-term buffers hold the tail the pipeline has not flushed yet.
        self._memory.embedding_retriever.reset()
        senmem = self._memory.senmem_buffer_manager
        senmem.buffer.clear()
        senmem.big_buffer.clear()
        senmem.token_count = 0
        shortmem = self._memory.shortmem_buffer_manager
        shortmem.buffer.clear()
        shortmem.token_count = 0
        self._pending = None
        self._last_timestamp = _FALLBACK_TIMESTAMP
        self._progress.start_variant()

    def observe(self, turn: Turn) -> None:
        # LightMem flushes its buffers only when asked to, so the previous turn
        # is held back until the next one arrives: the turn that turns out to be
        # the variant's last is the one that forces segmentation and extraction.
        if self._pending is not None:
            self._add(self._pending, force=False)
        self._last_timestamp = turn.timestamp or self._last_timestamp
        self._pending = {
            "role": turn.speaker,
            "content": turn.text,
            "time_stamp": _lightmem_timestamp(self._last_timestamp),
        }
        self._progress.update()

    def end_variant(self) -> None:
        if self._pending is not None:
            self._add(self._pending, force=True)
            self._pending = None
        self._progress.end_variant()

    def query(self, question: str) -> str:
        # Relearning probes observe their context turns after end_variant();
        # flush the held-back turn so those turns reach memory before probing.
        if self._pending is not None:
            self._add(self._pending, force=True)
            self._pending = None
        entries = _lightmem_retrieve(self._memory, question, self._retrieve_limit)
        if not entries:
            return "I don't have any relevant information stored."
        context = "\n".join(f"- {entry}" for entry in entries)
        return self.llm.chat(
            [{"role": "user", "content": _ANSWER_PROMPT.format(context=context, question=question)}],
            temperature=0.0,
            max_tokens=512,
            task="lightmem_answer",
        ).strip()

    @retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
    def _add(self, message: dict[str, str], *, force: bool) -> None:
        """add_memory with retry: a transient gateway error (502/504) or
        otherwise malformed memory_manager response crashes inside LightMem's
        process_extraction_results ('NoneType' object has no attribute 'get'),
        which killed the core eval run overnight. The failure happens while
        bookkeeping extraction results, before any store write, so a retry
        cannot duplicate entries."""
        self._memory.add_memory(
            messages=[message], force_segment=force, force_extract=force
        )
