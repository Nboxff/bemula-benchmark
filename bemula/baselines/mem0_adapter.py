"""Mem0 baseline adapter.

Wraps the official `mem0ai` pip package (declared under the `baselines`
extra, lazily imported). Mem0 is configured against the OpenAI-compatible
endpoint from .env: LLM = BEMULA_AGENT_MODEL, embedder = BEMULA_EMBED_MODEL,
vector store = local Qdrant (embedded path, no server).
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

from bemula.baselines.base import MemoryAdapter, TurnProgress
from bemula.baselines.serving import pin_client_serving
from bemula.llm import (
    BaseLLM,
    resolve_embed_endpoint,
    resolve_endpoint,
    thinking_extra_body,
)
from bemula.schemas import Turn

_INSTALL_HINT = (
    "Mem0Adapter requires the official 'mem0ai' package, which is not installed.\n"
    "Install the declared optional extra (do NOT add it to core dependencies):\n"
    "    uv sync --extra baselines\n"
    "and configure an OpenAI-compatible endpoint in .env. "
    "See https://github.com/mem0ai/mem0"
)

# Embedding dimensions for known models; required by the local vector store.
_EMBED_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
    "all-MiniLM-L6-v2": 384,
}

_ANSWER_PROMPT = (
    "You are a personal assistant with access to remembered conversation "
    "history. Answer the user's question using ONLY the context below. "
    "If the answer is not in the context, say you don't have that "
    "information. Keep the answer short.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
def _mem0_add(memory: Any, messages: list[dict[str, str]], user_id: str) -> Any:
    """mem0.add with retry: its internal LLM extraction surfaces transient
    gateway errors (502/504) and malformed extraction responses as exceptions,
    and a fresh attempt re-runs the LLM call. The observed failures are raised
    before any vector-store write, so a retry does not duplicate memories."""
    return memory.add(messages, user_id=user_id)


@retry(stop=stop_after_attempt(4), wait=wait_exponential(multiplier=1, min=2, max=30), reraise=True)
def _mem0_search(memory: Any, question: str, user_id: str) -> Any:
    """mem0.search with retry, same failure mode as _mem0_add."""
    return memory.search(question, filters={"user_id": user_id})


def _coerce_memory_schema(response: Any) -> Any:
    """Normalize a malformed mem0 extraction response back to the expected
    schema ({"memory": [{"text": ...}, ...]}).

    mem0's parser only checks that the response is valid JSON; a response
    whose "memory" value is a plain string or a list of strings slips through
    and crashes _add_to_vector_store ('str' object has no attribute 'get').
    The extraction LLM runs at temperature 0, so such a response reproduces
    deterministically and retrying cannot help; coercing the schema is the
    only way past that turn. Responses without a "memory" dict-list mismatch
    pass through untouched.
    """
    if not isinstance(response, str):
        return response
    from mem0.memory.utils import remove_code_blocks

    try:
        data = json.loads(remove_code_blocks(response), strict=False)
    except (json.JSONDecodeError, TypeError, ValueError):
        return response
    if not isinstance(data, dict):
        return response
    memories = data.get("memory")
    if isinstance(memories, str):
        data["memory"] = [{"text": memories}] if memories.strip() else []
    elif isinstance(memories, list) and any(not isinstance(m, dict) for m in memories):
        data["memory"] = [
            {"text": str(m)} if not isinstance(m, dict) else m for m in memories
        ]
    else:
        return response
    return json.dumps(data, ensure_ascii=False)


def _build_config(work_dir: Path, embedder_provider: str) -> dict[str, Any]:
    """Mem0 config pinned to the .env endpoint with a local vector store.

    Endpoint routing: the LLM follows bemula.llm.resolve_endpoint; the
    embedder follows bemula.llm.resolve_embed_endpoint (dedicated
    EMBED_PROXY_* endpoint when configured).
    """
    load_dotenv(override=True)
    agent_model = os.environ.get("BEMULA_AGENT_MODEL", "deepseek-v4-flash")
    embed_model = os.environ.get("BEMULA_EMBED_MODEL", "text-embedding-3-small")
    llm_base_url, llm_api_key = resolve_endpoint(agent_model)

    if embedder_provider == "openai":
        embed_base_url, embed_api_key = resolve_embed_endpoint(embed_model)
        embedder = {
            "provider": "openai",
            "config": {
                "model": embed_model,
                "api_key": embed_api_key,
                "openai_base_url": embed_base_url,
            },
        }
        dims = _EMBED_DIMS.get(embed_model, 1536)
    elif embedder_provider == "huggingface":
        # Local sentence-transformers model (declared in the baselines extra),
        # downloaded from hf-mirror.com on first use. Not taken from
        # BEMULA_EMBED_MODEL, which names an OpenAI-compatible model.
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        hf_model = os.environ.get("BEMULA_HF_EMBED_MODEL", "all-MiniLM-L6-v2")
        embedder = {"provider": "huggingface", "config": {"model": hf_model}}
        dims = _EMBED_DIMS.get(hf_model, 384)
    else:
        raise ValueError(f"unknown embedder provider: {embedder_provider}")

    # Reasoning-model families (GPT-5 series, DeepSeek) reject temperature/top_p;
    # mem0's name heuristic misses some of them, so classify explicitly. When
    # set, mem0 only sends messages/response_format/tools/reasoning_effort.
    reasoning_markers = ("gpt-5", "deepseek", "o1", "o3", "o4")
    is_reasoning = any(m in agent_model.lower() for m in reasoning_markers)
    # mem0 owns its OpenAI client, so the benchmark's thinking switch has to
    # travel through its config; reasoning deployments are pinned to
    # reasoning_effort="none" while BEMULA_THINKING is disabled.
    reasoning_effort = (thinking_extra_body(agent_model) or {}).get("reasoning_effort")

    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model": agent_model,
                "temperature": 0.0,
                # Reasoning models spend hidden tokens; a small budget yields
                # empty responses (same pitfall as in bemula/llm.py).
                "max_tokens": 16384,
                "api_key": llm_api_key,
                "openai_base_url": llm_base_url,
                "is_reasoning_model": is_reasoning,
                "reasoning_effort": reasoning_effort,
            },
        },
        "embedder": embedder,
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": "bemula",
                "embedding_model_dims": dims,
                "path": str(work_dir / "qdrant"),
            },
        },
        "history_db_path": str(work_dir / "history.db"),
        "version": "v1.1",
    }


class Mem0Adapter(MemoryAdapter):
    """Mem0 `Memory`: observe -> add turn pairs; query -> search + agent LLM."""

    def __init__(
        self,
        llm: BaseLLM,
        user_id: str = "bemula-user",
        embedder_provider: str = "openai",
        work_dir: str | Path | None = None,
    ) -> None:
        os.environ.setdefault("MEM0_TELEMETRY", "false")  # before importing mem0
        try:
            from mem0 import Memory  # type: ignore[import-not-found]
            from mem0.llms.openai import OpenAILLM  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc

        load_dotenv(override=True)
        agent_model = os.environ.get("BEMULA_AGENT_MODEL", "deepseek-v4-flash")
        # mem0 builds its own OpenAI client, so its internal extraction calls
        # need the benchmark's serving settings re-applied before Memory
        # constructs the LLM.
        pin_client_serving(OpenAILLM, agent_model)
        self.llm = llm
        self.user_id = user_id
        self.name = "mem0"
        self._work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="bemula-mem0-"))
        self._work_dir.mkdir(parents=True, exist_ok=True)
        # mem0 keeps a process-global home dir (~/.mem0) for migration state and
        # opens a Qdrant lock on it at Memory init, so two adapters in one
        # process conflict. Point the module-level home at this adapter's work
        # dir before construction; adapters are built sequentially.
        import mem0.memory.main as mem0_main

        mem0_main.mem0_dir = str(self._work_dir / "mem0_home")
        Path(mem0_main.mem0_dir).mkdir(parents=True, exist_ok=True)
        # Qdrant's local store probes the SQLite build options with a
        # table-valued pragma that older system SQLite builds (e.g. CentOS 7)
        # do not support. The same-thread check is redundant here because each
        # adapter instance is driven by a single worker thread.
        from qdrant_client.local.persistence import CollectionPersistence

        CollectionPersistence.CHECK_SAME_THREAD = False
        self._memory = Memory.from_config(_build_config(self._work_dir, embedder_provider))
        # mem0's extraction LLM (temperature 0) can deterministically return
        # JSON whose "memory" field is not a list of dicts; normalize such
        # responses so _add_to_vector_store does not crash on them.
        llm = self._memory.llm
        original_generate = llm.generate_response

        def generate_response(*args: Any, **kwargs: Any) -> Any:
            return _coerce_memory_schema(original_generate(*args, **kwargs))

        llm.generate_response = generate_response
        # Buffer to pair each user turn with the following assistant turn.
        self._pending_user: str | None = None
        self._progress = TurnProgress("Mem0")

    def begin_instance(self, instance: Any) -> None:
        self._progress.begin_instance(instance)

    def end_instance(self) -> None:
        self._progress.end_instance()

    def begin_session(self, session: Any) -> None:
        self._progress.begin_session(session)

    def end_variant(self) -> None:
        self._progress.end_variant()

    def reset(self) -> None:
        self._memory.delete_all(user_id=self.user_id)
        self._pending_user = None
        self._progress.start_variant()

    def observe(self, turn: Turn) -> None:
        if turn.speaker == "user":
            self._flush_pending()
            self._pending_user = turn.text
        else:
            messages = []
            if self._pending_user is not None:
                messages.append({"role": "user", "content": self._pending_user})
                self._pending_user = None
            messages.append({"role": "assistant", "content": turn.text})
            _mem0_add(self._memory, messages, self.user_id)
        self._progress.update()

    def query(self, question: str) -> str:
        self._flush_pending()  # a trailing user turn may still be buffered
        memories = self._search(question)
        if not memories:
            return "I don't have any relevant information stored."
        context = "\n".join(f"- {m}" for m in memories)
        return self.llm.chat(
            [{"role": "user", "content": _ANSWER_PROMPT.format(context=context, question=question)}],
            temperature=0.0,
            max_tokens=512,
            task="mem0_answer",
        ).strip()

    def _flush_pending(self) -> None:
        if self._pending_user is not None:
            _mem0_add(
                self._memory, [{"role": "user", "content": self._pending_user}], self.user_id
            )
            self._pending_user = None

    def _entries(self, results: Any) -> list[dict[str, Any]]:
        """Normalize search results across mem0 versions."""
        entries = results.get("results", results) if isinstance(results, dict) else results
        return [e for e in entries if isinstance(e, dict)]

    def _search(self, question: str) -> list[str]:
        results = _mem0_search(self._memory, question, self.user_id)
        return [str(e.get("memory", e.get("text", e))) for e in self._entries(results)]
