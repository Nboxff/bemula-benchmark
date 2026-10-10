"""SimpleMem baseline adapter.

Wraps the official SimpleMem repository (https://github.com/aiming-lab/SimpleMem).
SimpleMem compresses observed dialogue turns into structured memory entries
(lossless restatement + keywords + symbolic metadata) with multi-view indexing;
at query time its intent-aware hybrid retriever fetches candidate entries and
the benchmark's agent LLM answers from them, so the answer side stays uniform
across baselines.

The internal LLM is pinned to the OpenAI-compatible endpoint from .env
(BEMULA_AGENT_MODEL via resolve_endpoint). The default local
SentenceTransformer embedder is replaced by a shim backed by the shared
embedder from `rag.get_embedder` (API endpoint or local model, per
BEMULA_EMBED_BACKEND).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from bemula.baselines.base import MemoryAdapter, TurnProgress
from bemula.baselines.serving import last_finish_reason, pin_client_serving
from bemula.llm import BaseLLM, resolve_endpoint
from bemula.schemas import Turn

_REPO_URL = "https://github.com/aiming-lab/SimpleMem"
_INSTALL_HINT = (
    "SimpleMemAdapter requires the official SimpleMem repository, which is not "
    "available.\nClone it and point BEMULA_SIMPLEMEM_PATH at the clone:\n"
    f"    git clone {_REPO_URL}\n"
    "    export BEMULA_SIMPLEMEM_PATH=/path/to/SimpleMem\n"
    "Install its runtime requirements (lancedb, dateparser, "
    "sentence-transformers) into the baselines extra environment, then re-run."
)

_ANSWER_PROMPT = (
    "You are a personal assistant with access to remembered conversation "
    "history. Answer the user's question using ONLY the context below. "
    "If the answer is not in the context, say you don't have that "
    "information. Keep the answer short.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


# SimpleMem compresses a whole window (WINDOW_SIZE dialogues, 40 by default)
# with a single LLM call and returns an empty list when that reply cannot be
# parsed. A reply that outgrows the model's output budget is truncated
# mid-array, so parsing fails and the window - with every fact in it - would be
# dropped silently. Halving such a window keeps the same text but fits the
# budget; recursion is bounded so a genuinely empty window costs only a few
# extra calls.
_SPLIT_MIN_DIALOGUES = 8
_SPLIT_MAX_DEPTH = 2


def _install_window_split_retry(builder: Any, client: Any) -> None:
    """Retry an unparseable window as two halves instead of dropping it."""
    original = builder._generate_memory_entries  # noqa: SLF001 - wraps the system's own hook

    def generate(dialogues: list[Any], depth: int = 0) -> list[Any]:
        entries = original(dialogues)
        if entries or len(dialogues) < _SPLIT_MIN_DIALOGUES or depth >= _SPLIT_MAX_DEPTH:
            if not entries and dialogues:
                print(
                    f"[bemula] SimpleMem extracted no entries from a "
                    f"{len(dialogues)}-dialogue window "
                    f"(finish_reason={last_finish_reason(client)!r})"
                )
            return entries
        middle = len(dialogues) // 2
        return generate(dialogues[:middle], depth + 1) + generate(dialogues[middle:], depth + 1)

    builder._generate_memory_entries = generate  # noqa: SLF001 - see above


def _find_repo_path() -> Path:
    """Locate the SimpleMem clone.

    ``BEMULA_SIMPLEMEM_PATH`` takes precedence.  When it is not set, also try a
    sibling ``SimpleMem`` directory because this repository is developed
    alongside the SimpleMem clone in the same workspace.
    """
    env_path = os.environ.get("BEMULA_SIMPLEMEM_PATH")
    if env_path:
        return Path(env_path)

    sibling = Path(__file__).resolve().parents[3] / "SimpleMem"
    if sibling.is_dir():
        return sibling

    return Path("SimpleMem")


class _APIEmbeddingModel:
    """Route SimpleMem's embedding calls to the benchmark's API embedder.

    SimpleMem's ``EmbeddingModel`` loads a local SentenceTransformer; the vector
    store only relies on ``dimension``, ``encode_documents`` and
    ``encode_single``, so this shim provides exactly those against the
    OpenAI-compatible embedding endpoint used by the other baselines.
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder
        self.dimension = len(embedder.embed(["dimension probe"])[0])

    def encode_documents(self, texts: list[str]) -> list[np.ndarray]:
        return [np.asarray(v, dtype=np.float32) for v in self._embedder.embed(texts)]

    def encode_single(self, text: str, is_query: bool = False) -> np.ndarray:
        return np.asarray(self._embedder.embed([text])[0], dtype=np.float32)


class SimpleMemAdapter(MemoryAdapter):
    """SimpleMem structured-compression memory with hybrid retrieval."""

    name = "simplemem"

    def __init__(
        self,
        llm: BaseLLM,
        repo_path: str | Path | None = None,
        embedder: Any | None = None,
        work_dir: str | Path | None = None,
    ) -> None:
        path = Path(repo_path) if repo_path is not None else _find_repo_path()
        if not path.is_dir():
            raise RuntimeError(_INSTALL_HINT)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

        try:
            import main as simplemem_main  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                f"Could not import SimpleMem from {path}.\n{_INSTALL_HINT}"
            ) from exc

        if embedder is None:
            from bemula.baselines.rag import get_embedder

            embedder = get_embedder()
        load_dotenv(override=True)
        agent_model = os.environ.get("BEMULA_AGENT_MODEL", "deepseek-v4-flash")
        base_url, api_key = resolve_endpoint(agent_model)
        self.llm = llm

        # SimpleMem builds its own OpenAI client, so its internal calls need the
        # benchmark's serving settings re-applied before the system is built.
        from simplemem.core.utils.llm_client import LLMClient as _SimpleMemLLMClient

        pin_client_serving(_SimpleMemLLMClient, agent_model)
        self._work_dir = Path(work_dir) if work_dir else Path(
            tempfile.mkdtemp(prefix="bemula-simplemem-")
        )
        self._work_dir.mkdir(parents=True, exist_ok=True)

        # SimpleMemSystem instantiates the local-SentenceTransformer
        # EmbeddingModel through main's module global; point it at the
        # API-backed shim before construction so no local model is loaded.
        shim = _APIEmbeddingModel(embedder)
        simplemem_main.EmbeddingModel = lambda: shim
        self._system = simplemem_main.SimpleMemSystem(
            api_key=api_key,
            model=agent_model,
            base_url=base_url,
            db_path=str(self._work_dir / "lancedb"),
            clear_db=True,
            use_streaming=False,
        )
        builder = self._system.memory_builder
        _install_window_split_retry(builder, builder.llm_client)
        self._progress = TurnProgress("SimpleMem")

    def begin_instance(self, instance: Any) -> None:
        self._progress.begin_instance(instance)

    def end_instance(self) -> None:
        self._progress.end_instance()

    def begin_session(self, session: Any) -> None:
        self._progress.begin_session(session)

    def reset(self) -> None:
        # LanceDB's clear() drops and recreates the table; the builder's
        # buffer and counters are in-memory and must be reset alongside.
        self._system.vector_store.clear()
        builder = self._system.memory_builder
        builder.dialogue_buffer = []
        builder.processed_count = 0
        self._progress.start_variant()

    def observe(self, turn: Turn) -> None:
        self._system.add_dialogue(
            speaker=turn.speaker, content=turn.text, timestamp=turn.timestamp
        )
        self._progress.update()

    def end_variant(self) -> None:
        # Turns are buffered into WINDOW_SIZE batches; flush the tail window
        # so every observed turn is built into memory before probing.
        self._system.finalize()
        self._progress.end_variant()

    def query(self, question: str) -> str:
        # Relearning probes observe their context turns after end_variant();
        # flush the builder's tail buffer or those turns never reach memory.
        # process_remaining() is a no-op when the buffer is empty.
        self._system.finalize()
        entries = self._system.hybrid_retriever.retrieve(question)
        lines = [e.lossless_restatement for e in entries if e.lossless_restatement]
        if not lines:
            return "I don't have any relevant information stored."
        context = "\n".join(f"- {line}" for line in lines)
        return self.llm.chat(
            [{"role": "user", "content": _ANSWER_PROMPT.format(context=context, question=question)}],
            temperature=0.0,
            max_tokens=512,
            task="simplemem_answer",
        ).strip()
