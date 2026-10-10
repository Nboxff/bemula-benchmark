"""A-MEM baseline adapter.

Wraps the official A-MEM repository (https://github.com/WujiangXu/A-mem) using
its robust memory layer (`memory_layer_robust.py`), which avoids OpenAI JSON
schema mode and works with any OpenAI-compatible chat endpoint.  A-MEM stores
each observed turn as a structured note, builds metadata with an LLM, links
related notes, and retrieves neighboring notes at query time.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from bemula.baselines.base import MemoryAdapter, TurnProgress
from bemula.llm import BaseLLM
from bemula.schemas import Turn

_REPO_URL = "https://github.com/WujiangXu/A-mem"
_INSTALL_HINT = (
    "AmemAdapter requires the official A-MEM repository, which is not "
    "available.\nClone it and point BEMULA_AMEM_PATH at the clone:\n"
    f"    git clone {_REPO_URL}\n"
    "    export BEMULA_AMEM_PATH=/path/to/A-mem\n"
    "Install its requirements (see requirements.txt) into a compatible "
    "environment, then re-run."
)

_ANSWER_PROMPT = (
    "You are a personal assistant with access to remembered conversation "
    "history. Answer the user's question using ONLY the context below. "
    "If the answer is not in the context, say you don't have that "
    "information. Keep the answer short.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def _find_repo_path() -> Path:
    """Locate the A-MEM clone.

    ``BEMULA_AMEM_PATH`` takes precedence.  When it is not set, also try a
    sibling ``A-mem`` directory because this repository is developed alongside
    the A-MEM clone in the same workspace.
    """
    env_path = os.environ.get("BEMULA_AMEM_PATH")
    if env_path:
        return Path(env_path)

    sibling = Path(__file__).resolve().parents[3] / "A-mem"
    if sibling.is_dir():
        return sibling

    return Path("A-mem")


class _AmemLLM:
    """Expose the Bemula ``BaseLLM`` as A-MEM's simple completion interface.

    A-MEM's robust layer calls ``llm.get_completion(prompt, temperature=...)``
    for metadata extraction, evolution decisions, and (optionally) final
    answers.  This wrapper routes those calls through the same endpoint and
    model configuration as the rest of the benchmark.
    """

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        return self._llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=1024,
            task="amem_internal",
        ).strip()


class _APIEmbeddingModel:
    """Make A-MEM's retriever use the benchmark's shared embedder.

    A-MEM's original ``SimpleEmbeddingRetriever`` expects a
    ``SentenceTransformer``-like object with an ``encode()`` method.  This
    adapter points that interface at the same embedder used by the other
    baselines (``rag.get_embedder``: API endpoint or local model, per
    BEMULA_EMBED_BACKEND).
    """

    def __init__(self, embedder: Any) -> None:
        self._embedder = embedder

    def encode(self, texts: list[str]):
        return self._embedder.embed(texts)


class AmemAdapter(MemoryAdapter):
    """Structured-note A-MEM memory with note evolution and link retrieval."""

    name = "amem"

    def __init__(
        self,
        llm: BaseLLM,
        repo_path: str | Path | None = None,
        embedder: Any | None = None,
        retrieve_k: int = 10,
        progress: bool = True,
    ) -> None:
        path = Path(repo_path) if repo_path is not None else _find_repo_path()
        if not path.is_dir():
            raise RuntimeError(_INSTALL_HINT)
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

        try:
            from memory_layer_robust import (  # type: ignore[import-not-found]
                RobustAgenticMemorySystem,
                SimpleEmbeddingRetriever,
            )
        except ImportError as exc:
            raise RuntimeError(
                f"Could not import A-MEM from {path}.\n{_INSTALL_HINT}"
            ) from exc

        if embedder is None:
            from bemula.baselines.rag import get_embedder

            embedder = get_embedder()
        self.llm = llm
        self.embedder = embedder
        self.retrieve_k = retrieve_k
        self._progress = TurnProgress("A-MEM", enabled=progress)
        self._system_class = RobustAgenticMemorySystem
        self._retriever_class = SimpleEmbeddingRetriever
        self._system = self._create_system()

    def _new_retriever(self):
        """Create A-MEM's retriever backed by the API embedding endpoint."""
        retriever = object.__new__(self._retriever_class)
        retriever.model = _APIEmbeddingModel(self.embedder)
        retriever.corpus = []
        retriever.embeddings = None
        retriever.document_ids = {}
        return retriever

    def _create_system(self):
        """Build a fresh A-MEM system with its internal LLM routed to Bemula.

        The official A-MEM constructor loads a local SentenceTransformer.  We
        deliberately construct the same object without calling that constructor
        and install an API-backed retriever instead, so the baseline needs no
        local model download.
        """
        system = object.__new__(self._system_class)
        system.memories = {}
        system.retriever = self._new_retriever()
        system.llm_controller = SimpleNamespace(llm=_AmemLLM(self.llm))
        system.evo_cnt = 0
        system.evo_threshold = 100

        def _consolidate() -> None:
            """Keep using the API retriever when A-MEM consolidates memory."""
            system.retriever = self._new_retriever()
            for memory in system.memories.values():
                metadata_text = (
                    f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
                )
                system.retriever.add_documents([memory.content + " , " + metadata_text])

        system.consolidate_memories = _consolidate
        return system

    def begin_instance(self, instance: Any) -> None:
        self._progress.begin_instance(instance)

    def end_instance(self) -> None:
        self._progress.end_instance()

    def begin_session(self, session: Any) -> None:
        self._progress.begin_session(session)

    def end_variant(self) -> None:
        self._progress.end_variant()

    def reset(self) -> None:
        # A-MEM has no clear/reset API; rebuilding the whole in-memory system is
        # the only safe way to guarantee isolated state for each eval variant.
        self._system = self._create_system()
        self._progress.start_variant()

    def observe(self, turn: Turn) -> None:
        # Follow the official A-MEM LoCoMo harness: every dialogue turn becomes
        # one note, prefixed with the speaker and timestamped.
        content = f"Speaker {turn.speaker} says : {turn.text}"
        self._system.add_note(content, time=turn.timestamp)
        self._progress.update()

    def query(self, question: str) -> str:
        context = self._system.find_related_memories_raw(question, k=self.retrieve_k)
        if not context.strip():
            return "I don't have any relevant information stored."
        return self.llm.chat(
            [{"role": "user", "content": _ANSWER_PROMPT.format(context=context, question=question)}],
            temperature=0.0,
            max_tokens=512,
            task="amem_answer",
        ).strip()
