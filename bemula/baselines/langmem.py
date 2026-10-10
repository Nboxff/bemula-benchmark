"""LangMem baseline adapter.

Wraps the official LangMem package (https://github.com/langchain-ai/langmem).
LangMem's background extractor watches a conversation and maintains a collection
of natural-language memory notes in a LangGraph store: for each batch of turns
it searches the store for related notes, asks the model which notes to add,
update or remove, and writes the result back. At query time the benchmark's
agent LLM answers from the semantically nearest notes, so the answer side stays
uniform across baselines.

Deliberate deviations from the stock setup:

- Deletes are enabled (`enable_deletes=True`, LangMem ships with them off).
  Turning them on binds trustcall's `RemoveDoc` tool, which is the only way the
  extractor can express "drop this note", so a natural-language forget request
  has a channel to act on. The benchmark itself still never calls a delete
  interface; whether the system uses the tool is its own decision. Note that
  removals are scoped to the notes retrieved for the current batch
  (`query_limit`, default 5), so a target the search misses cannot be removed.
- The extraction LLM is pinned to the OpenAI-compatible endpoint from .env
  (BEMULA_AGENT_MODEL via resolve_endpoint) with the shared thinking switch, and
  the store's semantic index is backed by the shared embedder from
  `rag.get_embedder` rather than a provider string.

Otherwise the stock configuration is kept: notes are unstructured (`Memory`),
inserts and updates stay on (`create_memory_store_manager` exposes only
`enable_inserts` and `enable_deletes`; updates always stay on), and each observed
session is handed to the extractor as one conversation.

LangMem messages carry no timestamp, so `Turn.timestamp` is not passed on; the
default `Memory` schema has nowhere to put it.
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

from bemula.baselines.base import MemoryAdapter, TurnProgress
from bemula.llm import BaseLLM, resolve_endpoint, thinking_extra_body
from bemula.schemas import Turn

_INSTALL_HINT = (
    "LangmemAdapter requires the official LangMem package, which is not "
    "available.\nInstall it into the baselines extra environment:\n"
    "    uv sync --extra baselines\n"
    "    # or: uv pip install langmem"
)

_ANSWER_PROMPT = (
    "You are a personal assistant with access to remembered conversation "
    "history. Answer the user's question using ONLY the context below. "
    "If the answer is not in the context, say you don't have that "
    "information. Keep the answer short.\n\n"
    "Context:\n{context}\n\nQuestion: {question}"
)


def _memory_text(value: Any) -> str:
    """Pull the note text out of a store item value.

    The default schema stores `{"kind": "Memory", "content": {"content": ...}}`
    and the manager coerces it to a `Memory` model on search, but custom schemas
    can surface plain dicts or strings.
    """
    inner = getattr(value, "content", None)
    if inner is None and isinstance(value, dict):
        inner = value.get("content")
    if isinstance(inner, dict):
        inner = inner.get("content")
    if inner is None:
        inner = value
    return str(inner).strip()


class LangmemAdapter(MemoryAdapter):
    """LangMem background memory extraction over a LangGraph store."""

    name = "langmem"

    def __init__(
        self,
        llm: BaseLLM,
        embedder: Any | None = None,
        retrieve_limit: int = 10,
        namespace: tuple[str, ...] = ("memories",),
        enable_deletes: bool = True,
    ) -> None:
        try:
            from langchain.chat_models import init_chat_model
            from langgraph.store.memory import InMemoryStore
            from langmem import create_memory_store_manager
        except ImportError as exc:
            raise RuntimeError(_INSTALL_HINT) from exc

        if embedder is None:
            from bemula.baselines.rag import get_embedder

            embedder = get_embedder()
        load_dotenv(override=True)
        agent_model = os.environ.get("BEMULA_AGENT_MODEL", "deepseek-v4-flash")
        base_url, api_key = resolve_endpoint(agent_model)

        self.llm = llm
        self._namespace = namespace
        self._retrieve_limit = retrieve_limit
        self._embedder = embedder
        self._dims = len(embedder.embed(["dimension probe"])[0])
        self._store_class = InMemoryStore
        self._manager_factory = create_memory_store_manager
        self._enable_deletes = enable_deletes

        # LangChain's own client is configured through the model, so the
        # benchmark's thinking switch goes in as `extra_body` rather than
        # through `pin_client_serving`.
        extra_body = thinking_extra_body(agent_model)
        self._model = init_chat_model(
            f"openai:{agent_model}",
            base_url=base_url,
            api_key=api_key,
            temperature=0.0,
            **({"extra_body": extra_body} if extra_body else {}),
        )

        self._store: Any = None
        self._manager: Any = None
        self._messages: list[dict[str, str]] = []
        self._create_system()
        self._progress = TurnProgress("LangMem")

    def _embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(list(texts)).tolist()

    def _create_system(self) -> None:
        """Build a fresh store and extractor; called on every reset.

        `InMemoryStore` lives in the process, so rebuilding both is what keeps
        one dialogue variant from seeing another's notes.
        """
        self._store = self._store_class(
            index={"dims": self._dims, "embed": self._embed}
        )
        self._manager = self._manager_factory(
            self._model,
            namespace=self._namespace,
            store=self._store,
            enable_deletes=self._enable_deletes,
        )
        self._messages = []

    def begin_instance(self, instance: Any) -> None:
        self._progress.begin_instance(instance)

    def end_instance(self) -> None:
        self._progress.end_instance()

    def begin_session(self, session: Any) -> None:
        # The turns buffered so far belong to the previous session; extract them
        # before the new one starts so each session is distilled on its own.
        self._flush()
        self._progress.begin_session(session)

    def reset(self) -> None:
        self._create_system()
        self._progress.start_variant()

    def observe(self, turn: Turn) -> None:
        self._messages.append({"role": turn.speaker, "content": turn.text})
        self._progress.update()

    def end_variant(self) -> None:
        self._flush()
        self._progress.end_variant()

    def query(self, question: str) -> str:
        # Relearning probes observe their context turns after end_variant();
        # extract them before searching or they never reach the store.
        self._flush()
        entries = self._manager.search(query=question, limit=self._retrieve_limit)
        lines = [text for text in (_memory_text(item.value) for item in entries) if text]
        if not lines:
            return "I don't have any relevant information stored."
        context = "\n".join(f"- {line}" for line in lines)
        return self.llm.chat(
            [{"role": "user", "content": _ANSWER_PROMPT.format(context=context, question=question)}],
            temperature=0.0,
            max_tokens=512,
            task="langmem_answer",
        ).strip()

    def _flush(self) -> None:
        if not self._messages:
            return
        batch, self._messages = self._messages, []
        self._manager.invoke({"messages": batch})
