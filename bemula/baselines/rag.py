"""Naive RAG baseline, implemented in-repo.

Each observed turn is stored verbatim as a chunk and indexed on write: chunks
are embedded in write batches as the dialogue is ingested, so a probe only has
to embed its question and can retrieve top-k by cosine similarity. The retrieved
chunks are stuffed into an answer prompt and the agent LLM
(`BEMULA_AGENT_MODEL`) produces the answer.

This baseline has no forgetting mechanism and serves as the lower-bound
reference.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from dotenv import load_dotenv

from bemula.baselines.base import MemoryAdapter
from bemula.llm import BaseLLM, resolve_embed_endpoint
from bemula.schemas import Turn


class OpenAIEmbedder:
    """OpenAI-compatible embedding endpoint (BEMULA_EMBED_MODEL in .env).

    Endpoint routing follows bemula.llm.resolve_embed_endpoint: the dedicated
    EMBED_PROXY_* endpoint when configured, otherwise the standard
    resolve_endpoint routing.
    """

    def __init__(self, model: str | None = None) -> None:
        from openai import OpenAI

        load_dotenv(override=True)
        self.model = model or os.environ.get("BEMULA_EMBED_MODEL", "text-embedding-3-small")
        base_url, api_key = resolve_embed_endpoint(self.model)
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return np.array([d.embedding for d in resp.data], dtype=np.float32)


class LocalEmbedder:
    """Local sentence-transformers embedder (BEMULA_HF_EMBED_MODEL in .env).

    Declared in the `baselines` extra and lazily imported; the model weights
    are downloaded from hf-mirror.com (HF_ENDPOINT) into the Hugging Face
    cache on first use.
    """

    def __init__(self, model: str | None = None) -> None:
        load_dotenv(override=True)
        self.model = model or os.environ.get(
            "BEMULA_HF_EMBED_MODEL", "all-MiniLM-L6-v2"
        )
        self._st: Any = None

    def _load(self) -> Any:
        if self._st is None:
            # huggingface_hub reads HF_ENDPOINT at import time, so it must be
            # set before sentence_transformers is imported.
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for the local embedding "
                    "backend; install it with `uv sync --extra baselines`"
                ) from e
            self._st = SentenceTransformer(self.model)
        return self._st

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._load().encode(texts, convert_to_numpy=True), dtype=np.float32
        )


def get_embedder() -> OpenAIEmbedder | LocalEmbedder:
    """Factory: the embedding backend is chosen by BEMULA_EMBED_BACKEND
    ("api" default, "local" for a local sentence-transformers model)."""
    load_dotenv(override=True)
    backend = os.environ.get("BEMULA_EMBED_BACKEND", "api").strip().lower()
    if backend == "api":
        return OpenAIEmbedder()
    if backend == "local":
        return LocalEmbedder()
    raise ValueError(f"unknown BEMULA_EMBED_BACKEND: {backend!r} (expected api|local)")


def _cosine(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query) + 1e-9
    m_norm = np.linalg.norm(matrix, axis=1) + 1e-9
    return (matrix @ query) / (m_norm * q_norm)


# Upstream OpenAI-compatible embedding endpoints reject requests above 2048
# inputs; long instances store more chunks than that.
_MAX_EMBED_INPUTS = 2048
# The same endpoints cap a request at 300k tokens, which long dialogue turns
# reach on far fewer than 2048 chunks; a character budget keeps requests inside
# that cap with headroom for tokenizers that split text finer than 4:1.
_MAX_EMBED_CHARS = 600_000
# Embedding one turn per request would pay the endpoint's per-request overhead
# once per observed turn (tens of thousands of times per full run), so turns are
# indexed in write batches sized to the endpoint's own limit.
_WRITE_BATCH = _MAX_EMBED_INPUTS


def _embed_batches(texts: list[str]) -> list[list[str]]:
    """Split texts into requests the embedding endpoint accepts.

    Bounded by the input count and by the payload size: an instance with long
    turns overflows the request token cap well before reaching 2048 chunks.
    """
    batches: list[list[str]] = []
    current: list[str] = []
    size = 0
    for text in texts:
        if current and (len(current) >= _MAX_EMBED_INPUTS or size + len(text) > _MAX_EMBED_CHARS):
            batches.append(current)
            current, size = [], 0
        current.append(text)
        size += len(text)
    if current:
        batches.append(current)
    return batches


class RagAdapter(MemoryAdapter):
    """Chunk-store RAG baseline: index chunks on write, retrieve on probe."""

    name = "rag"

    def __init__(
        self,
        llm: BaseLLM,
        embedder: OpenAIEmbedder | LocalEmbedder,
        top_k: int = 5,
        write_batch: int = _WRITE_BATCH,
    ) -> None:
        if write_batch < 1:
            raise ValueError("write_batch must be >= 1")
        self.llm = llm
        self.embedder = embedder
        self.top_k = top_k
        self.write_batch = write_batch
        self._chunks: list[str] = []
        # Vectors are stored once per chunk and reused by every probe;
        # `_embedded` counts how many leading chunks have a row already.
        self._chunk_vecs: np.ndarray | None = None
        self._embedded = 0

    def reset(self) -> None:
        self._chunks = []
        self._chunk_vecs = None
        self._embedded = 0

    def observe(self, turn: Turn) -> None:
        self._chunks.append(f"{turn.speaker}: {turn.text}")
        if len(self._chunks) - self._embedded >= self.write_batch:
            self._index_pending()

    def _embed(self, texts: list[str]) -> np.ndarray:
        """Embed in endpoint-sized batches, preserving input order."""
        batches = _embed_batches(texts)
        if len(batches) == 1:
            return self._embed_request(batches[0])
        return np.concatenate([self._embed_request(b) for b in batches], axis=0)

    def _embed_request(self, texts: list[str]) -> np.ndarray:
        """Embed one request, halving it when the endpoint rejects the size."""
        try:
            return self.embedder.embed(texts)
        except Exception as exc:  # noqa: BLE001 - re-raised unless it is a size rejection
            if len(texts) <= 1 or getattr(exc, "status_code", None) != 400:
                raise
            mid = len(texts) // 2
            return np.concatenate(
                [self._embed_request(texts[:mid]), self._embed_request(texts[mid:])], axis=0
            )

    def _index_pending(self) -> None:
        """Embed every chunk written since the last index flush."""
        pending = self._chunks[self._embedded:]
        if not pending:
            return
        vecs = self._embed(pending)
        self._chunk_vecs = (
            vecs if self._chunk_vecs is None
            else np.concatenate([self._chunk_vecs, vecs], axis=0)
        )
        self._embedded = len(self._chunks)

    def query(self, question: str) -> str:
        if not self._chunks:
            return "I don't have any relevant information stored."
        # Relearning probes observe their short dialogue after the variant's
        # ingestion, so flush any turns written since the last batch.
        self._index_pending()
        assert self._chunk_vecs is not None
        query_vec = self.embedder.embed([question])[0]
        sims = _cosine(query_vec, self._chunk_vecs)
        top = np.argsort(-sims)[: self.top_k]
        context = "\n".join(f"- {self._chunks[i]}" for i in top)
        prompt = (
            "You are a personal assistant with access to remembered conversation "
            "history. Answer the user's question using ONLY the context below. "
            "If the answer is not in the context, say you don't have that "
            "information. Keep the answer short.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}"
        )
        return self.llm.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=512,
            task="rag_answer",
        ).strip()
