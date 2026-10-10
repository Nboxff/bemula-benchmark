"""LLM access for data generation.

Provides an OpenAI-compatible chat client (configured via .env) with tenacity
retries. The pipeline is LLM-only; offline tests use checked-in fixtures and
scripted LLM doubles.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

Message = dict[str, str]

# Output budget for chat calls. Provider defaults are often far below what a
# prompt needs: DeepSeek caps requests that omit `max_tokens` at 8192 and
# silently truncates longer answers, which breaks callers that parse the whole
# reply as one JSON document. The floor leaves room for hidden reasoning tokens.
DEFAULT_MAX_TOKENS = 16384
MIN_MAX_TOKENS = 4096


def resolve_endpoint(model: str) -> tuple[str | None, str]:
    """Pick the endpoint credentials for a model.

    Every model goes through the single OpenAI-compatible endpoint configured
    via OPENAI_BASE_URL / OPENAI_API_KEY in .env.
    """
    load_dotenv(override=True)
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            f"No API key configured for model {model!r}. "
            "Set OPENAI_API_KEY in .env."
        )
    return base_url, api_key


def resolve_embed_endpoint(model: str) -> tuple[str | None, str]:
    """Pick the endpoint credentials for an embedding model.

    When EMBED_PROXY_BASE_URL / EMBED_PROXY_API_KEY are set, embeddings go
    through that dedicated endpoint, kept separate from the LLM endpoint;
    either field left unset falls back to the standard resolve_endpoint
    routing.
    """
    load_dotenv(override=True)
    base_url = os.environ.get("EMBED_PROXY_BASE_URL")
    api_key = os.environ.get("EMBED_PROXY_API_KEY")
    routed_base_url, routed_api_key = resolve_endpoint(model)
    if not base_url and not api_key:
        return routed_base_url, routed_api_key
    return base_url or routed_base_url, api_key or routed_api_key


class BaseLLM:
    """Minimal chat interface used across the generation pipeline."""

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.9,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        task: str = "generic",
    ) -> str:
        raise NotImplementedError

    def chat_json(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.9,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        task: str = "generic",
    ) -> Any:
        """Chat and parse the response as JSON (array or object)."""
        text = self.chat(messages, temperature=temperature, max_tokens=max_tokens, task=task)
        return parse_json(text)


def parse_json(text: str) -> Any:
    """Extract the first JSON array/object from an LLM response."""
    match = re.search(r"(\[.*\]|\{.*\})", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON found in LLM response: {text[:200]!r}")
    return json.loads(match.group(1))


# Model families served as reasoning deployments: they reject sampling
# parameters and spend hidden reasoning tokens before emitting content.
REASONING_MODEL_PREFIXES = ("gpt-5", "o1", "o3", "o4")


def thinking_extra_body(model: str | None) -> dict[str, Any] | None:
    """Provider-specific request extension that switches thinking on or off.

    DeepSeek takes the `thinking` extension. Reasoning deployments (GPT-5 /
    o-series) instead default to a non-zero `reasoning_effort`, so every call
    pays hidden reasoning tokens unless that effort is pinned; "none" is the
    supported value that turns reasoning off. Qwen3 hybrids switch thinking
    per request through the chat template. Models without a known switch keep
    the provider default.
    """
    name = (model or "").lower()
    mode = os.environ.get("BEMULA_THINKING", "disabled").strip().lower()
    if "deepseek" in name:
        return {"thinking": {"type": mode}}
    if name.startswith(REASONING_MODEL_PREFIXES) and mode != "enabled":
        return {"reasoning_effort": "none"}
    if "qwen" in name and mode != "enabled":
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


# Dialogue agents that parse chat() replies as JSON (the user simulator and
# session verifier) and therefore need the same enforced output as chat_json
# callers. Non-DeepSeek models (e.g. gpt-5.6-luna) otherwise drift to free-form
# prose on long role-play prompts despite the Return ONLY JSON instruction.
_JSON_CHAT_TASKS = frozenset({"user_sim", "session_verify"})


class OpenAILLM(BaseLLM):
    """OpenAI-compatible chat client.

    Reads OPENAI_BASE_URL / OPENAI_API_KEY / BEMULA_GEN_MODEL from .env.
    Thinking is disabled by default (faster and cheaper for bulk generation
    and judging): DeepSeek through its `thinking` switch, reasoning
    deployments through `reasoning_effort="none"`, Qwen3 hybrids through the
    chat template's `enable_thinking`. Set BEMULA_THINKING=enabled to fall
    back to the provider default.
    """

    def __init__(self, model: str | None = None, max_retries: int = 3) -> None:
        from openai import OpenAI

        load_dotenv(override=True)
        self.model = model or os.environ.get("BEMULA_GEN_MODEL", "deepseek-v4-flash")
        base_url, api_key = resolve_endpoint(self.model)
        self.max_retries = max_retries
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def chat_json(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.9,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        task: str = "generic",
    ) -> Any:
        # Enforced JSON output; parse like chat_json callers.
        text = self.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            task=task,
            json_mode=True,
        )
        return parse_json(text)

    def chat(
        self,
        messages: list[Message],
        *,
        temperature: float = 0.9,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        task: str = "generic",
        json_mode: bool = False,
    ) -> str:
        @retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=2, max=30),
            reraise=True,
        )
        def _call() -> str:
            # Reasoning models (e.g. DeepSeek) spend part of the budget on hidden
            # reasoning tokens; a small max_tokens can yield empty content
            # (finish_reason='length'). Enforce a safe floor.
            budget = max(max_tokens, MIN_MAX_TOKENS)
            # Thinking off is family-specific: DeepSeek takes the `thinking`
            # extension, reasoning deployments take `reasoning_effort`.
            extra = thinking_extra_body(self.model)
            # Non-DeepSeek models can ignore Return ONLY JSON on long role-play
            # prompts; response_format pins the output to parseable JSON.
            # The DeepSeek endpoint has been reliable without it, so it
            # stays off that path.
            enforce_json = json_mode or task in _JSON_CHAT_TASKS
            json_args = (
                {"response_format": {"type": "json_object"}}
                if enforce_json and "deepseek" not in self.model.lower()
                else {}
            )
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=budget,
                extra_body=extra,
                **json_args,
            )
            content = resp.choices[0].message.content
            # Reasoning models can exhaust the token budget on hidden reasoning and
            # return empty content (finish_reason='length'); treat as retryable.
            if not content:
                raise ValueError("empty LLM response")
            return content

        return _call()


def get_llm(model: str | None = None, max_retries: int = 3) -> BaseLLM:
    """Factory: OpenAI-compatible client for the configured model."""
    return OpenAILLM(model=model, max_retries=max_retries)
