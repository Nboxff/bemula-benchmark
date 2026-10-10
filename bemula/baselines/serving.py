"""Pin the benchmark's serving settings on clients owned by external systems.

Wrapped baselines (SimpleMem, MemoryOS, ...) build their own OpenAI clients
internally, so their requests never pass through the payload assembled in
`bemula.llm`. Two settings matter. The thinking switch: left at the provider
default, every internal planning / reflection / memory-building call becomes a
reasoning call, which multiplies latency and token cost and can exhaust the
response budget before any content is emitted. The output budget: left unset,
the provider default applies, and a reply that needs more (DeepSeek stops at
8192) is truncated mid-document and dropped by the caller's parser.

These helpers re-apply the benchmark settings on the foreign client so all
baselines are served under the same configuration; the switch itself is
family-specific (`bemula.llm.thinking_extra_body`).
"""

from __future__ import annotations

from typing import Any

from bemula.llm import DEFAULT_MAX_TOKENS, thinking_extra_body

__all__ = ["pin_client_serving", "last_finish_reason", "thinking_extra_body"]


# Attribute where the wrapper records the upstream finish_reason, so adapters
# can tell a truncated reply (budget exhausted) from a genuinely empty one.
_FINISH_REASON_ATTR = "bemula_last_finish_reason"


def last_finish_reason(client: Any) -> str | None:
    """Finish reason of the most recent call routed through a pinned client."""
    return getattr(client, _FINISH_REASON_ATTR, None)


def _wrap_completions(client: Any, extra: dict[str, Any] | None) -> None:
    """Rebind the client's `create` so every request carries the pinned settings."""
    completions = client.chat.completions
    original = completions.create

    def create(*args: Any, **kwargs: Any) -> Any:
        if extra:
            body = dict(kwargs.get("extra_body") or {})
            body.update(extra)
            kwargs["extra_body"] = body
        # External systems leave the budget to the provider, whose default can
        # be smaller than a full reply. An explicit budget keeps structured
        # answers (one JSON document per call) complete.
        if "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs:
            kwargs["max_tokens"] = DEFAULT_MAX_TOKENS
        response = original(*args, **kwargs)
        choices = getattr(response, "choices", None)
        if choices:
            setattr(client, _FINISH_REASON_ATTR, getattr(choices[0], "finish_reason", None))
        return response

    completions.create = create


def pin_client_serving(cls: type, model: str | None) -> bool:
    """Route every client built by `cls` through the benchmark's settings.

    The wrapped class must expose its OpenAI client as `self.client`. Returns
    True when the class was patched, False when it was already pinned.
    """
    if getattr(cls, "_bemula_serving_pinned", False):
        return False
    extra = thinking_extra_body(model)
    original_init = cls.__init__

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        client = getattr(self, "client", None)
        if client is not None:
            _wrap_completions(client, extra)

    cls.__init__ = __init__  # type: ignore[method-assign]
    cls._bemula_serving_pinned = True  # type: ignore[attr-defined]
    return True
