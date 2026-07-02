"""Native Ollama ``/api/chat`` client (``api_mode='ollama_chat'``).

Why this exists: Ollama's OpenAI-compatible ``/v1`` endpoint silently drops
request-level controls — both ``options`` (e.g. ``num_ctx``) and the thinking
switch (``think`` / ``reasoning``). A thinking-capable model therefore keeps
emitting reasoning even when the user turns thinking off, because no ``/v1``
field reaches the model. Only native ``/api/chat`` honors ``think: false``.

This adapter speaks ``/api/chat`` directly via httpx and returns
**OpenAI-shaped** objects (``choices[0].message`` / streamed chunks with
``choices[0].delta``) so the existing chat_completions normalization and the
streaming-consumer loop in ``chat_completion_helpers`` work unchanged. The
matching transport (``agent/transports/ollama.py``) builds the native request
body and inherits normalization from ``ChatCompletionsTransport``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Iterator

import httpx

# Local generation can pause between tokens (model load, long prompts); keep
# the inter-chunk read timeout generous while still bounding a wedged socket.
_DEFAULT_TIMEOUT = httpx.Timeout(connect=15.0, read=300.0, write=60.0, pool=15.0)


def ollama_root_url(base_url: str | None) -> str:
    """Map a configured base_url (often ``…/v1``) to the Ollama API root.

    The custom provider is configured with an OpenAI-style ``…/v1`` base_url;
    native calls need the root (``/api/chat`` is appended by the caller).
    """
    b = (base_url or "http://localhost:11434").strip().rstrip("/")
    if b.lower().endswith("/v1"):
        b = b[: -len("/v1")]
    return b.rstrip("/") or "http://localhost:11434"


_thinking_capability_cache: dict[tuple[str, str], bool] = {}


def ollama_supports_thinking(base_url: str | None, model: str) -> bool:
    """Whether ``model`` exposes the ``thinking`` capability (cached).

    ``/api/chat`` rejects ``think`` for non-thinking models, so we probe
    ``/api/show`` once per (root, model). On any failure we assume **True** —
    the user opted this provider into native mode to control thinking, so a
    transient probe miss should not silently leave thinking on.
    """
    root = ollama_root_url(base_url)
    key = (root, model)
    cached = _thinking_capability_cache.get(key)
    if cached is not None:
        return cached
    supported = True
    try:
        with httpx.Client(timeout=httpx.Timeout(8.0)) as client:
            resp = client.post(root + "/api/show", json={"model": model})
            resp.raise_for_status()
            caps = resp.json().get("capabilities") or []
            supported = "thinking" in caps
    except Exception:
        supported = True
    _thinking_capability_cache[key] = supported
    return supported


def _resolve_timeout(agent: Any) -> httpx.Timeout:
    fn = getattr(agent, "_resolved_api_call_timeout", None)
    try:
        secs = fn() if callable(fn) else None
    except Exception:
        secs = None
    if isinstance(secs, (int, float)) and secs > 0:
        return httpx.Timeout(connect=15.0, read=float(secs), write=60.0, pool=15.0)
    return _DEFAULT_TIMEOUT


def _tool_calls_from_message(message: dict) -> list[SimpleNamespace] | None:
    """Ollama ``message.tool_calls`` → OpenAI-shaped tool_call list (or None)."""
    raw = message.get("tool_calls") or []
    if not raw:
        return None
    out: list[SimpleNamespace] = []
    for i, tc in enumerate(raw):
        fn = tc.get("function") or {}
        args = fn.get("arguments")
        # Ollama returns arguments as a JSON object; the agent loop expects a
        # JSON *string* (it re-parses downstream).
        args_str = json.dumps(args) if isinstance(args, (dict, list)) else (args or "")
        out.append(
            SimpleNamespace(
                index=i,
                id=tc.get("id") or f"ollama_call_{i}",
                type="function",
                function=SimpleNamespace(name=fn.get("name") or "", arguments=args_str),
                model_extra=None,
            )
        )
    return out


def _usage_from(d: dict) -> SimpleNamespace:
    prompt = int(d.get("prompt_eval_count") or 0)
    completion = int(d.get("eval_count") or 0)
    return SimpleNamespace(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )


def _finish_reason(done_reason: str | None, has_tool_calls: bool) -> str:
    dr = (done_reason or "").strip().lower()
    if has_tool_calls:
        return "tool_calls"
    if dr == "length":
        return "length"
    return "stop"


def _raise_for_ollama_error(data: dict) -> None:
    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"Ollama /api/chat error: {data['error']}")


def ollama_chat_create(agent: Any, api_kwargs: dict) -> SimpleNamespace:
    """Non-streaming ``POST /api/chat`` → OpenAI-shaped response object.

    Returns a ``SimpleNamespace`` mirroring an OpenAI ChatCompletion so the
    inherited ``normalize_response`` (and the agent loop) consume it unchanged.
    """
    body = dict(api_kwargs)
    body["stream"] = False
    url = ollama_root_url(getattr(agent, "base_url", None)) + "/api/chat"
    with httpx.Client(timeout=_resolve_timeout(agent)) as client:
        resp = client.post(url, json=body)
        resp.raise_for_status()
        data = resp.json()
    _raise_for_ollama_error(data)

    message = data.get("message") or {}
    tool_calls = _tool_calls_from_message(message)
    msg = SimpleNamespace(
        role=message.get("role") or "assistant",
        content=(message.get("content") or None),
        reasoning_content=(message.get("thinking") or None),
        tool_calls=tool_calls,
    )
    choice = SimpleNamespace(
        index=0,
        message=msg,
        finish_reason=_finish_reason(data.get("done_reason"), bool(tool_calls)),
    )
    return SimpleNamespace(
        id="ollama-" + str(data.get("created_at") or ""),
        model=data.get("model") or body.get("model"),
        choices=[choice],
        usage=_usage_from(data),
    )


def ollama_chat_stream(agent: Any, api_kwargs: dict) -> Iterator[SimpleNamespace]:
    """Streaming ``POST /api/chat`` (NDJSON) → OpenAI-shaped chunks.

    Ollama streams newline-delimited JSON: each line is a partial ``message``
    with ``content``/``thinking`` deltas; the final line carries ``done: true``
    plus token counts. We re-shape each line into an OpenAI ChatCompletionChunk
    so the streaming-consumer loop accumulates it like any other provider.
    Tool calls arrive whole (not split across deltas), so they're emitted once.
    """
    body = dict(api_kwargs)
    body["stream"] = True
    url = ollama_root_url(getattr(agent, "base_url", None)) + "/api/chat"
    saw_tool_calls = False

    with httpx.Client(timeout=_resolve_timeout(agent)) as client:
        with client.stream("POST", url, json=body) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line or not line.strip():
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                _raise_for_ollama_error(d)

                message = d.get("message") or {}
                tool_calls = _tool_calls_from_message(message)
                if tool_calls:
                    saw_tool_calls = True

                done = bool(d.get("done"))
                finish = (
                    _finish_reason(d.get("done_reason"), saw_tool_calls) if done else None
                )

                delta = SimpleNamespace(
                    content=(message.get("content") or None),
                    reasoning_content=(message.get("thinking") or None),
                    tool_calls=tool_calls,
                )
                choice = SimpleNamespace(index=0, delta=delta, finish_reason=finish)
                yield SimpleNamespace(
                    choices=[choice],
                    model=d.get("model"),
                    # Usage only meaningful on the final frame.
                    usage=_usage_from(d) if done else None,
                )
