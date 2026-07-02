"""Transport for api_mode='ollama_chat' — Ollama's native ``/api/chat``.

Ollama's OpenAI-compatible ``/v1`` endpoint ignores the thinking switch
(``think``/``reasoning``) and request ``options`` (``num_ctx``), so a
thinking-capable model keeps reasoning even when the user turns thinking off.
Only native ``/api/chat`` honors ``think: false``. This transport builds the
native request body; the actual HTTP call lives in ``agent/ollama_adapter.py``,
which returns OpenAI-shaped objects so normalization is inherited unchanged
from :class:`ChatCompletionsTransport`.
"""

from __future__ import annotations

import json
from typing import Any

from agent.transports import register_transport
from agent.transports.chat_completions import ChatCompletionsTransport


class OllamaTransport(ChatCompletionsTransport):
    """Native Ollama ``/api/chat`` request builder.

    Only ``build_kwargs`` and message/tool conversion are Ollama-specific; the
    adapter normalizes responses into OpenAI shape, so ``normalize_response`` /
    ``validate_response`` are inherited from the chat_completions transport.
    """

    @property
    def api_mode(self) -> str:
        return "ollama_chat"

    def convert_tools(self, tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # Ollama accepts the OpenAI ``[{type:function, function:{…}}]`` shape
        # verbatim (verified against a live tool call) — identity.
        return tools

    def convert_messages(self, messages: list[dict[str, Any]], **kwargs) -> list[dict[str, Any]]:
        """OpenAI-format messages → Ollama ``/api/chat`` messages.

        Differences handled:
        - vision: OpenAI ``content`` list with ``image_url`` parts → native
          ``images: [<base64>]`` on the message + text joined into ``content``.
        - assistant ``tool_calls``: arguments must be an object, not a JSON
          string (and Ollama ignores ``id``/``type``).
        - ``tool`` results: native wants ``content`` (+ optional ``tool_name``),
          not ``tool_call_id``.
        """
        out: list[dict[str, Any]] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            new: dict[str, Any] = {"role": role}

            text, images = _split_content(content)
            new["content"] = text
            if images:
                new["images"] = images

            if role == "assistant" and msg.get("tool_calls"):
                native_tcs = []
                for tc in msg["tool_calls"]:
                    fn = (tc or {}).get("function", {}) or {}
                    args = fn.get("arguments")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args) if args.strip() else {}
                        except json.JSONDecodeError:
                            args = {}
                    native_tcs.append({"function": {"name": fn.get("name") or "", "arguments": args or {}}})
                if native_tcs:
                    new["tool_calls"] = native_tcs

            if role == "tool":
                # Ollama matches tool results by name/order, not call id.
                name = msg.get("name") or msg.get("tool_name")
                if name:
                    new["tool_name"] = name

            out.append(new)
        return out

    def build_kwargs(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **params,
    ) -> dict[str, Any]:
        """Build the native ``/api/chat`` request body.

        ``stream`` is added by the adapter, not here.
        """
        body: dict[str, Any] = {
            "model": model,
            "messages": self.convert_messages(messages),
        }
        if tools:
            body["tools"] = self.convert_tools(tools)

        # Thinking switch — the whole reason this transport exists. Send
        # think:false only when reasoning is explicitly disabled AND the model
        # is thinking-capable (sending think to a non-thinking model errors).
        reasoning_config = params.get("reasoning_config")
        disabled = bool(
            isinstance(reasoning_config, dict)
            and reasoning_config.get("enabled") is False
        )
        if disabled:
            from agent.ollama_adapter import ollama_supports_thinking

            if ollama_supports_thinking(params.get("base_url"), model):
                body["think"] = False

        # options — native /api/chat HONORS these (unlike /v1).
        options: dict[str, Any] = {}
        temperature = params.get("temperature")
        if temperature is not None:
            options["temperature"] = temperature
        num_ctx = params.get("ollama_num_ctx") or params.get("num_ctx")
        if num_ctx:
            options["num_ctx"] = int(num_ctx)
        max_tokens = params.get("max_tokens")
        if max_tokens:
            options["num_predict"] = int(max_tokens)
        if options:
            body["options"] = options

        # request_overrides last (parity with chat_completions) — lets advanced
        # users inject extra native fields (e.g. keep_alive) via config.
        overrides = params.get("request_overrides")
        if isinstance(overrides, dict):
            for k, v in overrides.items():
                if k == "options" and isinstance(v, dict):
                    body.setdefault("options", {}).update(v)
                else:
                    body[k] = v

        return body


def _split_content(content: Any) -> tuple[str, list[str]]:
    """Return (text, images[base64]) from an OpenAI ``content`` value."""
    if content is None:
        return "", []
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []

    texts: list[str] = []
    images: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type")
        if ptype == "text":
            texts.append(part.get("text") or "")
        elif ptype in {"image_url", "input_image"}:
            url = ""
            iu = part.get("image_url")
            if isinstance(iu, dict):
                url = iu.get("url") or ""
            elif isinstance(iu, str):
                url = iu
            url = url or part.get("image_url", "") or ""
            # Native wants raw base64 (strip any ``data:…;base64,`` prefix).
            if isinstance(url, str) and url:
                images.append(url.split(",", 1)[1] if url.startswith("data:") else url)
    return "\n".join(t for t in texts if t), images


register_transport("ollama_chat", OllamaTransport)
