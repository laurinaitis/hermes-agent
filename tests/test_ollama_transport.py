"""Tests for the native Ollama transport (api_mode='ollama_chat').

Network is mocked throughout — no live Ollama required.
"""

import json
from types import SimpleNamespace

import pytest

from agent.transports import get_transport
import agent.ollama_adapter as oa
import hermes_cli.providers as providers


# ── helpers ────────────────────────────────────────────────────────────────
class _FakeResp:
    def __init__(self, payload=None, lines=None, status=200):
        self._payload = payload
        self._lines = lines or []
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload

    def iter_lines(self):
        yield from self._lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeClient:
    def __init__(self, *, post_json=None, get_json=None, stream_lines=None):
        self._post_json = post_json
        self._get_json = get_json
        self._stream_lines = stream_lines

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def post(self, url, json=None):
        return _FakeResp(payload=self._post_json)

    def get(self, url):
        return _FakeResp(payload=self._get_json)

    def stream(self, method, url, json=None):
        return _FakeResp(lines=self._stream_lines)


def _agent(base_url="http://localhost:11434/v1"):
    return SimpleNamespace(base_url=base_url, _resolved_api_call_timeout=lambda: 300.0)


# ── root url ───────────────────────────────────────────────────────────────
@pytest.mark.parametrize("inp,out", [
    ("http://localhost:11434/v1", "http://localhost:11434"),
    ("http://localhost:11434/v1/", "http://localhost:11434"),
    ("http://localhost:11434/", "http://localhost:11434"),
    (None, "http://localhost:11434"),
])
def test_root_url_strips_v1(inp, out):
    assert oa.ollama_root_url(inp) == out


# ── build_kwargs: the core think gate ───────────────────────────────────────
def test_build_kwargs_sends_think_false_when_disabled_and_capable(monkeypatch):
    monkeypatch.setattr(oa, "ollama_supports_thinking", lambda base, model: True)
    t = get_transport("ollama_chat")
    body = t.build_kwargs(
        model="gemma4:e4b-q8",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_config={"enabled": False},
        base_url="http://localhost:11434/v1",
        max_tokens=4096,
        ollama_num_ctx=8192,
    )
    assert body["think"] is False
    assert body["options"] == {"num_ctx": 8192, "num_predict": 4096}
    assert body["model"] == "gemma4:e4b-q8"


def test_build_kwargs_omits_think_when_enabled(monkeypatch):
    monkeypatch.setattr(oa, "ollama_supports_thinking", lambda base, model: True)
    t = get_transport("ollama_chat")
    body = t.build_kwargs(model="m", messages=[{"role": "user", "content": "hi"}],
                          reasoning_config=None, base_url="http://localhost:11434/v1")
    assert "think" not in body


def test_build_kwargs_omits_think_for_non_thinking_model(monkeypatch):
    monkeypatch.setattr(oa, "ollama_supports_thinking", lambda base, model: False)
    t = get_transport("ollama_chat")
    body = t.build_kwargs(model="m", messages=[{"role": "user", "content": "hi"}],
                          reasoning_config={"enabled": False}, base_url="http://x/v1")
    assert "think" not in body


def test_build_kwargs_passes_tools_through():
    t = get_transport("ollama_chat")
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    body = t.build_kwargs(model="m", messages=[], tools=tools, base_url="http://x")
    assert body["tools"] == tools


# ── message conversion ──────────────────────────────────────────────────────
def test_convert_messages_vision_and_toolcalls_and_toolresult():
    t = get_transport("ollama_chat")
    out = t.convert_messages([
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}},
        ]},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "x", "type": "function", "function": {"name": "f", "arguments": '{"a": 1}'}}
        ]},
        {"role": "tool", "name": "f", "content": "result"},
    ])
    # vision: text joined, image stripped to raw base64
    assert out[0]["content"] == "look"
    assert out[0]["images"] == ["QUJD"]
    # assistant tool_calls: arguments parsed to object, id/type dropped
    assert out[1]["tool_calls"] == [{"function": {"name": "f", "arguments": {"a": 1}}}]
    # tool result: tool_name set, content kept
    assert out[2] == {"role": "tool", "content": "result", "tool_name": "f"}


def test_convert_messages_bad_toolcall_args_default_to_empty():
    t = get_transport("ollama_chat")
    out = t.convert_messages([
        {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "f", "arguments": "not json"}}
        ]},
    ])
    assert out[0]["tool_calls"][0]["function"]["arguments"] == {}


# ── inherited normalization on the adapter's OpenAI shape ───────────────────
def test_normalize_response_maps_thinking_to_reasoning():
    t = get_transport("ollama_chat")
    msg = SimpleNamespace(role="assistant", content="7",
                          reasoning_content="because 12-5", tool_calls=None)
    resp = SimpleNamespace(
        id="ollama-1", model="m",
        choices=[SimpleNamespace(index=0, message=msg, finish_reason="stop")],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=1, total_tokens=4),
    )
    nr = t.normalize_response(resp)
    assert nr.content == "7"
    assert nr.reasoning_content == "because 12-5"
    assert nr.finish_reason == "stop"
    assert nr.usage.completion_tokens == 1


# ── non-streaming adapter ───────────────────────────────────────────────────
def test_ollama_chat_create_shapes_response(monkeypatch):
    payload = {
        "model": "m", "created_at": "t",
        "message": {"role": "assistant", "content": "7", "thinking": ""},
        "done_reason": "stop", "prompt_eval_count": 10, "eval_count": 2,
    }
    monkeypatch.setattr(oa.httpx, "Client", lambda *a, **k: _FakeClient(post_json=payload))
    r = oa.ollama_chat_create(_agent(), {"model": "m", "messages": []})
    assert r.choices[0].message.content == "7"
    assert r.choices[0].message.reasoning_content is None
    assert r.choices[0].finish_reason == "stop"
    assert r.usage.prompt_tokens == 10 and r.usage.completion_tokens == 2


def test_ollama_chat_create_tool_calls(monkeypatch):
    payload = {
        "model": "m",
        "message": {"role": "assistant", "content": "",
                    "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
        "done_reason": "stop", "prompt_eval_count": 5, "eval_count": 1,
    }
    monkeypatch.setattr(oa.httpx, "Client", lambda *a, **k: _FakeClient(post_json=payload))
    r = oa.ollama_chat_create(_agent(), {"model": "m", "messages": []})
    assert r.choices[0].finish_reason == "tool_calls"
    tc = r.choices[0].message.tool_calls[0]
    assert tc.function.name == "get_weather"
    assert json.loads(tc.function.arguments) == {"city": "Paris"}


def test_ollama_chat_create_raises_on_error(monkeypatch):
    monkeypatch.setattr(oa.httpx, "Client", lambda *a, **k: _FakeClient(post_json={"error": "boom"}))
    with pytest.raises(RuntimeError, match="boom"):
        oa.ollama_chat_create(_agent(), {"model": "m", "messages": []})


# ── streaming adapter (NDJSON → OpenAI-shaped chunks) ───────────────────────
def test_ollama_chat_stream_parses_ndjson(monkeypatch):
    lines = [
        json.dumps({"model": "m", "message": {"role": "assistant", "content": "7"}, "done": False}),
        json.dumps({"model": "m", "message": {"role": "assistant", "content": ""},
                    "done": True, "done_reason": "stop", "prompt_eval_count": 3, "eval_count": 1}),
    ]
    monkeypatch.setattr(oa.httpx, "Client", lambda *a, **k: _FakeClient(stream_lines=lines))
    content, finish, usage = [], None, None
    for ch in oa.ollama_chat_stream(_agent(), {"model": "m", "messages": []}):
        d = ch.choices[0].delta
        if d.content:
            content.append(d.content)
        if ch.choices[0].finish_reason:
            finish = ch.choices[0].finish_reason
        if ch.usage:
            usage = ch.usage
    assert "".join(content) == "7"
    assert finish == "stop"
    assert usage.completion_tokens == 1


def test_ollama_chat_stream_tool_calls_set_finish(monkeypatch):
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "",
                                "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}]},
                    "done": False}),
        json.dumps({"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "stop"}),
    ]
    monkeypatch.setattr(oa.httpx, "Client", lambda *a, **k: _FakeClient(stream_lines=lines))
    finish, saw_tc = None, False
    for ch in oa.ollama_chat_stream(_agent(), {"model": "m", "messages": []}):
        if ch.choices[0].delta.tool_calls:
            saw_tc = True
        if ch.choices[0].finish_reason:
            finish = ch.choices[0].finish_reason
    assert saw_tc
    assert finish == "tool_calls"


# ── capability probe + endpoint detection ───────────────────────────────────
def test_supports_thinking_reads_capabilities(monkeypatch):
    oa._thinking_capability_cache.clear()
    monkeypatch.setattr(oa.httpx, "Client",
                        lambda *a, **k: _FakeClient(post_json={"capabilities": ["completion", "thinking"]}))
    assert oa.ollama_supports_thinking("http://localhost:11434/v1", "gemma4") is True
    oa._thinking_capability_cache.clear()
    monkeypatch.setattr(oa.httpx, "Client",
                        lambda *a, **k: _FakeClient(post_json={"capabilities": ["completion"]}))
    assert oa.ollama_supports_thinking("http://localhost:11434/v1", "plain") is False


def test_determine_api_mode_ollama_vs_cloud(monkeypatch):
    providers._ollama_endpoint_cache.clear()
    # Local + /api/version present → ollama_chat
    monkeypatch.setattr(providers, "_is_ollama_endpoint", lambda url: True)
    assert providers.determine_api_mode("custom", "http://localhost:11434/v1") == "ollama_chat"
    # Cloud host is never probed (host-local gate) → falls through
    providers._ollama_endpoint_cache.clear()
    called = {"probed": False}

    def _spy(url):
        called["probed"] = True
        return False

    monkeypatch.setattr(providers, "_is_ollama_endpoint", _spy)
    mode = providers.determine_api_mode("anthropic", "https://api.anthropic.com")
    assert mode != "ollama_chat"
    assert called["probed"] is False  # cloud host gated out before probe


def test_is_ollama_endpoint_probe(monkeypatch):
    providers._ollama_endpoint_cache.clear()
    import httpx as _httpx
    monkeypatch.setattr(_httpx, "Client",
                        lambda *a, **k: _FakeClient(get_json={"version": "0.30.10"}))
    assert providers._is_ollama_endpoint("http://localhost:11434/v1") is True
    providers._ollama_endpoint_cache.clear()
    monkeypatch.setattr(_httpx, "Client", lambda *a, **k: _FakeClient(get_json={}))
    assert providers._is_ollama_endpoint("http://localhost:9999") is False
