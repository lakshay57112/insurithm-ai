import json

import httpx
import pytest

from insurithm.agent.llm import (
    FunctionLLM,
    GeminiLLM,
    LLMConfigError,
    LLMError,
    LLMOutputError,
    ScriptedLLM,
    parse_json_object,
)


def reply(obj, finish="STOP"):
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(obj)}]}, "finishReason": finish}]}


def make_llm(handler, **kw):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    llm = GeminiLLM("secret-key-123", "gemini-test", client=client, sleep=fake_sleep, **kw)
    return llm, sleeps


async def test_request_shape_and_secret_handling():
    seen = {}

    def handler(request: httpx.Request):
        seen["url"] = str(request.url)
        seen["key"] = request.headers.get("x-goog-api-key")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=reply({"ok": True}))

    llm, _ = make_llm(handler)
    assert await llm.complete_json("draft", "SYSTEM TEXT", "USER TEXT") == {"ok": True}

    assert seen["url"].endswith("/models/gemini-test:generateContent")
    assert "secret-key-123" not in seen["url"]
    assert seen["key"] == "secret-key-123"
    body = seen["body"]
    assert body["systemInstruction"]["parts"][0]["text"] == "SYSTEM TEXT"
    assert body["contents"][0]["parts"][0]["text"] == "USER TEXT"
    assert body["generationConfig"]["temperature"] == 0
    assert body["generationConfig"]["responseMimeType"] == "application/json"


async def test_temperature_and_token_limit_are_configurable(monkeypatch):
    bodies = []

    def handler(request):
        bodies.append(json.loads(request.content)["generationConfig"])
        return httpx.Response(200, json=reply({"ok": 1}))

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    await GeminiLLM("k", "m", client=client, temperature=None, max_output_tokens=123).complete_json("s", "sys", "p")
    await GeminiLLM("k", "m", client=client).complete_json("s", "sys", "p")
    assert "temperature" not in bodies[0] and bodies[0]["maxOutputTokens"] == 123
    assert bodies[1]["temperature"] == 0.0 and bodies[1]["maxOutputTokens"] == 8192

    monkeypatch.chdir("/")
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GEMINI_MODEL", "m")
    monkeypatch.setenv("GEMINI_TEMPERATURE", "none")
    monkeypatch.setenv("GEMINI_MAX_OUTPUT_TOKENS", "4096")
    llm = GeminiLLM.from_env()
    assert llm._temperature is None and llm._max_output_tokens == 4096
    await llm.aclose()


async def test_model_prefix_is_accepted():
    llm = GeminiLLM("k", "models/gemini-x")
    assert llm.model == "gemini-x"
    await llm.aclose()


async def test_retries_rate_limits_then_succeeds():
    attempts = []

    def handler(request):
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json=reply({"n": len(attempts)}))

    llm, sleeps = make_llm(handler)
    assert await llm.complete_json("s", "sys", "p") == {"n": 3}
    assert sleeps == [2.0, 4.0]  # exponential backoff


async def test_retry_after_header_is_honoured():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(429, headers={"retry-after": "7"}, json={})
        return httpx.Response(200, json=reply({"ok": 1}))

    llm, sleeps = make_llm(handler)
    await llm.complete_json("s", "sys", "p")
    assert sleeps == [7.0]


async def test_gives_up_after_max_retries():
    llm, sleeps = make_llm(lambda r: httpx.Response(503, json={}), max_retries=2)
    with pytest.raises(LLMError, match="3 attempts"):
        await llm.complete_json("s", "sys", "p")
    assert len(sleeps) == 2


async def test_client_errors_are_not_retried_and_do_not_leak_the_key():
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(400, json={"error": {"message": "API key not valid"}})

    llm, sleeps = make_llm(handler)
    with pytest.raises(LLMError) as err:
        await llm.complete_json("s", "sys", "p")
    assert "400" in str(err.value) and "secret-key-123" not in str(err.value)
    assert len(calls) == 1 and sleeps == []


async def test_network_errors_are_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ConnectError("boom")
        return httpx.Response(200, json=reply({"ok": 1}))

    llm, _ = make_llm(handler)
    assert await llm.complete_json("s", "sys", "p") == {"ok": 1}


async def test_blocked_prompt_and_truncated_and_invalid_output():
    llm, _ = make_llm(lambda r: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    with pytest.raises(LLMError, match="SAFETY"):
        await llm.complete_json("s", "sys", "p")

    llm, _ = make_llm(lambda r: httpx.Response(200, json=reply({"a": 1}, finish="MAX_TOKENS")))
    with pytest.raises(LLMOutputError, match="cut off"):
        await llm.complete_json("s", "sys", "p")

    garbage = {"candidates": [{"content": {"parts": [{"text": "I cannot do that"}]}}]}
    llm, _ = make_llm(lambda r: httpx.Response(200, json=garbage))
    with pytest.raises(LLMOutputError):
        await llm.complete_json("s", "sys", "p")


def test_from_env_requires_key_and_model(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    monkeypatch.chdir("/")  # make sure no .env file is picked up
    with pytest.raises(LLMConfigError, match="GEMINI_API_KEY"):
        GeminiLLM.from_env()
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    with pytest.raises(LLMConfigError, match="GEMINI_MODEL"):
        GeminiLLM.from_env()
    monkeypatch.setenv("GEMINI_MODEL", "m")
    assert GeminiLLM.from_env().model == "m"


def test_parse_json_object_variants():
    assert parse_json_object('{"a": 1}') == {"a": 1}
    assert parse_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Sure! Here it is: {"a": {"b": 2}} Hope that helps.') == {"a": {"b": 2}}
    for bad in ("", "no json", "[1, 2]", "{broken"):
        with pytest.raises(LLMOutputError):
            parse_json_object(bad)


async def test_scripted_llm_records_calls_and_runs_out():
    llm = ScriptedLLM([{"a": 1}, '{"b": 2}', RuntimeError("model down")])
    assert await llm.complete_json("investigate", "sys", "p1") == {"a": 1}
    assert await llm.complete_json("draft", "sys", "p2") == {"b": 2}
    with pytest.raises(RuntimeError):
        await llm.complete_json("draft", "sys", "p3")
    assert [c["stage"] for c in llm.calls] == ["investigate", "draft", "draft"]
    with pytest.raises(AssertionError, match="ran out"):
        await llm.complete_json("draft", "sys", "p4")


async def test_function_llm_routes_by_stage():
    llm = FunctionLLM(lambda stage, system, prompt: {"stage": stage})
    assert await llm.complete_json("draft", "s", "p") == {"stage": "draft"}
