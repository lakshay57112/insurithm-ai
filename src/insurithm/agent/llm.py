"""Language-model access.

The rest of the code only sees the small `LLM` protocol, so the model can be
swapped, and tests can script every answer without a network or an API key.

`GeminiLLM` talks to the Gemini REST API over httpx. The model name comes from
the environment (`GEMINI_MODEL`); it is deliberately not hard-coded because
free-tier model names change. The API key is sent in a header and is never put
in a URL, a log line or an error message.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import Any, Awaitable, Callable, Protocol, Sequence

import httpx

DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class LLMError(RuntimeError):
    """The model could not be reached or refused the request."""


class LLMConfigError(LLMError):
    """Missing API key or model name."""


class LLMOutputError(LLMError):
    """The model answered, but not with a usable JSON object."""


class LLM(Protocol):
    async def complete_json(self, stage: str, system: str, prompt: str) -> dict[str, Any]:
        """Return one JSON object. `stage` names the step (for routing, logs and test doubles)."""
        ...


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a model reply into a JSON object, tolerating code fences and stray prose."""
    candidate = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end <= start:
            raise LLMOutputError("model reply contained no JSON object") from None
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"model reply was not valid JSON: {exc.msg}") from None
    if not isinstance(value, dict):
        raise LLMOutputError("model reply was JSON but not an object")
    return value


class GeminiLLM:
    def __init__(
        self,
        api_key: str,
        model: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 60.0,
        max_retries: int = 3,
        backoff_seconds: float = 2.0,
        max_output_tokens: int = 8192,
        temperature: float | None = 0.0,
        client: httpx.AsyncClient | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ):
        if not api_key:
            raise LLMConfigError("GEMINI_API_KEY is not set")
        if not model:
            raise LLMConfigError("GEMINI_MODEL is not set (pick a model from your Gemini free tier)")
        self._api_key = api_key
        self.model = model.removeprefix("models/")
        self._url = f"{base_url.rstrip('/')}/models/{self.model}:generateContent"
        self._max_retries = max_retries
        self._backoff = backoff_seconds
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._sleep = sleep
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=timeout)

    @classmethod
    def from_env(cls, **kwargs: Any) -> "GeminiLLM":
        try:
            from dotenv import load_dotenv

            load_dotenv()
        except ImportError:  # python-dotenv is optional
            pass
        # Some newer models ignore or warn about `temperature`; GEMINI_TEMPERATURE=none leaves it out.
        raw_temperature = os.environ.get("GEMINI_TEMPERATURE", "").strip().lower()
        if raw_temperature in ("none", "off", "omit"):
            kwargs.setdefault("temperature", None)
        elif raw_temperature:
            kwargs.setdefault("temperature", float(raw_temperature))
        if os.environ.get("GEMINI_MAX_OUTPUT_TOKENS", "").strip():
            kwargs.setdefault("max_output_tokens", int(os.environ["GEMINI_MAX_OUTPUT_TOKENS"]))
        return cls(os.environ.get("GEMINI_API_KEY", ""), os.environ.get("GEMINI_MODEL", ""), **kwargs)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _body(self, system: str, prompt: str) -> dict[str, Any]:
        config: dict[str, Any] = {"responseMimeType": "application/json", "maxOutputTokens": self._max_output_tokens}
        if self._temperature is not None:
            config["temperature"] = self._temperature
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": config,
        }

    async def complete_json(self, stage: str, system: str, prompt: str) -> dict[str, Any]:
        headers = {"x-goog-api-key": self._api_key, "content-type": "application/json"}
        body = self._body(system, prompt)
        last_problem = "no attempt made"

        for attempt in range(self._max_retries + 1):
            try:
                response = await self._client.post(self._url, headers=headers, json=body)
            except httpx.TransportError as exc:
                last_problem = f"network error: {type(exc).__name__}"
            else:
                if response.status_code == 200:
                    return self._extract(response.json())
                if response.status_code not in RETRY_STATUSES:
                    raise LLMError(f"Gemini API error {response.status_code}: {self._api_message(response)}")
                last_problem = f"HTTP {response.status_code}"
                retry_after = response.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    await self._sleep(min(float(retry_after), 30.0))
                    continue
            if attempt < self._max_retries:
                await self._sleep(self._backoff * (2**attempt))

        raise LLMError(f"Gemini request failed after {self._max_retries + 1} attempts ({last_problem})")

    @staticmethod
    def _api_message(response: httpx.Response) -> str:
        try:
            return str(response.json()["error"]["message"])[:300]
        except (ValueError, KeyError, TypeError):
            return response.text[:300]

    @staticmethod
    def _extract(data: dict[str, Any]) -> dict[str, Any]:
        candidates = data.get("candidates") or []
        if not candidates:
            reason = (data.get("promptFeedback") or {}).get("blockReason", "no candidates returned")
            raise LLMError(f"Gemini returned no answer ({reason})")
        candidate = candidates[0]
        if candidate.get("finishReason") == "MAX_TOKENS":
            raise LLMOutputError("model reply was cut off at the output token limit (raise GEMINI_MAX_OUTPUT_TOKENS)")
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought"))
        if not text.strip():
            raise LLMOutputError("model reply was empty")
        return parse_json_object(text)


class ScriptedLLM:
    """Test double that returns pre-written replies in order and records every call."""

    def __init__(self, responses: Sequence[dict[str, Any] | str | Exception]):
        self._responses = list(responses)
        self.calls: list[dict[str, str]] = []

    async def complete_json(self, stage: str, system: str, prompt: str) -> dict[str, Any]:
        self.calls.append({"stage": stage, "system": system, "prompt": prompt})
        if not self._responses:
            raise AssertionError(f"ScriptedLLM ran out of replies at stage {stage!r}")
        reply = self._responses.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return parse_json_object(reply) if isinstance(reply, str) else reply

    @property
    def remaining(self) -> int:
        return len(self._responses)


class FunctionLLM:
    """Test double that answers by calling `fn(stage, system, prompt)`."""

    def __init__(self, fn: Callable[[str, str, str], dict[str, Any]]):
        self._fn = fn
        self.calls: list[dict[str, str]] = []

    async def complete_json(self, stage: str, system: str, prompt: str) -> dict[str, Any]:
        self.calls.append({"stage": stage, "system": system, "prompt": prompt})
        return self._fn(stage, system, prompt)


class ThrottledLLM:
    """Wraps another LLM and leaves at least `min_interval` seconds between calls (for free-tier rate limits)."""

    def __init__(self, inner: LLM, min_interval: float):
        self._inner = inner
        self._min_interval = min_interval
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def complete_json(self, stage: str, system: str, prompt: str) -> dict[str, Any]:
        async with self._lock:
            wait = self._last + self._min_interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()
        return await self._inner.complete_json(stage, system, prompt)


class CountingLLM:
    """Wraps another LLM and counts the calls made through it."""

    def __init__(self, inner: LLM):
        self._inner = inner
        self.count = 0

    async def complete_json(self, stage: str, system: str, prompt: str) -> dict[str, Any]:
        self.count += 1
        return await self._inner.complete_json(stage, system, prompt)
