"""
LLM provider abstractions — Groq, Gemini, and Ollama.

Each provider implements the same interface so the engine can
rotate between them transparently.
"""

from __future__ import annotations

import abc
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Optional

logger = logging.getLogger(__name__)


# ── Data Classes ─────────────────────────────────────────────

@dataclass
class LLMResponse:
    """Standardised response from any LLM provider."""
    text: str
    provider: str
    model: str
    tokens_used: int = 0
    latency_ms: float = 0.0


@dataclass
class ProviderHealth:
    """Tracks rate-limit state for a provider."""
    remaining_requests: Optional[int] = None
    reset_time: Optional[float] = None
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    cooldown_until: float = 0.0  # unix timestamp

    @property
    def is_cooling_down(self) -> bool:
        return time.time() < self.cooldown_until

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.last_error = None

    def record_failure(self, error: str, cooldown_seconds: int = 30) -> None:
        self.consecutive_failures += 1
        self.last_error = error
        self.cooldown_until = time.time() + cooldown_seconds


# ── Abstract Base ────────────────────────────────────────────

class BaseLLMProvider(abc.ABC):
    """Interface that every LLM provider must implement."""

    name: str = "base"
    health: ProviderHealth = field(default_factory=ProviderHealth)

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider can accept requests right now."""
        ...

    @abc.abstractmethod
    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Blocking generation — returns full text."""
        ...

    def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        """Generate and parse JSON. Falls back to extracting JSON from prose."""
        resp = self.generate(
            prompt, system=system, temperature=temperature,
            max_tokens=max_tokens, json_mode=True,
        )
        return _parse_json_tolerant(resp.text)


# ── Groq Provider ────────────────────────────────────────────

class GroqProvider(BaseLLMProvider):
    """Groq cloud — blazing-fast open-weight models (free tier)."""

    name = "groq"

    def __init__(self, api_key: str, model: str, fast_model: str | None = None):
        self.api_key = api_key
        self.model = model
        self.fast_model = fast_model or model
        self.health = ProviderHealth()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from groq import Groq
            self._client = Groq(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        if self.health.is_cooling_down:
            return False
        return True

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
        use_fast: bool = False,
    ) -> LLMResponse:
        client = self._get_client()
        model = self.fast_model if use_fast else self.model

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        t0 = time.time()
        try:
            resp = client.chat.completions.create(**kwargs)
            self.health.record_success()
            text = resp.choices[0].message.content or ""
            tokens = getattr(resp.usage, "total_tokens", 0) if resp.usage else 0
            return LLMResponse(
                text=text,
                provider=self.name,
                model=model,
                tokens_used=tokens,
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            error_str = str(e)
            cooldown = 60 if "rate_limit" in error_str.lower() else 10
            self.health.record_failure(error_str, cooldown_seconds=cooldown)
            raise


# ── Gemini Provider ──────────────────────────────────────────

class GeminiProvider(BaseLLMProvider):
    """Google Gemini via google-genai SDK (free tier)."""

    name = "gemini"

    def __init__(self, api_key: str, model: str):
        self.api_key = api_key
        self.model = model
        self.health = ProviderHealth()
        self._client = None

    def _get_client(self):
        if self._client is None:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def is_available(self) -> bool:
        if not self.api_key:
            return False
        if self.health.is_cooling_down:
            return False
        return True

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
        **kwargs,
    ) -> LLMResponse:
        client = self._get_client()
        from google.genai import types

        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if json_mode:
            config_kwargs["response_mime_type"] = "application/json"

        config = types.GenerateContentConfig(**config_kwargs)

        t0 = time.time()
        try:
            resp = client.models.generate_content(
                model=self.model,
                contents=full_prompt,
                config=config,
            )
            self.health.record_success()
            text = resp.text or ""
            return LLMResponse(
                text=text,
                provider=self.name,
                model=self.model,
                tokens_used=0,
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            error_str = str(e)
            cooldown = 60 if "429" in error_str or "quota" in error_str.lower() else 10
            self.health.record_failure(error_str, cooldown_seconds=cooldown)
            raise


# ── Ollama Provider ──────────────────────────────────────────

class OllamaProvider(BaseLLMProvider):
    """Ollama — local models, unlimited, no API key needed."""

    name = "ollama"

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.health = ProviderHealth()
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if self.health.is_cooling_down:
            return False
        # Cache availability check (Ollama might not be installed)
        if self._available is None:
            try:
                import requests
                r = requests.get(f"{self.base_url}/api/tags", timeout=2)
                self._available = r.status_code == 200
            except Exception:
                self._available = False
        return self._available

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
        **kwargs,
    ) -> LLMResponse:
        try:
            from ollama import Client
        except ImportError:
            raise RuntimeError("ollama package not installed")

        client = Client(host=self.base_url)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
        }

        t0 = time.time()
        try:
            resp = client.chat(
                model=self.model,
                messages=messages,
                format="json" if json_mode else "",
                options=options,
            )
            self.health.record_success()
            text = resp.get("message", {}).get("content", "")
            tokens = resp.get("eval_count", 0) + resp.get("prompt_eval_count", 0)
            return LLMResponse(
                text=text,
                provider=self.name,
                model=self.model,
                tokens_used=tokens,
                latency_ms=(time.time() - t0) * 1000,
            )
        except Exception as e:
            self._available = False
            self.health.record_failure(str(e), cooldown_seconds=30)
            raise


# ── Utility ──────────────────────────────────────────────────

def _parse_json_tolerant(text: str) -> dict:
    """
    Parse JSON from LLM output that may wrap JSON in markdown or prose.
    Scans for the first valid JSON object using raw_decode.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    if "```" in text:
        lines = text.split("\n")
        cleaned = []
        inside = False
        for line in lines:
            if line.strip().startswith("```"):
                inside = not inside
                continue
            if inside:
                cleaned.append(line)
        if cleaned:
            try:
                return json.loads("\n".join(cleaned))
            except json.JSONDecodeError:
                pass

    # Scan for first JSON object using raw_decode
    decoder = json.JSONDecoder()
    for i in range(len(text)):
        if text[i] == "{":
            try:
                obj, _ = decoder.raw_decode(text, i)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                continue

    # Last resort: return empty dict
    logger.warning("Could not parse JSON from LLM output: %s", text[:200])
    return {}
