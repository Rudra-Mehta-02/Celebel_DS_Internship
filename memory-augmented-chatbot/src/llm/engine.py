"""
Multi-provider LLM engine with automatic failover.

Call flow:
    try Groq → if 429/timeout → try Gemini → if 429/timeout → try Ollama → raise

The user never sees a provider error — the engine silently rotates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

from src.config import get_settings
from src.llm.providers import (
    BaseLLMProvider,
    GeminiProvider,
    GroqProvider,
    LLMResponse,
    OllamaProvider,
    _parse_json_tolerant,
)

logger = logging.getLogger(__name__)


@dataclass
class UsageStats:
    """Tracks cumulative usage across providers."""
    calls: dict[str, int] = field(default_factory=lambda: {})
    tokens: dict[str, int] = field(default_factory=lambda: {})
    total_latency_ms: dict[str, float] = field(default_factory=lambda: {})

    def record(self, resp: LLMResponse) -> None:
        self.calls[resp.provider] = self.calls.get(resp.provider, 0) + 1
        self.tokens[resp.provider] = self.tokens.get(resp.provider, 0) + resp.tokens_used
        self.total_latency_ms[resp.provider] = (
            self.total_latency_ms.get(resp.provider, 0.0) + resp.latency_ms
        )

    def summary(self) -> dict:
        return {
            "calls_per_provider": dict(self.calls),
            "tokens_per_provider": dict(self.tokens),
            "avg_latency_ms": {
                p: self.total_latency_ms[p] / self.calls[p]
                for p in self.calls
                if self.calls[p] > 0
            },
        }


class MultiProviderLLM:
    """
    Unified LLM interface that rotates across Groq → Gemini → Ollama.

    Usage:
        llm = MultiProviderLLM()
        response = llm.generate("What is deep learning?")
        print(response.text, response.provider)
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.providers: list[BaseLLMProvider] = []
        self.usage = UsageStats()

        # Register providers in priority order
        if settings.has_groq:
            self.providers.append(
                GroqProvider(
                    api_key=settings.groq_api_key,
                    model=settings.groq_model,
                    fast_model=settings.groq_fast_model,
                )
            )
            logger.info("✅ Groq provider registered (primary)")

        if settings.has_gemini:
            self.providers.append(
                GeminiProvider(
                    api_key=settings.google_api_key,
                    model=settings.gemini_model,
                )
            )
            logger.info("✅ Gemini provider registered (fallback)")

        # Ollama is always registered (no key needed)
        self.providers.append(
            OllamaProvider(
                base_url=settings.ollama_base_url,
                model=settings.ollama_model,
            )
        )
        logger.info("✅ Ollama provider registered (local fallback)")

    def generate(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        json_mode: bool = False,
    ) -> LLMResponse:
        """
        Generate text using the first available provider.
        Automatically fails over to the next provider on errors.
        """
        errors: list[str] = []

        for provider in self.providers:
            if not provider.is_available():
                errors.append(f"{provider.name}: cooling down / unavailable")
                continue

            try:
                resp = provider.generate(
                    prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=json_mode,
                )
                self.usage.record(resp)
                logger.debug(
                    "LLM response from %s (%s) in %.0fms",
                    resp.provider, resp.model, resp.latency_ms,
                )
                return resp
            except Exception as e:
                msg = f"{provider.name}: {e}"
                errors.append(msg)
                logger.warning("Provider %s failed: %s — trying next", provider.name, e)
                continue

        raise RuntimeError(
            "All LLM providers failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    def generate_json(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> dict:
        """Generate and parse JSON with tolerant parsing."""
        resp = self.generate(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
            json_mode=True,
        )
        return _parse_json_tolerant(resp.text)

    def generate_fast(
        self,
        prompt: str,
        *,
        system: str = "",
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> LLMResponse:
        """
        Use the fastest available model (Groq 8B or Gemini Flash-Lite).
        Used for routing decisions and query rewriting where speed matters.
        """
        errors: list[str] = []

        for provider in self.providers:
            if not provider.is_available():
                continue
            try:
                kwargs = dict(
                    prompt=prompt,
                    system=system,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    json_mode=False,
                )
                # Use fast model for Groq
                if isinstance(provider, GroqProvider):
                    kwargs["use_fast"] = True

                resp = provider.generate(**kwargs)
                self.usage.record(resp)
                return resp
            except Exception as e:
                errors.append(f"{provider.name}: {e}")
                continue

        raise RuntimeError("All providers failed for fast generation")

    def provider_status(self) -> list[dict]:
        """Return health status for all registered providers."""
        result = []
        for p in self.providers:
            result.append({
                "name": p.name,
                "available": p.is_available(),
                "consecutive_failures": p.health.consecutive_failures,
                "last_error": p.health.last_error,
                "cooling_down": p.health.is_cooling_down,
            })
        return result


# ── Singleton ────────────────────────────────────────────────
_engine: Optional[MultiProviderLLM] = None


def get_llm() -> MultiProviderLLM:
    """Return the cached MultiProviderLLM singleton."""
    global _engine
    if _engine is None:
        _engine = MultiProviderLLM()
    return _engine
