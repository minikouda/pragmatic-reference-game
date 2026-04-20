"""
Unified LLM client supporting OpenRouter (OpenAI-compatible) and Anthropic APIs.

Design
------
- `LLMClient` is a thin, stateless wrapper. All callers construct chat turns as
  `ChatMessage` dataclasses; the client converts them to the provider's format.
- Retry logic uses exponential back-off with jitter (essential for bulk runs).
- The `batch_complete` method runs requests concurrently via `ThreadPoolExecutor`,
  which is the main lever for throughput in the 500-scene evaluation loop.
"""

from __future__ import annotations

import os
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role:    str   # "system" | "user" | "assistant"
    content: str


@dataclass
class CompletionResult:
    text:           str
    model:          str
    input_tokens:   int
    output_tokens:  int
    latency_ms:     float


# ── Client ───────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified chat-completion client.

    Parameters
    ----------
    model      : model string, e.g. "anthropic/claude-sonnet-4-5" or "gpt-4o"
    provider   : "openrouter" | "anthropic" | "openai"
    api_key    : defaults to env var OPENROUTER_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY
    max_tokens : default generation budget
    temperature: default sampling temperature
    max_retries: number of retries on transient errors
    """

    def __init__(
        self,
        model:       str,
        provider:    str   = "openrouter",
        api_key:     str | None = None,
        max_tokens:  int   = 512,
        temperature: float = 0.0,
        max_retries: int   = 3,
    ) -> None:
        self.model       = model
        self.provider    = provider
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries

        self._client = self._build_client(provider, api_key)

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(self, messages: list[ChatMessage], **kwargs) -> str:
        """Synchronous single completion. Returns response text."""
        return self._complete_with_retry(messages, **kwargs).text

    def complete_full(self, messages: list[ChatMessage], **kwargs) -> CompletionResult:
        """Synchronous single completion. Returns full CompletionResult."""
        return self._complete_with_retry(messages, **kwargs)

    def batch_complete(
        self,
        message_batches: list[list[ChatMessage]],
        max_workers: int = 8,
        **kwargs,
    ) -> list[str]:
        """
        Run multiple completions concurrently.

        Returns responses in the same order as `message_batches`.
        """
        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(self.complete, msgs, **kwargs): i
                for i, msgs in enumerate(message_batches)
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    logger.error(f"Batch item {idx} failed: {e}")
                    results[idx] = ""
        return [results[i] for i in range(len(message_batches))]

    # ── Retry logic ────────────────────────────────────────────────────────────

    def _complete_with_retry(self, messages: list[ChatMessage], **kwargs) -> CompletionResult:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0 = time.monotonic()
                result = self._call(messages, **kwargs)
                result.latency_ms = (time.monotonic() - t0) * 1000
                return result
            except Exception as e:
                last_exc = e
                wait = (2 ** attempt) + random.uniform(0, 1)
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_exc

    # ── Provider dispatch ─────────────────────────────────────────────────────

    def _call(self, messages: list[ChatMessage], **kwargs) -> CompletionResult:
        if self.provider in ("openrouter", "openai"):
            return self._call_openai_compat(messages, **kwargs)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages, **kwargs)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai_compat(self, messages: list[ChatMessage], **kwargs) -> CompletionResult:
        raw = [{"role": m.role, "content": m.content} for m in messages]
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=raw,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        usage = resp.usage
        return CompletionResult(
            text=resp.choices[0].message.content,
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=0.0,
        )

    def _call_anthropic(self, messages: list[ChatMessage], **kwargs) -> CompletionResult:
        system = ""
        chat   = []
        for m in messages:
            if m.role == "system":
                system = m.content
            else:
                chat.append({"role": m.role, "content": m.content})
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            system=system,
            messages=chat,
        )
        usage = resp.usage
        return CompletionResult(
            text=resp.content[0].text,
            model=self.model,
            input_tokens=usage.input_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            latency_ms=0.0,
        )

    # ── Client construction ───────────────────────────────────────────────────

    @staticmethod
    def _build_client(provider: str, api_key: str | None):
        if provider == "openrouter":
            from openai import OpenAI
            key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
            return OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        elif provider == "openai":
            from openai import OpenAI
            key = api_key or os.environ.get("OPENAI_API_KEY", "")
            return OpenAI(api_key=key)
        elif provider == "anthropic":
            import anthropic
            key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            return anthropic.Anthropic(api_key=key)
        else:
            raise ValueError(f"Unknown provider: {provider}")


# ── Factory helpers ───────────────────────────────────────────────────────────

def openrouter(model: str = "anthropic/claude-haiku-4-5", **kwargs) -> LLMClient:
    return LLMClient(model=model, provider="openrouter", **kwargs)

def anthropic_client(model: str = "claude-sonnet-4-6", **kwargs) -> LLMClient:
    return LLMClient(model=model, provider="anthropic", **kwargs)

def openai_client(model: str = "gpt-4o", **kwargs) -> LLMClient:
    return LLMClient(model=model, provider="openai", **kwargs)
