"""
Unified LLM client supporting text and vision (multimodal) completions.

Providers
---------
- openrouter : OpenAI-compatible endpoint at openrouter.ai (supports vision models)
- anthropic  : Anthropic Messages API
- openai     : OpenAI Chat Completions API

Vision support
--------------
Pass `image_path` to `complete()` or `complete_full()` to attach a local image
as a base64-encoded inline data URL.  Both OpenAI-compat and Anthropic formats
are handled transparently.

Throughput
----------
`batch_complete` runs requests concurrently via ThreadPoolExecutor.
For 500 scenes × 3 speakers × 3 listeners with 8 workers, wall-clock time
is roughly total_calls / workers × avg_latency.
"""

from __future__ import annotations

import base64
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    role:    str   # "system" | "user" | "assistant"
    content: str


@dataclass
class CompletionResult:
    text:          str
    model:         str
    input_tokens:  int
    output_tokens: int
    latency_ms:    float


# ── Image encoding ────────────────────────────────────────────────────────────

def _encode_image(image_path: str | Path) -> tuple[str, str]:
    """Return (base64_data, mime_type) for a local image file."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg", ".gif": "image/gif",
            ".webp": "image/webp"}.get(suffix, "image/png")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return data, mime


# ── Client ────────────────────────────────────────────────────────────────────

class LLMClient:
    """
    Unified chat-completion client with optional image input.

    Parameters
    ----------
    model       : model string, e.g. "anthropic/claude-sonnet-4-5" or "gpt-4o"
    provider    : "openrouter" | "anthropic" | "openai"
    api_key     : defaults to env var OPENROUTER_API_KEY / ANTHROPIC_API_KEY / OPENAI_API_KEY
    max_tokens  : default generation budget
    temperature : default sampling temperature
    max_retries : retries on transient errors (exponential back-off + jitter)
    """

    def __init__(
        self,
        model:       str,
        provider:    str        = "openrouter",
        api_key:     str | None = None,
        max_tokens:  int        = 512,
        temperature: float      = 0.0,
        max_retries: int        = 6,
    ) -> None:
        self.model       = model
        self.provider    = provider
        self.max_tokens  = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self._client     = self._build_client(provider, api_key)

    # ── Public API ────────────────────────────────────────────────────────────

    def complete(
        self,
        messages:   list[ChatMessage],
        image_path: str | Path | None = None,
        **kwargs,
    ) -> str:
        return self._complete_with_retry(messages, image_path=image_path, **kwargs).text

    def complete_full(
        self,
        messages:   list[ChatMessage],
        image_path: str | Path | None = None,
        **kwargs,
    ) -> CompletionResult:
        return self._complete_with_retry(messages, image_path=image_path, **kwargs)

    def batch_complete(
        self,
        message_batches:  list[list[ChatMessage]],
        image_paths:      list[str | Path | None] | None = None,
        max_workers:      int = 8,
        **kwargs,
    ) -> list[str]:
        """Run multiple completions concurrently. Returns results in input order."""
        if image_paths is None:
            image_paths = [None] * len(message_batches)

        results: dict[int, str] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futs = {
                pool.submit(self.complete, msgs, img, **kwargs): i
                for i, (msgs, img) in enumerate(zip(message_batches, image_paths))
            }
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    logger.error(f"Batch item {idx} failed: {e}")
                    results[idx] = ""
        return [results[i] for i in range(len(message_batches))]

    # ── Retry ─────────────────────────────────────────────────────────────────

    def _complete_with_retry(
        self,
        messages:   list[ChatMessage],
        image_path: str | Path | None = None,
        **kwargs,
    ) -> CompletionResult:
        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                t0     = time.monotonic()
                result = self._call(messages, image_path=image_path, **kwargs)
                result.latency_ms = (time.monotonic() - t0) * 1000
                return result
            except Exception as e:
                last_exc = e
                # 429 = rate limit: wait much longer; 504 = transient: retry quickly
                is_rate_limit = "429" in str(e)
                base_wait = 30.0 if is_rate_limit else 2 ** attempt
                wait = base_wait + random.uniform(0, 2)
                logger.warning(f"LLM call failed (attempt {attempt+1}): {e}. Retry in {wait:.1f}s")
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_exc

    # ── Provider dispatch ─────────────────────────────────────────────────────

    def _call(
        self,
        messages:   list[ChatMessage],
        image_path: str | Path | None = None,
        **kwargs,
    ) -> CompletionResult:
        if self.provider in ("openrouter", "openai"):
            return self._call_openai_compat(messages, image_path=image_path, **kwargs)
        elif self.provider == "anthropic":
            return self._call_anthropic(messages, image_path=image_path, **kwargs)
        raise ValueError(f"Unknown provider: {self.provider}")

    def _call_openai_compat(
        self,
        messages:   list[ChatMessage],
        image_path: str | Path | None = None,
        **kwargs,
    ) -> CompletionResult:
        raw: list[dict] = []
        for m in messages:
            if m.role == "user" and image_path is not None:
                # Attach image inline as base64 data URL in the last user turn
                b64, mime = _encode_image(image_path)
                raw.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url",
                         "image_url": {"url": f"data:{mime};base64,{b64}"}},
                        {"type": "text", "text": m.content},
                    ],
                })
                image_path = None   # only attach once
            else:
                raw.append({"role": m.role, "content": m.content})

        resp  = self._client.chat.completions.create(
            model=self.model,
            messages=raw,  # type: ignore[arg-type]
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            temperature=kwargs.get("temperature", self.temperature),
        )
        usage = resp.usage
        return CompletionResult(
            text=resp.choices[0].message.content or "",
            model=self.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=0.0,
        )

    def _call_anthropic(
        self,
        messages:   list[ChatMessage],
        image_path: str | Path | None = None,
        **kwargs,
    ) -> CompletionResult:
        system = ""
        chat: list[dict] = []
        for m in messages:
            if m.role == "system":
                system = m.content
            elif m.role == "user" and image_path is not None:
                b64, mime = _encode_image(image_path)
                chat.append({
                    "role": "user",
                    "content": [
                        {"type": "image",
                         "source": {"type": "base64", "media_type": mime, "data": b64}},
                        {"type": "text", "text": m.content},
                    ],
                })
                image_path = None
            else:
                chat.append({"role": m.role, "content": m.content})

        resp  = self._client.messages.create(
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
            # Support both ANTHROPIC_API_KEY (standard) and ANTHROPIC (.env shorthand)
            key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC", "")
            return anthropic.Anthropic(api_key=key)
        raise ValueError(f"Unknown provider: {provider}")


# ── Factory helpers ───────────────────────────────────────────────────────────

def openrouter(model: str = "anthropic/claude-haiku-4-5", **kwargs) -> LLMClient:
    return LLMClient(model=model, provider="openrouter", **kwargs)

def anthropic_client(model: str = "claude-sonnet-4-6", **kwargs) -> LLMClient:
    return LLMClient(model=model, provider="anthropic", **kwargs)

def openai_client(model: str = "gpt-4o", **kwargs) -> LLMClient:
    return LLMClient(model=model, provider="openai", **kwargs)
