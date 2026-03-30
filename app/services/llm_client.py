from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the LLM call fails after all retries."""


class LLMClient:
    """
    Provider-agnostic async LLM client.

    Usage:
        client = LLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
        raw_json = await client.complete(system_prompt, user_prompt)
    """

    def __init__(self, provider: str, model: str):
        self.provider = provider.lower()
        self.model = model
        self._validate_provider()

    def _validate_provider(self):
        supported = {"anthropic", "openai", "huggingface"} #using huggingface for demo
        if self.provider not in supported:
            raise ValueError(f"Unsupported LLM provider '{self.provider}'. Choose from: {supported}")

    # ── Public interface ──────────────────────────────────────────────────────

    async def complete(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        """
        Send a completion request and return (raw_text, usage_metadata).

        Retries up to settings.llm_max_retries times with exponential backoff.
        Raises LLMError if all attempts fail.
        """
        last_exc: Exception | None = None
        for attempt in range(1, settings.llm_max_retries + 1):
            try:
                start = time.perf_counter()
                text, usage = await self._call(system_prompt, user_prompt)
                elapsed_ms = (time.perf_counter() - start) * 1000
                logger.info(
                    "LLM call succeeded | provider=%s model=%s attempt=%d latency=%.0fms "
                    "prompt_tokens=%s completion_tokens=%s",
                    self.provider,
                    self.model,
                    attempt,
                    elapsed_ms,
                    usage.get("prompt_tokens") or usage.get("input_tokens"),
                    usage.get("completion_tokens") or usage.get("output_tokens"),
                )
                usage["latency_ms"] = elapsed_ms
                return text, usage
            except Exception as exc:
                last_exc = exc
                if attempt < settings.llm_max_retries:
                    delay = settings.llm_retry_delay_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "LLM call failed (attempt %d/%d) — retrying in %.1fs | %s",
                        attempt,
                        settings.llm_max_retries,
                        delay,
                        exc,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "LLM call failed after %d attempts | %s", settings.llm_max_retries, exc
                    )

        raise LLMError(f"LLM call failed after {settings.llm_max_retries} attempts") from last_exc

    # Provider dispatcher

    async def _call(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        if self.provider == "anthropic":
            return await self._call_anthropic(system_prompt, user_prompt)
        elif self.provider == "openai":
            return await self._call_openai(system_prompt, user_prompt)
        elif self.provider == "huggingface":
            return await self._call_huggingface(system_prompt, user_prompt)
        raise LLMError(f"Unknown provider: {self.provider}")

    # Anthropic

    async def _call_anthropic(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        try:
            import anthropic
        except ImportError as e:
            raise LLMError("anthropic package not installed. Run: pip install anthropic") from e

        api_key = settings.anthropic_api_key
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set in environment / .env file")

        client = anthropic.AsyncAnthropic(api_key=api_key)

        response = await client.messages.create(
            model=self.model,
            max_tokens=settings.llm_max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )

        text = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return text, usage

    # OpenAI

    async def _call_openai(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        try:
            from openai import AsyncOpenAI
        except ImportError as e:
            raise LLMError("openai package not installed. Run: pip install openai") from e

        api_key = settings.openai_api_key
        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set in environment / .env file")

        client = AsyncOpenAI(api_key=api_key)

        response = await client.chat.completions.create(
            model=self.model,
            max_tokens=settings.llm_max_tokens,
            response_format={"type": "json_object"},  # JSON mode
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        text = response.choices[0].message.content or ""
        usage = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
        }
        return text, usage

    # HuggingFace

    async def _call_huggingface(self, system_prompt: str, user_prompt: str) -> tuple[str, dict]:
        try:
            from huggingface_hub import AsyncInferenceClient
        except ImportError as e:
            raise LLMError("huggingface_hub package not installed. Run: pip install huggingface_hub") from e

        api_key = settings.huggingface_api_key
        if not api_key:
            raise LLMError("HUGGINGFACE_API_KEY is not set in environment / .env file")

        client = AsyncInferenceClient(api_key=api_key)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await client.chat_completion(
                model=self.model,
                messages=messages,
                max_tokens=settings.llm_max_tokens,
            )
        except Exception as exc:
            raise LLMError(f"HuggingFace API call failed: {exc}") from exc

        text = response.choices[0].message.content
        if not text:
            raise LLMError("HuggingFace returned an empty response")

        usage = {
            "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
        }
        return text, usage
