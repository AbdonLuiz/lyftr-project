"""LiteLLM key pool, ordered failover, and response cache.

Cache lookup happens before any provider call. The cache key is the model plus
the exact message list (system prompt, stored history, and the new user message),
so a repeated first turn in a new session is a hit and a different transcript is not.
LiteLLM 1.111 removed its disk cache, so this uses LiteLLM's in-memory cache.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

import litellm
from litellm import acompletion
from litellm.caching.caching import Cache, CacheMode
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.types.caching import LiteLLMCacheType

from app.config import Settings
from app.simulation import status_for_key

logger = logging.getLogger(__name__)

litellm.suppress_debug_info = True
litellm.set_verbose = False
litellm.drop_params = True


class ProvidersUnavailable(Exception):
    """Every configured key failed or was simulated as failed."""


@dataclass(frozen=True)
class KeySlot:
    key_id: str
    api_key: str
    model: str


@dataclass(frozen=True)
class Completion:
    reply: str
    model: str
    input_tokens: int
    output_tokens: int
    key_id: str | None
    fallback_attempts: int
    resolution_source: str
    latency_ms: int


def parse_key_pool(settings: Settings) -> list[KeySlot]:
    keys = [item.strip() for item in settings.llm_api_keys.split(",") if item.strip()]
    model_names = [item.strip() for item in settings.llm_models.split(",") if item.strip()]
    if model_names and len(model_names) != len(keys):
        raise ValueError("LLM_MODELS must contain one model name for each entry in LLM_API_KEYS")
    slots: list[KeySlot] = []
    for index, api_key in enumerate(keys):
        model = model_names[index] if model_names else settings.llm_model.strip()
        if not model:
            raise ValueError("LLM_MODEL is empty")
        slots.append(KeySlot(key_id=f"key_{index + 1}", api_key=api_key, model=model))
    return slots


def cache_key_for(model: str, messages: list[dict[str, str]]) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{"role": str(item["role"]), "content": str(item["content"])} for item in messages]


def _elapsed_ms(started: float) -> int:
    return max(0, int(round((time.perf_counter() - started) * 1000)))


def _read_count(usage: object, name: str) -> int | None:
    if usage is None:
        return None
    if isinstance(usage, dict):
        value = usage.get(name)
    else:
        value = getattr(usage, name, None)
    if value is None:
        return None
    return max(0, int(value))


def _estimate_tokens(model: str, messages: list[dict[str, str]], reply: str) -> tuple[int, int]:
    try:
        prompt = litellm.token_counter(model=model, messages=messages)
        completion = litellm.token_counter(model=model, text=reply)
        return max(0, int(prompt)), max(0, int(completion))
    except Exception:
        prompt_chars = sum(len(item["content"]) for item in messages)
        return max(1, prompt_chars // 4), max(1, len(reply) // 4)


def extract_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "".join(parts)
    return ""


def _is_cache_hit(response: object) -> bool:
    hidden = getattr(response, "_hidden_params", None)
    return isinstance(hidden, dict) and hidden.get("cache_hit") is True


class LLMGateway:
    def __init__(self, slots: list[KeySlot], cache_ttl_seconds: int, timeout_seconds: float) -> None:
        self.slots = slots
        self.timeout_seconds = timeout_seconds
        self.cache_ttl_seconds = cache_ttl_seconds
        memory = InMemoryCache(
            max_size_in_memory=10_000,
            default_ttl=cache_ttl_seconds,
            max_size_per_item=4096,
        )
        litellm.cache = Cache(
            type=LiteLLMCacheType.LOCAL,
            mode=CacheMode.default_on,
            max_messages=None,
            ttl=cache_ttl_seconds,
            _backend=memory,
        )
        self.cache_enabled = True

    @classmethod
    def from_settings(cls, settings: Settings) -> LLMGateway:
        return cls(
            slots=parse_key_pool(settings),
            cache_ttl_seconds=settings.cache_ttl_seconds,
            timeout_seconds=settings.llm_timeout_seconds,
        )

    def _read_cache(self, model: str, messages: list[dict[str, str]]) -> dict[str, object] | None:
        key = cache_key_for(model, messages)
        try:
            cached = litellm.cache.get_cache(cache_key=key, messages=messages, cache={"ttl": self.cache_ttl_seconds})
        except Exception:
            logger.warning("cache lookup failed")
            return None
        if not isinstance(cached, dict):
            return None
        reply = cached.get("reply")
        if not isinstance(reply, str) or not reply.strip():
            return None
        return cached

    def _write_cache(
        self,
        model: str,
        messages: list[dict[str, str]],
        reply: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        key = cache_key_for(model, messages)
        payload = {
            "reply": reply,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
        try:
            litellm.cache.add_cache(
                payload,
                cache_key=key,
                messages=messages,
                cache={"ttl": self.cache_ttl_seconds},
            )
        except Exception:
            logger.warning("cache store failed")

    async def complete(
        self,
        messages: list[dict[str, str]],
        simulated: dict[str, int],
    ) -> Completion:
        if not self.slots:
            raise ProvidersUnavailable()

        normalized = normalize_messages(messages)
        started = time.perf_counter()
        seen_models: list[str] = []
        for slot in self.slots:
            if slot.model not in seen_models:
                seen_models.append(slot.model)

        for model in seen_models:
            cached = self._read_cache(model, normalized)
            if cached is None:
                continue
            return Completion(
                reply=str(cached["reply"]),
                model=str(cached.get("model") or model),
                input_tokens=int(cached.get("input_tokens") or 0),
                output_tokens=int(cached.get("output_tokens") or 0),
                key_id=None,
                fallback_attempts=0,
                resolution_source="LLM_CACHE",
                latency_ms=_elapsed_ms(started),
            )

        attempts = 0
        for slot in self.slots:
            simulated_status = status_for_key(simulated, slot.key_id)
            if simulated_status is not None:
                attempts += 1
                logger.info("simulated failure key_id=%s status=%s", slot.key_id, simulated_status)
                continue
            try:
                response = await acompletion(
                    model=slot.model,
                    messages=normalized,
                    api_key=slot.api_key,
                    timeout=self.timeout_seconds,
                    num_retries=0,
                    caching=True,
                )
                reply = extract_text(response)
                if not reply.strip():
                    raise ValueError("empty completion")
                usage = getattr(response, "usage", None)
                input_tokens = _read_count(usage, "prompt_tokens")
                output_tokens = _read_count(usage, "completion_tokens")
                if input_tokens is None or output_tokens is None:
                    estimated_in, estimated_out = _estimate_tokens(slot.model, normalized, reply)
                    input_tokens = estimated_in if input_tokens is None else input_tokens
                    output_tokens = estimated_out if output_tokens is None else output_tokens
            except Exception as exc:
                attempts += 1
                logger.warning(
                    "provider call failed key_id=%s error_type=%s status=%s",
                    slot.key_id,
                    type(exc).__name__,
                    getattr(exc, "status_code", None),
                )
                continue

            if _is_cache_hit(response):
                self._write_cache(slot.model, normalized, reply, input_tokens, output_tokens)
                return Completion(
                    reply=reply,
                    model=slot.model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    key_id=None,
                    fallback_attempts=0,
                    resolution_source="LLM_CACHE",
                    latency_ms=_elapsed_ms(started),
                )

            self._write_cache(slot.model, normalized, reply, input_tokens, output_tokens)
            return Completion(
                reply=reply,
                model=slot.model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                key_id=slot.key_id,
                fallback_attempts=attempts,
                resolution_source="LIVE_INFERENCE",
                latency_ms=_elapsed_ms(started),
            )

        raise ProvidersUnavailable()
