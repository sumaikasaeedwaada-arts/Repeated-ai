import os
import logging
import math
import threading
import time
from contextlib import contextmanager
from collections.abc import Callable, Iterator
from typing import NoReturn, TypeVar

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from AI_service import database, settings
from AI_service.context.business_context import SYSTEM_PROMPT
from AI_service.errors import (
    AIProviderAuthenticationError,
    AIProviderError,
    AIProviderResponseError,
    AIProviderTimeoutError,
)


logger = logging.getLogger(__name__)
MODEL = settings.GEMINI_MODEL
client = genai.Client(
    http_options=types.HttpOptions(timeout=settings.GEMINI_TIMEOUT_MS)
)
_CACHE_LOCKS_GUARD = threading.Lock()
_CACHE_LOCKS: dict[tuple[int, str, str], tuple[threading.Lock, int]] = {}
_CacheValue = TypeVar("_CacheValue")


def _log_provider_failure(
    failure_kind: str,
    error: Exception | None = None,
    provider_status: int | None = None
) -> None:
    logger.warning(
        "Gemini provider request failed.",
        extra={
            "event": "ai_provider_failure",
            "provider": "gemini",
            "failure_kind": failure_kind,
            "error_type": type(error).__name__ if error else None,
            "provider_status": provider_status
        }
    )


def _raise_provider_error(error: Exception) -> NoReturn:
    if isinstance(error, AIProviderError):
        raise error
    if isinstance(error, (TimeoutError, httpx.TimeoutException)):
        _log_provider_failure("timeout", error)
        raise AIProviderTimeoutError from error
    if isinstance(error, genai_errors.APIError):
        status = error.code
        message = (error.message or "").casefold()
        invalid_api_key = (
            status == 400
            and "api key" in message
            and ("invalid" in message or "not valid" in message)
        )
        if status in {401, 403} or invalid_api_key:
            _log_provider_failure("authentication", error, status)
            raise AIProviderAuthenticationError from error
        if status in {408, 504}:
            _log_provider_failure("timeout", error, status)
            raise AIProviderTimeoutError from error
        _log_provider_failure(
            "provider_rate_limit" if status == 429 else "unavailable",
            error,
            status
        )
        raise AIProviderError from error
    if isinstance(error, (httpx.NetworkError, ConnectionError, OSError)):
        _log_provider_failure("connection", error)
        raise AIProviderError from error
    _log_provider_failure("unexpected_provider_error", error)
    raise AIProviderError from error


def _ensure_api_key() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        _log_provider_failure("missing_api_key")
        raise AIProviderAuthenticationError


@contextmanager
def _cache_lock(
    cache_user_id: int,
    cache_type: str,
    cache_key: str
) -> Iterator[None]:
    lock_key = (cache_user_id, cache_type, cache_key)
    with _CACHE_LOCKS_GUARD:
        entry = _CACHE_LOCKS.get(lock_key)
        if entry is None:
            lock, references = threading.Lock(), 0
        else:
            lock, references = entry
        _CACHE_LOCKS[lock_key] = (lock, references + 1)

    lock.acquire()
    try:
        yield
    finally:
        lock.release()
        with _CACHE_LOCKS_GUARD:
            current_lock, references = _CACHE_LOCKS[lock_key]
            if references == 1:
                del _CACHE_LOCKS[lock_key]
            else:
                _CACHE_LOCKS[lock_key] = (current_lock, references - 1)


def _cached_call(
    cache_user_id: int | None,
    cache_type: str,
    cache_key: str,
    ttl_seconds: int,
    compute: Callable[[], _CacheValue],
    parse_cached: Callable[[object], _CacheValue | None]
) -> _CacheValue:
    if cache_user_id is None or ttl_seconds <= 0:
        return compute()

    with _cache_lock(cache_user_id, cache_type, cache_key):
        cached = database.get_cached_value(
            cache_user_id,
            cache_type,
            cache_key
        )
        parsed = parse_cached(cached)
        if parsed is not None:
            _log_cache_hit(cache_type)
            return parsed
        if cached is not None:
            database.delete_cached_value(
                cache_user_id,
                cache_type,
                cache_key
            )

        lease_seconds = max(
            60.0,
            settings.GEMINI_TIMEOUT_MS / 1000 + 10
        )
        deadline = time.monotonic() + lease_seconds + 1
        lease_token = None
        while lease_token is None:
            lease_token = database.claim_cache_entry(
                cache_user_id,
                cache_type,
                cache_key,
                lease_seconds
            )
            if lease_token is not None:
                break

            cached = database.get_cached_value(
                cache_user_id,
                cache_type,
                cache_key
            )
            parsed = parse_cached(cached)
            if parsed is not None:
                _log_cache_hit(cache_type)
                return parsed
            if cached is not None:
                database.delete_cached_value(
                    cache_user_id,
                    cache_type,
                    cache_key
                )
            if time.monotonic() >= deadline:
                logger.warning(
                    "AI cache single-flight wait expired; proceeding uncached."
                )
                return compute()
            time.sleep(0.1)

        try:
            result = compute()
            stored = database.store_claimed_value(
                cache_user_id,
                cache_type,
                cache_key,
                lease_token,
                result,
                ttl_seconds
            )
            if not stored:
                logger.warning("AI cache lease expired before storing a result.")
            return result
        except Exception:
            database.release_cache_claim(
                cache_user_id,
                cache_type,
                cache_key,
                lease_token
            )
            raise


def _parse_cached_answer(value: object) -> str | None:
    return value if isinstance(value, str) and value.strip() else None


def _parse_cached_embedding(value: object) -> list[float] | None:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            for item in value
        )
    ):
        return None
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return vector if all(math.isfinite(item) for item in vector) else None


def _log_cache_hit(cache_type: str) -> None:
    logger.info(
        "AI request reused from a user-scoped cache.",
        extra={"event": "ai_cache_hit", "cache_type": cache_type}
    )


def generate_answer(
    prompt: str,
    cache_user_id: int | None = None
) -> str:
    cache_key = database.make_cache_key(
        "gemini-response-v1",
        {
            "model": MODEL,
            "system_instruction": SYSTEM_PROMPT,
            "prompt": prompt
        }
    )

    def generate() -> str:
        _ensure_api_key()
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_PROMPT
                )
            )
        except Exception as exc:
            _raise_provider_error(exc)

        try:
            text = response.text
        except (AttributeError, TypeError, ValueError) as exc:
            _log_provider_failure("invalid_response", exc)
            raise AIProviderResponseError from exc
        if not isinstance(text, str) or not text.strip():
            _log_provider_failure("invalid_response")
            raise AIProviderResponseError
        return text.strip()

    return _cached_call(
        cache_user_id,
        "gemini-response",
        cache_key,
        settings.AI_RESPONSE_CACHE_TTL_SECONDS,
        generate,
        _parse_cached_answer
    )


def create_embedding(
    text: str,
    task_type: str,
    cache_user_id: int | None = None
) -> list[float]:
    cache_key = database.make_cache_key(
        "gemini-embedding-v1",
        {
            "model": settings.GEMINI_EMBEDDING_MODEL,
            "task_type": task_type,
            "text": text
        }
    )

    def embed() -> list[float]:
        _ensure_api_key()
        try:
            result = client.models.embed_content(
                model=settings.GEMINI_EMBEDDING_MODEL,
                contents=text,
                config=types.EmbedContentConfig(task_type=task_type)
            )
        except Exception as exc:
            _raise_provider_error(exc)

        try:
            values = result.embeddings[0].values
            if not values:
                raise ValueError("Empty embedding.")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in values
            ):
                raise ValueError("Non-numeric embedding.")
            vector = [float(value) for value in values]
            if not all(math.isfinite(value) for value in vector):
                raise ValueError("Non-finite embedding.")
        except (AttributeError, IndexError, TypeError, ValueError, OverflowError) as exc:
            _log_provider_failure("invalid_embedding", exc)
            raise AIProviderResponseError from exc
        return vector

    return _cached_call(
        cache_user_id,
        "gemini-embedding",
        cache_key,
        settings.AI_EMBEDDING_CACHE_TTL_SECONDS,
        embed,
        _parse_cached_embedding
    )
