from collections.abc import AsyncIterator
import json
from typing import Any

from loguru import logger


CHAT_COMPLETION_PATHS = {
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/responses",
}


def _extract_chat_content(response_data: dict[str, Any]) -> str | None:
    choices = response_data.get("choices")
    if not choices:
        return None
    for choice in choices:
        message = choice.get("message") or choice.get("delta")
        if message:
            content = message.get("content")
            reasoning = message.get("reasoning") or message.get("thinking")
            if reasoning:
                return f"<think>\n{reasoning}\n</think>\n{content or ''}"
            if content:
                return content
    return None


def _log_chat_response(response_content: bytes) -> None:
    try:
        data = json.loads(response_content)
        content = _extract_chat_content(data)
        if content:
            logger.info("Model response:\n{}", content)
        usage = _extract_usage(data)
        if usage:
            _log_usage(usage)
    except Exception:
        pass


def _extract_usage(response_data: dict[str, Any]) -> dict[str, Any] | None:
    usage = response_data.get("usage")
    if isinstance(usage, dict):
        return usage

    response = response_data.get("response")
    if isinstance(response, dict):
        nested_usage = response.get("usage")
        if isinstance(nested_usage, dict):
            return nested_usage

    return None


def _log_usage(usage: dict[str, Any]) -> None:
    logger.info(
        "Token usage: {}",
        json.dumps(usage, ensure_ascii=False, sort_keys=True),
    )


def _parse_sse_data(line: str) -> dict[str, Any] | None:
    if line.startswith("data: "):
        data_str = line[6:].strip()
        if data_str == "[DONE]":
            return None
        try:
            return json.loads(data_str)
        except Exception:
            return None
    return None


def _extract_stream_content_and_usage(
    chunks: list[bytes],
) -> tuple[str | None, dict[str, Any] | None]:
    content_parts = []
    reasoning_parts = []
    usage = None
    for chunk in chunks:
        try:
            chunk_str = chunk.decode("utf-8")
            for line in chunk_str.split("\n"):
                data = _parse_sse_data(line)
                if data is None:
                    continue
                extracted_usage = _extract_usage(data)
                if extracted_usage:
                    usage = extracted_usage
                choices = data.get("choices", [])
                for choice in choices:
                    delta = choice.get("delta", {})
                    content = delta.get("content")
                    if content:
                        content_parts.append(content)
                    reasoning = (
                        delta.get("reasoning_content")
                        or delta.get("reasoning")
                        or delta.get("thinking")
                    )
                    if reasoning:
                        reasoning_parts.append(reasoning)
        except Exception:
            continue

    if reasoning_parts:
        return (
            f"<think>\n{''.join(reasoning_parts)}\n</think>\n{''.join(content_parts)}",
            usage,
        )
    if content_parts:
        return "".join(content_parts), usage
    return None, usage


def _log_stream_chat_response(chunks: list[bytes]) -> None:
    content, usage = _extract_stream_content_and_usage(chunks)
    if content:
        logger.info("Model response:\n{}", content)
    if usage:
        _log_usage(usage)


async def _stream_backend_response_with_logging(
    response: Any,
    backend_url: str,
) -> AsyncIterator[bytes]:
    from openai_router.runtime import runtime_state

    chunks: list[bytes] = []
    try:
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
            yield chunk
        _log_stream_chat_response(chunks)
    except runtime_state.client.timeout.ReadTimeout:
        logger.error("Read timeout while streaming from backend {}: {}", backend_url)
    except Exception as exc:
        logger.exception(
            "An error occurred while streaming from backend {}: {}", backend_url, exc
        )
