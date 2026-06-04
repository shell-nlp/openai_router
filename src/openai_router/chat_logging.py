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
            tool_output = _format_tool_calls_from_message(message)
            return _join_model_output(content, reasoning, tool_output)
    return None


def _join_model_output(
    content: str | None,
    reasoning: str | None,
    tool_output: str | None,
) -> str | None:
    output_parts = []
    if reasoning:
        output_parts.append(f"<think>\n{reasoning}\n</think>")
    if content:
        output_parts.append(content)
    if tool_output:
        output_parts.append(tool_output)
    if output_parts:
        return "\n".join(output_parts)
    return None


def _format_tool_calls_from_message(message: dict[str, Any]) -> str | None:
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        return _format_tool_calls(tool_calls)

    function_call = message.get("function_call")
    if isinstance(function_call, dict):
        return _format_tool_calls([{"function": function_call}])

    return None


def _format_tool_calls(tool_calls: list[dict[str, Any]]) -> str | None:
    rendered_calls = []
    for tool_call in tool_calls:
        function = tool_call.get("function")
        if not isinstance(function, dict):
            function = tool_call

        name = function.get("name")
        if not isinstance(name, str) or not name:
            continue

        rendered = [f"<tool_call>\n<function={name}>\n"]
        for argument_name, argument_value in _parse_tool_arguments(
            function.get("arguments")
        ).items():
            rendered.append(f"<parameter={argument_name}>\n")
            if isinstance(argument_value, str):
                rendered.append(argument_value)
            else:
                rendered.append(
                    json.dumps(argument_value, ensure_ascii=False, sort_keys=True)
                )
            rendered.append("\n</parameter>\n")
        rendered.append("</function>\n</tool_call>")
        rendered_calls.append("".join(rendered))

    if rendered_calls:
        return "\n".join(rendered_calls)
    return None


def _parse_tool_arguments(arguments: Any) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    if isinstance(arguments, dict):
        return arguments
    return {}


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
    tool_calls_by_index: dict[int, dict[str, str]] = {}
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
                    _accumulate_stream_tool_calls(delta, tool_calls_by_index)
        except Exception:
            continue

    tool_output = _format_stream_tool_calls(tool_calls_by_index)
    if reasoning_parts:
        return (
            _join_model_output(
                "".join(content_parts),
                "".join(reasoning_parts),
                tool_output,
            ),
            usage,
        )
    if content_parts:
        return _join_model_output("".join(content_parts), None, tool_output), usage
    return tool_output, usage


def _accumulate_stream_tool_calls(
    delta: dict[str, Any],
    tool_calls_by_index: dict[int, dict[str, str]],
) -> None:
    tool_calls = delta.get("tool_calls")
    if isinstance(tool_calls, list):
        for fallback_index, tool_call in enumerate(tool_calls):
            if not isinstance(tool_call, dict):
                continue
            index = tool_call.get("index")
            if not isinstance(index, int):
                index = fallback_index
            stream_tool_call = tool_calls_by_index.setdefault(
                index,
                {"name": "", "arguments": ""},
            )
            function = tool_call.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                if isinstance(name, str) and not stream_tool_call["name"]:
                    stream_tool_call["name"] = name
                arguments = function.get("arguments")
                if isinstance(arguments, str):
                    stream_tool_call["arguments"] += arguments

    function_call = delta.get("function_call")
    if isinstance(function_call, dict):
        stream_tool_call = tool_calls_by_index.setdefault(
            0,
            {"name": "", "arguments": ""},
        )
        name = function_call.get("name")
        if isinstance(name, str) and not stream_tool_call["name"]:
            stream_tool_call["name"] = name
        arguments = function_call.get("arguments")
        if isinstance(arguments, str):
            stream_tool_call["arguments"] += arguments


def _format_stream_tool_calls(
    tool_calls_by_index: dict[int, dict[str, str]],
) -> str | None:
    tool_calls = [
        {"function": tool_calls_by_index[index]}
        for index in sorted(tool_calls_by_index)
    ]
    return _format_tool_calls(tool_calls)


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
