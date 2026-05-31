import json
from collections.abc import AsyncIterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

from openai_router.chat_logging import (
    CHAT_COMPLETION_PATHS,
    _log_chat_response,
    _log_stream_chat_response,
    _parse_sse_data,
    _stream_backend_response_with_logging,
)
from openai_router.runtime import runtime_state
from openai_router.services import route_service

HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


@dataclass
class ResolvedRouteRequest:
    backend_server_url: str
    backend_url: str
    json_body: dict[str, Any]
    backend_api_key: str | None
    routed_model_name: str


@dataclass
class ResolvedResponseRequest:
    backend_url: str
    backend_api_key: str | None


async def resolve_route_request(request: Request) -> ResolvedRouteRequest:
    try:
        json_body = await request.json()
    except Exception as exc:
        logger.error("Failed to parse request body: {}", exc)
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    original_json_body = deepcopy(json_body)
    model_name = json_body.get("model")
    if model_name is None:
        raise HTTPException(status_code=400, detail="'model' field is required in request body")

    logger.debug(
        "Original request body for model {}: {}",
        model_name,
        _serialize_json_body_for_logging(original_json_body),
    )

    routing_target = await run_in_threadpool(
        route_service.get_routing_target,
        model_name,
        json_body,
        dict(request.headers),
    )
    if routing_target.backend_server_url is None:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid model: {model_name}. "
                f"Available models: {list(routing_target.available_model_names)}"
            ),
        )

    backend_url = build_backend_url(routing_target.backend_server_url, request.url.path)
    try:
        json_body = apply_request_param_mapping(json_body, routing_target.request_param_mapping)
    except ValueError as exc:
        logger.error("Invalid request parameter mapping for model {}: {}", model_name, exc)
        raise HTTPException(
            status_code=500,
            detail="Invalid request parameter mapping configuration for routed model",
        ) from exc

    json_body["model"] = routing_target.routed_model_name
    logger.debug(
        "Final proxied request body for model {}: {}",
        model_name,
        _serialize_json_body_for_logging(json_body),
    )
    logger.info("Routing to backend_url: {} for model {}", backend_url, model_name)
    return ResolvedRouteRequest(
        backend_server_url=routing_target.backend_server_url,
        backend_url=backend_url,
        json_body=json_body,
        backend_api_key=routing_target.backend_api_key,
        routed_model_name=routing_target.routed_model_name,
    )


async def resolve_response_request(
    request: Request,
    response_id: str,
) -> ResolvedResponseRequest:
    stored_route = runtime_state.get_response_route(response_id)
    if stored_route is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "Unknown response_id. This router can only retrieve or cancel "
                "responses created through the same router instance recently."
            ),
        )

    backend_url = build_backend_url(stored_route.backend_server_url, request.url.path)
    logger.info("Routing response {} to backend_url: {}", response_id, backend_url)
    return ResolvedResponseRequest(
        backend_url=backend_url,
        backend_api_key=stored_route.backend_api_key,
    )


async def stream_proxy(
    backend_url: str,
    request: Request,
    json_body: dict[str, Any] | None,
    backend_api_key: str | None,
    backend_server_url: str | None = None,
) -> Response:
    headers = build_proxy_headers(request, backend_api_key)
    client = get_http_client()
    is_chat = request.url.path in CHAT_COMPLETION_PATHS

    try:
        response = await client.send(
            client.build_request(
                request.method,
                backend_url,
                params=request.query_params,
                json=json_body,
                headers=headers,
            ),
            stream=True,
        )
        if response.status_code >= 400:
            error_content = await response.aread()
            await response.aclose()
            logger.warning(
                "Backend error: {} - {}",
                response.status_code,
                error_content.decode(errors="replace"),
            )
            return Response(
                content=error_content,
                headers=filter_response_headers(response.headers),
                media_type=response.headers.get("Content-Type"),
                status_code=response.status_code,
            )

        if _should_track_response_route(request, backend_server_url):
            return StreamingResponse(
                _stream_backend_response_with_tracking(
                    response,
                    backend_url,
                    backend_server_url,
                    backend_api_key,
                ),
                headers=filter_response_headers(response.headers),
                media_type=response.headers.get("Content-Type"),
                status_code=response.status_code,
                background=BackgroundTask(response.aclose),
            )

        if is_chat:
            return StreamingResponse(
                _stream_backend_response_with_logging(response, backend_url),
                headers=filter_response_headers(response.headers),
                media_type=response.headers.get("Content-Type"),
                status_code=response.status_code,
                background=BackgroundTask(response.aclose),
            )
        return StreamingResponse(
            _stream_backend_response(response, backend_url),
            headers=filter_response_headers(response.headers),
            media_type=response.headers.get("Content-Type"),
            status_code=response.status_code,
            background=BackgroundTask(response.aclose),
        )
    except httpx.ConnectError as exc:
        logger.error("Connection error to backend {}: {}", backend_url, exc)
        return build_proxy_error_response(503, "Backend service unavailable")
    except httpx.ReadTimeout as exc:
        logger.error("Read timeout from backend {}: {}", backend_url, exc)
        return build_proxy_error_response(504, "Backend request timed out")
    except Exception as exc:
        logger.exception("An error occurred during streaming proxy: {}", exc)
        return build_proxy_error_response(500, f"Internal proxy error: {exc}")


async def _stream_backend_response(
    response: httpx.Response,
    backend_url: str,
) -> AsyncIterator[bytes]:
    try:
        async for chunk in response.aiter_bytes():
            yield chunk
    except httpx.ReadTimeout as exc:
        logger.error("Read timeout while streaming from backend {}: {}", backend_url, exc)
    except Exception as exc:
        logger.exception("An error occurred while streaming from backend {}: {}", backend_url, exc)


async def _stream_backend_response_with_tracking(
    response: httpx.Response,
    backend_url: str,
    backend_server_url: str,
    backend_api_key: str | None,
) -> AsyncIterator[bytes]:
    chunks: list[bytes] = []
    pending_sse_line = ""
    has_tracked_response = False

    try:
        async for chunk in response.aiter_bytes():
            chunks.append(chunk)
            if not has_tracked_response:
                pending_sse_line, has_tracked_response = _remember_response_route_from_sse_chunk(
                    pending_sse_line,
                    chunk,
                    backend_server_url,
                    backend_api_key,
                )
            yield chunk

        if not has_tracked_response and pending_sse_line:
            _remember_response_route_from_sse_lines(
                [pending_sse_line],
                backend_server_url,
                backend_api_key,
            )
        _log_stream_chat_response(chunks)
    except httpx.ReadTimeout as exc:
        logger.error("Read timeout while streaming from backend {}: {}", backend_url, exc)
    except Exception as exc:
        logger.exception(
            "An error occurred while streaming from backend {}: {}",
            backend_url,
            exc,
        )


async def non_stream_proxy(
    backend_url: str,
    request: Request,
    json_body: dict[str, Any] | None,
    backend_api_key: str | None,
    backend_server_url: str | None = None,
) -> Response:
    headers = build_proxy_headers(request, backend_api_key)
    client = get_http_client()
    is_chat = request.url.path in CHAT_COMPLETION_PATHS

    try:
        response = await client.request(
            request.method,
            backend_url,
            params=request.query_params,
            json=json_body,
            headers=headers,
        )
        if response.status_code >= 400:
            logger.warning(
                "Backend error: {} - {}",
                response.status_code,
                response.text,
            )
        elif _should_track_response_route(request, backend_server_url):
            _remember_response_route_from_body(
                response.content,
                backend_server_url,
                backend_api_key,
            )
        if is_chat:
            _log_chat_response(response.content)
        return Response(
            content=response.content,
            headers=filter_response_headers(response.headers),
            media_type=response.headers.get("Content-Type"),
            status_code=response.status_code,
        )
    except httpx.ConnectError as exc:
        logger.error("Connection error to backend {}: {}", backend_url, exc)
        return build_proxy_error_response(503, "Backend service unavailable")
    except httpx.ReadTimeout as exc:
        logger.error("Read timeout from backend {}: {}", backend_url, exc)
        return build_proxy_error_response(504, "Backend request timed out")
    except Exception as exc:
        logger.exception("An error occurred during non-streaming proxy: {}", exc)
        return build_proxy_error_response(500, f"Internal proxy error: {exc}")


def build_backend_url(server: str, path: str) -> str:
    parsed = urlsplit(server.rstrip("/"))
    base_path = parsed.path.rstrip("/")

    if not base_path:
        resolved_path = path
    elif path == base_path or path.startswith(f"{base_path}/"):
        resolved_path = path
    else:
        resolved_path = f"{base_path}{path}"

    return urlunsplit((parsed.scheme, parsed.netloc, resolved_path, "", ""))


def apply_request_param_mapping(
    json_body: dict[str, Any],
    request_param_mapping: str | None,
) -> dict[str, Any]:
    if not request_param_mapping:
        return json_body

    try:
        raw_mapping = json.loads(request_param_mapping)
    except json.JSONDecodeError as exc:
        raise ValueError("Mapping configuration is not valid JSON.") from exc

    if not isinstance(raw_mapping, dict):
        raise ValueError("Mapping configuration must be a JSON object.")

    for raw_source_path, raw_target_path in raw_mapping.items():
        if not isinstance(raw_source_path, str) or not isinstance(raw_target_path, str):
            raise ValueError("Mapping configuration keys and values must be strings.")

        source_segments = _split_request_param_path(raw_source_path)
        target_segments = _split_request_param_path(raw_target_path)
        found, value = _pop_nested_value(json_body, source_segments)
        if not found:
            continue
        _set_nested_value(json_body, target_segments, value)

    return json_body


def _serialize_json_body_for_logging(json_body: Mapping[str, Any]) -> str:
    return json.dumps(json_body, ensure_ascii=False, sort_keys=True)


def build_proxy_headers(request: Request, backend_api_key: str | None) -> dict[str, str]:
    headers = {
        name: value
        for name, value in request.headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }
    if backend_api_key:
        headers["Authorization"] = f"Bearer {backend_api_key}"
    return headers


def filter_response_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        name: value
        for name, value in headers.items()
        if name.lower() not in HOP_BY_HOP_HEADERS
    }


def build_proxy_error_response(status_code: int, detail: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": detail})


def get_http_client() -> httpx.AsyncClient:
    if runtime_state.client is None:
        raise RuntimeError("HTTP client is not initialized.")
    return runtime_state.client


def _split_request_param_path(path: str) -> list[str]:
    segments = [segment.strip() for segment in path.split(".")]
    if any(not segment for segment in segments):
        raise ValueError(f"Invalid request parameter mapping path: '{path}'.")
    return segments


def _pop_nested_value(
    json_body: dict[str, Any],
    path_segments: list[str],
) -> tuple[bool, Any]:
    current: Any = json_body
    parents: list[tuple[dict[str, Any], str]] = []

    for segment in path_segments[:-1]:
        if not isinstance(current, dict):
            return False, None
        next_value = current.get(segment)
        if not isinstance(next_value, dict):
            return False, None
        parents.append((current, segment))
        current = next_value

    if not isinstance(current, dict):
        return False, None

    leaf_key = path_segments[-1]
    if leaf_key not in current:
        return False, None

    value = current.pop(leaf_key)
    for parent, parent_key in reversed(parents):
        child = parent.get(parent_key)
        if isinstance(child, dict) and not child:
            parent.pop(parent_key, None)
            continue
        break

    return True, value


def _set_nested_value(
    json_body: dict[str, Any],
    path_segments: list[str],
    value: Any,
) -> None:
    current = json_body
    for segment in path_segments[:-1]:
        next_value = current.get(segment)
        if not isinstance(next_value, dict):
            next_value = {}
            current[segment] = next_value
        current = next_value

    current[path_segments[-1]] = value


def _should_track_response_route(
    request: Request,
    backend_server_url: str | None,
) -> bool:
    return (
        backend_server_url is not None
        and request.method.upper() == "POST"
        and request.url.path == "/v1/responses"
    )


def _remember_response_route_from_body(
    response_content: bytes,
    backend_server_url: str,
    backend_api_key: str | None,
) -> None:
    try:
        response_payload = json.loads(response_content)
    except json.JSONDecodeError:
        return

    if isinstance(response_payload, Mapping):
        _remember_response_route_from_payload(
            response_payload,
            backend_server_url,
            backend_api_key,
        )


def _remember_response_route_from_sse_chunk(
    pending_sse_line: str,
    chunk: bytes,
    backend_server_url: str,
    backend_api_key: str | None,
) -> tuple[str, bool]:
    sse_text = pending_sse_line + chunk.decode("utf-8", errors="ignore")
    lines = sse_text.split("\n")
    new_pending_sse_line = lines.pop()
    return new_pending_sse_line, _remember_response_route_from_sse_lines(
        [line.rstrip("\r") for line in lines],
        backend_server_url,
        backend_api_key,
    )


def _remember_response_route_from_sse_lines(
    lines: list[str],
    backend_server_url: str,
    backend_api_key: str | None,
) -> bool:
    for line in lines:
        payload = _parse_sse_data(line)
        if isinstance(payload, Mapping) and _remember_response_route_from_payload(
            payload,
            backend_server_url,
            backend_api_key,
        ):
            return True
    return False


def _remember_response_route_from_payload(
    payload: Mapping[str, Any],
    backend_server_url: str,
    backend_api_key: str | None,
) -> bool:
    response_id = _extract_response_id(payload)
    if response_id is None:
        return False

    runtime_state.remember_response_route(
        response_id,
        backend_server_url,
        backend_api_key,
    )
    logger.debug("Tracked response {} -> {}", response_id, backend_server_url)
    return True


def _extract_response_id(payload: Mapping[str, Any]) -> str | None:
    response_id = payload.get("id")
    if isinstance(response_id, str) and response_id:
        return response_id

    nested_response = payload.get("response")
    if isinstance(nested_response, Mapping):
        nested_response_id = nested_response.get("id")
        if isinstance(nested_response_id, str) and nested_response_id:
            return nested_response_id

    return None
