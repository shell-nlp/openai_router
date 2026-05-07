from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from fastapi import HTTPException, Request
from fastapi.responses import Response
from loguru import logger
from starlette.concurrency import run_in_threadpool

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
    backend_url: str
    json_body: dict[str, Any]
    backend_api_key: str | None
    routed_model_name: str


async def resolve_route_request(request: Request) -> ResolvedRouteRequest:
    try:
        json_body = await request.json()
    except Exception as exc:
        logger.error("Failed to parse request body: {}", exc)
        raise HTTPException(status_code=400, detail="Invalid JSON body") from exc

    model_name = json_body.get("model")
    if model_name is None:
        raise HTTPException(status_code=400, detail="'model' field is required in request body")

    server, available_models, backend_api_key, routed_model_name = await run_in_threadpool(
        route_service.get_routing_target,
        model_name,
        json_body,
        dict(request.headers),
    )
    if server is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model_name}. Available models: {available_models}",
        )

    backend_url = build_backend_url(server, request.url.path)
    json_body["model"] = routed_model_name
    logger.info("Routing to backend_url: {} for model {}", backend_url, model_name)
    return ResolvedRouteRequest(
        backend_url=backend_url,
        json_body=json_body,
        backend_api_key=backend_api_key,
        routed_model_name=routed_model_name,
    )


async def stream_proxy(
    backend_url: str,
    request: Request,
    json_body: dict[str, Any],
    backend_api_key: str | None,
):
    headers = build_proxy_headers(request, backend_api_key)
    client = get_http_client()

    try:
        async with client.stream(
            request.method,
            backend_url,
            params=request.query_params,
            json=json_body,
            headers=headers,
        ) as response:
            if response.status_code >= 400:
                error_content = await response.aread()
                logger.warning(
                    "Backend error: {} - {}",
                    response.status_code,
                    error_content.decode(),
                )
                raise HTTPException(
                    status_code=response.status_code,
                    detail=error_content.decode(),
                )

            async for chunk in response.aiter_bytes():
                yield chunk
    except httpx.ConnectError as exc:
        logger.error("Connection error to backend {}: {}", backend_url, exc)
        raise HTTPException(status_code=503, detail="Backend service unavailable") from exc


async def non_stream_proxy(
    backend_url: str,
    request: Request,
    json_body: dict[str, Any],
    backend_api_key: str | None,
) -> Response:
    headers = build_proxy_headers(request, backend_api_key)
    client = get_http_client()

    try:
        response = await client.request(
            request.method,
            backend_url,
            params=request.query_params,
            json=json_body,
            headers=headers,
        )
        return Response(
            content=response.content,
            headers=filter_response_headers(response.headers),
            media_type=response.headers.get("Content-Type"),
            status_code=response.status_code,
        )
    except httpx.ConnectError as exc:
        logger.error("Connection error to backend {}: {}", backend_url, exc)
        raise HTTPException(status_code=503, detail="Backend service unavailable") from exc
    except httpx.ReadTimeout as exc:
        logger.error("Read timeout from backend {}: {}", backend_url, exc)
        raise HTTPException(status_code=504, detail="Backend request timed out") from exc
    except Exception as exc:
        logger.error("An error occurred during non-streaming proxy: {}", exc)
        raise HTTPException(status_code=500, detail=f"Internal proxy error: {exc}") from exc


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


def get_http_client() -> httpx.AsyncClient:
    if runtime_state.client is None:
        raise RuntimeError("HTTP client is not initialized.")
    return runtime_state.client
