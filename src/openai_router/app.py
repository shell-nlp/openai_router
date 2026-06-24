import asyncio
import json
import warnings
from collections.abc import Mapping
from contextlib import asynccontextmanager
from pathlib import Path

import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from jinja2 import Environment, FileSystemLoader, select_autoescape
from loguru import logger
from starlette.concurrency import run_in_threadpool

from openai_router.admin import ADMIN_CSS, create_admin_ui
from openai_router.config import MODEL_SYNC_CHECK_INTERVAL_SECONDS
from openai_router.db import create_db_and_tables, dispose_engine, initialize_engine
from openai_router.proxy import (
    non_stream_proxy,
    resolve_response_request,
    resolve_route_request,
    stream_proxy,
)
from openai_router.runtime import runtime_state
from openai_router.services import route_service

TEMPLATE_DIR = Path(__file__).parent


def parse_tool_arguments(arguments):
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    if isinstance(arguments, Mapping):
        return arguments
    return {}


def parse_stream_parameter(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(),
)
# tojson 默认 ensure_ascii=True，会把 tools 中的中文转成 \uXXXX 转义，导致日志里显示为乱码
jinja_env.policies["json.dumps_kwargs"] = {"sort_keys": True, "ensure_ascii": False}
jinja_env.filters["parse_tool_arguments"] = parse_tool_arguments
chat_template = jinja_env.get_template("chat_template.jinja")

warnings.filterwarnings(
    "ignore",
    message="SSR mode is not supported with multi-page apps when mounting on a FastAPI app. Disabling SSR mode.",
    category=UserWarning,
)


async def periodic_model_sync() -> None:
    while True:
        try:
            synced_count = await run_in_threadpool(
                route_service.sync_due_backend_sources
            )
            if synced_count:
                logger.info(
                    "Periodic model sync processed {} backend source(s).", synced_count
                )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.exception("Periodic model sync failed: {}", exc)

        await asyncio.sleep(MODEL_SYNC_CHECK_INTERVAL_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_engine()
    await run_in_threadpool(create_db_and_tables)

    timeout = httpx.Timeout(10.0, connect=60.0, read=None, write=60.0)
    runtime_state.client = httpx.AsyncClient(timeout=timeout)
    runtime_state.sync_task = asyncio.create_task(periodic_model_sync())

    yield

    if runtime_state.sync_task is not None:
        runtime_state.sync_task.cancel()
        try:
            await runtime_state.sync_task
        except asyncio.CancelledError:
            pass
        runtime_state.sync_task = None

    if runtime_state.client is not None:
        await runtime_state.client.aclose()
        runtime_state.client = None
        logger.info("HTTPX client closed.")

    runtime_state.clear_response_routes()
    dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    @app.get(
        "/health",
    )
    async def health():
        return Response(status_code=200)

    @app.get("/v1/models", summary="List available models")
    async def list_models() -> dict:
        try:
            response_data = await run_in_threadpool(route_service.build_models_response)
            logger.info(
                "Returning {} unique available models for /v1/models.",
                len(response_data["data"]),
            )
            return response_data
        except Exception as exc:
            logger.error("Failed to list models: {}", exc)
            raise HTTPException(
                status_code=500,
                detail=f"Internal server error when retrieving models: {exc}",
            ) from exc

    async def route_model_request(request: Request):
        resolved_request = await resolve_route_request(request)
        messages = resolved_request.json_body.get("messages")
        if messages:
            rendered_prompt = chat_template.render(
                messages=messages,
                add_generation_prompt=True,
                tools=resolved_request.json_body.get("tools"),
            )
            logger.info("prompt:\n{}", rendered_prompt)
        if resolved_request.json_body.get("stream", False):
            return await stream_proxy(
                resolved_request.backend_url,
                request,
                resolved_request.json_body,
                resolved_request.backend_api_key,
                resolved_request.backend_server_url,
            )
        return await non_stream_proxy(
            resolved_request.backend_url,
            request,
            resolved_request.json_body,
            resolved_request.backend_api_key,
            resolved_request.backend_server_url,
        )

    @app.post("/v1/responses", summary="/v1/responses")
    async def responses_router(request: Request):
        return await route_model_request(request)

    @app.get("/v1/responses/{response_id}", summary="/v1/responses/{response_id}")
    @app.post(
        "/v1/responses/{response_id}/cancel",
        summary="/v1/responses/{response_id}/cancel",
    )
    async def response_operation_router(response_id: str, request: Request):
        resolved_request = await resolve_response_request(request, response_id)
        if request.method.upper() == "GET" and parse_stream_parameter(
            request.query_params.get("stream")
        ):
            return await stream_proxy(
                resolved_request.backend_url,
                request,
                None,
                resolved_request.backend_api_key,
            )
        return await non_stream_proxy(
            resolved_request.backend_url,
            request,
            None,
            resolved_request.backend_api_key,
        )

    @app.post("/tokenize", summary="/tokenize")
    @app.post("/detokenize", summary="/detokenize")
    @app.post("/v1/completions", summary="/v1/completions")
    @app.post("/v1/chat/completions", summary="/v1/chat/completions")
    @app.post("/v1/embeddings", summary="/v1/embeddings")
    @app.post("/v1/moderations", summary="/v1/moderations")
    @app.post("/v1/images/generations", summary="/v1/images/generations")
    @app.post("/v1/images/edits", summary="/v1/images/edits")
    @app.post("/v1/images/variations", summary="/v1/images/variations")
    @app.post("/v1/audio/transcriptions", summary="/v1/audio/transcriptions")
    @app.post("/v1/audio/speech", summary="/v1/audio/speech")
    @app.post("/v1/rerank", summary="/v1/rerank")
    async def router(request: Request):
        return await route_model_request(request)

    admin_interface = create_admin_ui()
    return gr.mount_gradio_app(
        app, admin_interface, path="/", css=ADMIN_CSS, ssr_mode=False
    )


app = create_app()
