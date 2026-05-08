import asyncio
from contextlib import asynccontextmanager
import warnings

import gradio as gr
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from loguru import logger
from starlette.concurrency import run_in_threadpool

from openai_router.admin import ADMIN_CSS, create_admin_ui
from openai_router.config import MODEL_SYNC_CHECK_INTERVAL_SECONDS
from openai_router.db import create_db_and_tables, dispose_engine, initialize_engine
from openai_router.proxy import non_stream_proxy, resolve_route_request, stream_proxy
from openai_router.runtime import runtime_state
from openai_router.services import route_service

warnings.filterwarnings(
    "ignore",
    message="SSR mode is not supported with multi-page apps when mounting on a FastAPI app. Disabling SSR mode.",
    category=UserWarning,
)


async def periodic_model_sync() -> None:
    while True:
        try:
            synced_count = await run_in_threadpool(route_service.sync_due_backend_sources)
            if synced_count:
                logger.info("Periodic model sync processed {} backend source(s).", synced_count)
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

    dispose_engine()


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)

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

    @app.post("/v1/responses", summary="/v1/responses ")
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
        resolved_request = await resolve_route_request(request)
        if resolved_request.json_body.get("stream", False):
            return StreamingResponse(
                stream_proxy(
                    resolved_request.backend_url,
                    request,
                    resolved_request.json_body,
                    resolved_request.backend_api_key,
                ),
                media_type="text/event-stream",
            )
        return await non_stream_proxy(
            resolved_request.backend_url,
            request,
            resolved_request.json_body,
            resolved_request.backend_api_key,
        )

    admin_interface = create_admin_ui()
    return gr.mount_gradio_app(app, admin_interface, path="/", css=ADMIN_CSS, ssr_mode=False)


app = create_app()
