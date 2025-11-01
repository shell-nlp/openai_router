from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse
import httpx
from loguru import logger
import os
from contextlib import asynccontextmanager


backend_servers = {}
# 创建一个可重用的 httpx 客户端
client: httpx.AsyncClient = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    # 启动时的逻辑
    models = os.environ.get("MODELS", "").split(",")
    for model in models:
        model = model.strip()
        if model:
            try:
                model_name, model_url = model.split("=")
                backend_servers[model_name] = model_url
            except ValueError:
                logger.warning(f"Skipping  misformatted model entry: {model}")

    logger.info(f"Backend  servers: {backend_servers}")

    # 初始化 httpx 客户端，设置一个合理的超时
    # read=None 意味着对读取操作不设置超时，这对于流式响应是必要的
    timeout = httpx.Timeout(10.0, connect=60.0, read=None, write=60.0)
    client = httpx.AsyncClient(timeout=timeout)

    yield  # 应用运行期间

    # 关闭时的逻辑
    if client:
        await client.aclose()


# 使用 lifespan 创建 FastAPI 实例
app = FastAPI(lifespan=lifespan)


async def _get_routing_info(request: Request):
    """
    辅助函数：解析请求体以获取模型和目标后端 URL。
    这是路由逻辑所必需的。
    """
    try:
        json_body = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse request body: {e}")
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    model = json_body.get("model")
    if model is None:
        raise HTTPException(
            status_code=400, detail="'model' field is required in request body"
        )

    server = backend_servers.get(model)
    if server is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available models: {list(backend_servers.keys())}",
        )

    backend_url = server + request.url.path
    logger.info(f"Routing to backend_url: {backend_url} for model {model}")

    return backend_url, json_body


async def _stream_proxy(backend_url: str, request: Request, json_body: dict):
    """
    这是一个异步生成器，用于代理流式响应。
    """
    # 准备转发给后端的请求头
    # 移除 'host' 和 'content-length'，因为它们将由 httpx 重新计算
    headers = {
        h: v
        for h, v in request.headers.items()
        if h.lower() not in ["host", "content-length"]
    }

    try:
        # 使用 client.stream 发起请求
        async with client.stream(
            request.method,
            backend_url,
            params=request.query_params,
            json=json_body,  # 我们已经读取了 body，所以作为 json 参数传递
            headers=headers,
        ) as response:

            # 在开始流式传输之前，检查后端的错误响应
            # 我们不能在 StreamingResponse 中途设置状态码，但我们可以选择不流式传输
            if response.status_code >= 400:
                # 如果后端出错，读取错误信息并作为 HTTP 异常抛出
                error_content = await response.aread()
                logger.warning(
                    f"Backend error: {response.status_code} - {error_content.decode()}"
                )
                raise HTTPException(
                    status_code=response.status_code, detail=error_content.decode()
                )

            # 迭代来自后端的流式数据块
            async for chunk in response.aiter_bytes():
                # 将每个数据块 yield 给 FastAPI
                yield chunk

    except httpx.ConnectError as e:
        logger.error(f"Connection error to backend {backend_url}: {e}")
        raise HTTPException(status_code=503, detail="Backend service unavailable")
    except Exception as e:
        logger.error(f"An error occurred during streaming proxy: {e}")
        # 此时可能已经发送了部分响应，所以我们不能再抛出 HTTPException
        # 只能记录错误并停止
        logger.error("Streaming interrupted due to an error.")


async def _non_stream_proxy(backend_url: str, request: Request, json_body: dict):
    """
    处理非流式请求的代理逻辑。
    """
    headers = {
        h: v
        for h, v in request.headers.items()
        if h.lower() not in ["host", "content-length"]
    }

    try:
        response = await client.post(
            backend_url, params=request.query_params, json=json_body, headers=headers
        )

        # 转发后端的响应，包括错误
        return Response(
            content=response.content,
            media_type=response.headers.get("Content-Type"),
            status_code=response.status_code,
        )

    except httpx.ConnectError as e:
        logger.error(f"Connection error to backend {backend_url}: {e}")
        raise HTTPException(status_code=503, detail="Backend service unavailable")
    except httpx.ReadTimeout as e:
        logger.error(f"Read timeout from backend {backend_url}: {e}")
        raise HTTPException(status_code=504, detail="Backend request timed out")
    except Exception as e:
        logger.error(f"An error occurred during non-streaming proxy: {e}")
        raise HTTPException(status_code=500, detail=f"Internal proxy error: {e}")


@app.post("/v1/completions", summary="/v1/completions")
@app.post("/v1/chat/completions", summary="/v1/chat/completions")
async def post_completions(request: Request):
    backend_url, json_body = await _get_routing_info(request)

    if json_body.get("stream", False):
        logger.info("Handling as STREAMING request")
        return StreamingResponse(
            _stream_proxy(backend_url, request, json_body),
            media_type="text/event-stream",
        )
    else:
        logger.info("Handling as NON-STREAMING request")
        return await _non_stream_proxy(backend_url, request, json_body)


if __name__ == "__main__":
    import uvicorn

    # 你需要通过环境变量来设置模型，例如：
    # MODELS="gpt-4=http://localhost:8080,llama=http://localhost:8081" uvicorn streaming_proxy:app --host 0.0.0.0 --port 8000
    os.environ["MODELS"] = "qwen3=https://miyun.archermind.com"
    if not os.environ.get("MODELS"):
        logger.warning(
            "MODELS environment variable is not set. Example: MODELS='model_name=http://backend_url'"
        )
    uvicorn.run(app, host="0.0.0.0", port=8000)
