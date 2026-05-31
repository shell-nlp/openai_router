import sys
import time
import webbrowser
import socket
from typing import Annotated

import typer
import uvicorn
from loguru import logger

from openai_router.app import app
from openai_router.log_store import loguru_sink
from openai_router.runtime import runtime_state


cli_app = typer.Typer()


def _configure_logging() -> None:
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(
        loguru_sink,
        level="INFO",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}",
    )


@cli_app.command()
def main(
    host: Annotated[str, typer.Option(help="指定监听的主机地址", show_default=True)] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="指定监听的主机端口", show_default=True)] = 28000,
) -> None:
    _configure_logging()
    display_host = _get_display_host(host)
    base_url = f"http://{host}:{port}"
    display_base_url = f"http://{display_host}:{port}"
    public_base_url = f"{display_base_url}/v1"
    runtime_state.public_base_url = public_base_url
    logger.info("监听地址: {}", base_url)
    logger.info("UI 界面: {}", display_base_url)
    logger.info("路由 Base URL: {}", public_base_url)
    logger.info("openAI API 文档: {}/docs", display_base_url)
    time.sleep(1)
    try:
        webbrowser.open_new_tab(display_base_url)
    except Exception as exc:
        logger.warning("无法自动打开浏览器: {}", exc)

    uvicorn.run(app, host=host, port=port)


def _get_display_host(host: str) -> str:
    if host != "0.0.0.0":
        return host

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip_address = sock.getsockname()[0]
            if ip_address:
                return ip_address
    except OSError:
        pass

    return "localhost"
