import time
import webbrowser
from typing import Annotated

import typer
import uvicorn
from loguru import logger

from openai_router.app import app


cli_app = typer.Typer()


@cli_app.command()
def main(
    host: Annotated[str, typer.Option(help="指定监听的主机地址", show_default=True)] = "localhost",
    port: Annotated[int, typer.Option(help="指定监听的主机端口", show_default=True)] = 28000,
) -> None:
    base_url = f"http://{host}:{port}"
    logger.info("UI 界面: {}", base_url)
    logger.info("openAI API 文档: {}/docs", base_url)
    time.sleep(1)
    try:
        browser_url = f"http://localhost:{port}" if host == "0.0.0.0" else base_url
        webbrowser.open_new_tab(browser_url)
    except Exception as exc:
        logger.warning("无法自动打开浏览器: {}", exc)

    uvicorn.run(app, host=host, port=port)
