import gradio as gr
import pandas as pd
from starlette.concurrency import run_in_threadpool

from openai_router.services import route_service


async def get_current_routes() -> list[list[str]]:
    return await run_in_threadpool(route_service.get_admin_routes)


async def add_or_update_route(
    model_name: str,
    model_url: str,
    api_key: str | None,
) -> tuple[str, list[list[str]]]:
    if not model_name or not model_name.strip() or not model_url or not model_url.strip():
        return "模型名称和 URL 不能为空", await get_current_routes()

    status_message = await run_in_threadpool(
        route_service.add_or_update_route,
        model_name,
        model_url,
        api_key,
    )
    return status_message, await get_current_routes()


async def delete_route(model_name: str, model_url: str) -> tuple[str, list[list[str]]]:
    if not model_name or not model_name.strip() or not model_url or not model_url.strip():
        return "要删除的模型名称和 URL 均不能为空", await get_current_routes()

    status_message = await run_in_threadpool(route_service.delete_route, model_name, model_url)
    return status_message, await get_current_routes()


def on_select_route(routes_data: pd.DataFrame, evt: gr.SelectData) -> tuple[str, str, str]:
    if evt.index is None:
        return "", "", ""

    selected_row = routes_data.iloc[evt.index[0]]
    model_name = selected_row.iloc[0]
    model_url = selected_row.iloc[1]
    return model_name, model_url, ""


def create_admin_ui() -> gr.Blocks:
    with gr.Blocks(
        title="模型路由管理器",
        css="footer {display: none !important}",
    ) as admin_ui:
        gr.Markdown("<h1 style='text-align:center;'>模型路由管理器</h1>", elem_id="title")
        gr.Markdown(
            """**将不同端口、不同服务的`openAI`的接口通过统一的url进行路由！兼容 `vLLM`、`SGLang`、`lmdeoply`、`Ollama`等。**\n
**注意：** 所有路由配置都持久化到 `routes.db` 数据库中。您需要手动添加初始路由。"""
        )

        with gr.Row():
            refresh_button = gr.Button("刷新路由列表")

        with gr.Row():
            with gr.Column(scale=2):
                routes_datagrid = gr.DataFrame(
                    headers=[
                        "模型名称 (Model Name)",
                        "后端 URL (Backend URL)",
                        "API 密钥 (API Key)",
                    ],
                    label="当前路由表 (同一模型可有多个URL)",
                    row_count=(1, "fixed"),
                    col_count=(3, "fixed"),
                    interactive=False,
                )
            with gr.Column(scale=1):
                gr.Markdown("### 管理路由")
                status_output = gr.Textbox(
                    label="操作状态",
                    interactive=False,
                    value="这里用于显示上一次的操作状态",
                )
                model_name_input = gr.Textbox(label="模型名称", value="gpt4")
                model_url_input = gr.Textbox(
                    label="后端 URL",
                    value="http://localhost:8082",
                )
                api_key_input = gr.Textbox(
                    label="后端 API 密钥 (可选)",
                    info="如果提供，路由器将使用此密钥覆盖原始请求中的 Authorization 标头。如果留空，将透传原始请求的密钥。",
                    type="password",
                )
                with gr.Row():
                    add_update_button = gr.Button("添加 / 更新")
                    delete_button = gr.Button("删除 (指定URL)", variant="stop")

        admin_ui.load(get_current_routes, outputs=routes_datagrid)
        refresh_button.click(get_current_routes, outputs=routes_datagrid)
        add_update_button.click(
            add_or_update_route,
            inputs=[model_name_input, model_url_input, api_key_input],
            outputs=[status_output, routes_datagrid],
        )
        delete_button.click(
            delete_route,
            inputs=[model_name_input, model_url_input],
            outputs=[status_output, routes_datagrid],
        )
        routes_datagrid.select(
            on_select_route,
            inputs=[routes_datagrid],
            outputs=[model_name_input, model_url_input, api_key_input],
        )

    return admin_ui
