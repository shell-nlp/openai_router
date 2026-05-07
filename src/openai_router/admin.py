import gradio as gr
import pandas as pd
from starlette.concurrency import run_in_threadpool

from openai_router.runtime import runtime_state
from openai_router.services import route_service


async def get_current_routes() -> list[list[str]]:
    return await run_in_threadpool(route_service.get_admin_routes)


async def get_current_backend_sources() -> list[list[str]]:
    return await run_in_threadpool(route_service.get_admin_backend_sources)


async def get_current_routing_policy() -> str:
    return await run_in_threadpool(route_service.get_routing_policy)


def get_router_base_url() -> str:
    return runtime_state.public_base_url or "服务启动后可用"


async def refresh_admin_tables() -> tuple[list[list[str]], list[list[str]]]:
    return await get_current_routes(), await get_current_backend_sources()


async def add_or_update_route(
    model_name: str,
    aliases_text: str | None,
    model_url: str,
    api_key: str | None,
) -> tuple[str, list[list[str]], list[list[str]]]:
    if not model_name or not model_name.strip():
        routes, sources = await refresh_admin_tables()
        return "模型名称不能为空", routes, sources
    if not model_url or not model_url.strip():
        routes, sources = await refresh_admin_tables()
        return "后端 URL 不能为空", routes, sources

    try:
        status_message = await run_in_threadpool(
            route_service.add_or_update_route,
            model_name,
            aliases_text or "",
            model_url,
            api_key,
        )
    except ValueError as exc:
        routes, sources = await refresh_admin_tables()
        return str(exc), routes, sources

    routes, sources = await refresh_admin_tables()
    return status_message, routes, sources


async def delete_route(
    model_name: str,
    model_url: str,
) -> tuple[str, list[list[str]], list[list[str]]]:
    if not model_name or not model_name.strip() or not model_url or not model_url.strip():
        routes, sources = await refresh_admin_tables()
        return "要删除的模型名称和 URL 均不能为空", routes, sources

    status_message = await run_in_threadpool(route_service.delete_route, model_name, model_url)
    routes, sources = await refresh_admin_tables()
    return status_message, routes, sources


async def add_or_update_backend_source(
    model_url: str,
    api_key: str | None,
    excluded_models_text: str | None,
    sync_interval_minutes: float | None,
) -> tuple[str, list[list[str]], list[list[str]]]:
    if not model_url or not model_url.strip():
        routes, sources = await refresh_admin_tables()
        return "后端源 URL 不能为空", routes, sources

    normalized_sync_interval = int(sync_interval_minutes or 15)
    if normalized_sync_interval < 1:
        routes, sources = await refresh_admin_tables()
        return "自动同步间隔必须大于等于 1 分钟", routes, sources

    try:
        status_message = await run_in_threadpool(
            route_service.add_or_update_backend_source,
            model_url,
            api_key,
            excluded_models_text or "",
            normalized_sync_interval,
        )
    except ValueError as exc:
        routes, sources = await refresh_admin_tables()
        return str(exc), routes, sources

    routes, sources = await refresh_admin_tables()
    return status_message, routes, sources


async def sync_backend_source(
    model_url: str,
) -> tuple[str, list[list[str]], list[list[str]]]:
    if not model_url or not model_url.strip():
        routes, sources = await refresh_admin_tables()
        return "要同步的后端源 URL 不能为空", routes, sources

    status_message = await run_in_threadpool(route_service.sync_backend_source_by_url, model_url)
    routes, sources = await refresh_admin_tables()
    return status_message, routes, sources


async def delete_backend_source(
    model_url: str,
) -> tuple[str, list[list[str]], list[list[str]]]:
    if not model_url or not model_url.strip():
        routes, sources = await refresh_admin_tables()
        return "要删除的后端源 URL 不能为空", routes, sources

    status_message = await run_in_threadpool(route_service.delete_backend_source, model_url)
    routes, sources = await refresh_admin_tables()
    return status_message, routes, sources


async def update_routing_policy(routing_policy: str) -> str:
    try:
        return await run_in_threadpool(route_service.update_routing_policy, routing_policy)
    except ValueError as exc:
        return str(exc)


def on_select_route(routes_data: pd.DataFrame, evt: gr.SelectData) -> tuple[str, str, str, str]:
    if evt.index is None:
        return "", "", "", ""

    selected_row = routes_data.iloc[evt.index[0]]
    model_name = selected_row.iloc[0]
    aliases_text = selected_row.iloc[1]
    if pd.isna(aliases_text):
        aliases_text = ""
    model_url = selected_row.iloc[2]
    return model_name, aliases_text, model_url, ""


def on_select_backend_source(
    sources_data: pd.DataFrame,
    evt: gr.SelectData,
) -> tuple[str, str, str, float]:
    if evt.index is None:
        return "", "", "", 15

    selected_row = sources_data.iloc[evt.index[0]]
    model_url = selected_row.iloc[0]
    excluded_models = selected_row.iloc[5]
    if pd.isna(excluded_models):
        excluded_models = ""
    sync_interval = float(selected_row.iloc[2])
    return model_url, "", excluded_models, sync_interval


def create_admin_ui() -> gr.Blocks:
    with gr.Blocks(
        title="模型路由管理器",
        css="footer {display: none !important}",
    ) as admin_ui:
        gr.Markdown("<h1 style='text-align:center;'>模型路由管理器</h1>", elem_id="title")
        router_base_url_output = gr.Textbox(
            label="当前路由后的 Base URL",
            value="服务启动后可用",
            interactive=False,
        )
        gr.Markdown(
            """**将不同端口、不同服务的`openAI`接口通过统一的 URL 进行路由！兼容 `vLLM`、`SGLang`、`lmdeploy`、`Ollama` 等。**\n
**注意：** 手动模型路由与“自动从 `/v1/models` 导入”的后端源已拆分为两个独立管理区。"""
        )

        with gr.Row():
            refresh_button = gr.Button("刷新全部")

        status_output = gr.Textbox(
            label="操作状态",
            interactive=False,
            value="这里用于显示上一次的操作状态",
        )

        with gr.Row():
            with gr.Column(scale=2):
                routes_datagrid = gr.DataFrame(
                    headers=[
                        "模型名称 (Model Name)",
                        "模型别名 (Aliases)",
                        "后端 URL (Backend URL)",
                        "API 密钥 (API Key)",
                        "管理方式 (Mode)",
                        "同步间隔 (Min)",
                        "最后同步 (UTC)",
                    ],
                    label="模型路由",
                    row_count=(1, "fixed"),
                    col_count=(7, "fixed"),
                    interactive=False,
                )
            with gr.Column(scale=1):
                gr.Markdown("### 路由策略")
                routing_policy_input = gr.Dropdown(
                    label="路由策略",
                    choices=["round_robin", "consistent_hash"],
                    value="round_robin",
                    info=(
                        "consistent_hash 会优先使用 X-Session-ID / X-User-ID 等请求头，"
                        "其次使用 session_params.session_id、user、session_id、user_id，"
                        "将同一会话稳定路由到同一后端。"
                    ),
                )
                save_policy_button = gr.Button("保存路由策略")
                gr.Markdown("### 管理手动路由")
                model_name_input = gr.Textbox(label="模型名称", value="gpt4")
                aliases_input = gr.Textbox(
                    label="模型别名",
                    value="",
                    info="可选。多个别名请用英文逗号分隔，例如：gpt-4o-latest,my-gpt4o。",
                )
                model_url_input = gr.Textbox(
                    label="后端 URL",
                    value="http://localhost:8082",
                )
                route_api_key_input = gr.Textbox(
                    label="后端 API 密钥 (可选)",
                    info="如果提供，路由器将使用此密钥覆盖原始请求中的 Authorization 标头。如果留空，将透传原始请求的密钥。",
                    type="password",
                )
                gr.Markdown("当前区域仅管理手动路由；自动导入后端源请在下方单独维护。")
                with gr.Row():
                    add_update_button = gr.Button("添加 / 更新路由")
                    delete_button = gr.Button("删除路由", variant="stop")

        with gr.Row():
            with gr.Column(scale=2):
                backend_sources_datagrid = gr.DataFrame(
                    headers=[
                        "后端源 URL (Backend Source URL)",
                        "API 密钥 (API Key)",
                        "同步间隔 (Min)",
                        "最后同步 (UTC)",
                        "最后错误 (Last Error)",
                        "排除模型 (Excluded Models)",
                    ],
                    label="自动同步后端源",
                    row_count=(1, "fixed"),
                    col_count=(6, "fixed"),
                    interactive=False,
                )
            with gr.Column(scale=1):
                gr.Markdown("### 管理自动同步后端源")
                source_url_input = gr.Textbox(
                    label="后端源 URL",
                    value="http://localhost:8082",
                )
                source_api_key_input = gr.Textbox(
                    label="后端源 API 密钥 (可选)",
                    type="password",
                )
                source_excluded_models_input = gr.Textbox(
                    label="排除模型",
                    value="",
                    info="可选。多个模型请用逗号分隔；这些模型即使出现在 `/v1/models` 中，也不会被自动导入。",
                )
                sync_interval_minutes_input = gr.Number(
                    label="自动同步间隔（分钟）",
                    value=15,
                    minimum=1,
                    precision=0,
                    info="保存后会立即拉取一次 `/v1/models`，之后按设定间隔自动同步。",
                )
                gr.Markdown("删除后端源时，会一并清理该后端源自动生成的路由；手动路由不会受影响。")
                with gr.Row():
                    add_update_source_button = gr.Button("添加 / 更新后端源")
                    sync_source_button = gr.Button("立即同步")
                    delete_source_button = gr.Button("删除后端源", variant="stop")

        admin_ui.load(get_router_base_url, outputs=router_base_url_output)
        admin_ui.load(get_current_routing_policy, outputs=routing_policy_input)
        admin_ui.load(refresh_admin_tables, outputs=[routes_datagrid, backend_sources_datagrid])
        refresh_button.click(
            refresh_admin_tables,
            outputs=[routes_datagrid, backend_sources_datagrid],
        )
        save_policy_button.click(
            update_routing_policy,
            inputs=[routing_policy_input],
            outputs=[status_output],
        )
        add_update_button.click(
            add_or_update_route,
            inputs=[
                model_name_input,
                aliases_input,
                model_url_input,
                route_api_key_input,
            ],
            outputs=[status_output, routes_datagrid, backend_sources_datagrid],
        )
        delete_button.click(
            delete_route,
            inputs=[model_name_input, model_url_input],
            outputs=[status_output, routes_datagrid, backend_sources_datagrid],
        )
        add_update_source_button.click(
            add_or_update_backend_source,
            inputs=[
                source_url_input,
                source_api_key_input,
                source_excluded_models_input,
                sync_interval_minutes_input,
            ],
            outputs=[status_output, routes_datagrid, backend_sources_datagrid],
        )
        sync_source_button.click(
            sync_backend_source,
            inputs=[source_url_input],
            outputs=[status_output, routes_datagrid, backend_sources_datagrid],
        )
        delete_source_button.click(
            delete_backend_source,
            inputs=[source_url_input],
            outputs=[status_output, routes_datagrid, backend_sources_datagrid],
        )
        routes_datagrid.select(
            on_select_route,
            inputs=[routes_datagrid],
            outputs=[model_name_input, aliases_input, model_url_input, route_api_key_input],
        )
        backend_sources_datagrid.select(
            on_select_backend_source,
            inputs=[backend_sources_datagrid],
            outputs=[
                source_url_input,
                source_api_key_input,
                source_excluded_models_input,
                sync_interval_minutes_input,
            ],
        )

    return admin_ui
