import html
import json
from typing import Any

import gradio as gr
import pandas as pd
from starlette.concurrency import run_in_threadpool

from openai_router.log_store import DEFAULT_LOG_VIEW_LIMIT, LOG_LEVELS, log_store
from openai_router.runtime import runtime_state
from openai_router.services import route_service

ADMIN_CSS = "footer {display: none !important}"

ROUTE_MAPPING_COLUMN_INDEX = 4
ROUTE_MODE_COLUMN_INDEX = 5
BACKEND_SOURCE_SYNC_INTERVAL_COLUMN_INDEX = 2
BACKEND_SOURCE_EXCLUSIONS_COLUMN_INDEX = 5
REQUEST_PARAM_MAPPING_HEADERS = ["原参数路径 (key)", "目标参数路径 (value)"]


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


async def get_recent_logs_page_data(levels: list[str] | None = None) -> tuple[str, str]:
    total_logs, filtered_logs, lines = await run_in_threadpool(
        log_store.snapshot,
        DEFAULT_LOG_VIEW_LIMIT,
        levels,
    )
    safe_logs = html.escape("\n".join(lines))
    rendered_logs = (
        "<div id='log-scrollbox' style='max-height: 70vh; overflow-y: auto; "
        "background: #f8fafc; color: #0f172a; border: 1px solid #cbd5e1; "
        "border-radius: 8px; padding: 12px;'>"
        "<pre style='margin: 0; font-family: monospace; font-size: 13px; "
        "line-height: 1.5; white-space: pre-wrap; word-break: break-word;'>"
        f"{safe_logs}</pre>"
        "</div>"
    )
    if levels:
        status = f"当前缓存 {total_logs} 条日志，等级筛选后 {filtered_logs} 条，显示最近 {len(lines)} 条"
    else:
        status = f"当前缓存 {total_logs} 条日志，显示最近 {len(lines)} 条"
    return status, rendered_logs


def _normalize_request_param_mapping_rows(rows: Any) -> list[list[str]]:
    return _coerce_request_param_mapping_rows(rows, include_empty=False)


def _coerce_request_param_mapping_rows(
    rows: Any,
    *,
    include_empty: bool,
) -> list[list[str]]:
    if rows is None:
        return []

    if isinstance(rows, pd.DataFrame):
        values = rows.values.tolist()
    else:
        values = rows

    if not isinstance(values, list):
        return []

    normalized_rows: list[list[str]] = []
    for row in values:
        if not isinstance(row, (list, tuple)):
            continue
        source = ""
        target = ""
        if len(row) > 0 and not pd.isna(row[0]):
            source = str(row[0]).strip()
        if len(row) > 1 and not pd.isna(row[1]):
            target = str(row[1]).strip()
        if include_empty or source or target:
            normalized_rows.append([source, target])

    return normalized_rows


def serialize_request_param_mapping_rows(rows: Any) -> str:
    normalized_rows = _normalize_request_param_mapping_rows(rows)
    if not normalized_rows:
        return ""

    mapping: dict[str, str] = {}
    for source, target in normalized_rows:
        if not source or not target:
            raise ValueError("请求参数映射的每一行都必须同时填写 key 和 value。")
        if source in mapping:
            raise ValueError(f"请求参数映射存在重复的 key: '{source}'")
        mapping[source] = target

    return json.dumps(mapping, ensure_ascii=False, separators=(",", ":"))


def deserialize_request_param_mapping_text(request_param_mapping_text: str | None) -> list[list[str]]:
    if request_param_mapping_text is None:
        return [["", ""]]

    normalized_text = request_param_mapping_text.strip()
    if not normalized_text:
        return [["", ""]]

    raw_mapping = json.loads(normalized_text)
    if not isinstance(raw_mapping, dict):
        raise ValueError("请求参数映射必须是 JSON 对象。")

    rows = [[str(source), str(target)] for source, target in raw_mapping.items()]
    return rows or [["", ""]]


def add_request_param_mapping_row(rows: Any) -> list[list[str]]:
    visible_rows = _coerce_request_param_mapping_rows(rows, include_empty=True)
    if not visible_rows:
        visible_rows = [["", ""]]
    return [*visible_rows, ["", ""]]


def remove_request_param_mapping_row(rows: Any) -> list[list[str]]:
    visible_rows = _coerce_request_param_mapping_rows(rows, include_empty=True)
    if len(visible_rows) <= 1:
        return [["", ""]]
    return visible_rows[:-1]


async def add_or_update_route(
    model_name: str,
    aliases_text: str | None,
    model_url: str,
    api_key: str | None,
    request_param_mapping_rows: Any,
) -> tuple[str, list[list[str]], list[list[str]]]:
    if not model_name or not model_name.strip():
        routes, sources = await refresh_admin_tables()
        return "模型名称不能为空", routes, sources
    if not model_url or not model_url.strip():
        routes, sources = await refresh_admin_tables()
        return "后端 URL 不能为空", routes, sources

    try:
        request_param_mapping_text = serialize_request_param_mapping_rows(
            request_param_mapping_rows
        )
        status_message = await run_in_threadpool(
            route_service.add_or_update_route,
            model_name,
            aliases_text or "",
            model_url,
            api_key,
            request_param_mapping_text,
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


def on_select_route(
    routes_data: pd.DataFrame,
    evt: gr.SelectData,
) -> tuple[str, str, str, str, list[list[str]]]:
    if evt.index is None:
        return "", "", "", "", [["", ""]]

    selected_row = routes_data.iloc[evt.index[0]]
    model_name = str(selected_row.iloc[0])
    aliases_text = "" if pd.isna(selected_row.iloc[1]) else str(selected_row.iloc[1])
    model_url = str(selected_row.iloc[2])
    request_param_mapping_text = (
        ""
        if pd.isna(selected_row.iloc[ROUTE_MAPPING_COLUMN_INDEX])
        else str(selected_row.iloc[ROUTE_MAPPING_COLUMN_INDEX])
    )

    try:
        request_param_mapping_rows = deserialize_request_param_mapping_text(
            request_param_mapping_text
        )
    except ValueError:
        request_param_mapping_rows = [["", ""]]

    return model_name, aliases_text, model_url, "", request_param_mapping_rows


def on_select_backend_source(
    sources_data: pd.DataFrame,
    evt: gr.SelectData,
) -> tuple[str, str, str, float]:
    if evt.index is None:
        return "", "", "", 15

    selected_row = sources_data.iloc[evt.index[0]]
    model_url = str(selected_row.iloc[0])
    excluded_models = (
        ""
        if pd.isna(selected_row.iloc[BACKEND_SOURCE_EXCLUSIONS_COLUMN_INDEX])
        else str(selected_row.iloc[BACKEND_SOURCE_EXCLUSIONS_COLUMN_INDEX])
    )
    sync_interval = float(selected_row.iloc[BACKEND_SOURCE_SYNC_INTERVAL_COLUMN_INDEX])
    return model_url, "", excluded_models, sync_interval


async def get_overview_data() -> tuple[str, str, str, str]:
    routes = await get_current_routes()
    sources = await get_current_backend_sources()
    routing_policy = await get_current_routing_policy()

    manual_routes = sum(
        1
        for route in routes
        if len(route) > ROUTE_MODE_COLUMN_INDEX and route[ROUTE_MODE_COLUMN_INDEX] == "手动配置"
    )
    auto_routes = sum(
        1
        for route in routes
        if len(route) > ROUTE_MODE_COLUMN_INDEX and route[ROUTE_MODE_COLUMN_INDEX] == "自动同步"
    )
    source_errors = sum(
        1 for source in sources if len(source) > 4 and str(source[4]).strip() not in {"", "-"}
    )

    route_summary = f"共 {len(routes)} 条，手动 {manual_routes} 条，自动 {auto_routes} 条"
    source_summary = f"共 {len(sources)} 个，异常 {source_errors} 个"
    return get_router_base_url(), route_summary, source_summary, routing_policy


async def add_or_update_route_page(
    model_name: str,
    aliases_text: str | None,
    model_url: str,
    api_key: str | None,
    request_param_mapping_rows: Any,
) -> tuple[str, list[list[str]]]:
    status_message, routes, _ = await add_or_update_route(
        model_name,
        aliases_text,
        model_url,
        api_key,
        request_param_mapping_rows,
    )
    return status_message, routes


async def delete_route_page(
    model_name: str,
    model_url: str,
) -> tuple[str, list[list[str]]]:
    status_message, routes, _ = await delete_route(model_name, model_url)
    return status_message, routes


async def add_or_update_backend_source_page(
    model_url: str,
    api_key: str | None,
    excluded_models_text: str | None,
    sync_interval_minutes: float | None,
) -> tuple[str, list[list[str]]]:
    status_message, _, sources = await add_or_update_backend_source(
        model_url,
        api_key,
        excluded_models_text,
        sync_interval_minutes,
    )
    return status_message, sources


async def sync_backend_source_page(
    model_url: str,
) -> tuple[str, list[list[str]]]:
    status_message, _, sources = await sync_backend_source(model_url)
    return status_message, sources


async def delete_backend_source_page(
    model_url: str,
) -> tuple[str, list[list[str]]]:
    status_message, _, sources = await delete_backend_source(model_url)
    return status_message, sources


async def get_backend_config_page_data() -> tuple[list[list[str]], str]:
    return await get_current_backend_sources(), await get_current_routing_policy()


def _render_page_header(
    title: str,
    description: str,
) -> gr.Textbox:
    gr.Markdown(f"## {title}")
    base_url_output = gr.Textbox(
        label="当前路由后的 Base URL",
        value="服务启动后可用",
        interactive=False,
    )
    gr.Markdown(description)
    return base_url_output


def _render_top_right_button(label: str, link: str) -> None:
    with gr.Row():
        with gr.Column(scale=10):
            gr.Markdown("")
        with gr.Column(scale=1, min_width=140):
            gr.Button(label, link=link, variant="secondary", size="sm")


LOG_VIEWER_JS = """
const scrollToBottom = () => {
  const box = element.querySelector('#log-scrollbox');
  if (box) {
    box.scrollTop = box.scrollHeight;
  }
};

const attachAutoScroll = () => {
  if (element.__logAutoScrollObserver) {
    element.__logAutoScrollObserver.disconnect();
  }

  element.__logAutoScrollObserver = new MutationObserver(() => {
    requestAnimationFrame(() => requestAnimationFrame(scrollToBottom));
  });

  element.__logAutoScrollObserver.observe(element, {
    childList: true,
    subtree: true,
    characterData: true,
  });

  scrollToBottom();
};

watch('value', () => {
  requestAnimationFrame(() => requestAnimationFrame(attachAutoScroll));
});

attachAutoScroll();
"""


def create_admin_ui() -> gr.Blocks:
    with gr.Blocks(title="模型路由管理器") as admin_ui:
        gr.Navbar(main_page_name="模型路由")
        _render_top_right_button("查看日志", "/logs")
        gr.Markdown("<h1 style='text-align:center;'>模型路由管理器</h1>", elem_id="title")
        overview_base_url_output = _render_page_header(
            "模型路由",
            (
                "将不同端口、不同服务的 OpenAI 兼容接口通过统一 URL 进行路由。"
                "支持模型别名、后端 API Key 覆盖，以及模型级请求参数映射，"
                "用于适配不同后端版本的请求体差异。"
            ),
        )
        with gr.Row():
            route_summary_output = gr.Textbox(label="模型路由概况", interactive=False)
            source_summary_output = gr.Textbox(label="后端配置概况", interactive=False)
            routing_policy_summary_output = gr.Textbox(label="当前路由策略", interactive=False)
        refresh_overview_button = gr.Button("刷新概览")

        routes_status_output = gr.Textbox(
            label="操作状态",
            interactive=False,
            value="这里用于显示上一次的路由操作状态",
        )

        with gr.Row():
            with gr.Column(scale=2):
                routes_datagrid = gr.DataFrame(
                    headers=[
                        "模型名称",
                        "模型别名",
                        "后端 URL",
                        "API 密钥",
                        "请求参数映射 (JSON)",
                        "管理方式",
                        "同步间隔 (Min)",
                        "最后同步 (UTC)",
                    ],
                    label="模型路由",
                    row_count=1,
                    column_count=8,
                    interactive=False,
                )
            with gr.Column(scale=1):
                model_name_input = gr.Textbox(label="模型名称", value="gpt-4")
                aliases_input = gr.Textbox(
                    label="模型别名",
                    value="",
                    info="可选。多个别名请用英文逗号分隔。",
                )
                model_url_input = gr.Textbox(label="后端 URL", value="http://localhost:8082")
                route_api_key_input = gr.Textbox(
                    label="后端 API 密钥 (可选)",
                    info="如果填写，将覆盖原始请求中的 Authorization 请求头；留空则透传原请求。",
                    type="password",
                )
                request_param_mapping_input = gr.DataFrame(
                    headers=REQUEST_PARAM_MAPPING_HEADERS,
                    value=[["", ""]],
                    row_count=(1, "dynamic"),
                    column_count=2,
                    interactive=True,
                    label="请求参数映射",
                )
                gr.Markdown(
                    "每一行表示一条映射规则：左侧填原参数路径，右侧填目标参数路径。"
                    "例如把 `enable_thinking` 移动到 `chat_template_kwargs.enable_thinking`。"
                )
                with gr.Row():
                    add_mapping_row_button = gr.Button("添加映射行")
                    remove_mapping_row_button = gr.Button("删除映射行")
                with gr.Row():
                    add_update_button = gr.Button("添加 / 更新路由")
                    delete_button = gr.Button("删除路由", variant="stop")

        admin_ui.load(
            get_overview_data,
            outputs=[
                overview_base_url_output,
                route_summary_output,
                source_summary_output,
                routing_policy_summary_output,
            ],
        )
        admin_ui.load(get_current_routes, outputs=routes_datagrid)
        refresh_overview_button.click(
            get_overview_data,
            outputs=[
                overview_base_url_output,
                route_summary_output,
                source_summary_output,
                routing_policy_summary_output,
            ],
        )
        refresh_overview_button.click(get_current_routes, outputs=routes_datagrid)
        add_mapping_row_button.click(
            add_request_param_mapping_row,
            inputs=[request_param_mapping_input],
            outputs=[request_param_mapping_input],
        )
        remove_mapping_row_button.click(
            remove_request_param_mapping_row,
            inputs=[request_param_mapping_input],
            outputs=[request_param_mapping_input],
        )
        add_update_button.click(
            add_or_update_route_page,
            inputs=[
                model_name_input,
                aliases_input,
                model_url_input,
                route_api_key_input,
                request_param_mapping_input,
            ],
            outputs=[routes_status_output, routes_datagrid],
        )
        delete_button.click(
            delete_route_page,
            inputs=[model_name_input, model_url_input],
            outputs=[routes_status_output, routes_datagrid],
        )
        routes_datagrid.select(
            on_select_route,
            inputs=[routes_datagrid],
            outputs=[
                model_name_input,
                aliases_input,
                model_url_input,
                route_api_key_input,
                request_param_mapping_input,
            ],
        )

    with admin_ui.route("后端配置", "/sources") as sources_page:
        gr.Navbar(main_page_name="模型路由")
        _render_top_right_button("查看日志", "/logs")
        gr.Markdown("## 后端配置")
        gr.Markdown(
            "当前页面用于管理全局路由策略和自动同步后端源。"
            "保存后会立即拉取一次 `/v1/models`，之后按设定间隔自动同步。"
        )
        sources_status_output = gr.Textbox(
            label="操作状态",
            interactive=False,
            value="这里用于显示上一次的后端配置操作状态",
        )
        gr.Markdown("### 路由策略")
        routing_policy_input = gr.Dropdown(
            label="路由策略",
            choices=["round_robin", "consistent_hash"],
            value="round_robin",
            info=(
                "consistent_hash 会优先使用 X-Session-ID / X-User-ID 等请求头，"
                "其次回退到请求体字段。"
            ),
        )
        save_policy_button = gr.Button("保存路由策略")

        with gr.Row():
            with gr.Column(scale=2):
                backend_sources_datagrid = gr.DataFrame(
                    headers=[
                        "后端源 URL",
                        "API 密钥",
                        "同步间隔 (Min)",
                        "最后同步 (UTC)",
                        "最后错误",
                        "排除模型",
                    ],
                    label="后端配置",
                    row_count=1,
                    column_count=6,
                    interactive=False,
                )
            with gr.Column(scale=1):
                gr.Markdown("### 自动同步后端源")
                source_url_input = gr.Textbox(label="后端源 URL", value="http://localhost:8082")
                source_api_key_input = gr.Textbox(
                    label="后端源 API 密钥 (可选)",
                    type="password",
                )
                source_excluded_models_input = gr.Textbox(
                    label="排除模型",
                    value="",
                    info=(
                        "可选。多个模型请用英文逗号分隔；这些模型即使出现在 `/v1/models` 中，"
                        "也不会被自动导入。"
                    ),
                )
                sync_interval_minutes_input = gr.Number(
                    label="自动同步间隔（分钟）",
                    value=15,
                    minimum=1,
                    precision=0,
                )
                gr.Markdown(
                    "删除后端源时，会一并清理该后端源自动生成的路由；手动路由不会受影响。"
                )
                with gr.Row():
                    add_update_source_button = gr.Button("添加 / 更新后端配置")
                    sync_source_button = gr.Button("立即同步")
                    delete_source_button = gr.Button("删除后端配置", variant="stop")

        sources_page.load(
            get_backend_config_page_data,
            outputs=[backend_sources_datagrid, routing_policy_input],
        )
        add_update_source_button.click(
            add_or_update_backend_source_page,
            inputs=[
                source_url_input,
                source_api_key_input,
                source_excluded_models_input,
                sync_interval_minutes_input,
            ],
            outputs=[sources_status_output, backend_sources_datagrid],
        )
        sync_source_button.click(
            sync_backend_source_page,
            inputs=[source_url_input],
            outputs=[sources_status_output, backend_sources_datagrid],
        )
        delete_source_button.click(
            delete_backend_source_page,
            inputs=[source_url_input],
            outputs=[sources_status_output, backend_sources_datagrid],
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
        save_policy_button.click(
            update_routing_policy,
            inputs=[routing_policy_input],
            outputs=[sources_status_output],
        )

    with admin_ui.route("日志", "/logs") as logs_page:
        gr.Navbar(main_page_name="模型路由")
        _render_top_right_button("返回模型路由", "/")
        gr.Markdown("## 实时日志")
        gr.Markdown("页面每秒自动刷新一次，展示当前进程中的最近日志。")
        log_level_filter = gr.CheckboxGroup(
            choices=list(LOG_LEVELS),
            value=[level for level in LOG_LEVELS if level != "DEBUG"],
            label="日志等级",
            info="支持多选；全部取消时默认显示所有等级。",
        )
        logs_status_output = gr.Textbox(label="日志状态", interactive=False)
        logs_output = gr.HTML(
            label="最近日志",
            autoscroll=False,
            js_on_load=LOG_VIEWER_JS,
            elem_id="live-log-output",
        )
        refresh_logs_button = gr.Button("立即刷新")
        logs_timer = gr.Timer(1)

        logs_page.load(
            get_recent_logs_page_data,
            inputs=[log_level_filter],
            outputs=[logs_status_output, logs_output],
        )
        logs_timer.tick(
            get_recent_logs_page_data,
            inputs=[log_level_filter],
            outputs=[logs_status_output, logs_output],
        )
        refresh_logs_button.click(
            get_recent_logs_page_data,
            inputs=[log_level_filter],
            outputs=[logs_status_output, logs_output],
        )
        log_level_filter.change(
            get_recent_logs_page_data,
            inputs=[log_level_filter],
            outputs=[logs_status_output, logs_output],
        )

    return admin_ui
