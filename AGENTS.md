# AGENTS.md

## 项目结构

- **应用代码**：`src/openai_router/`
  - `app.py`：FastAPI 应用入口，注册 OpenAI 兼容接口，挂载 Gradio 管理页，启动周期性后端同步任务。
  - `cli.py`：`openai-router` 命令行入口，负责启动 Uvicorn，并输出 UI / `/v1` 地址。
  - `admin.py`：Gradio 管理界面，包含“模型路由”和“后端配置”两页。
  - `proxy.py`：请求解析、目标路由选择、请求头处理、流式/非流式转发。
  - `services.py`：核心业务逻辑，负责路由缓存、别名、负载均衡、后端模型发现、自动同步。
  - `repositories.py`：SQLModel 数据访问层。
  - `models.py`：数据库模型定义。
  - `db.py`：SQLite engine 初始化、表创建、schema 变更检测与重建。
  - `chat_logging.py`：聊天请求/响应日志提取与流式日志拼装。
  - `chat_template.jinja`：`messages` 渲染模板，已通过 `package-data` 纳入发布产物。
- **测试**：`tests/`
  - `test_services.py`：路由策略、别名、同步逻辑。
  - `test_repositories.py`：后端源、自动路由、排除模型清理语义。
  - `test_proxy.py`：请求转发、错误处理、模型别名改写。
  - `test_chat_template.py`：Jinja 模板和参数解析。
- **静态资源**：`static/`，README 使用的截图和架构图。
- **数据目录**：`data/`，默认 SQLite 文件位于 `data/routes.db`。
- **打包配置**：`pyproject.toml`，使用 setuptools + wheel，源码布局为 `src/`。
- **容器相关**：`Dockerfile`、`docker-compose.yml`。

## 常用命令

```bash
# 安装依赖
uv sync

# 本地启动
uv run openai-router --host 0.0.0.0 --port 28000

# 可编辑安装
pip install -e .
openai-router --host 0.0.0.0 --port 28000

# 运行测试
uv run python -m unittest discover -s tests

# 构建发布产物
uv build

# Docker 启动
docker compose up --build -d
```

## 入口与页面

`src/openai_router/main.py` 是 Python 包导出的统一入口：

- `app`：供 ASGI / Uvicorn 使用。
- `cli_app` / `main`：供 `openai-router` 命令调用。

`src/openai_router/app.py` 创建并暴露唯一的 FastAPI 应用，主要职责：

- 初始化 SQLite engine 和表。
- 初始化全局 `httpx.AsyncClient`。
- 启动周期性后端模型同步任务。
- 挂载 Gradio 管理界面到 `/`。
- 暴露 OpenAI 兼容接口与健康检查接口。

管理界面页面：

- `/`：模型路由管理页。
- `/sources`：后端配置与路由策略页。

## API 路由

当前服务直接代理这些接口：

- `GET /health`
- `GET /v1/models`
- `POST /v1/responses`
- `POST /v1/completions`
- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `POST /v1/moderations`
- `POST /v1/images/generations`
- `POST /v1/images/edits`
- `POST /v1/images/variations`
- `POST /v1/audio/transcriptions`
- `POST /v1/audio/speech`
- `POST /v1/rerank`
- `POST /tokenize`
- `POST /detokenize`

其中 `/v1/chat/completions`、`/v1/completions`、`/v1/responses` 会启用聊天日志提取逻辑。

## 数据与运行时

`src/openai_router/config.py` 定义：

- `BASE_DIR`
- `DATA_DIR`
- `SQLITE_DB_FILE`
- `SQLITE_URL`
- `MODEL_SYNC_CHECK_INTERVAL_SECONDS`

`src/openai_router/runtime.py` 保存全局运行态：

- `client`：全局 `httpx.AsyncClient`
- `engine`：全局数据库 engine
- `sync_task`：周期性同步任务
- `public_base_url`：CLI 启动后写入，供 UI 展示

`src/openai_router/db.py` 有一个需要特别注意的行为：

- 如果检测到现有 SQLite 表结构和 `models.py` 不一致，会删除 `data/routes.db` 后重建。
- 改数据库 schema 时，必须同步更新测试，并明确评估这个“自动删库重建”行为是否仍符合预期。

## 核心业务规则

### 路由选择

- 默认策略是 `round_robin`。
- 可切换到 `consistent_hash`。
- `consistent_hash` 优先使用这些请求头作为哈希键：
  - `X-Session-ID`
  - `X-User-ID`
  - `X-Tenant-ID`
  - `X-Correlation-ID`
  - `X-Request-ID`
  - `X-Trace-ID`
- 若请求头缺失，则回退到 `session_params.session_id`、`user`、`session_id`、`user_id`，最后退化到整个请求体序列化结果。

### 模型名与别名

- `model` 可以直接命中真实模型名，也可以命中别名。
- 别名不能占用另一个真实模型名。
- 别名不能绑定到不同模型上。
- 删除某个模型的最后一条路由时，会一并清理该模型的全部别名。

### 后端源自动同步

- 后端源保存后会立即拉取一次模型列表。
- 自动发现优先尝试 `/v1/models`，其次尝试 `/models`。
- 自动同步生成的路由会标记为 `auto_managed=True`。
- 删除后端源时，只清理该后端源关联的自动路由；手动路由应保留。
- 排除模型列表保存在 `SourceModelExclusion` 表中。

### 请求代理

- `proxy.py` 会把传入请求路径拼接到配置的后端 URL 上。
- 如果某条路由配置了后端 API Key，会覆盖原始 `Authorization` 头。
- 若未配置后端 API Key，则透传客户端传入的 `Authorization`。
- 流式和非流式错误都返回标准 JSON：`{"detail": ...}`。

### 聊天日志

- 当请求体含有 `messages` 时，`app.py` 会使用 `chat_template.jinja` 渲染提示词并输出日志。
- 非流式聊天响应会直接解析 JSON 并打印 `content` / `reasoning`。
- 流式聊天响应会缓存 chunks，结束后统一拼装并记录日志。

## 开发注意事项

- 后端 URL 统一按“去空格、去末尾 `/`”处理；改相关逻辑时要保持一致。
- 改 `services.py` 时，优先补或改 `tests/test_services.py`。
- 改 `repositories.py` 或数据模型时，优先补或改 `tests/test_repositories.py`。
- 改代理、请求头、错误处理时，优先补或改 `tests/test_proxy.py`。
- 改模板或日志渲染时，优先补或改 `tests/test_chat_template.py`。
- 当前项目没有单独的 `.env` 配置体系；运行参数主要来自 CLI 选项和持久化数据库。
- 保持改动聚焦。这个仓库的核心是“模型路由 + 自动同步 + OpenAI 兼容转发”，不要顺手引入与主链路无关的框架或抽象。

## 发布注意事项

- 构建命令使用 `uv build`。
- `chat_template.jinja` 依赖 `[tool.setuptools.package-data]` 发布；若调整模板路径或文件名，必须同步更新 `pyproject.toml`。
- 发布前至少确认：
  - `uv run python -m unittest discover -s tests`
  - `uv build`

