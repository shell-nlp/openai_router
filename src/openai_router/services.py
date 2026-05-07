import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit, urlunsplit

import httpx

from loguru import logger

from openai_router.models import BackendSource, ModelRoute
from openai_router import repositories


class RouteService:
    def __init__(self) -> None:
        self._round_robin_counters = defaultdict(int)
        self._round_robin_locks = defaultdict(threading.Lock)

    def get_routing_target(self, model_name: str) -> tuple[str | None, list[str], str | None]:
        routes = repositories.list_routes_by_model(model_name)
        available_models = repositories.unique_model_names(repositories.list_model_names())

        if not routes:
            return None, available_models, None

        if len(routes) == 1:
            selected_route = routes[0]
            return selected_route.model_url, available_models, selected_route.api_key

        lock = self._round_robin_locks[model_name]
        with lock:
            current_index = self._round_robin_counters[model_name]
            self._round_robin_counters[model_name] = (current_index + 1) % len(routes)

        selected_route = routes[current_index]
        logger.debug(
            "Round-Robin: model={}, selected_index={}, total={}",
            model_name,
            current_index,
            len(routes),
        )
        return selected_route.model_url, available_models, selected_route.api_key

    def build_models_response(self) -> dict:
        models_by_name: dict[str, int] = {}
        for route in repositories.list_routes():
            created_timestamp = int(route.created.timestamp())
            previous_timestamp = models_by_name.get(route.model_name)
            if previous_timestamp is None:
                models_by_name[route.model_name] = created_timestamp
            else:
                models_by_name[route.model_name] = min(previous_timestamp, created_timestamp)

        models_data = [
            {
                "id": model_name,
                "object": "model",
                "created": created_timestamp,
                "owned_by": "openai_router",
                "permission": [],
            }
            for model_name, created_timestamp in sorted(models_by_name.items())
        ]
        return {"object": "list", "data": models_data}

    def get_admin_routes(self) -> list[list[str]]:
        rows: list[list[str]] = []
        source_map = {source.id: source for source in repositories.list_backend_sources()}
        routes = sorted(
            repositories.list_routes(),
            key=lambda route: (route.model_name, route.model_url),
        )
        for route in routes:
            masked_key = f"***{route.api_key[-4:]}" if route.api_key else "N/A (将透传)"
            source = source_map.get(route.source_id)
            sync_interval = str(source.sync_interval_minutes) if source else "-"
            last_synced_at = self._format_datetime(source.last_synced_at) if source else "-"
            mode = "自动同步" if route.auto_managed else "手动配置"
            rows.append(
                [
                    route.model_name,
                    route.model_url,
                    masked_key,
                    mode,
                    sync_interval,
                    last_synced_at,
                ]
            )
        return rows

    def add_or_update_route(
        self,
        model_name: str,
        model_url: str,
        api_key: str | None,
        auto_discover_models: bool = False,
        sync_interval_minutes: int = 15,
    ) -> str:
        normalized_model_url = self._normalize_backend_url(model_url)
        normalized_api_key = api_key.strip() if api_key else None
        normalized_api_key = normalized_api_key or None
        normalized_model_name = model_name.strip()
        normalized_sync_interval = max(1, int(sync_interval_minutes))

        if auto_discover_models:
            _, source = repositories.upsert_backend_source(
                normalized_model_url,
                normalized_api_key,
                normalized_sync_interval,
            )
            sync_result = self.sync_backend_source(source)
            message = (
                f"已为 '{normalized_model_url}' 启用自动模型同步，"
                f"同步间隔 {normalized_sync_interval} 分钟；"
                f"本次同步发现 {sync_result['discovered']} 个模型"
                f"（新增 {sync_result['created']}，更新 {sync_result['updated']}，删除 {sync_result['deleted']}）。"
            )
            logger.info("[Admin] {}", message)
            return message

        created, route = repositories.upsert_route(
            normalized_model_name,
            normalized_model_url,
            normalized_api_key,
        )
        if created:
            message = (
                f"新路由 '{route.model_name} -> {route.model_url}' 已添加 (用于负载均衡)。"
            )
        else:
            message = f"路由 '{route.model_name} -> {route.model_url}' 的 API 密钥已更新。"

        logger.info("[Admin] {}", message)
        return message

    def delete_route(self, model_name: str, model_url: str) -> str:
        normalized_model_name = model_name.strip()
        normalized_model_url = self._normalize_backend_url(model_url)
        deleted = repositories.delete_route(normalized_model_name, normalized_model_url)
        if not deleted:
            return f"错误: 未找到路由 '{normalized_model_name} -> {normalized_model_url}'。"

        message = f"路由 '{normalized_model_name} -> {normalized_model_url}' 已删除。"
        logger.info("[Admin] Route deleted: {} -> {}", normalized_model_name, normalized_model_url)
        return message

    @staticmethod
    def _normalize_backend_url(model_url: str) -> str:
        return model_url.strip().rstrip("/")

    def sync_due_backend_sources(self) -> int:
        now = datetime.now(timezone.utc)
        synced_count = 0
        for source in repositories.list_backend_sources():
            if not self._is_source_due_for_sync(source, now):
                continue
            self.sync_backend_source(source)
            synced_count += 1

        return synced_count

    def sync_backend_source(self, source: BackendSource) -> dict[str, int]:
        synchronized_at = datetime.now(timezone.utc)
        try:
            discovered_models = self._fetch_backend_models(source.model_url, source.api_key)
            created_count, updated_count, deleted_count = repositories.sync_auto_managed_routes(
                source.id,
                discovered_models,
                source.model_url,
                source.api_key,
            )
            repositories.update_backend_source_sync_status(
                source.id,
                synchronized_at,
                None,
            )
            logger.info(
                "Auto-sync completed for {}. discovered={}, created={}, updated={}, deleted={}",
                source.model_url,
                len(discovered_models),
                created_count,
                updated_count,
                deleted_count,
            )
            return {
                "discovered": len(discovered_models),
                "created": created_count,
                "updated": updated_count,
                "deleted": deleted_count,
            }
        except Exception as exc:
            repositories.update_backend_source_sync_status(source.id, source.last_synced_at, str(exc))
            raise

    def _fetch_backend_models(
        self,
        backend_url: str,
        api_key: str | None,
    ) -> list[str]:
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        last_error: str | None = None
        for discovery_url in self._build_model_discovery_urls(backend_url):
            try:
                response = httpx.get(discovery_url, headers=headers, timeout=15.0)
                response.raise_for_status()
                models = self._parse_model_names(response.json(), discovery_url)
                logger.info(
                    "Discovered {} models from {}.",
                    len(models),
                    discovery_url,
                )
                return models
            except (httpx.HTTPError, ValueError) as exc:
                last_error = f"{discovery_url}: {exc}"
                logger.warning("Model discovery failed for {}: {}", discovery_url, exc)

        error_detail = last_error or "unknown error"
        raise ValueError(f"无法从后端自动获取模型列表: {error_detail}")

    @staticmethod
    def _build_model_discovery_urls(backend_url: str) -> list[str]:
        parsed = urlsplit(backend_url)
        base_path = parsed.path.rstrip("/")
        candidate_paths: list[str]

        if not base_path:
            candidate_paths = ["/v1/models", "/models"]
        elif base_path == "/v1" or base_path.endswith("/v1"):
            candidate_paths = [f"{base_path}/models", "/v1/models"]
        else:
            candidate_paths = [f"{base_path}/v1/models", f"{base_path}/models"]

        urls: list[str] = []
        for path in candidate_paths:
            candidate_url = urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
            if candidate_url not in urls:
                urls.append(candidate_url)
        return urls

    @staticmethod
    def _parse_model_names(payload: dict, discovery_url: str) -> list[str]:
        model_items = payload.get("data")
        if not isinstance(model_items, list):
            raise ValueError(f"{discovery_url} 返回格式不正确，缺少 data 列表")

        model_names: list[str] = []
        for item in model_items:
            if not isinstance(item, dict):
                continue
            model_name = item.get("id")
            if isinstance(model_name, str) and model_name.strip():
                model_names.append(model_name.strip())

        unique_model_names = repositories.unique_model_names(model_names)
        if not unique_model_names:
            raise ValueError(f"{discovery_url} 未返回任何可用模型")

        return unique_model_names

    @staticmethod
    def _is_source_due_for_sync(
        source: BackendSource,
        now: datetime,
    ) -> bool:
        if source.last_synced_at is None:
            return True
        last_synced_at = RouteService._ensure_utc_datetime(source.last_synced_at)
        current_time = RouteService._ensure_utc_datetime(now)
        return current_time - last_synced_at >= timedelta(minutes=source.sync_interval_minutes)

    @staticmethod
    def _format_datetime(value: datetime | None) -> str:
        if value is None:
            return "-"
        value = RouteService._ensure_utc_datetime(value)
        return value.strftime("%Y-%m-%d %H:%M:%S UTC")

    @staticmethod
    def _ensure_utc_datetime(value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


route_service = RouteService()
