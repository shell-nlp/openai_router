import threading
from collections import defaultdict

from loguru import logger

from openai_router.models import ModelRoute
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
        routes = sorted(
            repositories.list_routes(),
            key=lambda route: (route.model_name, route.model_url),
        )
        for route in routes:
            masked_key = f"***{route.api_key[-4:]}" if route.api_key else "N/A (将透传)"
            rows.append([route.model_name, route.model_url, masked_key])
        return rows

    def add_or_update_route(
        self,
        model_name: str,
        model_url: str,
        api_key: str | None,
    ) -> str:
        normalized_model_name = model_name.strip()
        normalized_model_url = self._normalize_backend_url(model_url)
        normalized_api_key = api_key.strip() if api_key else None
        normalized_api_key = normalized_api_key or None

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


route_service = RouteService()
