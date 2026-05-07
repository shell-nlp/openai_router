import json
import threading
from bisect import bisect_left
from collections.abc import Mapping
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx
from loguru import logger

from openai_router import repositories
from openai_router.models import BackendSource, ModelRoute

SUPPORTED_ROUTING_POLICIES = ("round_robin", "consistent_hash")
CONSISTENT_HASH_HEADER_NAMES = (
    "x-session-id",
    "x-user-id",
    "x-tenant-id",
    "x-correlation-id",
    "x-request-id",
    "x-trace-id",
)
VIRTUAL_NODES_PER_ROUTE = 160


class RouteService:
    def __init__(self) -> None:
        self._round_robin_counters = defaultdict(int)
        self._round_robin_locks = defaultdict(threading.Lock)
        self._consistent_hash_lock = threading.Lock()
        self._consistent_hash_rings: dict[
            str,
            tuple[tuple[str, ...], list[tuple[int, ModelRoute]]],
        ] = {}

    def get_routing_target(
        self,
        model_name: str,
        request_payload: Mapping[str, Any] | None = None,
        request_headers: Mapping[str, str] | None = None,
    ) -> tuple[str | None, list[str], str | None, str | None]:
        routes = repositories.list_routes_by_model(model_name)
        resolved_model_name = model_name
        if not routes:
            alias = repositories.get_alias(model_name)
            if alias is not None:
                resolved_model_name = alias.model_name
                routes = repositories.list_routes_by_model(resolved_model_name)
        available_models = self._list_available_model_names()

        if not routes:
            return None, available_models, None, None

        selected_route = self._select_route(
            resolved_model_name,
            routes,
            request_payload,
            request_headers,
        )
        return (
            selected_route.model_url,
            available_models,
            selected_route.api_key,
            selected_route.model_name,
        )

    def get_routing_policy(self) -> str:
        setting = repositories.get_router_setting()
        if setting is None:
            return "round_robin"
        return self._validate_routing_policy(setting.routing_policy)

    def update_routing_policy(self, routing_policy: str) -> str:
        normalized_policy = self._validate_routing_policy(routing_policy)
        repositories.upsert_router_setting(normalized_policy)
        message = f"路由策略已更新为 '{normalized_policy}'。"
        logger.info("[Admin] {}", message)
        return message

    def build_models_response(self) -> dict:
        models_by_name = self._build_model_timestamps()
        for alias in repositories.list_model_aliases():
            if alias.model_name not in models_by_name:
                continue
            models_by_name[alias.alias_name] = int(alias.created.timestamp())

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
        aliases_by_model = {
            model_name: ", ".join(
                sorted(alias.alias_name for alias in repositories.list_aliases_by_model(model_name))
            )
            for model_name in repositories.unique_model_names(repositories.list_model_names())
        }
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
                    aliases_by_model.get(route.model_name, ""),
                    route.model_url,
                    masked_key,
                    mode,
                    sync_interval,
                    last_synced_at,
                ]
            )
        return rows

    def get_admin_backend_sources(self) -> list[list[str]]:
        rows: list[list[str]] = []
        sources = sorted(
            repositories.list_backend_sources(),
            key=lambda source: source.model_url,
        )
        for source in sources:
            masked_key = f"***{source.api_key[-4:]}" if source.api_key else "N/A (将透传)"
            exclusions = ", ".join(sorted(repositories.list_excluded_model_names(source.id)))
            rows.append(
                [
                    source.model_url,
                    masked_key,
                    str(source.sync_interval_minutes),
                    self._format_datetime(source.last_synced_at),
                    source.last_sync_error or "",
                    exclusions,
                ]
            )
        return rows

    def add_or_update_route(
        self,
        model_name: str,
        aliases_text: str | None,
        model_url: str,
        api_key: str | None,
    ) -> str:
        normalized_model_url = self._normalize_backend_url(model_url)
        normalized_api_key = api_key.strip() if api_key else None
        normalized_api_key = normalized_api_key or None
        normalized_model_name = model_name.strip()
        normalized_aliases_text = aliases_text or ""

        self._validate_model_name(normalized_model_name)
        normalized_aliases = self._normalize_aliases(normalized_aliases_text, normalized_model_name)
        self._validate_aliases(normalized_model_name, normalized_aliases)

        created, route = repositories.upsert_route(
            normalized_model_name,
            normalized_model_url,
            normalized_api_key,
        )
        alias_result = repositories.replace_model_aliases(normalized_model_name, normalized_aliases)

        if created:
            message = (
                f"新路由 '{route.model_name} -> {route.model_url}' 已添加"
                f"；别名 {len(normalized_aliases)} 个已同步。"
            )
        else:
            message = (
                f"路由 '{route.model_name} -> {route.model_url}' 已更新"
                f"；别名 {len(normalized_aliases)} 个已同步。"
            )

        logger.info(
            "[Admin] {} aliases synced: created={}, updated={}, deleted={}",
            message,
            alias_result[0],
            alias_result[1],
            alias_result[2],
        )
        return message

    def add_or_update_backend_source(
        self,
        model_url: str,
        api_key: str | None,
        excluded_models_text: str | None,
        sync_interval_minutes: int = 15,
    ) -> str:
        normalized_model_url = self._normalize_backend_url(model_url)
        normalized_api_key = api_key.strip() if api_key else None
        normalized_api_key = normalized_api_key or None
        normalized_sync_interval = max(1, int(sync_interval_minutes))
        normalized_excluded_models = self._normalize_model_name_list(excluded_models_text or "")

        _, source = repositories.upsert_backend_source(
            normalized_model_url,
            normalized_api_key,
            normalized_sync_interval,
        )
        exclusion_result = repositories.replace_source_model_exclusions(
            source.id,
            normalized_excluded_models,
        )
        sync_result = self.sync_backend_source(source)
        message = (
            f"后端源 '{normalized_model_url}' 已保存；"
            f"同步间隔 {normalized_sync_interval} 分钟；"
            f"排除模型 {len(normalized_excluded_models)} 个"
            f"（新增 {exclusion_result[0]}，删除 {exclusion_result[1]}）；"
            f"本次同步发现 {sync_result['discovered']} 个模型"
            f"（新增 {sync_result['created']}，更新 {sync_result['updated']}，删除 {sync_result['deleted']}）。"
        )
        logger.info("[Admin] {}", message)
        return message

    def sync_backend_source_by_url(self, model_url: str) -> str:
        normalized_model_url = self._normalize_backend_url(model_url)
        source = repositories.get_backend_source_by_url(normalized_model_url)
        if source is None:
            return f"错误: 未找到后端源 '{normalized_model_url}'。"

        sync_result = self.sync_backend_source(source)
        message = (
            f"后端源 '{normalized_model_url}' 同步完成；"
            f"发现 {sync_result['discovered']} 个模型"
            f"（新增 {sync_result['created']}，更新 {sync_result['updated']}，删除 {sync_result['deleted']}）。"
        )
        logger.info("[Admin] {}", message)
        return message

    def delete_backend_source(self, model_url: str) -> str:
        normalized_model_url = self._normalize_backend_url(model_url)
        deleted = repositories.delete_backend_source(normalized_model_url)
        if not deleted:
            return f"错误: 未找到后端源 '{normalized_model_url}'。"

        message = f"后端源 '{normalized_model_url}' 已删除，其自动同步生成的路由也已一并清理。"
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

    def _list_available_model_names(self) -> list[str]:
        return repositories.unique_model_names(
            repositories.list_model_names() + repositories.list_alias_names()
        )

    def _build_model_timestamps(self) -> dict[str, int]:
        models_by_name: dict[str, int] = {}
        for route in repositories.list_routes():
            created_timestamp = int(route.created.timestamp())
            previous_timestamp = models_by_name.get(route.model_name)
            if previous_timestamp is None:
                models_by_name[route.model_name] = created_timestamp
            else:
                models_by_name[route.model_name] = min(previous_timestamp, created_timestamp)
        return models_by_name

    @staticmethod
    def _normalize_aliases(aliases_text: str, model_name: str) -> list[str]:
        raw_aliases = RouteService._split_csv_like_text(aliases_text)
        normalized_aliases = [
            alias.strip()
            for alias in raw_aliases
            if alias.strip() and alias.strip() != model_name
        ]
        return repositories.unique_model_names(normalized_aliases)

    @staticmethod
    def _normalize_model_name_list(model_names_text: str) -> list[str]:
        return repositories.unique_model_names(
            [model_name.strip() for model_name in RouteService._split_csv_like_text(model_names_text) if model_name.strip()]
        )

    @staticmethod
    def _split_csv_like_text(value: str) -> list[str]:
        return value.replace("\n", ",").replace("，", ",").replace(";", ",").split(",")

    @staticmethod
    def _validate_model_name(model_name: str) -> None:
        alias = repositories.get_alias(model_name)
        if alias is not None and alias.model_name != model_name:
            raise ValueError(
                f"模型名 '{model_name}' 当前已被用作 '{alias.model_name}' 的别名，请先移除该别名。"
            )

    @staticmethod
    def _validate_aliases(model_name: str, aliases: list[str]) -> None:
        for alias_name in aliases:
            if repositories.model_has_routes(alias_name) and alias_name != model_name:
                raise ValueError(f"别名 '{alias_name}' 已经是一个真实模型名，不能重复占用。")

            existing_alias = repositories.get_alias(alias_name)
            if existing_alias is not None and existing_alias.model_name != model_name:
                raise ValueError(
                    f"别名 '{alias_name}' 已绑定到模型 '{existing_alias.model_name}'，请先解除原绑定。"
                )

    def _select_route(
        self,
        model_name: str,
        routes: list[ModelRoute],
        request_payload: Mapping[str, Any] | None,
        request_headers: Mapping[str, str] | None,
    ) -> ModelRoute:
        if len(routes) == 1:
            return routes[0]

        routing_policy = self.get_routing_policy()
        if routing_policy == "consistent_hash":
            return self._select_consistent_hash_route(
                model_name,
                routes,
                request_payload,
                request_headers,
            )

        return self._select_round_robin_route(model_name, routes)

    def _select_round_robin_route(
        self,
        model_name: str,
        routes: list[ModelRoute],
    ) -> ModelRoute:
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
        return selected_route

    def _select_consistent_hash_route(
        self,
        model_name: str,
        routes: list[ModelRoute],
        request_payload: Mapping[str, Any] | None,
        request_headers: Mapping[str, str] | None,
    ) -> ModelRoute:
        hash_key = self._extract_hash_key(request_payload, request_headers)
        ring = self._get_consistent_hash_ring(model_name, routes)
        hash_value = self._fbi_hash(hash_key)
        ring_hashes = [item[0] for item in ring]
        selected_index = bisect_left(ring_hashes, hash_value)
        if selected_index >= len(ring):
            selected_index = 0
        selected_route = ring[selected_index][1]
        logger.debug(
            "Consistent-Hash: model={}, hash_key={}, hash_value={:016x}, selected_url={}",
            model_name,
            hash_key,
            hash_value,
            selected_route.model_url,
        )
        return selected_route

    def _get_consistent_hash_ring(
        self,
        model_name: str,
        routes: list[ModelRoute],
    ) -> list[tuple[int, ModelRoute]]:
        sorted_routes = sorted(routes, key=lambda route: route.model_url)
        route_signature = tuple(route.model_url for route in sorted_routes)

        with self._consistent_hash_lock:
            cached = self._consistent_hash_rings.get(model_name)
            if cached is not None and cached[0] == route_signature:
                return cached[1]

            ring: list[tuple[int, ModelRoute]] = []
            for route in sorted_routes:
                for index in range(VIRTUAL_NODES_PER_ROUTE):
                    virtual_key = f"{route.model_url}:{index}"
                    ring.append((self._fbi_hash(virtual_key), route))
            ring.sort(key=lambda item: item[0])
            self._consistent_hash_rings[model_name] = (route_signature, ring)
            logger.info(
                "Consistent hash ring updated for model {} with {} routes and {} virtual nodes.",
                model_name,
                len(sorted_routes),
                len(ring),
            )
            return ring

    @staticmethod
    def _extract_hash_key(
        request_payload: Mapping[str, Any] | None,
        request_headers: Mapping[str, str] | None,
    ) -> str:
        if request_headers is not None:
            for header_name in CONSISTENT_HASH_HEADER_NAMES:
                value = RouteService._get_header_value(request_headers, header_name)
                if value:
                    return f"header:{header_name}:{value}"

        if request_payload is not None:
            session_params = request_payload.get("session_params")
            if isinstance(session_params, Mapping):
                session_id = session_params.get("session_id")
                if isinstance(session_id, str) and session_id.strip():
                    return f"session:{session_id.strip()}"

            user = request_payload.get("user")
            if isinstance(user, str) and user.strip():
                return f"user:{user.strip()}"

            session_id = request_payload.get("session_id")
            if isinstance(session_id, str) and session_id.strip():
                return f"session:{session_id.strip()}"

            user_id = request_payload.get("user_id")
            if isinstance(user_id, str) and user_id.strip():
                return f"user:{user_id.strip()}"

        serialized_payload = RouteService._serialize_request_payload(request_payload)
        if len(serialized_payload) > 100:
            return f"request_hash:{RouteService._fbi_hash(serialized_payload):016x}"
        return f"request:{serialized_payload}"

    @staticmethod
    def _serialize_request_payload(request_payload: Mapping[str, Any] | None) -> str:
        if request_payload is None:
            return ""
        return json.dumps(
            request_payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _get_header_value(headers: Mapping[str, str], header_name: str) -> str | None:
        direct_value = headers.get(header_name)
        if isinstance(direct_value, str) and direct_value.strip():
            return direct_value.strip()

        lower_headers = {key.lower(): value for key, value in headers.items()}
        lowered_value = lower_headers.get(header_name)
        if isinstance(lowered_value, str) and lowered_value.strip():
            return lowered_value.strip()

        return None

    @staticmethod
    def _validate_routing_policy(routing_policy: str) -> str:
        normalized_policy = routing_policy.strip().lower()
        if normalized_policy not in SUPPORTED_ROUTING_POLICIES:
            raise ValueError(
                f"不支持的路由策略 '{routing_policy}'，仅支持: {', '.join(SUPPORTED_ROUTING_POLICIES)}。"
            )
        return normalized_policy

    @staticmethod
    def _u64(value: int) -> int:
        return value & 0xFFFFFFFFFFFFFFFF

    @classmethod
    def _murmur_hash_64a(cls, key: bytes, seed: int) -> int:
        multiplier = 0xC6A4A7935BD1E995
        shift = 47

        hash_value = cls._u64(seed ^ (len(key) * multiplier))
        chunk_limit = len(key) - (len(key) % 8)

        for offset in range(0, chunk_limit, 8):
            chunk = key[offset:offset + 8]
            mixed = int.from_bytes(chunk, "little")
            mixed = cls._u64(mixed * multiplier)
            mixed ^= mixed >> shift
            mixed = cls._u64(mixed * multiplier)
            hash_value ^= mixed
            hash_value = cls._u64(hash_value * multiplier)

        remainder = key[chunk_limit:]
        remainder_length = len(remainder)
        if remainder_length >= 7:
            hash_value ^= remainder[6] << 48
        if remainder_length >= 6:
            hash_value ^= remainder[5] << 40
        if remainder_length >= 5:
            hash_value ^= remainder[4] << 32
        if remainder_length >= 4:
            hash_value ^= remainder[3] << 24
        if remainder_length >= 3:
            hash_value ^= remainder[2] << 16
        if remainder_length >= 2:
            hash_value ^= remainder[1] << 8
        if remainder_length >= 1:
            hash_value ^= remainder[0]
            hash_value = cls._u64(hash_value * multiplier)

        hash_value ^= hash_value >> shift
        hash_value = cls._u64(hash_value * multiplier)
        hash_value ^= hash_value >> shift
        return cls._u64(hash_value)

    @classmethod
    def _murmur_rehash_64a(cls, key: int) -> int:
        multiplier = 0xC6A4A7935BD1E995
        shift = 47
        seed = 4193360111

        hash_value = cls._u64(seed ^ (8 * multiplier))
        mixed = cls._u64(key * multiplier)
        mixed ^= mixed >> shift
        mixed = cls._u64(mixed * multiplier)
        hash_value ^= mixed
        hash_value = cls._u64(hash_value * multiplier)
        hash_value ^= hash_value >> shift
        hash_value = cls._u64(hash_value * multiplier)
        hash_value ^= hash_value >> shift
        return cls._u64(hash_value)

    @classmethod
    def _furc_get_bit(
        cls,
        key: bytes,
        index: int,
        hash_cache: list[int],
        old_ord: list[int],
    ) -> bool:
        seed = 4193360111
        ord_value = index >> 6

        if old_ord[0] < ord_value:
            for current_ord in range(old_ord[0] + 1, ord_value + 1):
                if current_ord == 0:
                    hash_value = cls._murmur_hash_64a(key, seed)
                else:
                    hash_value = cls._murmur_rehash_64a(hash_cache[current_ord - 1])

                if len(hash_cache) <= current_ord:
                    hash_cache.extend([0] * (current_ord + 1 - len(hash_cache)))
                hash_cache[current_ord] = hash_value
            old_ord[0] = ord_value

        hash_value = hash_cache[ord_value]
        bit_position = index & 0x3F
        return ((hash_value >> bit_position) & 0x1) != 0

    @classmethod
    def _furc_hash(cls, key: str, modulus: int) -> int:
        max_tries = 32
        furc_shift = 23

        if modulus <= 1:
            return 0

        key_bytes = key.encode()
        hash_cache: list[int] = []
        old_ord = [-1]

        depth = 0
        while modulus > (1 << depth):
            depth += 1

        bit_index = depth
        for _ in range(max_tries):
            while not cls._furc_get_bit(key_bytes, bit_index, hash_cache, old_ord):
                if depth == 0:
                    return 0
                depth -= 1
                bit_index = depth

            bit_index += furc_shift
            number = 1
            for _ in range(max(depth - 1, 0)):
                number = (number << 1) | int(
                    cls._furc_get_bit(key_bytes, bit_index, hash_cache, old_ord)
                )
                bit_index += furc_shift

            if number < modulus:
                return number

        return 0

    @classmethod
    def _fbi_hash(cls, key: str) -> int:
        large_modulus = (1 << 23) - 1
        furc_result = cls._furc_hash(key, large_modulus)
        return cls._murmur_hash_64a(furc_result.to_bytes(4, "little"), 4193360111)

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
