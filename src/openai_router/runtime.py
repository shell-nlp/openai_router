import asyncio
from collections import OrderedDict
from dataclasses import dataclass
from time import monotonic
import threading
from typing import Optional

import httpx
from sqlalchemy.engine import Engine

MAX_TRACKED_RESPONSE_ROUTES = 10_000
RESPONSE_ROUTE_TTL_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class ResponseRoute:
    backend_server_url: str
    backend_api_key: Optional[str]


class RuntimeState:
    def __init__(self) -> None:
        self.client: Optional[httpx.AsyncClient] = None
        self.engine: Optional[Engine] = None
        self.sync_task: Optional[asyncio.Task] = None
        self.public_base_url: Optional[str] = None
        self._response_routes_lock = threading.Lock()
        self._response_routes: OrderedDict[str, tuple[str, Optional[str], float]] = (
            OrderedDict()
        )

    def remember_response_route(
        self,
        response_id: str,
        backend_server_url: str,
        backend_api_key: Optional[str],
    ) -> None:
        now = monotonic()
        with self._response_routes_lock:
            self._prune_response_routes_locked(now)
            self._response_routes[response_id] = (
                backend_server_url,
                backend_api_key,
                now,
            )
            self._response_routes.move_to_end(response_id)
            while len(self._response_routes) > MAX_TRACKED_RESPONSE_ROUTES:
                self._response_routes.popitem(last=False)

    def get_response_route(self, response_id: str) -> ResponseRoute | None:
        now = monotonic()
        with self._response_routes_lock:
            self._prune_response_routes_locked(now)
            entry = self._response_routes.get(response_id)
            if entry is None:
                return None

            backend_server_url, backend_api_key, _ = entry
            self._response_routes[response_id] = (
                backend_server_url,
                backend_api_key,
                now,
            )
            self._response_routes.move_to_end(response_id)
            return ResponseRoute(
                backend_server_url=backend_server_url,
                backend_api_key=backend_api_key,
            )

    def clear_response_routes(self) -> None:
        with self._response_routes_lock:
            self._response_routes.clear()

    def _prune_response_routes_locked(self, now: float) -> None:
        expired_before = now - RESPONSE_ROUTE_TTL_SECONDS
        while self._response_routes:
            _, (_, _, last_seen_at) = next(iter(self._response_routes.items()))
            if last_seen_at > expired_before:
                break
            self._response_routes.popitem(last=False)


runtime_state = RuntimeState()
