import asyncio
from typing import Optional

import httpx
from sqlalchemy.engine import Engine


class RuntimeState:
    def __init__(self) -> None:
        self.client: Optional[httpx.AsyncClient] = None
        self.engine: Optional[Engine] = None
        self.sync_task: Optional[asyncio.Task] = None
        self.public_base_url: Optional[str] = None


runtime_state = RuntimeState()
