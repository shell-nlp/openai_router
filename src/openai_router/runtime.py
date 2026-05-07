from typing import Optional

import httpx
from sqlalchemy.engine import Engine


class RuntimeState:
    def __init__(self) -> None:
        self.client: Optional[httpx.AsyncClient] = None
        self.engine: Optional[Engine] = None


runtime_state = RuntimeState()
