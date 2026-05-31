from collections import deque
import threading


MAX_LOG_LINES = 5_000
DEFAULT_LOG_VIEW_LIMIT = 300


class LogStore:
    def __init__(self, max_lines: int = MAX_LOG_LINES) -> None:
        self._lock = threading.Lock()
        self._lines: deque[str] = deque(maxlen=max_lines)

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line.rstrip("\n"))

    def snapshot(self, limit: int = DEFAULT_LOG_VIEW_LIMIT) -> tuple[int, list[str]]:
        with self._lock:
            lines = list(self._lines)
            count = len(lines)

        if limit > 0:
            lines = lines[-limit:]
        else:
            lines = []
        return count, lines

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()


log_store = LogStore()


def loguru_sink(message: object) -> None:
    log_store.append(str(message))
