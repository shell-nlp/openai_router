from collections import deque
import re
import threading
from typing import Iterable


MAX_LOG_LINES = 5_000
DEFAULT_LOG_VIEW_LIMIT = 1_000
LOG_LEVELS = ("DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
_LOG_LEVEL_PATTERN = re.compile(r"\|\s*([A-Z]+)\s*\|")


class LogStore:
    def __init__(self, max_lines: int = MAX_LOG_LINES) -> None:
        self._lock = threading.Lock()
        self._lines: deque[str] = deque(maxlen=max_lines)

    def append(self, line: str) -> None:
        with self._lock:
            self._lines.append(line.rstrip("\n"))

    def snapshot(
        self,
        limit: int = DEFAULT_LOG_VIEW_LIMIT,
        levels: Iterable[str] | None = None,
    ) -> tuple[int, int, list[str]]:
        with self._lock:
            lines = list(self._lines)
        total_count = len(lines)

        normalized_levels = {
            str(level).strip().upper()
            for level in (levels or [])
            if str(level).strip().upper() in LOG_LEVELS
        }
        if normalized_levels:
            lines = [
                line
                for line in lines
                if self._extract_level(line) in normalized_levels
            ]
        filtered_count = len(lines)

        if limit > 0:
            lines = lines[-limit:]
        else:
            lines = []
        return total_count, filtered_count, lines

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()

    @staticmethod
    def _extract_level(line: str) -> str | None:
        match = _LOG_LEVEL_PATTERN.search(line)
        if match is None:
            return None
        return match.group(1)


log_store = LogStore()


def loguru_sink(message: object) -> None:
    log_store.append(str(message))
