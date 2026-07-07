from collections import deque
import re
import threading
from typing import Iterable


MAX_LOG_LINES = 5_000
DEFAULT_LOG_VIEW_LIMIT = 100
LOG_LEVELS = ("DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL")
_LOG_LEVEL_PATTERN = re.compile(r"\|\s*([A-Z]+)\s*\|")
MAX_LOG_LINE_LENGTH = 2000


class LogStore:
    def __init__(self, max_lines: int = MAX_LOG_LINES) -> None:
        self._lock = threading.Lock()
        self._lines: deque[tuple[int, str]] = deque(maxlen=max_lines)
        self._seq = 0

    def append(self, line: str) -> None:
        with self._lock:
            self._seq += 1
            self._lines.append((self._seq, line.rstrip("\n")))

    def snapshot(
        self,
        limit: int = DEFAULT_LOG_VIEW_LIMIT,
        levels: Iterable[str] | None = None,
    ) -> tuple[int, int, list[str]]:
        with self._lock:
            raw = list(self._lines)
            seq = self._seq
        total_count = len(raw)
        lines = [l for _, l in raw]

        normalized_levels = {
            str(level).strip().upper()
            for level in (levels or [])
            if str(level).strip().upper() in LOG_LEVELS
        }
        if normalized_levels:
            zipped = [
                (s, l) for s, l in raw
                if self._extract_level(l) in normalized_levels
            ]
            filtered_count = len(zipped)
            lines = [l for _, l in zipped]
        else:
            filtered_count = len(lines)

        if limit > 0:
            lines = lines[-limit:]
        else:
            lines = []
        return total_count, filtered_count, lines

    def get_lines_since(
        self,
        seq: int,
        limit: int = DEFAULT_LOG_VIEW_LIMIT,
        levels: Iterable[str] | None = None,
    ) -> tuple[int, int, list[str], int]:
        with self._lock:
            raw = list(self._lines)
            current_seq = self._seq

        if seq < raw[0][0] if raw else seq:
            oldest_seq = raw[0][0] if raw else current_seq
        else:
            oldest_seq = None

        candidates = [(s, l) for s, l in raw if s > seq]
        total_buffer = len(raw)

        if not candidates and total_buffer > 0:
            return total_buffer, 0, [], current_seq

        lines = [l for _, l in candidates]
        normalized_levels = {
            str(level).strip().upper()
            for level in (levels or [])
            if str(level).strip().upper() in LOG_LEVELS
        }
        if normalized_levels:
            candidates = [
                (s, l) for s, l in candidates
                if self._extract_level(l) in normalized_levels
            ]
            lines = [l for _, l in candidates]
        filtered_count = len(lines)

        if limit > 0:
            lines = lines[-limit:]

        return total_buffer, filtered_count, lines, current_seq

    def get_sequence(self) -> int:
        with self._lock:
            return self._seq

    def clear(self) -> None:
        with self._lock:
            self._lines.clear()
            self._seq = 0

    @staticmethod
    def _extract_level(line: str) -> str | None:
        match = _LOG_LEVEL_PATTERN.search(line)
        if match is None:
            return None
        return match.group(1)


log_store = LogStore()


def loguru_sink(message: object) -> None:
    log_store.append(str(message))
