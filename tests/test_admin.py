import asyncio
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai_router.admin import (
    deserialize_request_param_mapping_text,
    get_recent_logs_page_data,
    on_select_route,
    remove_request_param_mapping_row,
    serialize_request_param_mapping_rows,
)
from openai_router.log_store import log_store


class AdminTest(unittest.TestCase):
    def setUp(self) -> None:
        log_store.clear()

    def tearDown(self) -> None:
        log_store.clear()

    def test_serialize_request_param_mapping_rows(self) -> None:
        serialized = serialize_request_param_mapping_rows(
            [
                ["enable_thinking", "chat_template_kwargs.enable_thinking"],
                ["top_k", "sampling.top_k"],
            ]
        )

        self.assertEqual(
            serialized,
            '{"enable_thinking":"chat_template_kwargs.enable_thinking","top_k":"sampling.top_k"}',
        )

    def test_serialize_request_param_mapping_rows_rejects_partial_row(self) -> None:
        with self.assertRaisesRegex(ValueError, "同时填写"):
            serialize_request_param_mapping_rows([["enable_thinking", ""]])

    def test_deserialize_request_param_mapping_text(self) -> None:
        rows = deserialize_request_param_mapping_text(
            '{"enable_thinking":"chat_template_kwargs.enable_thinking"}'
        )

        self.assertEqual(
            rows,
            [["enable_thinking", "chat_template_kwargs.enable_thinking"]],
        )

    def test_remove_request_param_mapping_row(self) -> None:
        rows = remove_request_param_mapping_row(
            [
                ["enable_thinking", "chat_template_kwargs.enable_thinking"],
                ["top_k", "sampling.top_k"],
            ]
        )

        self.assertEqual(
            rows,
            [["enable_thinking", "chat_template_kwargs.enable_thinking"]],
        )

    def test_on_select_route_returns_request_param_mapping_rows(self) -> None:
        routes_data = pd.DataFrame(
            [
                [
                    "gpt-4",
                    "gpt-4o-latest",
                    "http://backend/v1",
                    "***key",
                    '{"enable_thinking":"chat_template_kwargs.enable_thinking"}',
                    "手动配置",
                    "-",
                    "-",
                ]
            ]
        )

        selected = on_select_route(routes_data, SimpleNamespace(index=(0, 0)))

        self.assertEqual(
            selected,
            (
                "gpt-4",
                "gpt-4o-latest",
                "http://backend/v1",
                "",
                [["enable_thinking", "chat_template_kwargs.enable_thinking"]],
            ),
        )

    def test_get_recent_logs_page_data(self) -> None:
        log_store.append("2026-05-31 12:00:00.000 | INFO     | hello")
        log_store.append("2026-05-31 12:00:01.000 | ERROR    | world")

        status, logs = asyncio.run(get_recent_logs_page_data())

        self.assertIn("当前缓存 2 条日志", status)
        self.assertIn("hello", logs)
        self.assertIn("world", logs)


if __name__ == "__main__":
    unittest.main()
