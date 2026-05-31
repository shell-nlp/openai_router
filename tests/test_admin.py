import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai_router.admin import (
    deserialize_request_param_mapping_text,
    on_select_route,
    remove_request_param_mapping_row,
    serialize_request_param_mapping_rows,
)


class AdminTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
