import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai_router.app import chat_template, parse_tool_arguments


class ChatTemplateTest(unittest.TestCase):
    def test_assistant_tool_call_accepts_json_string_arguments(self) -> None:
        rendered = chat_template.render(
            messages=[
                {"role": "user", "content": "执行以下 pwd"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "bash",
                                "arguments": '{"command":"pwd","description":"Shows current directory"}',
                            },
                        }
                    ],
                },
            ],
            add_generation_prompt=True,
        )

        self.assertIn("<function=bash>", rendered)
        self.assertIn("<parameter=command>\npwd\n</parameter>", rendered)
        self.assertIn(
            "<parameter=description>\nShows current directory\n</parameter>",
            rendered,
        )

    def test_parse_tool_arguments_rejects_non_mapping_values(self) -> None:
        self.assertEqual(parse_tool_arguments("[]"), {})
        self.assertEqual(parse_tool_arguments("not json"), {})
        self.assertEqual(parse_tool_arguments(["command", "pwd"]), {})


if __name__ == "__main__":
    unittest.main()
