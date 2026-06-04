import json
import sys
import unittest
from pathlib import Path
from unittest.mock import call, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai_router.chat_logging import _log_chat_response, _log_stream_chat_response


class ChatLoggingTest(unittest.TestCase):
    @patch("openai_router.chat_logging.logger.info")
    def test_log_chat_response_logs_final_token_usage(self, mock_info) -> None:
        _log_chat_response(
            json.dumps(
                {
                    "choices": [
                        {"message": {"content": "hello"}},
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "total_tokens": 18,
                    },
                }
            ).encode("utf-8")
        )

        self.assertEqual(
            mock_info.call_args_list,
            [
                call("Model response:\n{}", "hello"),
                call(
                    "Token usage: {}",
                    '{"completion_tokens": 7, "prompt_tokens": 11, "total_tokens": 18}',
                ),
            ],
        )

    @patch("openai_router.chat_logging.logger.info")
    def test_log_stream_chat_response_logs_usage_from_final_stream_event(
        self,
        mock_info,
    ) -> None:
        _log_stream_chat_response(
            [
                b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
                b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
                (
                    'data: {"type":"response.completed","response":{"usage":'
                    '{"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}}}\n\n'
                ).encode("utf-8"),
                b"data: [DONE]\n\n",
            ]
        )

        self.assertEqual(
            mock_info.call_args_list,
            [
                call("Model response:\n{}", "hello"),
                call(
                    "Token usage: {}",
                    '{"input_tokens": 5, "output_tokens": 3, "total_tokens": 8}',
                ),
            ],
        )

    @patch("openai_router.chat_logging.logger.info")
    def test_log_chat_response_logs_tool_call_output(self, mock_info) -> None:
        _log_chat_response(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "",
                                "tool_calls": [
                                    {
                                        "type": "function",
                                        "function": {
                                            "name": "bash",
                                            "arguments": '{"command":"pwd"}',
                                        },
                                    }
                                ],
                            }
                        },
                    ],
                }
            ).encode("utf-8")
        )

        self.assertEqual(
            mock_info.call_args_list,
            [
                call(
                    "Model response:\n{}",
                    (
                        "<tool_call>\n"
                        "<function=bash>\n"
                        "<parameter=command>\n"
                        "pwd\n"
                        "</parameter>\n"
                        "</function>\n"
                        "</tool_call>"
                    ),
                ),
            ],
        )

    @patch("openai_router.chat_logging.logger.info")
    def test_log_stream_chat_response_logs_tool_call_output(self, mock_info) -> None:
        _log_stream_chat_response(
            [
                (
                    "data: "
                    + json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_1",
                                                "type": "function",
                                                "function": {
                                                    "name": "bash",
                                                    "arguments": '{"command"',
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    )
                    + "\n\n"
                ).encode("utf-8"),
                (
                    "data: "
                    + json.dumps(
                        {
                            "choices": [
                                {
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "function": {
                                                    "arguments": ':"pwd"}',
                                                },
                                            }
                                        ]
                                    }
                                }
                            ]
                        }
                    )
                    + "\n\n"
                ).encode("utf-8"),
                b"data: [DONE]\n\n",
            ]
        )

        self.assertEqual(
            mock_info.call_args_list,
            [
                call(
                    "Model response:\n{}",
                    (
                        "<tool_call>\n"
                        "<function=bash>\n"
                        "<parameter=command>\n"
                        "pwd\n"
                        "</parameter>\n"
                        "</function>\n"
                        "</tool_call>"
                    ),
                ),
            ],
        )


if __name__ == "__main__":
    unittest.main()
