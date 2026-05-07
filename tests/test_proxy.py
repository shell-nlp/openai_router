import unittest
from pathlib import Path
import sys

from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai_router.proxy import build_backend_url, build_proxy_headers, filter_response_headers


def build_request(headers: list[tuple[bytes, bytes]]) -> Request:
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/chat/completions",
        "query_string": b"",
        "headers": headers,
    }
    return Request(scope)


class ProxyHelpersTest(unittest.TestCase):
    def test_build_backend_url_normalizes_trailing_slash(self) -> None:
        self.assertEqual(
            build_backend_url("http://localhost:8000/v1/", "/v1/chat/completions"),
            "http://localhost:8000/v1/chat/completions",
        )

    def test_build_proxy_headers_strips_hop_by_hop_headers(self) -> None:
        request = build_request(
            [
                (b"host", b"router.local"),
                (b"content-length", b"120"),
                (b"content-type", b"application/json"),
                (b"authorization", b"Bearer original"),
            ]
        )

        headers = build_proxy_headers(request, "override-key")

        self.assertEqual(headers["content-type"], "application/json")
        self.assertEqual(headers["Authorization"], "Bearer override-key")
        self.assertNotIn("host", headers)
        self.assertNotIn("content-length", headers)

    def test_filter_response_headers_strips_hop_by_hop_headers(self) -> None:
        filtered = filter_response_headers(
            {
                "content-type": "application/json",
                "content-length": "42",
                "x-request-id": "abc",
            }
        )

        self.assertEqual(filtered, {"content-type": "application/json", "x-request-id": "abc"})


if __name__ == "__main__":
    unittest.main()
