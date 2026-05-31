import asyncio
import json
import unittest
from pathlib import Path
import sys
from unittest.mock import AsyncMock, Mock, patch

import httpx

from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai_router.proxy import (
    apply_request_param_mapping,
    build_backend_url,
    build_proxy_headers,
    filter_response_headers,
    non_stream_proxy,
    resolve_response_request,
    resolve_route_request,
    stream_proxy,
)
from openai_router.runtime import runtime_state
from openai_router.services import RoutingTarget


class AsyncByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk

    async def aclose(self) -> None:
        return None


def build_request(
    headers: list[tuple[bytes, bytes]],
    *,
    method: str = "POST",
    path: str = "/v1/chat/completions",
) -> Request:
    return build_json_request(headers, b"", method=method, path=path)


def build_json_request(
    headers: list[tuple[bytes, bytes]],
    body: bytes,
    *,
    method: str = "POST",
    path: str = "/v1/chat/completions",
) -> Request:
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": b"",
        "headers": headers,
    }
    return Request(scope, receive)


async def read_streaming_body(response) -> bytes:
    body = b""
    async for chunk in response.body_iterator:
        body += chunk
    return body


class ProxyHelpersTest(unittest.TestCase):
    def tearDown(self) -> None:
        runtime_state.client = None
        runtime_state.clear_response_routes()

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

    @patch("openai_router.proxy.route_service.get_routing_target")
    def test_resolve_route_request_rewrites_alias_to_real_model(
        self,
        mock_get_routing_target,
    ) -> None:
        mock_get_routing_target.return_value = (
            RoutingTarget(
                "http://backend-1/v1",
                ("gpt-4", "gpt-4o-latest"),
                "key-1",
                "gpt-4",
                '{"enable_thinking":"chat_template_kwargs.enable_thinking"}',
            )
        )
        request = build_json_request(
            [(b"content-type", b"application/json")],
            b'{"model":"gpt-4o-latest","messages":[],"enable_thinking":false}',
        )

        resolved = asyncio.run(resolve_route_request(request))

        self.assertEqual(resolved.backend_url, "http://backend-1/v1/chat/completions")
        self.assertEqual(resolved.backend_api_key, "key-1")
        self.assertEqual(resolved.routed_model_name, "gpt-4")
        self.assertEqual(resolved.json_body["model"], "gpt-4")
        self.assertEqual(
            resolved.json_body["chat_template_kwargs"],
            {"enable_thinking": False},
        )
        self.assertNotIn("enable_thinking", resolved.json_body)

    def test_apply_request_param_mapping_moves_nested_value(self) -> None:
        payload = {
            "enable_thinking": False,
            "metadata": {"request": {"trace_id": "req_1"}},
        }

        mapped = apply_request_param_mapping(
            payload,
            json.dumps(
                {
                    "enable_thinking": "chat_template_kwargs.enable_thinking",
                    "metadata.request.trace_id": "request_metadata.trace_id",
                }
            ),
        )

        self.assertEqual(
            mapped,
            {
                "chat_template_kwargs": {"enable_thinking": False},
                "request_metadata": {"trace_id": "req_1"},
            },
        )

    def test_resolve_response_request_uses_tracked_backend(self) -> None:
        runtime_state.remember_response_route(
            "resp_123",
            "http://backend-1/v1",
            "key-1",
        )
        request = build_request(
            [(b"content-type", b"application/json")],
            method="GET",
            path="/v1/responses/resp_123",
        )

        resolved = asyncio.run(resolve_response_request(request, "resp_123"))

        self.assertEqual(
            resolved.backend_url,
            "http://backend-1/v1/responses/resp_123",
        )
        self.assertEqual(resolved.backend_api_key, "key-1")

    def test_stream_proxy_returns_backend_error_response(self) -> None:
        request = build_json_request([(b"content-type", b"application/json")], b'{"stream": true}')
        backend_request = httpx.Request("POST", "http://backend/v1/chat/completions")
        backend_response = httpx.Response(
            429,
            content=b'{"error":"rate_limited"}',
            headers={"content-type": "application/json", "x-request-id": "req_123"},
            request=backend_request,
        )
        client = Mock()
        client.build_request.return_value = backend_request
        client.send = AsyncMock(return_value=backend_response)
        runtime_state.client = client

        response = asyncio.run(
            stream_proxy(
                "http://backend/v1/chat/completions",
                request,
                {"stream": True},
                None,
            )
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.body, b'{"error":"rate_limited"}')
        self.assertEqual(response.headers["content-type"], "application/json")
        self.assertEqual(response.headers["x-request-id"], "req_123")

    def test_stream_proxy_returns_json_response_on_connect_error(self) -> None:
        request = build_json_request([(b"content-type", b"application/json")], b'{"stream": true}')
        backend_request = httpx.Request("POST", "http://backend/v1/chat/completions")
        client = Mock()
        client.build_request.return_value = backend_request
        client.send = AsyncMock(
            side_effect=httpx.ConnectError("boom", request=backend_request)
        )
        runtime_state.client = client

        response = asyncio.run(
            stream_proxy(
                "http://backend/v1/chat/completions",
                request,
                {"stream": True},
                None,
            )
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.body, b'{"detail":"Backend service unavailable"}')
        self.assertEqual(response.headers["content-type"], "application/json")

    def test_stream_proxy_tracks_response_route_for_streaming_responses_create(self) -> None:
        request = build_json_request(
            [(b"content-type", b"application/json")],
            b'{"model":"gpt-4.1","stream":true}',
            path="/v1/responses",
        )
        backend_request = httpx.Request("POST", "http://backend/v1/responses")
        backend_response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            request=backend_request,
            stream=AsyncByteStream(
                [
                    b"event: response.created\n",
                    b'data: {"type":"response.created","response":{"id":"resp_stream"}}\n\n',
                    b"data: [DONE]\n\n",
                ]
            ),
        )
        client = Mock()
        client.build_request.return_value = backend_request
        client.send = AsyncMock(return_value=backend_response)
        runtime_state.client = client

        response = asyncio.run(
            stream_proxy(
                "http://backend/v1/responses",
                request,
                {"model": "gpt-4.1", "stream": True},
                "backend-key",
                backend_server_url="http://backend/v1",
            )
        )

        body = asyncio.run(read_streaming_body(response))
        tracked_route = runtime_state.get_response_route("resp_stream")
        self.assertIsNotNone(tracked_route)
        self.assertEqual(
            body,
            b'event: response.created\ndata: {"type":"response.created","response":{"id":"resp_stream"}}\n\ndata: [DONE]\n\n',
        )
        self.assertEqual(tracked_route.backend_server_url, "http://backend/v1")
        self.assertEqual(tracked_route.backend_api_key, "backend-key")

    def test_non_stream_proxy_tracks_response_route_for_responses_create(self) -> None:
        request = build_json_request(
            [(b"content-type", b"application/json")],
            b'{"model":"gpt-4.1"}',
            path="/v1/responses",
        )
        backend_request = httpx.Request("POST", "http://backend/v1/responses")
        backend_response = httpx.Response(
            200,
            content=b'{"id":"resp_123","object":"response"}',
            headers={"content-type": "application/json"},
            request=backend_request,
        )
        client = Mock()
        client.request = AsyncMock(return_value=backend_response)
        runtime_state.client = client

        response = asyncio.run(
            non_stream_proxy(
                "http://backend/v1/responses",
                request,
                {"model": "gpt-4.1"},
                "backend-key",
                backend_server_url="http://backend/v1",
            )
        )

        tracked_route = runtime_state.get_response_route("resp_123")
        self.assertIsNotNone(tracked_route)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(tracked_route.backend_server_url, "http://backend/v1")
        self.assertEqual(tracked_route.backend_api_key, "backend-key")

    def test_non_stream_proxy_returns_json_response_on_connect_error(self) -> None:
        request = build_json_request([(b"content-type", b"application/json")], b'{}')
        backend_request = httpx.Request("POST", "http://backend/v1/chat/completions")
        client = Mock()
        client.request = AsyncMock(
            side_effect=httpx.ConnectError("boom", request=backend_request)
        )
        runtime_state.client = client

        response = asyncio.run(
            non_stream_proxy(
                "http://backend/v1/chat/completions",
                request,
                {},
                None,
            )
        )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.body, b'{"detail":"Backend service unavailable"}')
        self.assertEqual(response.headers["content-type"], "application/json")


if __name__ == "__main__":
    unittest.main()
