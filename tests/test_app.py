import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import Response
from fastapi.routing import APIRoute
from starlette.requests import Request

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai_router.app import create_app, parse_stream_parameter


def build_request(
    *,
    method: str,
    path: str,
    query_string: bytes = b"",
) -> Request:
    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    scope = {
        "type": "http",
        "method": method,
        "path": path,
        "query_string": query_string,
        "headers": [],
    }
    return Request(scope, receive)


def build_test_app():
    with patch("openai_router.app.create_admin_ui", return_value=object()), patch(
        "openai_router.app.gr.mount_gradio_app",
        side_effect=lambda app, *_args, **_kwargs: app,
    ):
        return create_app()


def find_route_endpoint(app, path: str, method: str):
    for route in app.routes:
        if isinstance(route, APIRoute) and route.path == path and method in route.methods:
            return route.endpoint
    raise AssertionError(f"Route not found: {method} {path}")


class AppRoutingTest(unittest.TestCase):
    def test_parse_stream_parameter(self) -> None:
        self.assertTrue(parse_stream_parameter("true"))
        self.assertTrue(parse_stream_parameter("1"))
        self.assertTrue(parse_stream_parameter("ON"))
        self.assertFalse(parse_stream_parameter(None))
        self.assertFalse(parse_stream_parameter("false"))

    def test_response_get_with_stream_query_uses_stream_proxy(self) -> None:
        app = build_test_app()
        endpoint = find_route_endpoint(app, "/v1/responses/{response_id}", "GET")
        request = build_request(
            method="GET",
            path="/v1/responses/resp_123",
            query_string=b"stream=true",
        )
        resolved_request = SimpleNamespace(
            backend_url="http://backend/v1/responses/resp_123",
            backend_api_key="backend-key",
        )
        expected_response = Response(content=b"", media_type="text/event-stream")

        with patch(
            "openai_router.app.resolve_response_request",
            AsyncMock(return_value=resolved_request),
        ), patch(
            "openai_router.app.stream_proxy",
            AsyncMock(return_value=expected_response),
        ) as mock_stream_proxy, patch(
            "openai_router.app.non_stream_proxy",
            AsyncMock(),
        ) as mock_non_stream_proxy:
            actual_response = asyncio.run(endpoint("resp_123", request))

        self.assertIs(actual_response, expected_response)
        mock_stream_proxy.assert_awaited_once_with(
            "http://backend/v1/responses/resp_123",
            request,
            None,
            "backend-key",
        )
        mock_non_stream_proxy.assert_not_awaited()

    def test_response_get_without_stream_query_uses_non_stream_proxy(self) -> None:
        app = build_test_app()
        endpoint = find_route_endpoint(app, "/v1/responses/{response_id}", "GET")
        request = build_request(
            method="GET",
            path="/v1/responses/resp_123",
        )
        resolved_request = SimpleNamespace(
            backend_url="http://backend/v1/responses/resp_123",
            backend_api_key="backend-key",
        )
        expected_response = Response(content=b"{}")

        with patch(
            "openai_router.app.resolve_response_request",
            AsyncMock(return_value=resolved_request),
        ), patch(
            "openai_router.app.stream_proxy",
            AsyncMock(),
        ) as mock_stream_proxy, patch(
            "openai_router.app.non_stream_proxy",
            AsyncMock(return_value=expected_response),
        ) as mock_non_stream_proxy:
            actual_response = asyncio.run(endpoint("resp_123", request))

        self.assertIs(actual_response, expected_response)
        mock_non_stream_proxy.assert_awaited_once_with(
            "http://backend/v1/responses/resp_123",
            request,
            None,
            "backend-key",
        )
        mock_stream_proxy.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
