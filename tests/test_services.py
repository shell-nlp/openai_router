import unittest
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai_router.models import ModelRoute
from openai_router.services import RouteService


class RouteServiceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = RouteService()

    @patch("openai_router.services.repositories.list_model_names")
    @patch("openai_router.services.repositories.list_routes_by_model")
    def test_get_routing_target_round_robin(
        self,
        mock_list_routes_by_model,
        mock_list_model_names,
    ) -> None:
        mock_list_routes_by_model.return_value = [
            ModelRoute(model_name="gpt-4", model_url="http://backend-1/v1", api_key="key-1"),
            ModelRoute(model_name="gpt-4", model_url="http://backend-2/v1", api_key="key-2"),
        ]
        mock_list_model_names.return_value = ["gpt-4", "gpt-4"]

        first = self.service.get_routing_target("gpt-4")
        second = self.service.get_routing_target("gpt-4")

        self.assertEqual(first, ("http://backend-1/v1", ["gpt-4"], "key-1"))
        self.assertEqual(second, ("http://backend-2/v1", ["gpt-4"], "key-2"))

    @patch("openai_router.services.repositories.list_routes")
    def test_build_models_response_uses_earliest_timestamp(self, mock_list_routes) -> None:
        mock_list_routes.return_value = [
            ModelRoute(
                model_name="gpt-4",
                model_url="http://backend-2/v1",
                created=datetime(2025, 1, 2, tzinfo=timezone.utc),
            ),
            ModelRoute(
                model_name="gpt-4",
                model_url="http://backend-1/v1",
                created=datetime(2025, 1, 1, tzinfo=timezone.utc),
            ),
            ModelRoute(
                model_name="text-embedding-3-large",
                model_url="http://embedding/v1",
                created=datetime(2025, 1, 3, tzinfo=timezone.utc),
            ),
        ]

        response = self.service.build_models_response()

        self.assertEqual(response["object"], "list")
        self.assertEqual([item["id"] for item in response["data"]], ["gpt-4", "text-embedding-3-large"])
        self.assertEqual(response["data"][0]["created"], 1735689600)

    def test_add_route_normalizes_backend_url(self) -> None:
        normalized = self.service._normalize_backend_url(" http://localhost:8000/v1/ ")
        self.assertEqual(normalized, "http://localhost:8000/v1")


if __name__ == "__main__":
    unittest.main()
