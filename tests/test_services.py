import unittest
from datetime import datetime, timezone
from pathlib import Path
import sys
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai_router.models import BackendSource, ModelRoute
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

    def test_build_model_discovery_urls_for_v1_backend(self) -> None:
        urls = self.service._build_model_discovery_urls("http://localhost:8000/v1")
        self.assertEqual(urls, ["http://localhost:8000/v1/models"])

    def test_build_model_discovery_urls_for_root_backend(self) -> None:
        urls = self.service._build_model_discovery_urls("http://localhost:8000")
        self.assertEqual(
            urls,
            ["http://localhost:8000/v1/models", "http://localhost:8000/models"],
        )

    def test_parse_model_names_deduplicates_results(self) -> None:
        payload = {
            "data": [
                {"id": "gpt-4"},
                {"id": "gpt-4"},
                {"id": "text-embedding-3-large"},
                {"id": ""},
            ]
        }

        models = self.service._parse_model_names(payload, "http://localhost:8000/v1/models")

        self.assertEqual(models, ["gpt-4", "text-embedding-3-large"])

    @patch.object(RouteService, "sync_backend_source")
    @patch("openai_router.services.repositories.upsert_backend_source")
    def test_add_or_update_route_auto_discovers_models(
        self,
        mock_upsert_backend_source,
        mock_sync_backend_source,
    ) -> None:
        source = BackendSource(
            id=1,
            model_url="http://localhost:8000/v1",
            api_key="sk-test",
            sync_interval_minutes=15,
        )
        mock_upsert_backend_source.return_value = (True, source)
        mock_sync_backend_source.return_value = {
            "discovered": 2,
            "created": 2,
            "updated": 0,
            "deleted": 0,
        }

        message = self.service.add_or_update_route(
            "",
            "http://localhost:8000/v1/",
            "sk-test",
            auto_discover_models=True,
        )

        mock_upsert_backend_source.assert_called_once_with(
            "http://localhost:8000/v1",
            "sk-test",
            15,
        )
        mock_sync_backend_source.assert_called_once_with(source)
        self.assertIn("同步间隔 15 分钟", message)
        self.assertIn("本次同步发现 2 个模型", message)

    def test_source_without_last_sync_is_due(self) -> None:
        source = BackendSource(
            id=1,
            model_url="http://localhost:8000/v1",
            sync_interval_minutes=15,
            last_synced_at=None,
        )

        self.assertTrue(self.service._is_source_due_for_sync(source, datetime.now(timezone.utc)))

    def test_source_not_due_before_interval(self) -> None:
        now = datetime.now(timezone.utc)
        source = BackendSource(
            id=1,
            model_url="http://localhost:8000/v1",
            sync_interval_minutes=15,
            last_synced_at=now,
        )

        self.assertFalse(self.service._is_source_due_for_sync(source, now))


if __name__ == "__main__":
    unittest.main()
