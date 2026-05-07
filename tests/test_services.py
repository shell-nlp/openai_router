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
    @patch("openai_router.services.repositories.list_alias_names")
    @patch("openai_router.services.repositories.get_router_setting")
    @patch("openai_router.services.repositories.get_alias")
    @patch("openai_router.services.repositories.list_routes_by_model")
    def test_get_routing_target_round_robin(
        self,
        mock_list_routes_by_model,
        mock_get_alias,
        mock_get_router_setting,
        mock_list_alias_names,
        mock_list_model_names,
    ) -> None:
        mock_list_routes_by_model.return_value = [
            ModelRoute(model_name="gpt-4", model_url="http://backend-1/v1", api_key="key-1"),
            ModelRoute(model_name="gpt-4", model_url="http://backend-2/v1", api_key="key-2"),
        ]
        mock_get_alias.return_value = None
        mock_get_router_setting.return_value = None
        mock_list_alias_names.return_value = []
        mock_list_model_names.return_value = ["gpt-4", "gpt-4"]

        first = self.service.get_routing_target("gpt-4")
        second = self.service.get_routing_target("gpt-4")

        self.assertEqual(first, ("http://backend-1/v1", ["gpt-4"], "key-1", "gpt-4"))
        self.assertEqual(second, ("http://backend-2/v1", ["gpt-4"], "key-2", "gpt-4"))

    @patch("openai_router.services.repositories.list_model_names")
    @patch("openai_router.services.repositories.list_alias_names")
    @patch("openai_router.services.repositories.get_router_setting")
    @patch("openai_router.services.repositories.get_alias")
    @patch("openai_router.services.repositories.list_routes_by_model")
    def test_get_routing_target_resolves_alias(
        self,
        mock_list_routes_by_model,
        mock_get_alias,
        mock_get_router_setting,
        mock_list_alias_names,
        mock_list_model_names,
    ) -> None:
        mock_list_routes_by_model.side_effect = [
            [],
            [ModelRoute(model_name="gpt-4", model_url="http://backend-1/v1", api_key="key-1")],
        ]
        mock_get_alias.return_value = type("Alias", (), {"model_name": "gpt-4"})()
        mock_get_router_setting.return_value = None
        mock_list_alias_names.return_value = ["gpt-4o-latest"]
        mock_list_model_names.return_value = ["gpt-4"]

        resolved = self.service.get_routing_target("gpt-4o-latest")

        self.assertEqual(
            resolved,
            ("http://backend-1/v1", ["gpt-4", "gpt-4o-latest"], "key-1", "gpt-4"),
        )

    @patch("openai_router.services.repositories.list_model_names")
    @patch("openai_router.services.repositories.list_alias_names")
    @patch("openai_router.services.repositories.get_router_setting")
    @patch("openai_router.services.repositories.get_alias")
    @patch("openai_router.services.repositories.list_routes_by_model")
    def test_get_routing_target_uses_consistent_hash_for_same_session(
        self,
        mock_list_routes_by_model,
        mock_get_alias,
        mock_get_router_setting,
        mock_list_alias_names,
        mock_list_model_names,
    ) -> None:
        mock_list_routes_by_model.return_value = [
            ModelRoute(model_name="gpt-4", model_url="http://backend-1/v1", api_key="key-1"),
            ModelRoute(model_name="gpt-4", model_url="http://backend-2/v1", api_key="key-2"),
            ModelRoute(model_name="gpt-4", model_url="http://backend-3/v1", api_key="key-3"),
        ]
        mock_get_alias.return_value = None
        mock_get_router_setting.return_value = type(
            "Setting",
            (),
            {"routing_policy": "consistent_hash"},
        )()
        mock_list_alias_names.return_value = []
        mock_list_model_names.return_value = ["gpt-4"]

        first = self.service.get_routing_target(
            "gpt-4",
            {"model": "gpt-4", "messages": [], "user": "user-1"},
            {"x-session-id": "session-123"},
        )
        second = self.service.get_routing_target(
            "gpt-4",
            {"model": "gpt-4", "messages": [], "user": "user-2"},
            {"x-session-id": "session-123"},
        )
        third = self.service.get_routing_target(
            "gpt-4",
            {"model": "gpt-4", "messages": [], "user": "user-3"},
            {"x-session-id": "session-456"},
        )

        self.assertEqual(first[0], second[0])
        self.assertEqual(first[2], second[2])
        self.assertIn(third[0], {first[0], "http://backend-2/v1", "http://backend-3/v1"})

    def test_extract_hash_key_prefers_headers_over_body(self) -> None:
        hash_key = self.service._extract_hash_key(
            {
                "model": "gpt-4",
                "user": "body-user",
                "session_params": {"session_id": "body-session"},
            },
            {"x-user-id": "header-user"},
        )

        self.assertEqual(hash_key, "header:x-user-id:header-user")

    def test_extract_hash_key_uses_nested_session_id_before_user(self) -> None:
        hash_key = self.service._extract_hash_key(
            {
                "model": "gpt-4",
                "user": "body-user",
                "session_params": {"session_id": "body-session"},
            },
            None,
        )

        self.assertEqual(hash_key, "session:body-session")

    @patch("openai_router.services.repositories.list_routes")
    @patch("openai_router.services.repositories.list_model_aliases")
    def test_build_models_response_uses_earliest_timestamp(
        self,
        mock_list_model_aliases,
        mock_list_routes,
    ) -> None:
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
        mock_list_model_aliases.return_value = [
            type(
                "Alias",
                (),
                {
                    "alias_name": "gpt-4o-latest",
                    "model_name": "gpt-4",
                    "created": datetime(2025, 1, 4, tzinfo=timezone.utc),
                },
            )()
        ]

        response = self.service.build_models_response()

        self.assertEqual(response["object"], "list")
        self.assertEqual(
            [item["id"] for item in response["data"]],
            ["gpt-4", "gpt-4o-latest", "text-embedding-3-large"],
        )
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
    @patch("openai_router.services.repositories.replace_source_model_exclusions")
    @patch("openai_router.services.repositories.upsert_backend_source")
    def test_add_or_update_backend_source_syncs_models(
        self,
        mock_upsert_backend_source,
        mock_replace_source_model_exclusions,
        mock_sync_backend_source,
    ) -> None:
        source = BackendSource(
            id=1,
            model_url="http://localhost:8000/v1",
            api_key="sk-test",
            sync_interval_minutes=15,
        )
        mock_upsert_backend_source.return_value = (True, source)
        mock_replace_source_model_exclusions.return_value = (2, 0)
        mock_sync_backend_source.return_value = {
            "discovered": 2,
            "created": 2,
            "updated": 0,
            "deleted": 0,
        }

        message = self.service.add_or_update_backend_source(
            "http://localhost:8000/v1/",
            "sk-test",
            "unknown, test-model",
            15,
        )

        mock_upsert_backend_source.assert_called_once_with(
            "http://localhost:8000/v1",
            "sk-test",
            15,
        )
        mock_replace_source_model_exclusions.assert_called_once_with(
            1,
            ["unknown", "test-model"],
        )
        mock_sync_backend_source.assert_called_once_with(source)
        self.assertIn("同步间隔 15 分钟", message)
        self.assertIn("排除模型 2 个", message)
        self.assertIn("本次同步发现 2 个模型", message)

    @patch("openai_router.services.repositories.replace_model_aliases")
    @patch("openai_router.services.repositories.upsert_route")
    @patch("openai_router.services.repositories.get_alias")
    @patch("openai_router.services.repositories.model_has_routes")
    def test_add_or_update_route_saves_aliases(
        self,
        mock_model_has_routes,
        mock_get_alias,
        mock_upsert_route,
        mock_replace_model_aliases,
    ) -> None:
        mock_model_has_routes.return_value = False
        mock_get_alias.return_value = None
        mock_upsert_route.return_value = (
            True,
            ModelRoute(model_name="gpt-4", model_url="http://backend-1/v1", api_key="key-1"),
        )
        mock_replace_model_aliases.return_value = (2, 0, 0)

        message = self.service.add_or_update_route(
            "gpt-4",
            "gpt-4o-latest, my-gpt4",
            "http://backend-1/v1",
            "key-1",
        )

        mock_replace_model_aliases.assert_called_once_with(
            "gpt-4",
            ["gpt-4o-latest", "my-gpt4"],
        )
        self.assertIn("别名 2 个已同步", message)

    @patch("openai_router.services.repositories.delete_backend_source")
    def test_delete_backend_source_returns_cleanup_message(
        self,
        mock_delete_backend_source,
    ) -> None:
        mock_delete_backend_source.return_value = True

        message = self.service.delete_backend_source("http://backend/v1/")

        mock_delete_backend_source.assert_called_once_with("http://backend/v1")
        self.assertIn("自动同步生成的路由", message)

    @patch("openai_router.services.repositories.get_alias")
    @patch("openai_router.services.repositories.model_has_routes")
    def test_validate_aliases_rejects_real_model_name(
        self,
        mock_model_has_routes,
        mock_get_alias,
    ) -> None:
        mock_model_has_routes.return_value = True
        mock_get_alias.return_value = None

        with self.assertRaisesRegex(ValueError, "已经是一个真实模型名"):
            self.service.add_or_update_route(
                "gpt-4",
                "text-embedding-3-large",
                "http://backend-1/v1",
                "key-1",
            )

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

    def test_source_due_check_supports_naive_last_sync_time(self) -> None:
        now = datetime(2026, 5, 7, 5, 34, 58, tzinfo=timezone.utc)
        source = BackendSource(
            id=1,
            model_url="http://localhost:8000/v1",
            sync_interval_minutes=3,
            last_synced_at=datetime(2026, 5, 7, 5, 31, 52),
        )

        self.assertTrue(self.service._is_source_due_for_sync(source, now))


if __name__ == "__main__":
    unittest.main()
