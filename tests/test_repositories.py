import tempfile
import unittest
from pathlib import Path
import sys

from sqlmodel import SQLModel, Session, create_engine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openai_router import repositories
from openai_router.models import BackendSource, ModelAlias, ModelRoute
from openai_router.runtime import runtime_state


class RepositorySyncExclusionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "test.db"
        self.original_engine = runtime_state.engine
        runtime_state.engine = create_engine(f"sqlite:///{self.db_path}", echo=False)
        SQLModel.metadata.create_all(runtime_state.engine)

    def tearDown(self) -> None:
        if runtime_state.engine is not None:
            runtime_state.engine.dispose()
        runtime_state.engine = self.original_engine
        self.temp_dir.cleanup()

    def test_deleted_auto_managed_model_is_recreated_by_future_sync(self) -> None:
        with Session(runtime_state.engine) as session:
            source = BackendSource(model_url="http://backend/v1", sync_interval_minutes=15)
            session.add(source)
            session.commit()
            session.refresh(source)
            source_id = source.id

            route = ModelRoute(
                model_name="unknown",
                model_url="http://backend/v1",
                auto_managed=True,
                source_id=source_id,
            )
            session.add(route)
            session.commit()

        deleted = repositories.delete_route("unknown", "http://backend/v1")
        self.assertTrue(deleted)
        self.assertEqual(repositories.list_excluded_model_names(source_id), [])

        created_count, updated_count, deleted_count = repositories.sync_auto_managed_routes(
            source_id,
            ["unknown"],
            "http://backend/v1",
            None,
        )

        self.assertEqual((created_count, updated_count, deleted_count), (1, 0, 0))
        routes = repositories.list_routes_by_model("unknown")
        self.assertEqual(len(routes), 1)
        self.assertTrue(routes[0].auto_managed)

    def test_manual_readd_clears_exclusion(self) -> None:
        with Session(runtime_state.engine) as session:
            source = BackendSource(model_url="http://backend/v1", sync_interval_minutes=15)
            session.add(source)
            session.commit()
            session.refresh(source)
            source_id = source.id

            route = ModelRoute(
                model_name="unknown",
                model_url="http://backend/v1",
                auto_managed=True,
                source_id=source_id,
            )
            session.add(route)
            session.commit()

        repositories.replace_source_model_exclusions(source_id, ["unknown"])
        self.assertEqual(repositories.list_excluded_model_names(source_id), ["unknown"])

        repositories.upsert_route("unknown", "http://backend/v1", None)

        self.assertEqual(repositories.list_excluded_model_names(source_id), [])

    def test_upsert_route_persists_request_param_mapping(self) -> None:
        repositories.upsert_route(
            "mapped-model",
            "http://backend/v1",
            "backend-key",
            '{"enable_thinking":"chat_template_kwargs.enable_thinking"}',
        )

        routes = repositories.list_routes_by_model("mapped-model")

        self.assertEqual(len(routes), 1)
        self.assertEqual(
            routes[0].request_param_mapping,
            '{"enable_thinking":"chat_template_kwargs.enable_thinking"}',
        )

    def test_delete_backend_source_removes_auto_managed_routes_only(self) -> None:
        with Session(runtime_state.engine) as session:
            source = BackendSource(model_url="http://backend/v1", sync_interval_minutes=15)
            session.add(source)
            session.commit()
            session.refresh(source)
            source_id = source.id

            session.add(
                ModelRoute(
                    model_name="auto-model",
                    model_url="http://backend/v1",
                    auto_managed=True,
                    source_id=source_id,
                )
            )
            session.add(
                ModelRoute(
                    model_name="manual-model",
                    model_url="http://backend/v1",
                    auto_managed=False,
                    source_id=None,
                )
            )
            session.add(ModelAlias(alias_name="auto-alias", model_name="auto-model"))
            session.commit()

        deleted = repositories.delete_backend_source("http://backend/v1")

        self.assertTrue(deleted)
        self.assertIsNone(repositories.get_backend_source_by_url("http://backend/v1"))
        self.assertEqual(repositories.list_routes_by_model("auto-model"), [])
        self.assertEqual(repositories.list_aliases_by_model("auto-model"), [])
        self.assertEqual(len(repositories.list_routes_by_model("manual-model")), 1)

    def test_replace_source_model_exclusions_replaces_existing_values(self) -> None:
        with Session(runtime_state.engine) as session:
            source = BackendSource(model_url="http://backend/v1", sync_interval_minutes=15)
            session.add(source)
            session.commit()
            session.refresh(source)
            source_id = source.id

        repositories.replace_source_model_exclusions(source_id, ["unknown", "old-model"])
        created_count, deleted_count = repositories.replace_source_model_exclusions(
            source_id,
            ["unknown", "new-model"],
        )

        self.assertEqual((created_count, deleted_count), (1, 1))
        self.assertEqual(
            sorted(repositories.list_excluded_model_names(source_id)),
            ["new-model", "unknown"],
        )


if __name__ == "__main__":
    unittest.main()
