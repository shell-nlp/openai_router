from collections.abc import Iterable
from datetime import datetime, timezone

from sqlmodel import Session, select

from openai_router.db import get_engine
from openai_router.models import BackendSource, ModelRoute


def list_model_names() -> list[str]:
    with Session(get_engine()) as session:
        statement = select(ModelRoute.model_name)
        return list(session.exec(statement).all())


def list_routes() -> list[ModelRoute]:
    with Session(get_engine()) as session:
        statement = select(ModelRoute)
        return list(session.exec(statement).all())


def list_backend_sources() -> list[BackendSource]:
    with Session(get_engine()) as session:
        statement = select(BackendSource)
        return list(session.exec(statement).all())


def list_routes_by_model(model_name: str) -> list[ModelRoute]:
    with Session(get_engine()) as session:
        statement = select(ModelRoute).where(ModelRoute.model_name == model_name)
        return list(session.exec(statement).all())


def find_route(model_name: str, model_url: str) -> ModelRoute | None:
    with Session(get_engine()) as session:
        statement = select(ModelRoute).where(
            ModelRoute.model_name == model_name,
            ModelRoute.model_url == model_url,
        )
        return session.exec(statement).first()


def save_route(route: ModelRoute) -> None:
    with Session(get_engine()) as session:
        session.add(route)
        session.commit()


def get_backend_source_by_url(model_url: str) -> BackendSource | None:
    with Session(get_engine()) as session:
        statement = select(BackendSource).where(BackendSource.model_url == model_url)
        return session.exec(statement).first()


def upsert_backend_source(
    model_url: str,
    api_key: str | None,
    sync_interval_minutes: int,
) -> tuple[bool, BackendSource]:
    now = datetime.now(timezone.utc)
    with Session(get_engine()) as session:
        statement = select(BackendSource).where(BackendSource.model_url == model_url)
        source = session.exec(statement).first()
        created = source is None

        if source is None:
            source = BackendSource(
                model_url=model_url,
                api_key=api_key,
                sync_interval_minutes=sync_interval_minutes,
            )
        else:
            source.api_key = api_key
            source.sync_interval_minutes = sync_interval_minutes
            source.updated = now

        session.add(source)
        session.commit()
        session.refresh(source)
        return created, source


def update_backend_source_sync_status(
    source_id: int,
    last_synced_at: datetime | None,
    last_sync_error: str | None,
) -> None:
    with Session(get_engine()) as session:
        source = session.get(BackendSource, source_id)
        if source is None:
            return

        source.last_synced_at = last_synced_at
        source.last_sync_error = last_sync_error
        source.updated = datetime.now(timezone.utc)
        session.add(source)
        session.commit()


def upsert_route(model_name: str, model_url: str, api_key: str | None) -> tuple[bool, ModelRoute]:
    with Session(get_engine()) as session:
        statement = select(ModelRoute).where(
            ModelRoute.model_name == model_name,
            ModelRoute.model_url == model_url,
        )
        route = session.exec(statement).first()
        created = route is None

        if route is None:
            route = ModelRoute(model_name=model_name, model_url=model_url, api_key=api_key)
        else:
            route.api_key = api_key
            route.auto_managed = False
            route.source_id = None

        session.add(route)
        session.commit()
        session.refresh(route)
        return created, route


def bulk_upsert_routes(
    model_names: list[str],
    model_url: str,
    api_key: str | None,
) -> tuple[int, int]:
    created_count = 0
    updated_count = 0

    with Session(get_engine()) as session:
        for model_name in model_names:
            statement = select(ModelRoute).where(
                ModelRoute.model_name == model_name,
                ModelRoute.model_url == model_url,
            )
            route = session.exec(statement).first()

            if route is None:
                route = ModelRoute(
                    model_name=model_name,
                    model_url=model_url,
                    api_key=api_key,
                )
                created_count += 1
            else:
                route.api_key = api_key
                updated_count += 1

            session.add(route)

        session.commit()

    return created_count, updated_count


def sync_auto_managed_routes(
    source_id: int,
    model_names: list[str],
    model_url: str,
    api_key: str | None,
) -> tuple[int, int, int]:
    created_count = 0
    updated_count = 0
    deleted_count = 0
    now = datetime.now(timezone.utc)

    with Session(get_engine()) as session:
        source_routes_statement = select(ModelRoute).where(ModelRoute.source_id == source_id)
        existing_routes = {
            route.model_name: route
            for route in session.exec(source_routes_statement).all()
            if route.auto_managed
        }
        backend_routes_statement = select(ModelRoute).where(ModelRoute.model_url == model_url)
        backend_routes = {
            route.model_name: route
            for route in session.exec(backend_routes_statement).all()
        }
        desired_model_names = set(model_names)

        for model_name in model_names:
            route = existing_routes.get(model_name) or backend_routes.get(model_name)
            if route is None:
                route = ModelRoute(
                    model_name=model_name,
                    model_url=model_url,
                    api_key=api_key,
                    auto_managed=True,
                    source_id=source_id,
                    created=now,
                )
                created_count += 1
            else:
                updated_count += 1
                route.model_url = model_url
                route.api_key = api_key
                route.auto_managed = True
                route.source_id = source_id

            session.add(route)

        for model_name, route in existing_routes.items():
            if model_name in desired_model_names:
                continue
            session.delete(route)
            deleted_count += 1

        session.commit()

    return created_count, updated_count, deleted_count


def delete_route(model_name: str, model_url: str) -> bool:
    with Session(get_engine()) as session:
        statement = select(ModelRoute).where(
            ModelRoute.model_name == model_name,
            ModelRoute.model_url == model_url,
        )
        route = session.exec(statement).first()
        if route is None:
            return False

        session.delete(route)
        session.commit()
        return True


def unique_model_names(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(names))
