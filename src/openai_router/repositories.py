from collections.abc import Iterable
from datetime import datetime, timezone

from sqlmodel import Session, select

from openai_router.db import get_engine
from openai_router.models import (
    BackendSource,
    ModelAlias,
    ModelRoute,
    RouterSetting,
    SourceModelExclusion,
)


def list_model_names() -> list[str]:
    with Session(get_engine()) as session:
        statement = select(ModelRoute.model_name)
        return list(session.exec(statement).all())


def list_alias_names() -> list[str]:
    with Session(get_engine()) as session:
        statement = select(ModelAlias.alias_name)
        return list(session.exec(statement).all())


def list_routes() -> list[ModelRoute]:
    with Session(get_engine()) as session:
        statement = select(ModelRoute)
        return list(session.exec(statement).all())


def list_model_aliases() -> list[ModelAlias]:
    with Session(get_engine()) as session:
        statement = select(ModelAlias)
        return list(session.exec(statement).all())


def list_backend_sources() -> list[BackendSource]:
    with Session(get_engine()) as session:
        statement = select(BackendSource)
        return list(session.exec(statement).all())


def get_router_setting() -> RouterSetting | None:
    with Session(get_engine()) as session:
        return session.get(RouterSetting, 1)


def upsert_router_setting(routing_policy: str) -> RouterSetting:
    with Session(get_engine()) as session:
        setting = session.get(RouterSetting, 1)
        if setting is None:
            setting = RouterSetting(id=1, routing_policy=routing_policy)
        else:
            setting.routing_policy = routing_policy
            setting.updated = datetime.now(timezone.utc)

        session.add(setting)
        session.commit()
        session.refresh(setting)
        return setting


def list_routes_by_model(model_name: str) -> list[ModelRoute]:
    with Session(get_engine()) as session:
        statement = select(ModelRoute).where(ModelRoute.model_name == model_name)
        return list(session.exec(statement).all())


def list_aliases_by_model(model_name: str) -> list[ModelAlias]:
    with Session(get_engine()) as session:
        statement = select(ModelAlias).where(ModelAlias.model_name == model_name)
        return list(session.exec(statement).all())


def get_alias(alias_name: str) -> ModelAlias | None:
    with Session(get_engine()) as session:
        statement = select(ModelAlias).where(ModelAlias.alias_name == alias_name)
        return session.exec(statement).first()


def model_has_routes(model_name: str) -> bool:
    with Session(get_engine()) as session:
        statement = select(ModelRoute.id).where(ModelRoute.model_name == model_name)
        return session.exec(statement).first() is not None


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

        source = session.exec(
            select(BackendSource).where(BackendSource.model_url == model_url)
        ).first()
        if source is not None:
            _delete_source_model_exclusion(session, source.id, model_name)

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


def replace_model_aliases(model_name: str, alias_names: list[str]) -> tuple[int, int, int]:
    created_count = 0
    updated_count = 0
    deleted_count = 0
    desired_aliases = set(alias_names)

    with Session(get_engine()) as session:
        existing_aliases = {
            alias.alias_name: alias
            for alias in session.exec(
                select(ModelAlias).where(ModelAlias.model_name == model_name)
            ).all()
        }

        for alias_name in alias_names:
            alias = existing_aliases.get(alias_name)
            if alias is None:
                alias = session.exec(
                    select(ModelAlias).where(ModelAlias.alias_name == alias_name)
                ).first()

            if alias is None:
                session.add(ModelAlias(alias_name=alias_name, model_name=model_name))
                created_count += 1
                continue

            alias.model_name = model_name
            session.add(alias)
            updated_count += 1

        for alias_name, alias in existing_aliases.items():
            if alias_name in desired_aliases:
                continue
            session.delete(alias)
            deleted_count += 1

        session.commit()

    return created_count, updated_count, deleted_count


def list_excluded_model_names(source_id: int) -> list[str]:
    with Session(get_engine()) as session:
        statement = select(SourceModelExclusion.model_name).where(
            SourceModelExclusion.source_id == source_id
        )
        return list(session.exec(statement).all())


def replace_source_model_exclusions(source_id: int, model_names: list[str]) -> tuple[int, int]:
    created_count = 0
    deleted_count = 0
    desired_model_names = set(model_names)

    with Session(get_engine()) as session:
        existing_exclusions = {
            exclusion.model_name: exclusion
            for exclusion in session.exec(
                select(SourceModelExclusion).where(SourceModelExclusion.source_id == source_id)
            ).all()
        }

        for model_name in model_names:
            if model_name in existing_exclusions:
                continue
            session.add(SourceModelExclusion(source_id=source_id, model_name=model_name))
            created_count += 1

        for model_name, exclusion in existing_exclusions.items():
            if model_name in desired_model_names:
                continue
            session.delete(exclusion)
            deleted_count += 1

        session.commit()

    return created_count, deleted_count


def delete_backend_source(model_url: str) -> bool:
    with Session(get_engine()) as session:
        source = session.exec(
            select(BackendSource).where(BackendSource.model_url == model_url)
        ).first()
        if source is None:
            return False

        source_routes = session.exec(
            select(ModelRoute).where(ModelRoute.source_id == source.id)
        ).all()
        affected_model_names = {route.model_name for route in source_routes}
        for route in source_routes:
            session.delete(route)

        exclusions = session.exec(
            select(SourceModelExclusion).where(SourceModelExclusion.source_id == source.id)
        ).all()
        for exclusion in exclusions:
            session.delete(exclusion)

        session.delete(source)

        for model_name in affected_model_names:
            remaining_route = session.exec(
                select(ModelRoute).where(ModelRoute.model_name == model_name)
            ).first()
            if remaining_route is None:
                for alias in session.exec(
                    select(ModelAlias).where(ModelAlias.model_name == model_name)
                ).all():
                    session.delete(alias)

        session.commit()
        return True


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
        excluded_model_names = set(
            session.exec(
                select(SourceModelExclusion.model_name).where(
                    SourceModelExclusion.source_id == source_id
                )
            ).all()
        )
        desired_model_names = {
            model_name for model_name in model_names if model_name not in excluded_model_names
        }

        for model_name in model_names:
            if model_name in excluded_model_names:
                continue
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

            remaining_route = session.exec(
                select(ModelRoute).where(
                    ModelRoute.model_name == model_name,
                    ModelRoute.id != route.id,
                )
            ).first()
            if remaining_route is None:
                for alias in session.exec(
                    select(ModelAlias).where(ModelAlias.model_name == model_name)
                ).all():
                    session.delete(alias)

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
        remaining_route = session.exec(
            select(ModelRoute).where(
                ModelRoute.model_name == model_name,
                ModelRoute.id != route.id,
            )
        ).first()
        if remaining_route is None:
            for alias in session.exec(
                select(ModelAlias).where(ModelAlias.model_name == model_name)
            ).all():
                session.delete(alias)
        session.commit()
        return True


def _delete_source_model_exclusion(
    session: Session,
    source_id: int,
    model_name: str,
) -> None:
    exclusion = session.exec(
        select(SourceModelExclusion).where(
            SourceModelExclusion.source_id == source_id,
            SourceModelExclusion.model_name == model_name,
        )
    ).first()
    if exclusion is not None:
        session.delete(exclusion)


def unique_model_names(names: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(names))
