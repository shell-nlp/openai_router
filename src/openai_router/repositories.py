from collections.abc import Iterable

from sqlmodel import Session, select

from openai_router.db import get_engine
from openai_router.models import ModelRoute


def list_model_names() -> list[str]:
    with Session(get_engine()) as session:
        statement = select(ModelRoute.model_name)
        return list(session.exec(statement).all())


def list_routes() -> list[ModelRoute]:
    with Session(get_engine()) as session:
        statement = select(ModelRoute)
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

        session.add(route)
        session.commit()
        session.refresh(route)
        return created, route


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
