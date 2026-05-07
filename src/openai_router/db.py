import os

from loguru import logger
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlmodel import SQLModel, create_engine

from openai_router.config import SQLITE_DB_FILE, SQLITE_URL
from openai_router.models import BackendSource, ModelAlias, ModelRoute
from openai_router.runtime import runtime_state


def get_engine() -> Engine:
    if runtime_state.engine is None:
        raise RuntimeError("Database engine is not initialized.")
    return runtime_state.engine


def initialize_engine() -> Engine:
    engine = _rebuild_database_if_schema_changed(create_engine(SQLITE_URL, echo=False))
    runtime_state.engine = engine
    return engine


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(get_engine())


def dispose_engine() -> None:
    if runtime_state.engine is not None:
        runtime_state.engine.dispose()
        runtime_state.engine = None
        logger.info("Database engine disposed.")


def _rebuild_database_if_schema_changed(engine: Engine) -> Engine:
    try:
        inspector = inspect(engine)
        if not inspector.has_table(ModelRoute.__tablename__):
            logger.info("Database or table '{}' not found. A new database will be created.", ModelRoute.__tablename__)
            return engine

        for table_model in (ModelRoute, BackendSource, ModelAlias):
            if not inspector.has_table(table_model.__tablename__):
                continue

            actual_columns = {
                column["name"] for column in inspector.get_columns(table_model.__tablename__)
            }
            expected_columns = {column.name for column in table_model.__table__.columns}

            if actual_columns == expected_columns:
                continue

            logger.warning(
                "Database schema mismatch detected for {}. expected={}, actual={}",
                table_model.__tablename__,
                expected_columns,
                actual_columns,
            )
            logger.warning("Removing outdated database file {} to rebuild.", SQLITE_DB_FILE)

            engine.dispose()
            try:
                os.remove(SQLITE_DB_FILE)
                logger.info("Removed outdated database: {}", SQLITE_DB_FILE)
            except OSError as exc:
                logger.error("Failed to remove database {}: {}", SQLITE_DB_FILE, exc)

            return create_engine(SQLITE_URL, echo=False)
    except (OperationalError, DBAPIError):
        logger.info("Database or table 'modelroute' not found. A new database will be created.")
    except Exception as exc:
        logger.exception("Unexpected error while checking database schema: {}", exc)
    return engine
