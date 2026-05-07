from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class BackendSource(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    model_url: str = Field(index=True, unique=True)
    api_key: str | None = Field(default=None)
    sync_interval_minutes: int = Field(default=15, nullable=False)
    last_synced_at: datetime | None = Field(default=None, nullable=True)
    last_sync_error: str | None = Field(default=None)
    created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ModelRoute(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    model_name: str = Field(index=True)
    model_url: str
    api_key: str | None = Field(default=None)
    auto_managed: bool = Field(default=False, nullable=False)
    source_id: int | None = Field(default=None, foreign_key="backendsource.id", index=True)
    created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class ModelAlias(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    alias_name: str = Field(index=True, unique=True)
    model_name: str = Field(index=True)
    created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class RouterSetting(SQLModel, table=True):
    id: int | None = Field(default=1, primary_key=True)
    routing_policy: str = Field(default="round_robin", nullable=False)
    updated: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class SourceModelExclusion(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    source_id: int = Field(foreign_key="backendsource.id", index=True)
    model_name: str = Field(index=True)
    created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
