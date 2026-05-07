from datetime import datetime, timezone

from sqlmodel import Field, SQLModel


class ModelRoute(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    model_name: str = Field(index=True)
    model_url: str
    api_key: str | None = Field(default=None)
    created: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
