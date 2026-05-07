from openai_router.app import app
from openai_router.cli import cli_app, main


__all__ = ["app", "cli_app", "main"]


if __name__ == "__main__":
    cli_app()
