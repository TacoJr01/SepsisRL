"""FastAPI application entry point for the Sepsis environment."""

from openenv.core.env_server import create_fastapi_app

from ..models import SepsisAction, SepsisObservation
from .sepsis_environment import SepsisEnvironment

app = create_fastapi_app(
    env=SepsisEnvironment,
    action_cls=SepsisAction,
    observation_cls=SepsisObservation,
)


def main() -> None:
    """Run the OpenEnv server via Uvicorn."""
    import uvicorn

    uvicorn.run(
        "sepsis_env.server.app:app",
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    main()
