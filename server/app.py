"""FastAPI application entry point for the Sepsis environment."""

from fastapi.responses import HTMLResponse

from openenv.core.env_server import create_fastapi_app

from models import SepsisAction, SepsisObservation
from server.sepsis_environment import SepsisEnvironment

app = create_fastapi_app(
    env=SepsisEnvironment,
    action_cls=SepsisAction,
    observation_cls=SepsisObservation,
)


_HOME_HTML = (
    "<!doctype html>"
    "<html lang=\"en\">"
    "<head><meta charset=\"utf-8\"><title>Sepsis OpenEnv</title></head>"
    "<body style=\"font-family:Arial,sans-serif;padding:24px\">"
    "<h1>Sepsis OpenEnv</h1>"
    "<p>Environment server is running.</p>"
    "<button onclick=\"location.href='/docs'\">Open API Docs</button>"
    "</body></html>"
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def root() -> str:
    return _HOME_HTML


@app.get("/home", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    return _HOME_HTML


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Run the OpenEnv server via Uvicorn."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
