"""FastAPI inference service.

Two rules shape this module. A modality with no verified artifact returns 503
rather than a guess, and every response says which model answered. Both are
direct reactions to v1 shipping predictions from randomly initialised weights.

Unimplemented modalities are declared 501 in the OpenAPI document rather than
omitted, so the contract states what is missing instead of hiding it.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Request, Response
from fastapi.responses import JSONResponse

from sentimentsphere import __version__
from sentimentsphere.core.config import Settings
from sentimentsphere.core.types import Prediction
from sentimentsphere.inference.artifacts import ModelUnavailableError
from sentimentsphere.inference.predictors import PredictorRegistry
from sentimentsphere.serving.schemas import (
    ErrorResponse,
    HealthResponse,
    ModelsResponse,
    PredictionResponse,
    TextRequest,
)

logger = logging.getLogger("sentimentsphere.serving")

ERRORS: dict[int | str, dict[str, Any]] = {
    422: {"model": ErrorResponse, "description": "Request failed validation"},
    429: {"model": ErrorResponse, "description": "Rate limit exceeded"},
    503: {"model": ErrorResponse, "description": "No verified model installed"},
}


class RateLimiter:
    """Fixed-window-free sliding limiter, per client address.

    ponytail: in-process and single-worker. Put a real limiter in the proxy
    before running more than one worker.
    """

    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}

    def allow(self, client: str) -> bool:
        now = time.monotonic()
        seen = self._hits.setdefault(client, deque())
        while seen and now - seen[0] > 60.0:
            seen.popleft()
        if len(seen) >= self.per_minute:
            return False
        seen.append(now)
        return True


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load and warm the models at startup, so no user request pays for init."""
    settings: Settings = app.state.settings
    app.state.registry = PredictorRegistry(settings)
    app.state.limiter = RateLimiter(settings.requests_per_minute)
    try:
        app.state.registry.get("text").predict("warm up")
        logger.info("text model warmed")
    except (ModelUnavailableError, ValueError) as error:
        logger.warning("text model unavailable at startup: %s", error)
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    app = FastAPI(
        title="SentimentSphere",
        version=__version__,
        description="Calibrated multimodal emotion recognition over a fixed seven-label space.",
        lifespan=lifespan,
    )
    app.state.settings = settings or Settings()

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        client = request.client.host if request.client else "unknown"
        if request.url.path.startswith("/v1/predict") and not app.state.limiter.allow(client):
            return error_response(429, "Rate limit exceeded", "rate_limited", request_id)
        try:
            response = await call_next(request)
        except Exception:
            logger.exception("unhandled error", extra={"request_id": request_id})
            return error_response(500, "Internal error", "internal_error", request_id)
        elapsed = (time.perf_counter() - started) * 1000
        response.headers["x-request-id"] = request_id
        logger.info(
            "%s %s -> %s in %.1fms",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
            extra={"request_id": request_id},
        )
        return response

    register_routes(app)
    return app


def error_response(status: int, detail: str, code: str, request_id: str) -> JSONResponse:
    payload = ErrorResponse(detail=detail, code=code, request_id=request_id)
    return JSONResponse(
        status_code=status,
        content=payload.model_dump(),
        headers={"x-request-id": request_id},
    )


def get_registry(request: Request) -> PredictorRegistry:
    registry: PredictorRegistry = request.app.state.registry
    return registry


def to_response(prediction: Prediction, request_id: str) -> PredictionResponse:
    return PredictionResponse(**prediction.model_dump(), request_id=request_id)


def register_routes(app: FastAPI) -> None:
    @app.get("/healthz", response_model=HealthResponse, tags=["ops"])
    def healthz() -> HealthResponse:
        """Liveness only; model availability is reported by /v1/models."""
        return HealthResponse(status="ok", version=__version__)

    @app.get("/v1/models", response_model=ModelsResponse, tags=["ops"])
    def models(registry: PredictorRegistry = Depends(get_registry)) -> ModelsResponse:
        return ModelsResponse(models=registry.describe())

    @app.post(
        "/v1/predict/text",
        response_model=PredictionResponse,
        responses=ERRORS,
        tags=["predict"],
    )
    def predict_text(
        body: TextRequest,
        request: Request,
        registry: PredictorRegistry = Depends(get_registry),
    ) -> Response:
        request_id = request.state.request_id
        settings: Settings = request.app.state.settings
        if len(body.text) > settings.max_text_chars:
            return error_response(
                422,
                f"Text exceeds {settings.max_text_chars} characters",
                "text_too_long",
                request_id,
            )
        try:
            prediction = registry.get("text").predict(body.text)
        except ModelUnavailableError as error:
            return error_response(503, str(error), "model_unavailable", request_id)
        except ValueError as error:
            return error_response(422, str(error), "invalid_input", request_id)
        return JSONResponse(content=to_response(prediction, request_id).model_dump())

    for modality in ("audio", "image", "video"):

        @app.post(
            f"/v1/predict/{modality}",
            responses={501: {"model": ErrorResponse, "description": "Not implemented yet"}},
            tags=["predict"],
        )
        def predict_pending(request: Request, name: str = modality) -> Response:
            """Declared but not implemented: no {name} head has been trained yet."""
            return error_response(
                501,
                f"The {name} modality is not implemented; no trained head exists",
                "not_implemented",
                request.state.request_id,
            )


app = create_app()
