"""FastAPI serving layer.

Exposes the champion model over HTTP for the UI and any other consumer. The
model and the match history are loaded once at startup and refreshed through
``POST /reload``, so a newly promoted model can be picked up without a restart.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from padel_predictor.common.config import get_pipeline_config, get_settings
from padel_predictor.common.logging import get_logger
from padel_predictor.inference.predict import (
    ModelNotAvailableError,
    Predictor,
    known_players,
    load_history,
)

logger = get_logger(__name__)


class _State:
    """Process-wide model and history, refreshable at runtime."""

    predictor: Predictor | None = None
    history: pd.DataFrame | None = None
    error: str | None = None

    @classmethod
    def load(cls) -> None:
        try:
            cls.predictor = Predictor.load(get_settings(), get_pipeline_config())
            cls.history = load_history(get_settings())
            cls.error = None
            logger.info("api ready with model version %s", cls.predictor.model_version)
        except (ModelNotAvailableError, RuntimeError) as exc:
            # Start anyway: /health must be able to report *why* the app is not ready.
            cls.predictor, cls.history, cls.error = None, None, str(exc)
            logger.error("api started without a model: %s", exc)

    @classmethod
    def require(cls) -> tuple[Predictor, pd.DataFrame]:
        if cls.predictor is None or cls.history is None:
            raise HTTPException(status_code=503, detail=cls.error or "model not loaded")
        return cls.predictor, cls.history


@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001 -- FastAPI passes the app
    _State.load()
    yield


app = FastAPI(
    title="Padel Predictor",
    description="Predicts the winner of a professional padel match.",
    version="0.1.0",
    lifespan=lifespan,
)


class MatchupRequest(BaseModel):
    """Two teams of two players each."""

    team_a: list[str] = Field(min_length=2, max_length=2)
    team_b: list[str] = Field(min_length=2, max_length=2)
    tier: str = "p1"

    @field_validator("team_a", "team_b")
    @classmethod
    def _non_empty(cls, players: list[str]) -> list[str]:
        cleaned = [p.strip() for p in players]
        if any(not p for p in cleaned):
            raise ValueError("player names must not be empty")
        return cleaned


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness and readiness, including the deployed model version."""
    ready = _State.predictor is not None
    return {
        "status": "ok" if ready else "degraded",
        "model_loaded": ready,
        "model_name": _State.predictor.handle.name if ready else None,
        "model_version": _State.predictor.model_version if ready else None,
        "matches_in_history": 0 if _State.history is None else len(_State.history),
        "detail": _State.error,
    }


@app.get("/players")
def players() -> dict[str, Any]:
    """Players present in the match history, for populating the UI."""
    _, history = _State.require()
    return {"players": known_players(history)}


@app.get("/upcoming")
def upcoming() -> dict[str, Any]:
    """Predictions for the scheduled matches inside the configured horizon."""
    predictor, history = _State.require()
    frame = predictor.predict_upcoming(history)
    columns = [
        "match_id",
        "played_at",
        "tournament",
        "tier",
        "round",
        "team_a_player_1",
        "team_a_player_2",
        "team_b_player_1",
        "team_b_player_2",
        "probability_team_a",
        "predicted_winner",
        "model_version",
    ]
    records = frame[columns].assign(played_at=frame["played_at"].astype(str)).to_dict("records")
    return {"count": len(records), "model_version": predictor.model_version, "matches": records}


@app.post("/predict")
def predict(request: MatchupRequest) -> dict[str, Any]:
    """Predict an arbitrary matchup."""
    predictor, history = _State.require()
    prediction = predictor.predict_matchup(
        history=history,
        team_a=(request.team_a[0], request.team_a[1]),
        team_b=(request.team_b[0], request.team_b[1]),
        tier=request.tier,
    )
    return prediction.as_dict()


@app.post("/reload")
def reload_model() -> dict[str, Any]:
    """Re-read the champion model and the match history."""
    _State.load()
    return health()


def main() -> None:
    """Entrypoint used by the Docker image and ``padel serve``."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "padel_predictor.inference.api:app",
        host=settings.api_host,
        port=settings.api_port,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover
    main()
