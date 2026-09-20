"""Prediction.

Everything here builds its features with
:func:`padel_predictor.common.features.build_feature_frame` -- the same function
the feature pipeline uses. An ad-hoc prediction for two teams is served by
appending a synthetic upcoming match to the real history and replaying the state,
so a prediction made from the UI goes through exactly the code path that produced
the training rows.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

from padel_predictor.common import storage
from padel_predictor.common.config import (
    PipelineConfig,
    Settings,
    get_pipeline_config,
    get_settings,
)
from padel_predictor.common.features import (
    FEATURE_COLUMNS,
    build_feature_frame,
    unlabelled,
)
from padel_predictor.common.logging import get_logger
from padel_predictor.common.schemas import deduplicate_matches, normalise_raw_matches
from padel_predictor.training import registry
from padel_predictor.training.registry import ModelHandle

logger = get_logger(__name__)

AD_HOC_MATCH_ID = "ad-hoc"


class ModelNotAvailableError(RuntimeError):
    """Raised when no model carries the champion alias yet."""


@dataclass(frozen=True)
class Prediction:
    """One prediction, with the provenance needed to audit it later."""

    match_id: str
    played_at: str
    tournament: str
    tier: str
    team_a: str
    team_b: str
    probability_team_a: float
    predicted_winner: str
    model_name: str
    model_version: str
    predicted_at: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_history(settings: Settings | None = None) -> pd.DataFrame:
    """Read the deduplicated raw match history from the feature store."""
    settings = settings or get_settings()
    raw = storage.read_dataset(settings.feature_store_uri, storage.RAW_MATCHES)
    if raw.empty:
        raise RuntimeError(
            f"no raw matches under {settings.feature_store_uri!r}; run the feature pipeline first"
        )
    return deduplicate_matches(raw)


class Predictor:
    """The champion model plus the feature logic needed to feed it."""

    def __init__(self, handle: ModelHandle, config: PipelineConfig | None = None) -> None:
        self.handle = handle
        self.config = config or get_pipeline_config()

    @classmethod
    def load(
        cls,
        settings: Settings | None = None,
        config: PipelineConfig | None = None,
    ) -> Predictor:
        """Load the model currently aliased as champion in the registry."""
        settings = settings or get_settings()
        handle = registry.load_champion(settings)
        if handle is None:
            raise ModelNotAvailableError(
                f"no model aliased {settings.model_alias!r} under "
                f"{settings.registered_model_name!r}; run the training pipeline first"
            )
        logger.info("loaded model %s version %s", handle.name, handle.version)
        return cls(handle, config)

    @property
    def model_version(self) -> str:
        """The deployed model version -- surfaced by the API, UI and predictions."""
        return self.handle.version

    def probabilities(self, features: pd.DataFrame) -> pd.Series:
        """Probability that team A wins, for pre-built feature rows."""
        if features.empty:
            return pd.Series(dtype="float64")
        proba = self.handle.model.predict_proba(features[list(FEATURE_COLUMNS)])[:, 1]
        return pd.Series(proba, index=features.index, name="probability_team_a")

    def predict_frame(self, features: pd.DataFrame) -> pd.DataFrame:
        """Attach predictions and model provenance to a feature frame."""
        out = features.copy()
        out["probability_team_a"] = self.probabilities(features)
        out["predicted_winner"] = out["probability_team_a"].map(lambda p: "A" if p >= 0.5 else "B")
        out["model_name"] = self.handle.name
        out["model_version"] = self.handle.version
        out["predicted_at"] = pd.Timestamp.now(tz="UTC")
        return out

    def predict_upcoming(
        self,
        history: pd.DataFrame,
        horizon_days: int | None = None,
        now: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Predict every unplayed match inside the horizon."""
        horizon = horizon_days if horizon_days is not None else self.config.inference.horizon_days
        now = now or pd.Timestamp.now(tz="UTC")

        features = build_feature_frame(history, self.config.features)
        upcoming = unlabelled(features)
        cutoff = now + pd.Timedelta(days=horizon)
        upcoming = upcoming.loc[upcoming["played_at"] <= cutoff].reset_index(drop=True)

        logger.info("predicting %d upcoming matches within %d days", len(upcoming), horizon)
        return self.predict_frame(upcoming)

    def predict_matchup(
        self,
        history: pd.DataFrame,
        team_a: tuple[str, str],
        team_b: tuple[str, str],
        tier: str = "p1",
        played_at: pd.Timestamp | None = None,
        tournament: str = "ad-hoc",
    ) -> Prediction:
        """Predict a hypothetical matchup that is not in the schedule.

        The matchup is appended to the history as an unplayed match and the
        feature state is replayed, so the resulting row is built by the same
        code that built every training row.
        """
        played_at = played_at or pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)
        rank_points = _recent_rank_points(history)

        candidate = pd.DataFrame(
            [
                {
                    "match_id": AD_HOC_MATCH_ID,
                    "played_at": played_at,
                    "tournament": tournament,
                    "tier": tier,
                    "round": "unknown",
                    "team_a_player_1": team_a[0],
                    "team_a_player_2": team_a[1],
                    "team_b_player_1": team_b[0],
                    "team_b_player_2": team_b[1],
                    "team_a_rank_points": sum(rank_points.get(p, 0.0) for p in team_a),
                    "team_b_rank_points": sum(rank_points.get(p, 0.0) for p in team_b),
                    "winner": None,
                }
            ]
        )
        # Normalise the candidate first so both frames share dtypes; concatenating a
        # raw all-None `winner` column against a typed one changes dtypes under pandas 3.
        combined = normalise_raw_matches(
            pd.concat(
                [history[list(candidate.columns)], normalise_raw_matches(candidate)],
                ignore_index=True,
            )
        )

        features = build_feature_frame(combined, self.config.features)
        row = features.loc[features["match_id"] == AD_HOC_MATCH_ID]
        if row.empty:  # pragma: no cover -- defensive
            raise RuntimeError("the ad-hoc match did not survive feature building")

        predicted = self.predict_frame(row).iloc[0]
        return Prediction(
            match_id=AD_HOC_MATCH_ID,
            played_at=str(predicted["played_at"]),
            tournament=tournament,
            tier=tier,
            team_a=" / ".join(team_a),
            team_b=" / ".join(team_b),
            probability_team_a=float(predicted["probability_team_a"]),
            predicted_winner=str(predicted["predicted_winner"]),
            model_name=self.handle.name,
            model_version=self.handle.version,
            predicted_at=str(predicted["predicted_at"]),
        )


def _recent_rank_points(history: pd.DataFrame) -> dict[str, float]:
    """Each player's most recent per-player ranking points, halved from the team total.

    The raw schema carries combined team points, so a single player's share is
    approximated as half of their team's most recent total. Replace this with
    per-player points if the live source exposes them.
    """
    points: dict[str, float] = {}
    if history.empty:
        return points
    for match in history.sort_values("played_at").itertuples(index=False):
        for player, total in (
            (match.team_a_player_1, match.team_a_rank_points),
            (match.team_a_player_2, match.team_a_rank_points),
            (match.team_b_player_1, match.team_b_rank_points),
            (match.team_b_player_2, match.team_b_rank_points),
        ):
            if pd.notna(total):
                points[str(player)] = float(total) / 2.0
    return points


def known_players(history: pd.DataFrame) -> list[str]:
    """Sorted list of every player seen in the history -- used to populate the UI."""
    if history.empty:
        return []
    columns = [
        "team_a_player_1",
        "team_a_player_2",
        "team_b_player_1",
        "team_b_player_2",
    ]
    names = pd.unique(history[columns].to_numpy().ravel())
    return sorted(str(n) for n in names if pd.notna(n))


def upcoming_horizon_cutoff(config: PipelineConfig | None = None) -> dt.datetime:
    """The timestamp beyond which upcoming matches are ignored."""
    config = config or get_pipeline_config()
    return dt.datetime.now(dt.UTC) + dt.timedelta(days=config.inference.horizon_days)
