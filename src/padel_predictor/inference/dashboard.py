"""What the front end shows, computed without Streamlit so it can be tested.

Three things: the upcoming matches with their predictions, the numbers behind a
matchup (read from the same rolling state the features are built from), and the
track record -- stored predictions scored against the results that came in later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from padel_predictor.common import storage
from padel_predictor.common.config import (
    PipelineConfig,
    Settings,
    get_pipeline_config,
    get_settings,
)
from padel_predictor.common.features import (
    TARGET_COLUMN,
    MatchState,
    build_feature_frame,
    labelled,
    replay_state,
)
from padel_predictor.common.schemas import PLAYER_COLUMNS
from padel_predictor.inference.predict import Predictor, load_history

#: Probabilities are clipped this far from 0 and 1 before taking the log.
_EPS = 1e-15

SCORED_COLUMNS: tuple[str, ...] = (
    "match_id",
    "played_at",
    "tournament",
    "tier",
    "round",
    *PLAYER_COLUMNS,
    "probability_team_a",
    "predicted_at",
    "model_version",
    "provider_elo_prob",
    TARGET_COLUMN,
    "correct",
    "log_loss",
    "provider_log_loss",
)


@dataclass(frozen=True)
class Dashboard:
    """Everything one page load needs, read from the store once."""

    predictor: Predictor
    history: pd.DataFrame
    #: Feature rows of the whole history, finished and upcoming.
    features: pd.DataFrame
    #: Upcoming matches inside the horizon, with the champion's prediction.
    upcoming: pd.DataFrame
    #: Rolling state after the last finished match.
    state: MatchState
    #: Stored pre-match predictions of finished matches, scored.
    scored: pd.DataFrame
    loaded_at: pd.Timestamp


def load_dashboard(
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
) -> Dashboard:
    """Load the champion and the store, and derive what the front end shows."""
    settings = settings or get_settings()
    config = config or get_pipeline_config()

    predictor = Predictor.load(settings, config)
    history = load_history(settings)
    features = build_feature_frame(history, config.features, rankings=predictor.rankings)
    stored = storage.read_dataset(settings.feature_store_uri, storage.PREDICTIONS)
    return Dashboard(
        predictor=predictor,
        history=history,
        features=features,
        upcoming=predictor.predict_upcoming(history, features=features),
        state=replay_state(history, config.features),
        scored=score_predictions(stored, features),
        loaded_at=pd.Timestamp.now(tz="UTC"),
    )


# --------------------------------------------------------------------------- #
# The numbers behind a matchup
# --------------------------------------------------------------------------- #
def player_table(history: pd.DataFrame, state: MatchState) -> pd.DataFrame:
    """One row per player: in-house Elo, recent form, matches and last match."""
    finished = history.loc[history["winner"].notna(), list(PLAYER_COLUMNS)]
    counts = pd.Series(finished.to_numpy().ravel()).value_counts()
    names = sorted(str(n) for n in pd.unique(history[list(PLAYER_COLUMNS)].to_numpy().ravel()))
    rows = [
        {
            "player": name,
            "elo": float(state.elo[name]),
            "form": state.team_form([name]),
            "matches": int(counts.get(name, 0)),
            "last_played": state.last_played.get(name, pd.NaT),
        }
        for name in names
    ]
    table = pd.DataFrame(rows, columns=["player", "elo", "form", "matches", "last_played"])
    table["last_played"] = pd.to_datetime(table["last_played"], utc=True)
    return table


def team_comparison(
    state: MatchState,
    team_a: Sequence[str],
    team_b: Sequence[str],
    played_at: pd.Timestamp,
) -> pd.DataFrame:
    """Both pairs' values of the state-based features, side by side.

    Read through the same :class:`MatchState` readers that build the feature
    row, so the difference of the two columns is the feature the model saw.
    """
    rows = [
        ("In-house Elo", state.team_elo(team_a), state.team_elo(team_b)),
        ("Recent form (win rate)", state.team_form(team_a), state.team_form(team_b)),
        (
            "Rest days",
            state.team_rest_days(team_a, played_at),
            state.team_rest_days(team_b, played_at),
        ),
        (
            "Matches as a pair",
            float(state.partnership_matches(team_a)),
            float(state.partnership_matches(team_b)),
        ),
    ]
    return pd.DataFrame(rows, columns=["measure", "team_a", "team_b"])


# --------------------------------------------------------------------------- #
# Track record
# --------------------------------------------------------------------------- #
def _log_loss(outcome: pd.Series, probability: pd.Series) -> pd.Series:
    p = probability.astype("float64").clip(_EPS, 1 - _EPS)
    y = outcome.astype("float64")
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score_predictions(predictions: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Score the last pre-match prediction of every finished match.

    The inference pipeline stores a prediction per match and run, so a match
    drawn a week ahead has several. The one that counts is the last made before
    the match: before its start time, or, for a match without an announced time
    (stored at midnight of its day), before the end of that day. The outcome
    and the provider's Elo probability come from the feature rows, which carry
    the same team orientation as the predictions.
    """
    if predictions.empty or features.empty:
        return pd.DataFrame(columns=list(SCORED_COLUMNS))

    stored = predictions[["match_id", "probability_team_a", "predicted_at", "model_version"]].copy()
    stored["match_id"] = stored["match_id"].astype("string")
    stored["predicted_at"] = pd.to_datetime(stored["predicted_at"], utc=True)
    stored["model_version"] = stored["model_version"].astype("string")

    results = labelled(features)[
        ["match_id", "played_at", "tournament", "tier", "round", *PLAYER_COLUMNS]
        + ["provider_elo_prob", TARGET_COLUMN]
    ].copy()
    results["match_id"] = results["match_id"].astype("string")

    scored = stored.merge(results, on="match_id", how="inner")
    no_time = scored["played_at"] == scored["played_at"].dt.normalize()
    deadline = scored["played_at"] + pd.to_timedelta(no_time.astype(int), unit="D")
    scored = (
        scored.loc[scored["predicted_at"] < deadline]
        .sort_values("predicted_at", kind="stable")
        .drop_duplicates(subset="match_id", keep="last")
        .sort_values("played_at", kind="stable")
        .reset_index(drop=True)
    )

    won = scored[TARGET_COLUMN].astype("int64")
    scored["correct"] = (scored["probability_team_a"] >= 0.5) == (won == 1)
    scored["log_loss"] = _log_loss(won, scored["probability_team_a"])
    scored["provider_log_loss"] = _log_loss(won, scored["provider_elo_prob"])
    return scored[list(SCORED_COLUMNS)]


def weekly_performance(scored: pd.DataFrame) -> pd.DataFrame:
    """Accuracy and log loss per calendar week of the scored predictions.

    ``log_loss_rated`` and ``provider_log_loss`` cover only the matches the
    provider has a rating for, so the two are comparable: on the others the
    provider has no opinion to score.
    """
    columns = [
        "week",
        "matches",
        "accuracy",
        "log_loss",
        "rated",
        "log_loss_rated",
        "provider_log_loss",
    ]
    if scored.empty:
        return pd.DataFrame(columns=columns)

    frame = scored.assign(
        week=scored["played_at"].dt.tz_convert(None).dt.to_period("W-SUN").dt.start_time,
        rated=scored["provider_elo_prob"].notna(),
    )
    frame["log_loss_rated"] = frame["log_loss"].where(frame["rated"])
    weekly = frame.groupby("week", as_index=False).agg(
        matches=("match_id", "count"),
        accuracy=("correct", "mean"),
        log_loss=("log_loss", "mean"),
        rated=("rated", "sum"),
        log_loss_rated=("log_loss_rated", "mean"),
        provider_log_loss=("provider_log_loss", "mean"),
    )
    return weekly[columns]
