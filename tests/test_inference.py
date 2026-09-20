"""Prediction and serving.

A stub estimator stands in for the registered model so these tests cover the
feature plumbing and the API contract without needing MLflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from padel_predictor.common.features import FEATURE_COLUMNS, build_feature_frame
from padel_predictor.inference.predict import (
    Predictor,
    known_players,
    load_history,
)
from padel_predictor.training.registry import ModelHandle


class StubModel:
    """Predicts from ``elo_diff`` alone, so its output is easy to reason about."""

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        assert list(x.columns) == list(FEATURE_COLUMNS), "served the wrong feature order"
        p_a = 1.0 / (1.0 + np.exp(-x["elo_diff"].to_numpy() / 100.0))
        return np.column_stack([1.0 - p_a, p_a])


@pytest.fixture
def predictor(config) -> Predictor:
    handle = ModelHandle(
        model=StubModel(), name="padel-match-winner", version="7", alias="champion"
    )
    return Predictor(handle, config)


@pytest.fixture
def history(raw_matches):
    from padel_predictor.common.schemas import normalise_raw_matches

    return normalise_raw_matches(raw_matches)


def test_predict_frame_adds_probability_and_provenance(predictor, history, config):
    features = build_feature_frame(history, config.features)
    out = predictor.predict_frame(features)

    assert out["probability_team_a"].between(0, 1).all()
    assert set(out["predicted_winner"]) <= {"A", "B"}
    assert (out["model_version"] == "7").all()


def test_predicted_winner_follows_the_probability(predictor, history, config):
    features = build_feature_frame(history, config.features)
    out = predictor.predict_frame(features)
    expected = np.where(out["probability_team_a"] >= 0.5, "A", "B")
    assert out["predicted_winner"].tolist() == list(expected)


def test_matchup_favours_the_stronger_pair(predictor, history):
    """P1/P2 won their earlier meetings, so they should be favoured over P3/P4."""
    strong = predictor.predict_matchup(history, ("P1", "P2"), ("P3", "P4"))
    weak = predictor.predict_matchup(history, ("P3", "P4"), ("P1", "P2"))

    assert strong.probability_team_a > 0.5
    assert strong.predicted_winner == "A"
    assert weak.probability_team_a < 0.5
    # Swapping the sides mirrors the probability.
    assert strong.probability_team_a == pytest.approx(1.0 - weak.probability_team_a, abs=1e-9)


def test_matchup_reports_the_model_version(predictor, history):
    prediction = predictor.predict_matchup(history, ("P1", "P2"), ("P3", "P4"))
    assert prediction.model_version == "7"
    assert prediction.as_dict()["probability_team_a"] == prediction.probability_team_a


def test_matchup_does_not_mutate_the_history(predictor, history):
    before = len(history)
    predictor.predict_matchup(history, ("P1", "P2"), ("P3", "P4"))
    assert len(history) == before


def test_unknown_players_are_accepted_as_newcomers(predictor, history):
    """A player with no history gets the default rating rather than an error."""
    prediction = predictor.predict_matchup(history, ("Newcomer 1", "Newcomer 2"), ("P3", "P4"))
    assert 0.0 <= prediction.probability_team_a <= 1.0


def test_predict_upcoming_respects_the_horizon(predictor, config):
    from padel_predictor.common.schemas import normalise_raw_matches
    from tests.conftest import make_match

    now = pd.Timestamp.now(tz="UTC")
    records = [
        make_match("past", str(now - pd.Timedelta(days=3)), ("P1", "P2"), ("P3", "P4"), "A"),
        make_match("soon", str(now + pd.Timedelta(days=2)), ("P1", "P2"), ("P3", "P4"), None),
        make_match("later", str(now + pd.Timedelta(days=90)), ("P1", "P2"), ("P3", "P4"), None),
    ]
    frame = normalise_raw_matches(pd.DataFrame(records))

    out = predictor.predict_upcoming(frame, horizon_days=7, now=now)
    assert out["match_id"].tolist() == ["soon"]


def test_known_players_are_sorted_and_unique(history):
    players = known_players(history)
    assert players == sorted(set(players))
    assert "P1" in players


def test_load_history_without_a_store_fails_loudly(settings):
    with pytest.raises(RuntimeError, match="no raw matches"):
        load_history(settings)
