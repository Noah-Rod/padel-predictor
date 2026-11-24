"""What the front end shows: the state behind a matchup and the track record."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from padel_predictor.common.features import build_feature_frame, replay_state
from padel_predictor.common.schemas import normalise_raw_matches
from padel_predictor.inference.dashboard import (
    player_table,
    score_predictions,
    team_comparison,
    weekly_performance,
)
from tests.conftest import make_match


@pytest.fixture
def history(raw_matches):
    return normalise_raw_matches(raw_matches)


def stored(match_id: str, probability: float, predicted_at: str, version: str = "v1") -> dict:
    return {
        "match_id": match_id,
        "probability_team_a": probability,
        "predicted_at": predicted_at,
        "model_version": version,
    }


def test_comparison_differences_are_the_features_the_model_sees(history, feature_config):
    """The table next to a prediction must not be a second feature definition."""
    upcoming = make_match("next", "2026-01-12T10:00:00Z", ("P1", "P2"), ("P3", "P4"), None)
    combined = pd.concat([history, normalise_raw_matches(pd.DataFrame([upcoming]))])
    row = build_feature_frame(combined, feature_config).set_index("match_id").loc["next"]

    state = replay_state(history, feature_config)
    table = team_comparison(
        state, ("P1", "P2"), ("P3", "P4"), pd.Timestamp("2026-01-12T10:00:00Z")
    ).set_index("measure")
    difference = table["team_a"] - table["team_b"]

    assert difference["In-house Elo"] == pytest.approx(row["elo_diff"])
    assert difference["Recent form (win rate)"] == pytest.approx(row["form_diff"])
    assert difference["Rest days"] == pytest.approx(row["rest_days_diff"])
    assert difference["Matches as a pair"] == pytest.approx(row["partnership_diff"])


def test_replay_ignores_matches_without_a_result(history, feature_config):
    upcoming = make_match("next", "2026-01-12T10:00:00Z", ("P1", "P2"), ("P3", "P4"), None)
    combined = pd.concat([history, normalise_raw_matches(pd.DataFrame([upcoming]))])
    assert replay_state(combined, feature_config).elo == replay_state(history, feature_config).elo


def test_player_table_lists_every_player_once(history, feature_config):
    table = player_table(history, replay_state(history, feature_config))
    assert table["player"].tolist() == ["P1", "P2", "P3", "P4", "P5", "P6"]
    assert table.set_index("player").loc["P1", "matches"] == 4
    assert table["form"].between(0, 1).all()


def test_the_last_prediction_before_the_match_is_the_one_scored(history, feature_config):
    features = build_feature_frame(history, feature_config)
    predictions = pd.DataFrame(
        [
            stored("m5", 0.20, "2026-01-07T06:00:00Z"),
            stored("m5", 0.90, "2026-01-09T06:00:00Z", "v2"),
            # Stored after the match started: never counts.
            stored("m5", 0.01, "2026-01-09T12:00:00Z", "v3"),
        ]
    )
    scored = score_predictions(predictions, features)

    assert len(scored) == 1
    row = scored.iloc[0]
    assert row["probability_team_a"] == pytest.approx(0.90)
    assert row["model_version"] == "v2"
    won = features.set_index("match_id").loc["m5", "team_a_won"]
    assert row["correct"] == (won == 1)
    assert row["log_loss"] == pytest.approx(-np.log(0.90 if won == 1 else 0.10))


def test_a_match_without_a_start_time_counts_predictions_of_its_day(feature_config):
    """Such a match is stored at midnight; the 06:00 run of that day is still pre-match."""
    matches = normalise_raw_matches(
        pd.DataFrame([make_match("m1", "2026-01-05T00:00:00Z", ("P1", "P2"), ("P3", "P4"), "A")])
    )
    features = build_feature_frame(matches, feature_config)
    same_day = score_predictions(
        pd.DataFrame([stored("m1", 0.7, "2026-01-05T06:00:00Z")]), features
    )
    next_day = score_predictions(
        pd.DataFrame([stored("m1", 0.7, "2026-01-06T06:00:00Z")]), features
    )
    assert len(same_day) == 1
    assert next_day.empty


def test_unplayed_and_unknown_matches_are_not_scored(history, feature_config):
    upcoming = make_match("next", "2026-01-12T10:00:00Z", ("P1", "P2"), ("P3", "P4"), None)
    combined = pd.concat([history, normalise_raw_matches(pd.DataFrame([upcoming]))])
    features = build_feature_frame(combined, feature_config)
    predictions = pd.DataFrame(
        [stored("next", 0.6, "2026-01-11T06:00:00Z"), stored("gone", 0.6, "2026-01-01T06:00:00Z")]
    )
    assert score_predictions(predictions, features).empty
    assert score_predictions(pd.DataFrame(), features).empty


def test_weekly_performance_compares_model_and_provider_on_rated_matches_only(
    history, feature_config
):
    features = build_feature_frame(history, feature_config)
    # The provider rates m1 only; m2 falls in the same week, m5 as well.
    features.loc[features["match_id"] == "m1", "provider_elo_prob"] = 0.5
    predictions = pd.DataFrame(
        [
            stored("m1", 0.8, "2026-01-04T06:00:00Z"),
            stored("m2", 0.8, "2026-01-05T06:00:00Z"),
        ]
    )
    scored = score_predictions(predictions, features)
    weekly = weekly_performance(scored)

    assert len(weekly) == 1
    week = weekly.iloc[0]
    assert week["week"] == pd.Timestamp("2026-01-05")
    assert week["matches"] == 2
    assert week["rated"] == 1
    assert week["provider_log_loss"] == pytest.approx(np.log(2))
    rated = scored.loc[scored["match_id"] == "m1", "log_loss"].iloc[0]
    assert week["log_loss_rated"] == pytest.approx(rated)
    assert weekly_performance(scored.iloc[0:0]).empty
