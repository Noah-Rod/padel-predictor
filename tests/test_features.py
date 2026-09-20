"""Feature building -- above all, that it does not leak the label."""

from __future__ import annotations

import pandas as pd
import pytest

from padel_predictor.common.features import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    MatchState,
    build_feature_frame,
    expected_score,
    labelled,
    unlabelled,
)
from tests.conftest import make_match


def test_frame_has_features_keys_and_target(raw_matches, feature_config):
    out = build_feature_frame(raw_matches, feature_config)
    assert len(out) == len(raw_matches)
    for column in (*FEATURE_COLUMNS, TARGET_COLUMN, "match_id", "played_at"):
        assert column in out.columns
    assert out[list(FEATURE_COLUMNS)].notna().all().all()


def test_empty_input_returns_empty_frame_with_columns(feature_config):
    columns = list(make_match("x", "2026-01-01", ("a", "b"), ("c", "d")).keys())
    out = build_feature_frame(pd.DataFrame(columns=columns), feature_config)
    assert out.empty
    assert TARGET_COLUMN in out.columns


def test_first_match_features_are_all_neutral(raw_matches, feature_config):
    """With no history, both teams look identical apart from ranking points."""
    first = build_feature_frame(raw_matches, feature_config).iloc[0]
    assert first["elo_diff"] == pytest.approx(0.0)
    assert first["form_diff"] == pytest.approx(0.0)
    assert first["rest_days_diff"] == pytest.approx(0.0)
    assert first["partnership_diff"] == pytest.approx(0.0)


def test_features_do_not_leak_the_label(raw_matches, feature_config):
    """Flipping a match's own result must not change that match's features.

    This is the property that makes the training set honest: a feature row is
    built from state that contains only strictly earlier matches.
    """
    flipped = raw_matches.copy()
    last = flipped.index[-1]
    flipped.loc[last, "winner"] = "B" if flipped.loc[last, "winner"] == "A" else "A"

    original = build_feature_frame(raw_matches, feature_config)
    altered = build_feature_frame(flipped, feature_config)

    pd.testing.assert_frame_equal(
        original[list(FEATURE_COLUMNS)],
        altered[list(FEATURE_COLUMNS)],
    )
    # The label itself must of course differ.
    assert original.iloc[-1][TARGET_COLUMN] != altered.iloc[-1][TARGET_COLUMN]


def test_unplayed_match_gets_features_but_no_label(raw_matches, feature_config):
    upcoming = pd.concat(
        [
            raw_matches,
            pd.DataFrame(
                [make_match("m6", "2026-02-01T10:00:00Z", ("P1", "P2"), ("P3", "P4"), None)]
            ),
        ],
        ignore_index=True,
    )
    out = build_feature_frame(upcoming, feature_config)

    assert len(labelled(out)) == 5
    assert len(unlabelled(out)) == 1
    pending = unlabelled(out).iloc[0]
    assert pd.isna(pending[TARGET_COLUMN])
    assert pending[list(FEATURE_COLUMNS)].notna().all()
    # It has seen the earlier P1/P2 wins, so team A is rated higher.
    assert pending["elo_diff"] > 0


def test_winning_raises_elo_and_losing_lowers_it(feature_config):
    state = MatchState(feature_config)
    team_a, team_b = ("P1", "P2"), ("P3", "P4")
    state.update(team_a, team_b, pd.Timestamp("2026-01-05T10:00:00Z"), team_a_won=1)

    assert state.team_elo(team_a) > feature_config.elo_initial
    assert state.team_elo(team_b) < feature_config.elo_initial
    # Elo is zero-sum across the four players.
    assert state.team_elo(team_a) + state.team_elo(team_b) == pytest.approx(
        2 * feature_config.elo_initial
    )


def test_expected_score_is_symmetric():
    assert expected_score(1500, 1500) == pytest.approx(0.5)
    assert expected_score(1700, 1500) + expected_score(1500, 1700) == pytest.approx(1.0)
    assert expected_score(1700, 1500) > 0.5


def test_partnership_counts_are_order_independent(feature_config):
    state = MatchState(feature_config)
    state.update(("P1", "P2"), ("P3", "P4"), pd.Timestamp("2026-01-05T10:00:00Z"), 1)
    assert state.partnership_matches(("P2", "P1")) == 1


def test_unknown_tier_maps_to_zero(feature_config):
    state = MatchState(feature_config)
    assert state.tier_level("major") == feature_config.tier_levels["major"]
    assert state.tier_level("not-a-tier") == 0
    assert state.tier_level(None) == 0


def test_building_is_deterministic(raw_matches, feature_config):
    first = build_feature_frame(raw_matches, feature_config)
    second = build_feature_frame(raw_matches.sample(frac=1, random_state=0), feature_config)
    pd.testing.assert_frame_equal(first, second)
