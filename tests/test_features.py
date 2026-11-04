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


# --------------------------------------------------------------------------- #
# Ranking snapshots: as-of join on publication time (editorial point E1)
# --------------------------------------------------------------------------- #
def _snapshot(player, label, published, points, elo=1500.0):
    return {
        "player": player,
        "snapshot_date": label,
        "published_at": published,
        "rank_points": points,
        "provider_elo": elo,
    }


def _week_rankings(points_before, points_leaky, published_leaky):
    """Two snapshots per player: the previous week's, and one labelled with the
    tournament week itself (Monday 2026-01-05) but published at ``published_leaky``."""
    rows = []
    for player in ("P1", "P2", "P3", "P4"):
        rows.append(_snapshot(player, "2025-12-29", "2025-12-29T08:00:00Z", points_before, 1500.0))
        rows.append(_snapshot(player, "2026-01-05", published_leaky, points_leaky, 1500.0))
    return pd.DataFrame(rows)


def _thursday_match(winner="A"):
    # Tournament week Wed 2026-01-07 to Sun 2026-01-11; this match is on Thursday.
    return pd.DataFrame(
        [make_match("qf", "2026-01-08T12:00:00Z", ("P1", "P2"), ("P3", "P4"), winner)]
    )


def test_snapshot_labelled_with_the_week_but_published_after_it_is_not_used(feature_config):
    """The editorial's case: a snapshot dated inside the tournament week, published once
    the week is over, already contains this match's result and must not be joined."""
    rankings = _week_rankings(100.0, 999.0, published_leaky="2026-01-12T08:00:00Z")
    # The leaky snapshot rewards the winners of this very match.
    winners = rankings["player"].isin(["P1", "P2"]) & (rankings["rank_points"] == 999.0)
    rankings.loc[winners, "rank_points"] = 5000.0

    row = build_feature_frame(_thursday_match(), feature_config, rankings=rankings).iloc[0]

    assert row["rank_points_diff"] == pytest.approx(0.0)  # both pairs: 2 x 100 points
    assert row["ranking_published_at"] == pd.Timestamp("2025-12-29T08:00:00Z")


def test_snapshot_published_exactly_at_the_start_is_not_used(feature_config):
    """Strictly before: a snapshot published at the scheduled start is too late."""
    rankings = _week_rankings(100.0, 999.0, published_leaky="2026-01-08T12:00:00Z")
    row = build_feature_frame(_thursday_match(), feature_config, rankings=rankings).iloc[0]
    assert row["ranking_published_at"] == pd.Timestamp("2025-12-29T08:00:00Z")


def test_snapshot_published_before_the_start_is_used(feature_config):
    rankings = _week_rankings(100.0, 999.0, published_leaky="2026-01-08T11:59:00Z")
    row = build_feature_frame(_thursday_match(), feature_config, rankings=rankings).iloc[0]
    assert row["ranking_published_at"] == pd.Timestamp("2026-01-08T11:59:00Z")


def test_later_snapshots_cannot_change_a_match_s_features(feature_config):
    """Leakage test for rankings: rewriting every snapshot published after the start
    -- where the match's own result would show up -- leaves its row unchanged."""
    honest = _week_rankings(100.0, 999.0, published_leaky="2026-01-12T08:00:00Z")
    rewritten = honest.copy()
    late = pd.to_datetime(rewritten["published_at"], utc=True) >= pd.Timestamp("2026-01-08T12:00Z")
    rewritten.loc[late, ["rank_points", "provider_elo"]] = [123456.0, 3000.0]

    columns = [*FEATURE_COLUMNS, "provider_elo_prob"]
    pd.testing.assert_frame_equal(
        build_feature_frame(_thursday_match(), feature_config, rankings=honest)[columns],
        build_feature_frame(_thursday_match(), feature_config, rankings=rewritten)[columns],
    )


def test_team_points_are_summed_and_provider_elo_is_averaged(feature_config):
    rankings = pd.DataFrame(
        [
            _snapshot("P1", "2026-01-01", "2026-01-01T08:00:00Z", 300.0, 1700.0),
            _snapshot("P2", "2026-01-01", "2026-01-01T08:00:00Z", 100.0, 1500.0),
            _snapshot("P3", "2026-01-01", "2026-01-01T08:00:00Z", 50.0, 1500.0),
            _snapshot("P4", "2026-01-01", "2026-01-01T08:00:00Z", 50.0, 1500.0),
        ]
    )
    row = build_feature_frame(_thursday_match(), feature_config, rankings=rankings).iloc[0]

    assert row["rank_points_diff"] == pytest.approx(400.0 - 100.0)
    assert row["provider_elo_prob"] == pytest.approx(expected_score(1600.0, 1500.0))


def test_no_snapshot_means_no_provider_baseline_and_zero_points(feature_config):
    empty = pd.DataFrame(
        columns=["player", "snapshot_date", "published_at", "rank_points", "provider_elo"]
    )
    row = build_feature_frame(_thursday_match(), feature_config, rankings=empty).iloc[0]
    assert pd.isna(row["provider_elo_prob"])
    assert row["rank_points_diff"] == pytest.approx(0.0)


def test_rankings_override_rank_points_carried_by_the_match(feature_config):
    """With a ranking history the match's own (possibly current) points are ignored."""
    match = _thursday_match()
    match["team_a_rank_points"] = 99999.0
    rankings = _week_rankings(100.0, 999.0, published_leaky="2026-01-12T08:00:00Z")
    row = build_feature_frame(match, feature_config, rankings=rankings).iloc[0]
    assert row["rank_points_diff"] == pytest.approx(0.0)


# --------------------------------------------------------------------------- #
# Team aggregation (review point B1)
# --------------------------------------------------------------------------- #
def test_new_partnership_keeps_each_player_s_strength(feature_config):
    """P1 and P3 have both won before but never played together: their partnership
    count is 0, while Elo and form still reflect their individual records."""
    history = pd.DataFrame(
        [
            make_match("m1", "2026-01-05T10:00:00Z", ("P1", "P2"), ("P5", "P6"), "A"),
            make_match("m2", "2026-01-06T10:00:00Z", ("P3", "P4"), ("P7", "P8"), "A"),
            make_match("m3", "2026-01-07T10:00:00Z", ("P1", "P3"), ("P5", "P7"), None),
        ]
    )
    new_pair = build_feature_frame(history, feature_config).iloc[-1]

    assert new_pair["partnership_diff"] == pytest.approx(0.0)
    assert new_pair["elo_diff"] > 0
    assert new_pair["form_diff"] == pytest.approx(1.0)  # 1.0 for P1/P3, 0.0 for P5/P7


def test_form_uses_each_player_s_last_matches_with_any_partner(feature_config):
    state = MatchState(feature_config)
    t = pd.Timestamp("2026-01-05T10:00:00Z")
    state.update(("P1", "P2"), ("P3", "P4"), t, team_a_won=1)
    state.update(("P1", "P9"), ("P3", "P8"), t + pd.Timedelta(days=1), team_a_won=0)
    # P1: won then lost = 0.5; P2: one win = 1.0 -> pair form is the mean.
    assert state.team_form(("P1", "P2")) == pytest.approx(0.75)


def test_rest_is_the_gap_of_the_pair_s_most_recently_active_player(feature_config):
    state = MatchState(feature_config)
    state.update(("P1", "P5"), ("P6", "P7"), pd.Timestamp("2026-01-01T10:00:00Z"), 1)
    state.update(("P2", "P8"), ("P6", "P9"), pd.Timestamp("2026-01-08T10:00:00Z"), 1)
    rest = state.team_rest_days(("P1", "P2"), pd.Timestamp("2026-01-10T10:00:00Z"))
    assert rest == pytest.approx(2.0)
