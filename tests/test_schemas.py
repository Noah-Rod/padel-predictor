"""The raw-match contract."""

from __future__ import annotations

import pandas as pd
import pytest

from padel_predictor.common.schemas import (
    RAW_MATCH_COLUMNS,
    SchemaError,
    deduplicate_matches,
    empty_raw_matches,
    normalise_raw_matches,
)
from tests.conftest import make_match


def test_empty_frame_has_the_full_schema():
    empty = empty_raw_matches()
    assert list(empty.columns) == list(RAW_MATCH_COLUMNS)
    assert empty.empty


def test_normalise_sorts_by_time_and_coerces_types(raw_matches):
    shuffled = raw_matches.iloc[::-1].reset_index(drop=True)
    out = normalise_raw_matches(shuffled)

    assert out["match_id"].tolist() == ["m1", "m2", "m3", "m4", "m5"]
    assert str(out["played_at"].dtype) == "datetime64[ns, UTC]"
    assert out["team_a_rank_points"].dtype == "Float64"


def test_missing_columns_raise():
    with pytest.raises(SchemaError, match="missing required columns"):
        normalise_raw_matches(pd.DataFrame({"match_id": ["m1"]}))


def test_unparseable_timestamp_raises():
    bad = pd.DataFrame([make_match("m1", "not-a-date", ("P1", "P2"), ("P3", "P4"))])
    with pytest.raises(SchemaError, match="played_at"):
        normalise_raw_matches(bad)


def test_unknown_winner_becomes_null():
    frame = pd.DataFrame(
        [make_match("m1", "2026-01-05T10:00:00Z", ("P1", "P2"), ("P3", "P4"), winner="draw")]
    )
    assert normalise_raw_matches(frame)["winner"].isna().all()


def test_lowercase_winner_is_accepted():
    frame = pd.DataFrame(
        [make_match("m1", "2026-01-05T10:00:00Z", ("P1", "P2"), ("P3", "P4"), winner="b")]
    )
    assert normalise_raw_matches(frame)["winner"].tolist() == ["B"]


def test_deduplicate_keeps_the_newest_ingest():
    """A match ingested as upcoming, then re-ingested with a result, keeps the result."""
    early = make_match("m1", "2026-01-05T10:00:00Z", ("P1", "P2"), ("P3", "P4"), winner=None)
    late = make_match("m1", "2026-01-05T10:00:00Z", ("P1", "P2"), ("P3", "P4"), winner="A")
    frame = normalise_raw_matches(pd.DataFrame([early, late]))
    frame["ingested_at"] = [
        pd.Timestamp("2026-01-04T00:00:00Z"),
        pd.Timestamp("2026-01-06T00:00:00Z"),
    ]

    out = deduplicate_matches(frame)
    assert len(out) == 1
    assert out.iloc[0]["winner"] == "A"


def test_orient_teams_is_deterministic_and_ignores_source_order():
    from padel_predictor.common.schemas import orient_teams

    seeded_first = normalise_raw_matches(
        pd.DataFrame([make_match("m1", "2026-01-05", ("X1", "X2"), ("Y1", "Y2"), "A")])
    )
    swapped = normalise_raw_matches(
        pd.DataFrame([make_match("m1", "2026-01-05", ("Y2", "Y1"), ("X2", "X1"), "B")])
    )
    a, b = orient_teams(seeded_first).iloc[0], orient_teams(swapped).iloc[0]

    pair = lambda row, side: frozenset((row[f"team_{side}_player_1"], row[f"team_{side}_player_2"]))  # noqa: E731
    assert pair(a, "a") == pair(b, "a")
    # The same pair won, so after orientation the label agrees as well.
    assert a["winner"] == b["winner"]


def test_orient_teams_balances_a_seed_ordered_feed():
    """A feed that always lists the winner first must not yield 'A always wins'."""
    from padel_predictor.common.schemas import orient_teams

    rows = [
        make_match(f"m{i}", "2026-01-05", (f"S{i}a", f"S{i}b"), (f"U{i}a", f"U{i}b"), "A")
        for i in range(400)
    ]
    out = orient_teams(normalise_raw_matches(pd.DataFrame(rows)))
    share_a = (out["winner"] == "A").mean()
    assert 0.4 < share_a < 0.6


def test_rankings_deduplicate_to_the_first_publication():
    from padel_predictor.common.schemas import deduplicate_rankings

    rows = pd.DataFrame(
        [
            {
                "player": "P1",
                "snapshot_date": "2026-01-05",
                "published_at": "2026-01-05T08:00Z",
                "rank_points": 10,
                "provider_elo": 1500,
            },
            {
                "player": "P1",
                "snapshot_date": "2026-01-05",
                "published_at": "2026-01-09T08:00Z",
                "rank_points": 10,
                "provider_elo": 1500,
            },
        ]
    )
    out = deduplicate_rankings(rows)
    assert len(out) == 1
    assert out.iloc[0]["published_at"] == pd.Timestamp("2026-01-05T08:00Z")


def test_rankings_without_publication_time_are_rejected():
    from padel_predictor.common.schemas import normalise_raw_rankings

    rows = pd.DataFrame(
        [
            {
                "player": "P1",
                "snapshot_date": "2026-01-05",
                "published_at": None,
                "rank_points": 10,
                "provider_elo": 1500,
            }
        ]
    )
    with pytest.raises(SchemaError, match="published_at"):
        normalise_raw_rankings(rows)


def test_rounds_on_the_same_timestamp_replay_in_draw_order():
    """Without a start time, a day's rounds share midnight; QF must not precede R16."""
    rows = [
        make_match("1", "2026-03-07", ("A1", "A2"), ("B1", "B2")),
        make_match("2", "2026-03-07", ("A1", "A2"), ("C1", "C2")),
    ]
    frame = pd.DataFrame(rows)
    frame["round"] = ["QF", "R16"]
    out = normalise_raw_matches(frame)
    assert out["round"].tolist() == ["R16", "QF"]
