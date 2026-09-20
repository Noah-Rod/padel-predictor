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
