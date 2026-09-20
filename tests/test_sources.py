"""Ingestion adapters."""

from __future__ import annotations

import datetime as dt

import pytest

from padel_predictor.common.config import IngestionConfig, Settings
from padel_predictor.common.schemas import RAW_MATCH_COLUMNS
from padel_predictor.features.sources import HttpMatchSource, SampleMatchSource, get_source


def test_sample_source_is_deterministic():
    window = (dt.date(2026, 3, 1), dt.date(2026, 3, 14))
    first = SampleMatchSource().fetch(*window)
    second = SampleMatchSource().fetch(*window)

    assert not first.empty
    assert first["match_id"].tolist() == second["match_id"].tolist()
    assert first["winner"].tolist() == second["winner"].tolist()


def test_sample_source_respects_the_schema_and_window():
    start, end = dt.date(2026, 3, 1), dt.date(2026, 3, 14)
    matches = SampleMatchSource().fetch(start, end)

    assert list(matches.columns) == list(RAW_MATCH_COLUMNS)
    assert matches["played_at"].dt.date.min() >= start
    assert matches["played_at"].dt.date.max() <= end
    assert matches["match_id"].is_unique


def test_sample_source_leaves_future_matches_unlabelled():
    today = dt.datetime.now(dt.UTC).date()
    matches = SampleMatchSource().fetch(today - dt.timedelta(days=7), today + dt.timedelta(days=7))
    future = matches.loc[matches["played_at"] > dt.datetime.now(dt.UTC)]

    assert not future.empty
    assert future["winner"].isna().all()


def test_sample_source_rejects_a_reversed_window():
    with pytest.raises(ValueError, match="after end"):
        SampleMatchSource().fetch(dt.date(2026, 3, 14), dt.date(2026, 3, 1))


def test_http_source_needs_a_base_url(settings):
    with pytest.raises(ValueError, match="PADEL_API_BASE_URL"):
        HttpMatchSource(settings, IngestionConfig())


def test_http_records_are_mapped_onto_the_schema():
    records = [
        {
            "id": 42,
            "scheduled_at": "2026-03-01T12:00:00Z",
            "tournament": {"name": "Test Open", "tier": "p1"},
            "round": "SF",
            "teams": [
                {"players": [{"name": "A1"}, {"name": "A2"}], "rank_points": 3100},
                {"players": [{"name": "B1"}, {"name": "B2"}], "rank_points": 2400},
            ],
            "winner": "A",
        }
    ]
    out = HttpMatchSource.parse_records(records)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["match_id"] == "42"
    assert row["team_a_player_1"] == "A1"
    assert row["team_b_rank_points"] == 2400
    assert row["winner"] == "A"


def test_malformed_http_records_are_dropped_not_filled():
    records = [
        {"id": 1},  # no teams at all
        {
            "id": 2,
            "scheduled_at": "2026-03-01T12:00:00Z",
            "tournament": {"name": "T", "tier": "p2"},
            "teams": [
                {"players": [{"name": "A1"}, {"name": "A2"}]},
                {"players": [{"name": "B1"}, {"name": "B2"}]},
            ],
            "winner": None,
        },
    ]
    out = HttpMatchSource.parse_records(records)
    assert out["match_id"].tolist() == ["2"]


def test_empty_http_payload_returns_the_empty_schema():
    assert HttpMatchSource.parse_records([]).empty


def test_source_registry(settings):
    assert isinstance(get_source("sample", settings, IngestionConfig()), SampleMatchSource)
    with pytest.raises(ValueError, match="unknown source"):
        get_source("carrier-pigeon", settings, IngestionConfig())


def test_http_source_resolves_when_configured(monkeypatch):
    monkeypatch.setenv("PADEL_API_BASE_URL", "https://example.test/api")
    source = get_source("http", Settings(), IngestionConfig())
    assert isinstance(source, HttpMatchSource)
    assert source.base_url == "https://example.test/api"
