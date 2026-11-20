"""The feature pipeline end to end, against a temporary store."""

from __future__ import annotations

import datetime as dt

import pytest

from padel_predictor.common import storage
from padel_predictor.features import pipeline


def test_the_report_lists_when_each_ranking_was_first_seen():
    """The measurement behind the publication lag (review point A1)."""
    import pandas as pd

    from padel_predictor.common.schemas import normalise_raw_rankings
    from padel_predictor.features.report import DataReport, ranking_delays

    def row(player, label, published):
        return {
            "player": player,
            "snapshot_date": label,
            "published_at": published,
            "rank_points": 1.0,
            "provider_elo": 1500.0,
        }

    rankings = normalise_raw_rankings(
        pd.DataFrame(
            [
                row("Ana", "2026-03-02", "2026-03-09T00:00Z"),  # backfilled: label + 7 days
                row("Ana", "2026-03-09", "2026-03-10T04:15Z"),  # seen live a day later
                row("Bea", "2026-03-09", "2026-03-12T04:15Z"),  # read on her own, later
                row("Ana", "2026-03-16", "2026-03-19T04:15Z"),  # seen live three days later
            ]
        )
    )

    delays = ranking_delays(rankings)

    assert delays["observed"].tolist() == [False, True, True]
    assert delays["delay_days"].iloc[1] == pytest.approx(1 + 4.25 / 24)
    assert delays["first_seen_at"].iloc[1] == pd.Timestamp("2026-03-10T04:15Z")

    def report(lag):
        return "\n".join(
            DataReport(
                counts=pd.DataFrame(),
                ranking_rows=0,
                ranking_snapshots=0,
                first_match="-",
                last_match="-",
                baseline={},
                test_rows=0,
                test_start="-",
                ranking_delays=delays,
                publication_lag_days=lag,
            ).ranking_delay_markdown()
        )

    text = report(7)
    assert "2 ranking(s), first seen 1.2 to 3.2 days after their date" in text
    assert "| 2026-03-16 | 2026-03-19 04:15 | 3.2 |" in text
    assert "1 ranking(s) were backfilled" in text
    assert "too early" not in text
    # A lag below what was observed would let a ranking in before it existed.
    assert "**The rule is too early:**" in report(2)


def test_run_writes_raw_and_feature_snapshots(settings, config):
    result = pipeline.run(lookback_days=30, horizon_days=7, settings=settings, config=config)

    assert result.fetched_matches > 0
    assert result.feature_rows == result.labelled_rows + result.upcoming_rows
    assert result.upcoming_rows > 0, "the window reaches into the future, so fixtures are expected"

    raw = storage.read_dataset(settings.feature_store_uri, storage.RAW_MATCHES)
    features = storage.read_latest_snapshot(settings.feature_store_uri, storage.FEATURES)
    assert len(raw) == result.fetched_matches
    assert len(features) == result.feature_rows


def test_rerunning_does_not_duplicate_matches(settings, config):
    first = pipeline.run(lookback_days=30, horizon_days=7, settings=settings, config=config)
    second = pipeline.run(lookback_days=30, horizon_days=7, settings=settings, config=config)

    # Both runs wrote a snapshot, but the deduplicated history is unchanged.
    assert second.total_matches == first.total_matches
    assert second.feature_rows == first.feature_rows


def test_compute_features_without_ingest_fails_loudly(settings, config):
    with pytest.raises(RuntimeError, match="no raw matches"):
        pipeline.compute_features(settings.feature_store_uri, config)


def test_backfill_loads_a_range_in_chunks(settings, config):
    end = dt.date(2026, 6, 30)
    start = dt.date(2026, 4, 1)
    result = pipeline.backfill(
        start=start, end=end, chunk_days=30, settings=settings, config=config
    )

    assert result.fetched_matches > 0
    assert result.labelled_rows > 0
    raw = storage.read_dataset(settings.feature_store_uri, storage.RAW_MATCHES)
    assert raw["played_at"].dt.date.min() >= start
    assert raw["played_at"].dt.date.max() <= end


def test_backfill_rejects_a_reversed_range(settings, config):
    with pytest.raises(ValueError, match="after end"):
        pipeline.backfill(
            start=dt.date(2026, 6, 30), end=dt.date(2026, 4, 1), settings=settings, config=config
        )


def test_run_stores_rankings_and_the_provider_baseline(settings, config):
    pipeline.run(lookback_days=30, horizon_days=7, settings=settings, config=config)

    rankings = storage.read_dataset(settings.feature_store_uri, storage.RAW_RANKINGS)
    features = storage.read_latest_snapshot(settings.feature_store_uri, storage.FEATURES)
    assert not rankings.empty
    assert features["provider_elo_prob"].notna().any()
    # Every joined snapshot was published before the match it was joined to.
    joined = features.dropna(subset=["ranking_published_at"])
    assert (joined["ranking_published_at"] < joined["played_at"]).all()


def _rankings(*rows):
    import pandas as pd

    columns = ["player", "snapshot_date", "published_at", "rank_points", "provider_elo"]
    return pd.DataFrame([dict(zip(columns, r, strict=True)) for r in rows])


def test_a_stored_snapshot_is_rewritten_only_to_fill_a_missing_value():
    import pandas as pd

    known = _rankings(
        ("Ana", "2026-03-02", "2026-03-09T00:00Z", 300.0, None),
        ("Bo", "2026-03-02", "2026-03-09T00:00Z", 280.0, 1450.0),
    )
    fetched = _rankings(
        ("Ana", "2026-03-02", "2026-11-17T00:00Z", 999.0, 1420.0),  # fills Ana's Elo
        ("Bo", "2026-03-02", "2026-11-17T00:00Z", 280.0, 1450.0),  # nothing new
        ("Cy", "2026-03-02", "2026-03-09T00:00Z", 120.0, 1380.0),  # never seen
    )

    out = pipeline._new_ranking_rows(fetched, known).set_index("player")

    assert sorted(out.index) == ["Ana", "Cy"]
    # Ana keeps the stored publication time and points; only the gap is filled.
    assert out.loc["Ana", "published_at"] == pd.Timestamp("2026-03-09T00:00Z")
    assert (out.loc["Ana", "rank_points"], out.loc["Ana", "provider_elo"]) == (300.0, 1420.0)


def test_free_plan_backfill_is_clamped_to_six_months(monkeypatch, config):
    from padel_predictor.common.config import Settings

    monkeypatch.setenv("PADEL_SOURCE", "http")
    monkeypatch.setenv("PADEL_API_PLAN", "free")
    today = dt.datetime.now(dt.UTC).date()
    clamped = pipeline._clamp_to_plan(today - dt.timedelta(days=900), Settings(), config)
    assert clamped == today - dt.timedelta(days=config.ingestion.free_plan_history_days)


def test_paid_plan_backfill_is_not_clamped(config, settings):
    start = dt.date(2023, 1, 1)
    assert pipeline._clamp_to_plan(start, settings, config) == start
