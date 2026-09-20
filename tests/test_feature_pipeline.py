"""The feature pipeline end to end, against a temporary store."""

from __future__ import annotations

import datetime as dt

import pytest

from padel_predictor.common import storage
from padel_predictor.features import pipeline


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
