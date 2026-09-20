"""Feature pipeline: live source -> raw snapshot -> features -> feature store.

Run on a schedule by ``.github/workflows/feature-pipeline.yml``. The same code
path serves a normal incremental run and a historical backfill; only the date
window differs.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd

from padel_predictor.common import storage
from padel_predictor.common.config import (
    PipelineConfig,
    Settings,
    get_pipeline_config,
    get_settings,
)
from padel_predictor.common.features import build_feature_frame, labelled, unlabelled
from padel_predictor.common.logging import get_logger
from padel_predictor.common.schemas import deduplicate_matches, normalise_raw_matches
from padel_predictor.features.sources import MatchSource, get_source

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureRunResult:
    """What one run of the feature pipeline did."""

    window_start: dt.date
    window_end: dt.date
    fetched_matches: int
    total_matches: int
    feature_rows: int
    labelled_rows: int
    upcoming_rows: int
    raw_uri: str
    features_uri: str

    def summary(self) -> str:
        return (
            f"window={self.window_start}..{self.window_end} "
            f"fetched={self.fetched_matches} store_total={self.total_matches} "
            f"features={self.feature_rows} labelled={self.labelled_rows} "
            f"upcoming={self.upcoming_rows}"
        )


def ingest(
    start: dt.date,
    end: dt.date,
    base_uri: str,
    source: MatchSource,
    snapshot_date: dt.date | None = None,
) -> tuple[pd.DataFrame, str]:
    """Fetch a date window from the live source and append a raw snapshot."""
    matches = source.fetch(start, end)
    matches = normalise_raw_matches(matches)
    matches["ingested_at"] = pd.Timestamp.now(tz="UTC")
    uri = storage.write_snapshot(matches, base_uri, storage.RAW_MATCHES, snapshot_date)
    logger.info("ingested %d matches from source %r", len(matches), source.name)
    return matches, uri


def compute_features(
    base_uri: str,
    config: PipelineConfig,
    snapshot_date: dt.date | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Rebuild features from the full raw history and write a feature snapshot.

    Features are always recomputed from scratch rather than appended to. Elo and
    form are path-dependent: a late-arriving result changes every later row, so
    an incremental append would quietly desynchronise the store.
    """
    raw = storage.read_dataset(base_uri, storage.RAW_MATCHES)
    if raw.empty:
        raise RuntimeError(
            f"no raw matches under {base_uri!r}; run the ingest step or a backfill first"
        )

    history = deduplicate_matches(raw)
    features = build_feature_frame(history, config.features)
    uri = storage.write_snapshot(features, base_uri, storage.FEATURES, snapshot_date)
    return history, features, uri


def run(
    lookback_days: int | None = None,
    horizon_days: int | None = None,
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
    today: dt.date | None = None,
) -> FeatureRunResult:
    """One scheduled run: ingest a recent window, then rebuild the feature store.

    The window reaches into the future as well as the past: upcoming matches
    carry no result but still need feature rows, because they are what the
    inference pipeline predicts.
    """
    settings = settings or get_settings()
    config = config or get_pipeline_config()
    today = today or dt.datetime.now(dt.UTC).date()

    lookback = config.ingestion.lookback_days if lookback_days is None else lookback_days
    horizon = config.inference.horizon_days if horizon_days is None else horizon_days
    start, end = today - dt.timedelta(days=lookback), today + dt.timedelta(days=horizon)

    logger.info("feature pipeline starting | %s", settings.masked())
    source = get_source(settings.source, settings, config.ingestion)

    fetched, raw_uri = ingest(start, end, settings.feature_store_uri, source, today)
    history, features, features_uri = compute_features(settings.feature_store_uri, config, today)

    result = FeatureRunResult(
        window_start=start,
        window_end=end,
        fetched_matches=len(fetched),
        total_matches=len(history),
        feature_rows=len(features),
        labelled_rows=len(labelled(features)),
        upcoming_rows=len(unlabelled(features)),
        raw_uri=raw_uri,
        features_uri=features_uri,
    )
    logger.info("feature pipeline done | %s", result.summary())
    return result


def backfill(
    start: dt.date,
    end: dt.date,
    chunk_days: int = 30,
    settings: Settings | None = None,
    config: PipelineConfig | None = None,
) -> FeatureRunResult:
    """Load a historical date range in chunks, then rebuild the feature store.

    Chunking keeps each request to the live source small enough to retry, and
    makes a partial failure resumable: already-written snapshots are kept and
    re-ingesting a window is idempotent (matches deduplicate by ``match_id``).
    """
    if start > end:
        raise ValueError(f"start {start} is after end {end}")

    settings = settings or get_settings()
    config = config or get_pipeline_config()
    source = get_source(settings.source, settings, config.ingestion)

    fetched_total = 0
    raw_uri = ""
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days - 1), end)
        logger.info("backfill chunk %s..%s", cursor, chunk_end)
        chunk, raw_uri = ingest(cursor, chunk_end, settings.feature_store_uri, source, chunk_end)
        fetched_total += len(chunk)
        cursor = chunk_end + dt.timedelta(days=1)

    history, features, features_uri = compute_features(settings.feature_store_uri, config, end)

    result = FeatureRunResult(
        window_start=start,
        window_end=end,
        fetched_matches=fetched_total,
        total_matches=len(history),
        feature_rows=len(features),
        labelled_rows=len(labelled(features)),
        upcoming_rows=len(unlabelled(features)),
        raw_uri=raw_uri,
        features_uri=features_uri,
    )
    logger.info("backfill done | %s", result.summary())
    return result
