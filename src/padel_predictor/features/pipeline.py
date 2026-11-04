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
from padel_predictor.common.schemas import (
    deduplicate_matches,
    deduplicate_rankings,
    normalise_raw_matches,
    normalise_raw_rankings,
)
from padel_predictor.features.sources import MatchSource, get_source

logger = get_logger(__name__)


@dataclass(frozen=True)
class FeatureRunResult:
    """What one run of the feature pipeline did."""

    window_start: dt.date
    window_end: dt.date
    fetched_matches: int
    fetched_rankings: int
    total_matches: int
    feature_rows: int
    labelled_rows: int
    upcoming_rows: int
    raw_uri: str
    features_uri: str

    def summary(self) -> str:
        return (
            f"window={self.window_start}..{self.window_end} "
            f"fetched={self.fetched_matches} rankings={self.fetched_rankings} "
            f"store_total={self.total_matches} "
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


def ingest_rankings(
    start: dt.date,
    end: dt.date,
    base_uri: str,
    source: MatchSource,
    config: PipelineConfig,
    snapshot_date: dt.date | None = None,
) -> pd.DataFrame:
    """Fetch the ranking snapshots in force during a window and store them raw.

    The window is widened backwards so the snapshot that applies to the first
    match of the window is always present.
    """
    lookback = dt.timedelta(days=config.ingestion.rankings_lookback_days)
    rankings = normalise_raw_rankings(source.fetch_rankings(start - lookback, end))

    # Only snapshots not seen before are written: the first time a snapshot was
    # seen is when it was available, and an upsert must not move that time.
    known = load_rankings(base_uri)
    if not known.empty and not rankings.empty:
        keys = ["player", "snapshot_date"]
        seen = pd.MultiIndex.from_frame(known[keys])
        rankings = rankings.loc[~pd.MultiIndex.from_frame(rankings[keys]).isin(seen)]
        rankings = rankings.reset_index(drop=True)

    storage.write_snapshot(rankings, base_uri, storage.RAW_RANKINGS, snapshot_date)
    logger.info("ingested %d new ranking rows from source %r", len(rankings), source.name)
    return rankings


def load_rankings(base_uri: str) -> pd.DataFrame:
    """Every ranking snapshot in the store, one row per player and snapshot."""
    return deduplicate_rankings(storage.read_dataset(base_uri, storage.RAW_RANKINGS))


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
    rankings = load_rankings(base_uri)
    features = build_feature_frame(history, config.features, rankings=rankings)
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
    rankings = ingest_rankings(start, end, settings.feature_store_uri, source, config, today)
    history, features, features_uri = compute_features(settings.feature_store_uri, config, today)

    result = FeatureRunResult(
        window_start=start,
        window_end=end,
        fetched_matches=len(fetched),
        fetched_rankings=len(rankings),
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
    start = _clamp_to_plan(start, settings, config)

    fetched_total = 0
    rankings_total = 0
    raw_uri = ""
    cursor = start
    while cursor <= end:
        chunk_end = min(cursor + dt.timedelta(days=chunk_days - 1), end)
        logger.info("backfill chunk %s..%s", cursor, chunk_end)
        chunk, raw_uri = ingest(cursor, chunk_end, settings.feature_store_uri, source, chunk_end)
        fetched_total += len(chunk)
        ranks = ingest_rankings(
            cursor, chunk_end, settings.feature_store_uri, source, config, chunk_end
        )
        rankings_total += len(ranks)
        cursor = chunk_end + dt.timedelta(days=1)

    history, features, features_uri = compute_features(settings.feature_store_uri, config, end)

    result = FeatureRunResult(
        window_start=start,
        window_end=end,
        fetched_matches=fetched_total,
        fetched_rankings=rankings_total,
        total_matches=len(history),
        feature_rows=len(features),
        labelled_rows=len(labelled(features)),
        upcoming_rows=len(unlabelled(features)),
        raw_uri=raw_uri,
        features_uri=features_uri,
    )
    logger.info("backfill done | %s", result.summary())
    return result


def _clamp_to_plan(start: dt.date, settings: Settings, config: PipelineConfig) -> dt.date:
    """On the free plan the archive only reaches back six months."""
    if settings.source != "http" or settings.api_plan.strip().lower() != "free":
        return start
    earliest = dt.datetime.now(dt.UTC).date() - dt.timedelta(
        days=config.ingestion.free_plan_history_days
    )
    if start < earliest:
        logger.warning(
            "PADEL_API_PLAN=free only reaches back to %s; backfill starts there, not at %s",
            earliest,
            start,
        )
        return earliest
    return start
