"""Feature store access.

A "poor man's feature store": versioned Parquet snapshots under a single base
URI, which is either a local directory or an object-storage prefix (``gs://``,
``s3://``). Everything else in the codebase goes through this module, so
swapping in Hopsworks or Feast later means reimplementing one file.

Layout::

    <base>/raw_matches/ingest_date=YYYY-MM-DD/matches.parquet
    <base>/features/ingest_date=YYYY-MM-DD/features.parquet
    <base>/predictions/run_date=YYYY-MM-DD/predictions.parquet

Each run writes a new dated snapshot instead of overwriting, so any past state
of the store can be reproduced.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from padel_predictor.common.logging import get_logger

logger = get_logger(__name__)

RAW_MATCHES = "raw_matches"
FEATURES = "features"
PREDICTIONS = "predictions"

_DATASET_FILENAMES = {
    RAW_MATCHES: "matches.parquet",
    FEATURES: "features.parquet",
    PREDICTIONS: "predictions.parquet",
}
_PARTITION_KEYS = {
    RAW_MATCHES: "ingest_date",
    FEATURES: "ingest_date",
    PREDICTIONS: "run_date",
}


def is_remote(uri: str) -> bool:
    """True for object-storage URIs, which pandas reads through fsspec."""
    return "://" in uri and not uri.startswith("file://")


def _strip_scheme(uri: str) -> str:
    return uri[len("file://") :] if uri.startswith("file://") else uri


def join_uri(base: str, *parts: str) -> str:
    """Join URI segments without mangling the ``scheme://`` prefix."""
    cleaned = [str(p).strip("/") for p in parts if str(p).strip("/")]
    if is_remote(base):
        return "/".join([base.rstrip("/"), *cleaned])
    return str(Path(_strip_scheme(base)).joinpath(*cleaned))


def snapshot_uri(base_uri: str, dataset: str, snapshot_date: dt.date) -> str:
    """URI of one dated snapshot file of ``dataset``."""
    key = _PARTITION_KEYS[dataset]
    return join_uri(
        base_uri, dataset, f"{key}={snapshot_date.isoformat()}", _DATASET_FILENAMES[dataset]
    )


def write_snapshot(
    df: pd.DataFrame,
    base_uri: str,
    dataset: str,
    snapshot_date: dt.date | None = None,
) -> str:
    """Write ``df`` as a new dated snapshot and return the URI written to."""
    snapshot_date = snapshot_date or dt.datetime.now(dt.UTC).date()
    target = snapshot_uri(base_uri, dataset, snapshot_date)
    if not is_remote(base_uri):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    logger.info("wrote %d rows to %s", len(df), target)
    return target


def _local_snapshot_files(base_uri: str, dataset: str) -> list[Path]:
    root = Path(_strip_scheme(base_uri)) / dataset
    if not root.is_dir():
        return []
    return sorted(root.glob(f"*/{_DATASET_FILENAMES[dataset]}"))


def read_dataset(base_uri: str, dataset: str) -> pd.DataFrame:
    """Read every snapshot of ``dataset`` and concatenate them.

    Returns an empty frame when the dataset does not exist yet, so a first run
    on an empty store behaves the same as any other run.
    """
    if is_remote(base_uri):
        # fsspec expands the glob; a missing prefix raises, which we treat as empty.
        pattern = join_uri(base_uri, dataset, "*", _DATASET_FILENAMES[dataset])
        try:
            return pd.read_parquet(pattern)
        except (FileNotFoundError, OSError, ValueError):
            logger.warning("no snapshots found for %s under %s", dataset, base_uri)
            return pd.DataFrame()

    files = _local_snapshot_files(base_uri, dataset)
    if not files:
        logger.warning("no snapshots found for %s under %s", dataset, base_uri)
        return pd.DataFrame()
    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def read_latest_snapshot(base_uri: str, dataset: str) -> pd.DataFrame:
    """Read only the most recent snapshot of ``dataset``."""
    if is_remote(base_uri):
        # Without a listing API we cannot cheaply find the newest prefix, so fall
        # back to reading everything; callers deduplicate by key anyway.
        return read_dataset(base_uri, dataset)
    files = _local_snapshot_files(base_uri, dataset)
    if not files:
        logger.warning("no snapshots found for %s under %s", dataset, base_uri)
        return pd.DataFrame()
    return pd.read_parquet(files[-1])
