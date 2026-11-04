"""Feature store access.

Everything else in the codebase goes through this module. The backend is chosen
by the scheme of ``PADEL_FEATURE_STORE_URI``:

``hopsworks://<project>``
    The production feature store. Each dataset is a Hopsworks feature group
    with a primary key and an event time; writes are upserts on the primary
    key, so re-ingesting a match replaces its row instead of duplicating it.
    Hosted, so the data survives between GitHub Actions runs, which start
    with an empty disk every time.
anything else (a local path, ``gs://``, ``s3://``)
    Versioned Parquet snapshots, for offline development, the tests and a
    fresh clone without credentials.

Feature groups (all version 1):

===============  =========================  ==============
dataset          primary key                event time
===============  =========================  ==============
raw_matches      match_id                   played_at
raw_rankings     player, snapshot_date      published_at
features         match_id                   played_at
predictions      match_id, run_date         predicted_at
===============  =========================  ==============

Parquet layout::

    <base>/raw_matches/ingest_date=YYYY-MM-DD/matches.parquet
    <base>/raw_rankings/ingest_date=YYYY-MM-DD/rankings.parquet
    <base>/features/ingest_date=YYYY-MM-DD/features.parquet
    <base>/predictions/run_date=YYYY-MM-DD/predictions.parquet

Each run writes a new dated snapshot instead of overwriting, so any past state
of the store can be reproduced.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from padel_predictor.common.logging import get_logger

logger = get_logger(__name__)

HOPSWORKS_SCHEME = "hopsworks://"

RAW_MATCHES = "raw_matches"
RAW_RANKINGS = "raw_rankings"
FEATURES = "features"
PREDICTIONS = "predictions"

_DATASET_FILENAMES = {
    RAW_MATCHES: "matches.parquet",
    RAW_RANKINGS: "rankings.parquet",
    FEATURES: "features.parquet",
    PREDICTIONS: "predictions.parquet",
}
_PARTITION_KEYS = {
    RAW_MATCHES: "ingest_date",
    RAW_RANKINGS: "ingest_date",
    FEATURES: "ingest_date",
    PREDICTIONS: "run_date",
}


def is_hopsworks(uri: str) -> bool:
    """True when the feature store is a Hopsworks project."""
    return uri.startswith(HOPSWORKS_SCHEME)


def is_remote(uri: str) -> bool:
    """True for object-storage URIs, which pandas reads through fsspec."""
    return "://" in uri and not uri.startswith("file://") and not is_hopsworks(uri)


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
    if is_hopsworks(base_uri):
        return _hopsworks_write(df, base_uri, dataset, snapshot_date)
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
    if is_hopsworks(base_uri):
        return _hopsworks_read(base_uri, dataset)
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
    """Read only the most recent snapshot of ``dataset``.

    On Hopsworks the feature group *is* the latest state: every run upserts on
    the primary key.
    """
    if is_hopsworks(base_uri):
        return _hopsworks_read(base_uri, dataset)
    if is_remote(base_uri):
        # Without a listing API we cannot cheaply find the newest prefix, so fall
        # back to reading everything; callers deduplicate by key anyway.
        return read_dataset(base_uri, dataset)
    files = _local_snapshot_files(base_uri, dataset)
    if not files:
        logger.warning("no snapshots found for %s under %s", dataset, base_uri)
        return pd.DataFrame()
    return pd.read_parquet(files[-1])


# --------------------------------------------------------------------------- #
# Hopsworks
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureGroupSpec:
    name: str
    primary_key: tuple[str, ...]
    event_time: str
    description: str
    version: int = 1


FEATURE_GROUPS: dict[str, FeatureGroupSpec] = {
    RAW_MATCHES: FeatureGroupSpec(
        "raw_matches", ("match_id",), "played_at", "Matches as ingested, newest record per match"
    ),
    RAW_RANKINGS: FeatureGroupSpec(
        "raw_rankings",
        ("player", "snapshot_date"),
        "published_at",
        "Weekly ranking snapshots per player; joined on published_at",
    ),
    FEATURES: FeatureGroupSpec(
        "match_features",
        ("match_id",),
        "played_at",
        "Model-ready feature rows from common/features.py",
    ),
    PREDICTIONS: FeatureGroupSpec(
        "predictions",
        ("match_id", "run_date"),
        "predicted_at",
        "Daily predictions with the model version that served them",
    ),
}


def hopsworks_project_name(uri: str) -> str:
    """``hopsworks://padel_predictor`` -> ``padel_predictor``."""
    name = uri[len(HOPSWORKS_SCHEME) :].strip("/")
    if not name or "/" in name:
        raise ValueError(f"expected hopsworks://<project>, got {uri!r}")
    return name


@lru_cache(maxsize=4)
def _feature_store(project: str) -> Any:
    """Log in once per process and return the project's feature store."""
    try:
        import hopsworks
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hopsworks:// needs the optional dependency: uv sync --extra hopsworks"
        ) from exc

    from padel_predictor.common.config import get_settings

    settings = get_settings()
    key = settings.hopsworks_api_key.get_secret_value()
    if not key:
        raise RuntimeError("HOPSWORKS_API_KEY is not set; see .env.example")
    login = hopsworks.login(
        host=settings.hopsworks_host or None, project=project, api_key_value=key
    )
    return login.get_feature_store()


def to_hopsworks_frame(df: pd.DataFrame, dataset: str, snapshot_date: dt.date) -> pd.DataFrame:
    """Coerce a frame to types every Hopsworks engine accepts.

    Nullable pandas extension types become plain floats and objects, and the
    predictions get the ``run_date`` that completes their primary key.
    """
    out = df.copy()
    if dataset == PREDICTIONS:
        out["run_date"] = snapshot_date.isoformat()
    for column in out.columns:
        dtype = out[column].dtype
        if isinstance(dtype, pd.Int64Dtype | pd.Float64Dtype):
            out[column] = out[column].astype("float64")
        elif isinstance(dtype, pd.StringDtype):
            out[column] = out[column].astype(object)
    return out


def _hopsworks_write(df: pd.DataFrame, base_uri: str, dataset: str, snapshot_date: dt.date) -> str:
    spec = FEATURE_GROUPS[dataset]
    target = f"{base_uri.rstrip('/')}/{spec.name}_v{spec.version}"
    if df.empty:
        logger.info("nothing to write to %s", target)
        return target
    fs = _feature_store(hopsworks_project_name(base_uri))
    group = fs.get_or_create_feature_group(
        name=spec.name,
        version=spec.version,
        description=spec.description,
        primary_key=list(spec.primary_key),
        event_time=spec.event_time,
        online_enabled=False,
    )
    group.insert(to_hopsworks_frame(df, dataset, snapshot_date), wait=True)
    logger.info("upserted %d rows into %s", len(df), target)
    return target


def _hopsworks_read(base_uri: str, dataset: str) -> pd.DataFrame:
    spec = FEATURE_GROUPS[dataset]
    fs = _feature_store(hopsworks_project_name(base_uri))
    try:
        group = fs.get_feature_group(name=spec.name, version=spec.version)
    except Exception as exc:  # noqa: BLE001 -- hsfs raises its own types for "not found"
        logger.warning("feature group %s not found (%s)", spec.name, type(exc).__name__)
        return pd.DataFrame()
    if group is None:
        return pd.DataFrame()
    frame = group.read(dataframe_type="pandas")
    if dataset == PREDICTIONS:
        frame = frame.drop(columns=["run_date"], errors="ignore")
    return frame
