"""Feature store reads and writes."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from padel_predictor.common import storage


def test_missing_dataset_reads_as_empty(tmp_path):
    assert storage.read_dataset(str(tmp_path), storage.RAW_MATCHES).empty
    assert storage.read_latest_snapshot(str(tmp_path), storage.FEATURES).empty


def test_snapshots_accumulate_and_read_back(tmp_path):
    base = str(tmp_path / "store")
    storage.write_snapshot(pd.DataFrame({"a": [1]}), base, storage.RAW_MATCHES, dt.date(2026, 1, 1))
    storage.write_snapshot(pd.DataFrame({"a": [2]}), base, storage.RAW_MATCHES, dt.date(2026, 1, 2))

    combined = storage.read_dataset(base, storage.RAW_MATCHES)
    assert combined["a"].tolist() == [1, 2]

    latest = storage.read_latest_snapshot(base, storage.RAW_MATCHES)
    assert latest["a"].tolist() == [2]


def test_snapshot_path_is_partitioned_by_date(tmp_path):
    base = str(tmp_path / "store")
    written = storage.write_snapshot(
        pd.DataFrame({"a": [1]}), base, storage.PREDICTIONS, dt.date(2026, 3, 4)
    )
    assert "run_date=2026-03-04" in written
    assert Path(written).is_file()


def test_rewriting_the_same_date_replaces_that_snapshot(tmp_path):
    base = str(tmp_path / "store")
    day = dt.date(2026, 1, 1)
    storage.write_snapshot(pd.DataFrame({"a": [1]}), base, storage.FEATURES, day)
    storage.write_snapshot(pd.DataFrame({"a": [9]}), base, storage.FEATURES, day)
    assert storage.read_dataset(base, storage.FEATURES)["a"].tolist() == [9]


def test_remote_uris_are_detected_and_joined():
    assert storage.is_remote("gs://bucket/prefix")
    assert not storage.is_remote("/local/path")
    assert not storage.is_remote("file:///local/path")
    assert storage.join_uri("gs://bucket", "features", "x.parquet") == (
        "gs://bucket/features/x.parquet"
    )
