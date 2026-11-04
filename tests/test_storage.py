"""Feature store reads and writes."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

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


def test_hopsworks_uris_select_the_hopsworks_backend():
    assert storage.is_hopsworks("hopsworks://padel_predictor")
    assert not storage.is_remote("hopsworks://padel_predictor")
    assert storage.hopsworks_project_name("hopsworks://padel_predictor/") == "padel_predictor"


def test_a_malformed_hopsworks_uri_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="hopsworks://<project>"):
        storage.hopsworks_project_name("hopsworks://")


def test_every_dataset_has_a_feature_group_with_its_key_and_event_time():
    for dataset in (
        storage.RAW_MATCHES,
        storage.RAW_RANKINGS,
        storage.FEATURES,
        storage.PREDICTIONS,
    ):
        spec = storage.FEATURE_GROUPS[dataset]
        assert spec.primary_key and spec.event_time


def test_frames_are_coerced_for_hopsworks():
    frame = pd.DataFrame(
        {
            "match_id": pd.array(["m1"], dtype="string"),
            "team_a_won": pd.array([pd.NA], dtype="Int64"),
            "team_a_rank_points": pd.array([1.5], dtype="Float64"),
        }
    )
    out = storage.to_hopsworks_frame(frame, storage.PREDICTIONS, dt.date(2026, 11, 4))
    assert out["team_a_won"].dtype == "float64"
    assert out["match_id"].dtype == object
    assert out["run_date"].tolist() == ["2026-11-04"]


def test_hopsworks_writes_upsert_into_the_dataset_s_feature_group(monkeypatch):
    """The backend is driven against a stand-in feature store: no network."""
    calls = {}

    class Group:
        def insert(self, df, wait):
            calls["rows"], calls["wait"] = len(df), wait

    class Store:
        def get_or_create_feature_group(self, **kwargs):
            calls.update(kwargs)
            return Group()

    monkeypatch.setattr(storage, "_feature_store", lambda project: Store())
    uri = storage.write_snapshot(
        pd.DataFrame({"match_id": ["m1"], "played_at": [pd.Timestamp("2026-01-01", tz="UTC")]}),
        "hopsworks://padel_predictor",
        storage.RAW_MATCHES,
        dt.date(2026, 1, 1),
    )
    assert uri == "hopsworks://padel_predictor/raw_matches_v1"
    assert calls["primary_key"] == ["match_id"]
    assert calls["event_time"] == "played_at"
    assert calls["rows"] == 1 and calls["wait"] is True


def test_an_empty_hopsworks_host_logs_in_to_the_hosted_app(monkeypatch):
    """Actions passes an unset HOPSWORKS_HOST as an empty string, not as absent."""
    hopsworks = pytest.importorskip("hopsworks")
    from hopsworks_common.constants import HOSTS

    from padel_predictor.common.config import reset_caches

    calls = []

    class Login:
        def get_feature_store(self):
            return "fs"

    def fake_login(**kwargs):
        calls.append(kwargs)
        return Login()

    monkeypatch.setenv("HOPSWORKS_HOST", "")
    monkeypatch.setenv("HOPSWORKS_API_KEY", "key")
    monkeypatch.setattr(hopsworks, "login", fake_login)
    reset_caches()
    storage._feature_store.cache_clear()
    try:
        assert storage._feature_store("padel") == "fs"
    finally:
        storage._feature_store.cache_clear()
    assert calls[0]["host"] == HOSTS.SAAS_HOST
    assert calls[0]["project"] == "padel"
