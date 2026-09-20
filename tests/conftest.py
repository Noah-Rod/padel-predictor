"""Shared fixtures.

Every test runs against a temporary feature store and a patched environment, so
the suite never touches a developer's real ``data/`` directory or MLflow server.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from padel_predictor.common.config import Settings, get_pipeline_config, reset_caches


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Point every configuration lookup at a throwaway directory."""
    monkeypatch.setenv("PADEL_SOURCE", "sample")
    monkeypatch.setenv("PADEL_FEATURE_STORE_URI", str(tmp_path / "feature_store"))
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.setenv("PADEL_API_BASE_URL", "")
    monkeypatch.setenv("PADEL_API_KEY", "")
    monkeypatch.delenv("PADEL_MODEL_ALIAS", raising=False)
    reset_caches()
    yield
    reset_caches()


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def config():
    return get_pipeline_config()


@pytest.fixture
def feature_config(config):
    return config.features


def make_match(
    match_id: str,
    played_at: str,
    team_a: tuple[str, str],
    team_b: tuple[str, str],
    winner: str | None = "A",
    tier: str = "p1",
    rank_a: float = 2000.0,
    rank_b: float = 2000.0,
) -> dict:
    """Build one raw-match record."""
    return {
        "match_id": match_id,
        "played_at": played_at,
        "tournament": "Test Open",
        "tier": tier,
        "round": "QF",
        "team_a_player_1": team_a[0],
        "team_a_player_2": team_a[1],
        "team_b_player_1": team_b[0],
        "team_b_player_2": team_b[1],
        "team_a_rank_points": rank_a,
        "team_b_rank_points": rank_b,
        "winner": winner,
    }


@pytest.fixture
def raw_matches() -> pd.DataFrame:
    """A small, fully labelled history with repeated players and pairings."""
    records = [
        make_match("m1", "2026-01-05T10:00:00Z", ("P1", "P2"), ("P3", "P4"), "A"),
        make_match("m2", "2026-01-06T10:00:00Z", ("P1", "P2"), ("P5", "P6"), "A"),
        make_match("m3", "2026-01-07T10:00:00Z", ("P3", "P4"), ("P5", "P6"), "B"),
        make_match("m4", "2026-01-08T10:00:00Z", ("P1", "P3"), ("P2", "P4"), "B"),
        make_match("m5", "2026-01-09T10:00:00Z", ("P1", "P2"), ("P3", "P4"), "A"),
    ]
    return pd.DataFrame(records)


@pytest.fixture
def many_matches() -> pd.DataFrame:
    """A longer synthetic history, enough rows for a train/test split."""
    from padel_predictor.features.sources import SampleMatchSource

    end = dt.datetime.now(dt.UTC).date()
    return SampleMatchSource().fetch(end - dt.timedelta(days=200), end - dt.timedelta(days=1))
